"""diag-run8 F3: which row cells (``return T.loc[key]``, zero arguments) have only field reads, so their
readers read the table row natively; every escape keeps the Record path."""

import textwrap
from types import SimpleNamespace

import libcst as cst

from modelx_cython import row_cells
from modelx_cython.holds import hold_table
from modelx_cython.transformer import ModuleTransformer

HEADER = '''\
class _c_Projection:
    def _f_model_point(self):
        """A row."""
        return self.data.points().loc[self.policy_id]

    def _f_sa(self):
        return float(self.model_point()["sa"])

    def _f_sex(self):
        mp = self.model_point()
        return self.Sex[mp["sex"]] if mp["age"] > 0 else 0
'''


def analyse(extra='', other=''):
    source = HEADER + textwrap.indent(textwrap.dedent(extra), '    ') + textwrap.dedent(other)
    return row_cells.analyse(cst.parse_module(source), '_c_Projection')


def test_field_reads_direct_and_through_a_local_are_accepted():
    accepted, refused = analyse()
    assert list(accepted) == ['model_point'] and not refused
    assert cst.Module(body=[]).code_for_node(accepted['model_point']) == 'self.data.points().loc[self.policy_id]'


ESCAPES = {
    'returned': 'def _f_x(self):\n    return self.model_point()\n',
    'passed': 'def _f_x(self):\n    return len(self.model_point())\n',
    'stored and returned': 'def _f_x(self):\n    v = self.model_point()\n    return v\n',
    'iterated': 'def _f_x(self):\n    return sum(1 for f in self.model_point())\n',
    'compared': 'def _f_x(self):\n    return self.model_point() is None\n',
    'attribute': 'def _f_x(self):\n    return self.model_point().index[0]\n',
    'variable field': 'def _f_x(self, f):\n    return self.model_point()[f]\n',
    'stored field': 'def _f_x(self):\n    self.model_point()["sa"] = 1.0\n    return 0\n',
    'nested lambda': 'def _f_x(self):\n    g = lambda: self.model_point()["sa"]\n    return g()\n',
    'local assigned twice': 'def _f_x(self, t):\n    v = self.model_point()\n    v = self.model_point()\n    return v["sa"]\n',
    'local in a branch': 'def _f_x(self, t):\n    if t:\n        v = self.model_point()\n        return v["sa"]\n    return 0\n',
    'local also passed': 'def _f_x(self):\n    v = self.model_point()\n    return v["sa"] + len(v)\n',
    'bound method': 'def _f_x(self):\n    f = self.model_point\n    return f()["sa"]\n',
}


def test_a_field_read_inside_a_comprehension_is_not_an_escape():
    accepted, _ = analyse('def _f_x(self, t):\n    return [self.model_point()["sa"] for i in range(t)]\n')
    assert list(accepted) == ['model_point']


def test_each_escape_keeps_the_record_path():
    for name, extra in ESCAPES.items():
        accepted, refused = analyse(extra)
        assert not accepted and 'model_point' in refused, (name, accepted, refused)


def test_another_class_naming_the_cell_keeps_the_record_path():
    other = ('\n\nclass _c_Portfolio:\n    def _f_total(self):\n'
             '        return self.Projection[1].model_point()["sa"]\n')
    assert not analyse(other=other)[0]
    derived = ('\n\nclass _c_Derived:\n    def _f_model_point(self):\n'
               '        return self.data.other().loc[self.policy_id]\n')
    assert not analyse(other=derived)[0]


def test_only_a_plain_row_formula_is_a_row_cell():
    for formula in ('return self.data.points().loc[self.policy_id, "sa"]',     # a value, not a row
                    'return self.data.points().loc[policy_id]',                # a free name in the key
                    'return self.data.points(1).loc[self.policy_id]',          # a table Cell with arguments
                    'x = 1\n        return self.data.points().loc[self.policy_id]'):
        source = HEADER.replace('return self.data.points().loc[self.policy_id]', formula)
        accepted, _ = row_cells.analyse(cst.parse_module(source), '_c_Projection')
        assert not accepted, formula


def test_identical_table_prefix_requires_independent_generated_spaces():
    def space(name, base='_mx_sys.BaseSpace', key='self.policy_id'):
        source = HEADER.replace('_c_Projection:', name + '(' + base + '):')
        return cst.parse_module(source.replace('self.policy_id', key)).body[0]

    own = space('_c_First')
    expr = row_cells._row_expression(own.body.body[0])
    assert row_cells.independent_prefix(own, space('_c_Second'), 'model_point', expr)
    assert not row_cells.independent_prefix(own, space('_c_Derived', '_c_First'), 'model_point', expr)
    assert not row_cells.independent_prefix(own, space('_c_Second', key='self.other'), 'model_point', expr)


def test_sibling_row_fields_preserve_cross_space_escape_detection():
    own = HEADER.replace('_c_Projection:', '_c_Projection(_mx_sys.BaseSpace):')
    sibling = own[:own.index('    def _f_sex')].replace('_c_Projection', '_c_Second')

    def accepted(extra):
        module = cst.parse_module(own + '\n' + sibling + textwrap.indent(extra, '    '))
        return row_cells.analyse(module, '_c_Projection')[0]

    assert 'model_point' in accepted('')
    assert 'model_point' not in accepted('def _f_x(self):\n    return self.other.model_point()\n')
    assert 'model_point' not in accepted('def _f_x(self):\n    self = self.other\n    return self.model_point()["x"]\n')


def test_inline_replaces_only_calls_of_accepted_cells():
    accepted, _ = analyse()
    node = cst.parse_module('def _f_y(self):\n    return float(self.model_point()["sa"]) + self.other()\n')
    out = node.visit(row_cells.Inline(accepted))
    assert out.code == ('def _f_y(self):\n    return float(self.data.points().loc[self.policy_id]["sa"]) '
                        '+ self.other()\n'), out.code


def test_unused_row_assignment_is_left_as_a_public_call():
    accepted, _ = analyse()
    node = cst.parse_module(
        'def _f_y(self):\n'
        '    unused = self.model_point()\n'
        '    return self.model_point()["sa"]\n'
    )
    inline = row_cells.Inline(accepted)
    out = node.visit(inline)
    assert 'unused = self.model_point()' in out.code
    assert 'return self.data.points().loc[self.policy_id]["sa"]' in out.code
    assert not inline.assignments

def test_unrelated_native_read_cannot_admit_an_unlowered_row_field():
    accepted, _ = analyse()
    original = cst.parse_statement(
        'def _f_y(self):\n'
        '    row = self.model_point()\n'
        '    return row["sa"] + row["age"] + self.other().loc[1]["sa"]\n'
    )
    cls = SimpleNamespace()
    cells = SimpleNamespace(name='y', read_audit=['existing'])
    original_audit = cells.read_audit
    original_provider = 'existing.provider'
    resolver = SimpleNamespace(table_signatures={'existing': 0})

    class Probe:
        _resolver = resolver
        helper_providers = {original_provider}

        def _row_cells(self, cls_info):
            return accepted

        def _table_reads(self, node, cls_info, cell, keys, *, row_inline=None):
            if row_inline is not None:
                assert len(row_inline.sites) == 2
                row_inline.lowered_sites.add(next(iter(row_inline.sites)))
                cell.frame_reads = [{'form': 'L2'}, {'form': 'L2'}]
                cell.read_audit = ['discarded']
                hold_table(cls_info).field('prepared', 'discarded')
                self._resolver.table_signatures['discarded'] = 1
                self.helper_providers.add('discarded.provider')
                return node
            assert node is original
            assert not hasattr(cls_info, 'holds')
            assert cell.read_audit is original_audit
            assert not hasattr(cell, 'frame_reads')
            assert self._resolver.table_signatures == {'existing': 0}
            assert self.helper_providers == {original_provider}
            return node

    out = ModuleTransformer._table_reads_with_row_fields(Probe(), original, cls, cells, ())
    assert out is original and cells.row_fields_inlined == 0


def test_frame_lowering_failure_clears_derivation_proof():
    function = cst.parse_statement('def _f_x(self):\n    return 0\n')
    site = cst.parse_expression('self.model_point()["sa"]')
    inline = SimpleNamespace(sites={site: 'model_point'}, lowered_sites=set())

    def fail_after_marking(node):
        inline.lowered_sites.add(site)
        raise RuntimeError('injected error after the last read')

    owner = SimpleNamespace(module=SimpleNamespace(spec=SimpleNamespace(policy=object())),
                            _resolver=SimpleNamespace(table_signatures={}))
    frames = SimpleNamespace(lower=fail_after_marking, row_inline=inline, lowered=[],
                             lowered_nodes=set(), lowered_text=set(), param_readers={})
    result = ModuleTransformer._lower_frames(owner, frames, function, SimpleNamespace(),
                                             SimpleNamespace(name='x'))
    assert result is function
    assert not inline.lowered_sites
    assert 'injected error after the last read' in frames.declined[0]['reason']
