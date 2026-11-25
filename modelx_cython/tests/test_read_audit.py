"""Source matching and model-wide classification for the table-read audit."""
from types import SimpleNamespace

import libcst as cst

from modelx_cython.frame_reads import PROVIDER, FrameReads, audit_subtables, without_fallbacks
from modelx_cython.helpers import module_name
from modelx_cython.usage import CellsResolver


def _table_note(op, observation, count):
    return {'seen': {observation}, 'count': count, 'table_reads': {(op, observation)}, 'table_count': count}


def test_subtable_audit_checks_readers_across_spaces():
    def cell(cls, name, **kwargs):
        return SimpleNamespace(fqname=f'm.{cls}._f_{name}', **kwargs)

    native = cell('_c_Projection', 'native', read_audit=[{'per_step': True}, {'per_step': True}])
    mixed = cell('_c_Projection', 'mixed', read_audit=[{'per_step': True}])
    aliased = cell('_c_Projection', 'aliased', read_audit=[{'per_step': True}])
    receiver = cell('_c_Projection', 'receiver', read_audit=[{'per_step': False}, {'per_step': False}])
    calc = cell('_c_Projection', 'calc', read_param_readers={'native', 'mixed', 'aliased'},
                frame_reads=[{'receiver': 'self.receiver()', 'form': 'L5'}])
    projection = {'native': native, 'mixed': mixed, 'aliased': aliased, 'receiver': receiver, 'calc': calc}
    cls = SimpleNamespace(fqname='m._c_Projection', cells=projection)
    cross = SimpleNamespace(fqname='m._c_Cross', cells={})
    modules = {'m': SimpleNamespace(classes={'_c_Projection': cls, '_c_Cross': cross})}
    resolver = CellsResolver({cls.fqname: {name: value.fqname for name, value in projection.items()},
                              cross.fqname: {}},
                             {cross.fqname: {'proj': cls.fqname}}, {})
    provider = module_name(PROVIDER)
    syntax = cst.parse_module(f"""\
class _c_Projection:
    def _f_calc(self, t):
        return 0.0 if {provider}.slot(table, 0) > 0 else self.native(t).loc[t]

class _c_Cross:
    def _f_unlowered(self, t):
        proj = self.proj
        return self.proj.mixed(t) + proj.aliased(t)
""")
    audit_subtables(modules, {'m': SimpleNamespace(syntax=syntax)}, resolver)
    assert all(not entry['per_step'] and 'parameterised sub-table' in entry['reason'] for entry in native.read_audit)
    assert mixed.read_audit == [{'per_step': True}]
    assert aliased.read_audit == [{'per_step': True}]
    assert all(not entry['per_step'] and 'sub-table' in entry['reason'] for entry in receiver.read_audit)


def test_audit_keeps_the_pandas_operation_evaluated_as_a_native_key():
    cls = SimpleNamespace(name='_c_Projection')
    cell = SimpleNamespace(name='rates', has_args=lambda: True)
    key = 'source.index.get_loc(t)'
    eliminated = 'offsets.index.max()'
    expression = f'table.loc[{key} + {eliminated}, "rate"]'
    resolver = SimpleNamespace(read_notes={(cls.name, '_f_rates'): {
        key: _table_note('call', ('index:table:Data.source', 'get_loc', 'int'), 3),
        eliminated: _table_note('call', ('index:table:Data.offsets', 'max', 'int'), 3),
        expression: _table_note('sub', ('loc:table:Data.rates', ('num', 'str'), 'float', 'rate'), 3),
    }})
    reads = FrameReads(resolver, cls, cell, set(), None, builtin_int=True)
    reads.lowered_nodes = [cst.parse_expression(expression)]
    provider = module_name(PROVIDER)
    compiled = cst.parse_statement(f'return {provider}.f64_1(hold, 0, {key} + limit) '
                                   f'if {provider}.slot(hold, 0) > 0 else {expression}\n')
    assert [entry['expression'] for entry in reads.audit(without_fallbacks(compiled))] == [key]
