"""Source matching and model-wide classification for the table-read audit."""
from types import SimpleNamespace

import libcst as cst

from modelx_cython.frame_reads import PROVIDER, FrameReads, audit_subtables
from modelx_cython.helpers import module_name
from modelx_cython.usage import CellsResolver


def test_renamed_comprehension_target_matches_sample_notes():
    cell = SimpleNamespace(name='rates', has_args=lambda: True, vector_loop_renames={'_mx_v10_ci': 'i'})
    cls = SimpleNamespace(name='_c_Projection')
    expression = 's.loc[s.index[i], "rate"]'
    observation = ('loc:table:Data.rates', ('num', 'str'), 'float', 'rate')
    resolver = SimpleNamespace(read_notes={('_c_Projection', '_f_rates'): {
        expression: {'ops': {'sub'}, 'seen': {observation}, 'count': 3}
    }})
    reads = FrameReads(resolver, cls, cell, set(), None)
    reads.lowered_expressions = [
        's.loc[s.index[_mx_v10_ci], "rate"]',
        'other.loc[8, "rate"]',
    ]
    assert reads.audit() == []
    assert reads.unsampled_lowered == 1


def test_subtable_audit_checks_readers_across_spaces():
    def cell(cls, name, **kwargs):
        return SimpleNamespace(fqname=f'm.{cls}._f_{name}', **kwargs)

    native = cell('_c_Projection', 'native', read_audit=[{'per_step': True}, {'per_step': True}])
    mixed = cell('_c_Projection', 'mixed', read_audit=[{'per_step': True}])
    receiver = cell('_c_Projection', 'receiver', read_audit=[{'per_step': False}, {'per_step': False}])
    calc = cell('_c_Projection', 'calc', read_param_readers={'native', 'mixed'},
                frame_reads=[{'receiver': 'self.receiver()', 'form': 'L5'}])
    projection = {'native': native, 'mixed': mixed, 'receiver': receiver, 'calc': calc}
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
        return self.proj.mixed(t)
""")
    audit_subtables(modules, {'m': SimpleNamespace(syntax=syntax)}, resolver)
    assert all(not entry['per_step'] and 'parameterised sub-table' in entry['reason'] for entry in native.read_audit)
    assert mixed.read_audit == [{'per_step': True}]
    assert all(not entry['per_step'] and 'sub-table' in entry['reason'] for entry in receiver.read_audit)
