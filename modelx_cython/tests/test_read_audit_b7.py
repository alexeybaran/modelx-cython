"""Source matching and model-wide classification for the table-read audit: ordered walks and index positions."""
from types import SimpleNamespace

import libcst as cst
import pandas as pd

from modelx_cython import read_notes
from modelx_cython.frame_reads import FrameReads

from test_read_audit import _table_note


def test_renamed_comprehension_target_matches_sample_notes():
    cell = SimpleNamespace(name='rates', has_args=lambda: True, vector_loop_renames={'_mx_v10_ci': 'i'})
    cls = SimpleNamespace(name='_c_Projection')
    expression = 's.loc[s.index[i], "rate"]'
    observation = ('loc:table:Data.rates', ('num', 'str'), 'float', 'rate')
    resolver = SimpleNamespace(read_notes={('_c_Projection', '_f_rates'): {
        expression: _table_note('sub', observation, 3),
    }})
    reads = FrameReads(resolver, cls, cell, set(), None, builtin_int=True)
    reads.lowered_nodes = [cst.parse_expression(source) for source in (
        's.loc[s.index[_mx_v10_ci], "rate"]', 'other.loc[8, "rate"]',
    )]
    assert reads.audit(cst.parse_statement('return 0.0\n')) == []
    assert reads.unsampled_lowered == 1


def test_audit_counts_only_pandas_operations_at_an_items_loop_source():
    class _c_Data:
        def series(self):
            return pd.Series([0.5, 0.25], index=[1, 2])

    cls = SimpleNamespace(name='_c_Projection')
    cell = SimpleNamespace(name='rates', has_args=lambda: True)
    site = (cls.name, '_f_rates', 'series.items()')
    with read_notes.recording('unused'):
        series = read_notes.r_call((cls.name, '_f_rates', 'self.data.series()', 'call'), _c_Data(), 'series')
        for _ in range(4):
            source = read_notes.r_call((*site, 'call'), series, 'items')
            assert list(read_notes.r_it((*site, 'it'), source)) == [(1, 0.5), (2, 0.25)]
        assert read_notes.COUNTS[(*site, 'call')] == read_notes.COUNTS[(*site, 'it')] == 4
        resolver = SimpleNamespace(read_notes=read_notes.by_text())
    reads = FrameReads(resolver, cls, cell, set(), None, builtin_int=True)
    native = cst.parse_statement('for key, rate in series.items():\n    pass\n')
    assert [(entry['expression'], entry['evaluations'], entry['pandas_operation'])
            for entry in reads.audit(native)] == [('series.items()', 4, True)]
