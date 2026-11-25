"""Reproductions of the defects the r4 table reviews found in the read lowering: each cell equals live modelx.

One model per test (a build each); the frames are cells of a locked Space ``Data``.
"""
from table_live import differences

FRAMES = '''
T1 = pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0], "n": [7, 8, 9, 10]}, index=pd.Index([1, 2, 3, 4], name="policy_id"))
T2 = pd.DataFrame({"rate": [0.01 * i for i in range(1, 13)]},
                  index=pd.MultiIndex.from_product([["a", "b", "c"], [0, 1, 2, 3]], names=["g", "t"]))
TS = pd.DataFrame({"sa": [1.0, 2.0, 3.0]}, index=pd.Index(["a", "b", "c"]))
REFS = {"T1r": T1, "T2r": T2, "TSr": TS, "use_opt": True, "names": ["a", "b"]}
RUN_REFS = {"use_opt": False}
'''
DATA = {
    'T1': ('', 'return T1r'), 'T2': ('', 'return T2r'), 'TS': ('', 'return TSr'), 'T1c': ('', 'return T1r.copy()'),
    'opt': ('', 'if not use_opt:\n    raise ValueError("no optional table")\nreturn T1r'),
}
CELLS = {
    # a view of a prefix with no row: the assignment raises as live, whether or not a read of the view runs
    'view_keyerror': ('t', 'v = data.T2().loc["zz" if policy_id == 2 else "a"]\n'
                           'return v.loc[t % 4, "rate"] if policy_id != 2 else 0.0'),
    # an edit of the frame through another local bound to it is read as it is made
    'alias_edit': ('t', 'df = data.T1c()\na = df.loc[policy_id, "sa"]\nw = df\nw.loc[policy_id, "sa"] = 5.0 + policy_id\n'
                        'return a * 0.0 + df.loc[policy_id, "sa"]'),
    # the type of a read is pandas' (a numpy scalar), not the C number the lowered read is
    'isinstance_int': ('', 'return 1.0 if isinstance(data.T1().loc[policy_id, "n"], int) else 0.0'),
    'type_name': ('', 'return type(data.T1().loc[policy_id, "sa"]).__name__'),
    # a key cell or a table cell that raises is not evaluated for a branch that is not taken
    'gcode': ('', 'if policy_id == 3:\n    raise ValueError("no group")\nreturn "abc"[policy_id - 1]'),
    'hold_key': ('t', 'if policy_id != 3:\n    return data.T2().loc[(gcode(), t % 4), "rate"]\nreturn 0.0'),
    'hold_receiver': ('t', 'if data.use_opt:\n    return data.opt().loc[policy_id, "sa"]\nreturn 0.0'),
    # a key that selects several rows at run time, on a text level, whether or not the sample reached the read
    'ts_scalar': ('', 'return data.TS().loc["a", "sa"]'),
    'list_key_sampled': ('', 'k = data.names if policy_id > 0 else "a"\nreturn float(np.sum(data.TS().loc[k, "sa"]))'),
    'list_key_unsampled': ('', 'k = data.names if policy_id > 0 else "a"\nreturn float(np.sum(data.TS().loc[k, "sa"]))'),
    # `x[k,]` is `x[(k,)]`, not `x[k]`
    's_scalar': ('', 'return data.S1()[policy_id]'),
    'comma_sampled': ('', 'return data.S1()[policy_id,]'),
    'comma_unsampled': ('', 'return data.S1()[policy_id,]'),
}
DATA['S1'] = ('', 'return T1r["sa"]')
UNSAMPLED = ('list_key_unsampled', 'comma_unsampled')

import pytest


@pytest.fixture(scope='module')
def found(tmp_path_factory):
    return differences(tmp_path_factory.mktemp('defects'), FRAMES, DATA, CELLS, unsampled=UNSAMPLED)


def _equal(found, *cells):
    assert not {cell: found[cell] for cell in cells if cell in found}


def test_dropped_view_assignment_raises_as_live(found):
    _equal(found, 'view_keyerror')


def test_edit_through_an_alias_is_read(found):
    _equal(found, 'alias_edit')


def test_type_of_a_read_is_pandas(found):
    _equal(found, 'isinstance_int', 'type_name')


def test_holds_do_not_raise_before_a_branch_not_taken(found):
    _equal(found, 'hold_key', 'hold_receiver')


def test_list_key_on_a_text_level_without_sample(found):
    _equal(found, 'ts_scalar', 'list_key_sampled', 'list_key_unsampled')


def test_trailing_comma_key(found):
    _equal(found, 's_scalar', 'comma_sampled', 'comma_unsampled')
