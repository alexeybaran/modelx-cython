"""A frame edited in place through another local bound to it (an alias, a loop target) is read on pandas, as live
reads the edited frame."""
from forms_probe import compare

FRAMES = {'T1': 'pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0]}, index=pd.Index([1, 2, 3, 4], name="pid"))'}
CELLS = {'alias_edit': ('', 'df = data.T1()\na = df.loc[policy_id, "sa"]\nw = df\n'
                            'w.loc[policy_id, "sa"] = 5.0 + policy_id\nreturn df.loc[policy_id, "sa"] + a'),
         'loop_edit': ('', 'df = data.T1()\na = df.loc[policy_id, "sa"]\nfor w in (df,):\n'
                           '    w.loc[policy_id, "sa"] = 1.0 + policy_id\nreturn df.loc[policy_id, "sa"] + a')}


def test_edit_through_an_alias_is_read_as_live(tmp_path):
    _, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4))
    assert not failures, '\n'.join(failures)
