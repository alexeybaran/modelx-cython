"""An item's own row read a few times per item stays a plain pandas subscript: it is not prepared.

``death_mth(life)`` reads ``model_point()["death_mth_1" if life == 1 else "death_mth_2"]``, ``model_point()``
being the item's own row. Preparing that row for native reads costs a bind per item, worth it only for a reader
called at least the prepare minimum of times per item; the sample here calls it three times per item. A build
that prepared it anyway (``bind_own`` and ``object_field`` in the generated code) paid the bind on every item.
Values equal live.
"""
from forms_probe import compare

FRAMES = {'MP': 'pd.DataFrame({"death_mth_1": [3.0, 5.0, 7.0, 9.0], "death_mth_2": [4.0, 6.0, 8.0, 10.0]}, '
                'index=pd.Index([1, 2, 3, 4], name="policy_id"))'}
CELLS = {
    'model_point': ('', 'return data.MP().loc[policy_id]'),
    'death_mth': ('life', 'return model_point()["death_mth_1" if life == 1 else "death_mth_2"]'),
}


def test_an_own_row_read_a_few_times_per_item_is_not_prepared(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4))
    assert not failures, '\n'.join(failures)
    source = (tmp_path / 'probe_nomx_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    prepared = [line.strip() for line in source.splitlines() if 'bind_own(' in line or 'object_field(' in line]
    assert not prepared, prepared
