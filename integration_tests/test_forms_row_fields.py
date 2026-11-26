"""Fields of a row read per step without building the row: a row local, a row whose key is fixed per item
(held once per item), a row cell's field chosen at run time under ``float()``, and a key that is missing or not
one label (pandas raises or selects as live)."""
from forms_probe import compare, native_reads

FRAMES = {
    'MP': 'pd.DataFrame({"sa": [100.0, 200.0, 300.0], "age": [30, 40, 50], "cost": ["A", "B", "A"]}, '
          'index=pd.Index([1, 2, 3], name="policy_id"))',
    'CO': 'pd.DataFrame({"fee": [1.5, 2.5], "rate": [0.01, 0.02], "n": [3, 4]}, index=pd.Index(["A", "B"]))',
}
CELLS = {
    'point': ('', 'return data.MP().loc[policy_id]'),
    'cost_id': ('', 'return data.MP().loc[policy_id]["cost"]'),
    'fee': ('t', 'row = data.CO().loc[cost_id()]\nreturn row["fee"] * t + row["rate"]'),
    'per_step_row': ('t', 'row = data.CO().loc["A" if t % 2 else "B"]\nreturn row["n"] + t'),
    'chosen': ('t', 'return float(point()["sa" if t else "age"])'),
    'chosen_local': ('t', 'row = data.CO().loc[cost_id()]\nreturn float(row["rate" if t else "n"])'),
    'missing': ('t', 'return data.CO().loc["Z" if policy_id == 3 else "A"]["fee"] + t'),
}


def test_row_fields_per_step(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 2))
    assert not failures, '\n'.join(failures)
    assert native_reads(source, 'fee') == ['f64_row', 'f64_row'], source
    assert native_reads(source, 'per_step_row') == ['i64_row'], source      # an int field of a float row
    assert '_mx_cy.cast(_mx_cy.double, _mx_helper_' in source                # ... read as a float, as pandas
    assert native_reads(source, 'chosen') == ['field_f64'], source
    assert native_reads(source, 'chosen_local') == ['field_f64'], source
    assert native_reads(source, 'missing') == ['f64_row'], source
    assert native_reads(source, 'point') == [], source
