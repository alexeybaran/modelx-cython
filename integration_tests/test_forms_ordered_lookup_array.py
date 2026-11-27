"""An ordered lookup (``searchsorted``, ``asof``) whose argument holds several values is pandas' own and returns an
array as live; the native lookup takes a single number. A C integer argument needs no test; any other is looked up
natively only while it is one label at run time."""
import subprocess
import sys
import textwrap

from forms_probe import compare, lowered

FRAMES = {
    'T': 'pd.DataFrame({"q": [0.1, 0.2, 0.3, 0.4]}, index=pd.Index([18, 19, 20, 22]))',
    'G': 'pd.DataFrame({"q": [1.0, 2.0, 3.0]}, '
         'index=pd.MultiIndex.from_product([["A"], [2**53, 2**53 + 1, 2**53 + 3]]))',
}
CELLS = {
    'many': ('t', 'x = np.array([19, 20 + t])\nreturn data.T().index.searchsorted(x).tolist()'),
    'one': ('t', 'x = 18.0 + t if policy_id else None\nreturn int(data.T().index.searchsorted(x))'),
    'below': ('t', 'x = 18.5 + t if policy_id else None\nreturn int(data.T().index.asof(x))'),
    'typed': ('t', 'return int(data.T().index.searchsorted(18 + t))'),
    'typed_key': ('t', 'return 2**53 + t if policy_id < 4 else 2**53 - 1'),
    'view_typed': ('t', 'sub = data.G().loc["A"]\n'
                       'return float(sub.loc[sub.index.asof(typed_key(t)), "q"])'),
    'view_once': ('t', 'sub = data.G().loc["A"]\n'
                      'keys = [t, t]\n'
                      'result = float(sub.loc[sub.index.asof(typed_key(keys.pop())), "q"])\n'
                      'return result, len(keys)'),
    'view_clamped_once': ('t', 'sub = data.G().loc["A"]\n'
                              'keys = [t, t]\n'
                              'result = float(sub.loc[sub.index.asof(max(typed_key(keys.pop()), 9007199254740993)), "q"])\n'
                              'return result, len(keys)'),
    'view_signed_min': ('t', 'sub = data.G().loc["A"]\n'
                            'return float(sub.loc[sub.index.asof(max(typed_key(t), -9223372036854775808)), "q"])'),
    'view_signed_max': ('t', 'sub = data.G().loc["A"]\n'
                            'return float(sub.loc[sub.index.asof(min(typed_key(t), 9223372036854775807)), "q"])'),
    'view_out_of_range': ('t', 'sub = data.G().loc["A"]\n'
                               'return float(sub.loc[sub.index.asof(max(typed_key(t), 9223372036854775808)), "q"])'),
    'view_shadowed': ('t', 'sub = data.G().loc["A"]\n'
                          'max = min\n'
                          'return float(sub.loc[sub.index.asof(max(typed_key(t), 9007199254740993)), "q"])'),
    'search_pop': ('t', 'values = [22.0, np.array([19, 20 + t])] if policy_id >= 3 else [22.0, 18.0 + t]\n'
                       'result = data.T().index.searchsorted(values.pop())\n'
                       'return len(values), str(result)'),
    'asof_pop': ('t', 'values = [22.0, 17.0 if policy_id >= 3 else 18.5 + t]\n'
                     'result = data.T().index.asof(values.pop())\n'
                     'return len(values), str(result)'),
    'int_asof_pop': ('t', 'values = [22.0, float("nan") if policy_id >= 3 else 18.5 + t]\n'
                         'try:\n'
                         '    result = int(data.T().index.asof(values.pop()))\n'
                         'except (TypeError, ValueError):\n'
                         '    result = None\n'
                         'return len(values), result'),
}


def test_an_ordered_lookup_of_several_values_is_pandas(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 3))
    assert not failures, '\n'.join(failures)
    assert lowered(reads, 'one', 'searchsorted(x)'), reads
    assert lowered(reads, 'below', 'asof(x)'), reads
    assert lowered(reads, 'typed', 'searchsorted(18 + t)'), reads
    assert lowered(reads, 'view_typed', 'sub.index.asof(self.typed_key(t))'), reads
    assert lowered(reads, 'view_once', 'sub.index.asof('), reads
    ordered_cells = {
        name.rsplit('.', 1)[-1]
        for name, entries in reads.items() if isinstance(entries, list)
        if any(isinstance(entry, dict) and entry.get('form') == 'L6' for entry in entries)
    }
    assert {'view_clamped_once', 'view_signed_min', 'view_signed_max'} <= ordered_cells, reads
    assert ordered_cells.isdisjoint({'view_out_of_range', 'view_shadowed'}), reads
    assert not lowered(reads, 'search_pop', 'searchsorted('), reads
    assert not lowered(reads, 'asof_pop', 'asof('), reads
    assert not lowered(reads, 'int_asof_pop', 'asof('), reads

    probe = textwrap.dedent('''\
        import pandas as pd
        from probe_nomx import mx_model as exported
        from probe_nomx_cy import mx_model as compiled

        base = 2**53
        a, b, c = ("A", base), ("A", base + 1), ("A", base + 3)

        def lookup(model, rows, values, duplicate_columns, name):
            fresh = type(model)()
            index = pd.MultiIndex.from_tuples(rows)
            frame = pd.DataFrame({"q": values}, index=index)
            if duplicate_columns:
                frame = pd.concat([frame, frame], axis=1)
            fresh.Data.raw_G = frame
            try:
                return getattr(fresh.P[1], name)(1)
            except Exception as error:
                return type(error).__name__

        cases = [
            ([a, ("B", base), b, c], [2.0, 100.0, 9.0, 5.0], False),
            ([c, b, a], [5.0, 9.0, 2.0], False),
            ([a, c, b], [2.0, 5.0, 9.0], False),
            ([a, b, b, c], [2.0, 9.0, 8.0, 5.0], False),
            ([a, b, c], [2.0, 9.0, 5.0], True),
        ]
        for name in ("view_once", "view_clamped_once"):
            for case in cases:
                assert lookup(compiled, *case, name) == lookup(exported, *case, name), (name, case)
            assert lookup(compiled, *cases[0], name) == (9.0, 1)
        ''')
    checked = subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
