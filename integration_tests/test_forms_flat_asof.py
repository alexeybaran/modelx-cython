"""Read the signed asof row of a flat prepared frame, with its original pandas fallback."""
import subprocess
import sys
import textwrap

from forms_probe import compare, lowered


FRAMES = {
    'T': 'pd.DataFrame({"q": [2.0, 9.0, 5.0]}, index=pd.Index([2**53, 2**53 + 1, 2**53 + 3]))',
}
CELLS = {
    'typed_key': ('t', 'return 2**53 + t if policy_id < 4 else 2**53 - 1'),
    'flat': ('t', 'table = data.T()\n'
                  'keys = [t, t]\n'
                  'value = float(table.loc[table.index.asof(typed_key(keys.pop())), "q"])\n'
                  'return value, len(keys)'),
}


def test_flat_asof_uses_the_signed_prepared_row_once_or_preserves_pandas(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 2))
    assert not failures, '\n'.join(failures)
    assert lowered(reads, 'flat', 'table.index.asof('), reads
    source = (tmp_path / 'probe_nomx_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    assert '.source is table' in source
    assert '.asof_at_ll(' in source

    probe = textwrap.dedent('''\
        import pandas as pd
        from probe_nomx import mx_model as exported
        from probe_nomx_cy import mx_model as compiled

        class CustomFrame(pd.DataFrame):
            @property
            def loc(self):
                access = super().loc
                class Access:
                    def __getitem__(self, key):
                        return access[key] + 100
                return Access()

        class CustomIndex(pd.Index):
            def asof(self, label):
                return self[-1]

        def lookup(model, frame, policy=1):
            fresh = type(model)()
            fresh.Data.raw_T = frame
            try:
                return fresh.P[policy].flat(1)
            except Exception as error:
                return type(error).__name__

        a, b, c = 2**53, 2**53 + 1, 2**53 + 3
        ordered = pd.DataFrame({"q": [2.0, 9.0, 5.0]}, index=[a, b, c])
        descending = ordered.iloc[::-1]
        unsorted = ordered.iloc[[0, 2, 1]]
        object_column = ordered.astype({"q": "object"})
        duplicate_columns = pd.concat([ordered, ordered], axis=1)
        custom_frame = CustomFrame(ordered)
        custom_index = pd.Index([a, b, c])
        custom_index.__class__ = CustomIndex
        indexed = ordered.set_axis(custom_index)
        assert type(indexed.index) is CustomIndex

        for frame in (ordered, descending, unsorted, object_column, duplicate_columns, custom_frame, indexed):
            assert lookup(compiled, frame) == lookup(exported, frame), type(frame)
        assert lookup(compiled, ordered) == (9.0, 1)
        assert lookup(compiled, ordered, policy=4) == "KeyError"
        assert lookup(compiled, custom_frame) == (109.0, 1)
        assert lookup(compiled, indexed) == (5.0, 1)
        ''')
    checked = subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
