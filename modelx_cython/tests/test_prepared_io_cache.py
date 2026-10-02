"""Verified export snapshots must preserve original changed-file loading."""
import importlib
import sys

import pandas as pd

from modelx_cython.prepared_io_cache import emit


def test_verified_io_snapshot_fresh_ownership_and_changed_file(tmp_path, monkeypatch):
    import modelx as mx
    m = mx.new_model('VerifiedIO')
    m.new_pandas('values', 'points.csv', pd.DataFrame({'amount': [1.5, -1.0]}), file_type='csv')
    package = tmp_path / 'verified_io'
    m.export(package)
    m.close()
    monkeypatch.syspath_prepend(str(tmp_path))
    receipt = emit(package)
    assert receipt and 'points.csv' in receipt['sources']
    for name in tuple(sys.modules):
        if name == 'verified_io' or name.startswith('verified_io.'):
            del sys.modules[name]
    calls = []
    original = pd.read_csv
    def read(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', read)
    module = importlib.import_module('verified_io')
    first = module.mx_model
    second = type(first)()
    assert not calls
    pd.testing.assert_frame_equal(first.values, second.values)
    assert first.values is not second.values
    first.values.iloc[0, 0] = 7.0
    assert second.values.iloc[0, 0] == 1.5
    pd.DataFrame({'amount': [8.0, -2.0]}).to_csv(package / 'points.csv')
    third = type(first)()
    assert len(calls) == 1
    assert third.values.iloc[0, 0] == 8.0
    assert second.values.iloc[0, 0] == 1.5
    # Corrupt snapshots must never bypass normal readers.
    (package / '_mx_io_snapshot.pkl').write_bytes(b'not a pickle')
    fourth = type(first)()
    assert len(calls) == 2 and fourth.values.iloc[0, 0] == 8.0
