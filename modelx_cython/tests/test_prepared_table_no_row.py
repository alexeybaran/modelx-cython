"""Preparing a table decides its row dtype from the column dtypes, without building a row.

A prepared table once read ``frame.iloc[0]`` to learn whether pandas widens a row to a number dtype: a row
Series built for every table of every model. For number, bool and text columns the decision now follows the
dtypes alone. Row reads of the frame (``frame.iloc[i]``) are counted while each table is prepared: there must be none.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'


@pytest.fixture(scope='module')
def tables(tmp_path_factory):
    folder = tmp_path_factory.mktemp('prepared_no_row')
    shutil.copy(RUNTIME / '_tables.pyx', folder / '_tables.pyx')
    shutil.copy(RUNTIME / '_tables.pxd', folder / '_tables.pxd')
    (folder / 'setup.py').write_text(
        'from setuptools import setup\nfrom Cython.Build import cythonize\n'
        'setup(ext_modules=cythonize(["_tables.pyx"], '
        'compiler_directives={"freethreading_compatible": True, "language_level": 3}))\n', encoding='utf-8')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=folder, check=True,
                   stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(folder))
    try:
        import _tables
        yield _tables
    finally:
        sys.path.remove(str(folder))


COLUMNS = [
    {'q': np.array([0.5, 1.5, 2.5]), 'n': np.array([1, 2, 3], dtype='int32')},     # numbers: widened to float64
    {'q': np.array([0.5, 1.5, 2.5]), 'b': np.array([True, False, True])},          # bool beside a number: object
    {'n': np.array([1, 2, 3], dtype='int64'), 's': ['x', 'y', 'z']},                # text: object
    {'q': np.array([0.5, 1.5, 2.5])},
]


@pytest.mark.parametrize('columns', COLUMNS, ids=lambda c: '-'.join(str(v.dtype) if hasattr(v, 'dtype') else 'str'
                                                                     for v in c.values()))
def test_preparing_a_table_builds_no_row(tables, monkeypatch, columns):
    frame = pd.DataFrame(columns, index=pd.Index([10, 11, 12]))
    reads = []
    getitem = pd.core.indexing._iLocIndexer.__getitem__

    def counted(self, key):
        if self.obj is frame and isinstance(key, (int, np.integer)):      # frame.iloc[i]: a row Series
            reads.append(key)
        return getitem(self, key)

    monkeypatch.setattr(pd.core.indexing._iLocIndexer, '__getitem__', counted)
    table = tables.prepare(frame)
    monkeypatch.undo()
    assert table is not None
    assert reads == [], f'row reads frame.iloc{reads} while preparing {frame.dtypes.to_dict()}'
