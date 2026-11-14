"""The read-only frame of a compiled table cell answers pandas reads as a DataFrame does (D101)."""
import pandas as pd
import pytest

from modelx_cython.runtime.table_cells import ReadOnlyFrame


def test_nested_tuple_reads_on_a_multiindex_and_refused_writes():
    index = pd.MultiIndex.from_tuples([('M', 1, 85), ('F', 2, 85)], names=['sex', 'cls', 'age'])
    plain = pd.DataFrame({'rate': [0.5, 0.6], 'tag': ['x', 'y']}, index=index)
    frame = ReadOnlyFrame._wrap(plain.copy(), None, 'Data.rates')
    assert frame.loc[('M', 1, 85), 'rate'] == plain.loc[('M', 1, 85), 'rate']   # pandas' _getitem_axis path
    assert frame.loc(axis=0)[('F', 2, 85)]['tag'] == 'y'
    assert frame.iloc[1, 0] == 0.6 and frame.at[('M', 1, 85), 'tag'] == 'x'
    for write in (lambda: frame.loc.__setitem__((('M', 1, 85), 'rate'), 1.0),
                  lambda: frame.loc(axis=0).__setitem__(('M', 1, 85), 1.0),
                  lambda: frame.loc._setitem_with_indexer((0, 0), 1.0, 'loc')):
        with pytest.raises(TypeError, match=r'Data\.rates\[\(\)\] = t'):
            write()
    assert frame.equals(plain)
