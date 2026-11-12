"""Native construction uses existing owners and copied primitive report storage."""
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

from modelx_helpers import pandas_inputs,typed_inputs
from modelx_helpers.immutable_inputs import Record,Table
from modelx_helpers.table_operations import row,to_pandas
from modelx_cython.tests.helper_contracts.test_table_operations import native


def frames():
    for index in [pd.RangeIndex(3,name='t'),pd.RangeIndex(9,3,-2,name='t'),
                  pd.Index([9,2,7],name='t'),pd.Index([],dtype='object',name='t'),
                  pd.RangeIndex(7,7,name='t')]:
        yield pd.DataFrame({'flow':np.arange(len(index),dtype='float64'),
                            'state':np.full(len(index),np.nan)},index=index)


@pytest.mark.parametrize('frame',list(frames()))
@pytest.mark.parametrize('with_rows',[False,True])
def test_native_report_preparation_metadata_snapshot_and_pickle(native,frame,with_rows):
    frame = frame.copy(deep=True)
    frame.columns.name='cashflow';frame.flags.allows_duplicate_labels=False
    original=frame.copy(deep=True)
    forbidden={pandas_inputs.table.__code__,Table.__init__.__code__,Record.__init__.__code__}
    calls=[]
    def profile(frame,event,arg):
        if event=='call' and frame.f_code in forbidden:
            calls.append(frame.f_code.co_name)
    sys.setprofile(profile)
    try:owned=native.prepare(pandas_inputs.table,frame,with_rows)
    finally:sys.setprofile(None)
    assert calls==[]
    assert type(owned)is Table
    for value in [owned,pickle.loads(pickle.dumps(owned))]:
        pd.testing.assert_frame_equal(to_pandas(value),original,check_exact=True)
        assert to_pandas(value).flags.allows_duplicate_labels is False
        for key in original.index:
            assert typed_inputs.lookup_float(value,key,'flow')==original.loc[key,'flow']
            if with_rows:
                selected=native.select(row,value,key)
                pd.testing.assert_series_equal(to_pandas(selected),original.loc[key],check_exact=True)
                assert type(selected['flow'])is type(original.loc[key]['flow'])
                assert type(selected._metadata[3])is type(original.loc[key].name)
        with pytest.raises(KeyError):native.select(row,value,False)if with_rows else typed_inputs.lookup_float(value,False,'flow')
    if len(frame):frame.iloc[0,0]=99.0
    frame.index.name='changed';frame.columns.name='changed'
    pd.testing.assert_frame_equal(to_pandas(owned),original,check_exact=True)


def test_replacement_keeps_constructor_keyword_layout(native):
    frame=pd.DataFrame({'flow':[1.0]});observed=[]
    def current(*,report,frame):
        observed.append((report,frame));return 'replacement'
    assert native.prepare_keywords(current,frame)=='replacement'
    assert observed==[(True,frame)]


def test_unadmitted_mixed_or_attributed_reports_keep_reference_contract(native):
    for frame in [pd.DataFrame({'age':[42],'rate':[0.1]}),pd.DataFrame({'rate':[0.1]})]:
        frame.attrs={'basis':['original']}
        actual=native.prepare(pandas_inputs.table,frame,True)
        pd.testing.assert_frame_equal(to_pandas(actual),frame,check_exact=True)
        pd.testing.assert_series_equal(to_pandas(row(actual,0)),frame.loc[0],check_exact=True)
