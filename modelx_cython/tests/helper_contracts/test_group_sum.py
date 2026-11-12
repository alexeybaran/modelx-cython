"""Native annual-flow grouping and exact pandas reference contracts."""
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

from modelx_helpers import pandas_inputs as inputs
from modelx_helpers.immutable_inputs import Table
from modelx_helpers.table_operations import group_sum, to_pandas
from modelx_cython.tests.helper_contracts.test_table_operations import native


def reference(frame, labels, first=(), last=()):
    if isinstance(labels,(tuple,list)):labels=pd.Index(labels)
    out=frame[[c for c in frame.columns if c not in first+last]].groupby(labels).sum()
    for name in reversed(last):out.insert(0,name,frame[name].groupby(labels).last())
    for name in reversed(first):out.insert(0,name,frame[name].groupby(labels).first())
    return out


@pytest.mark.parametrize('labels', [[2,1,2,1], [2.0,1.0,2.0,1.0],
                                    [2.0,float('nan'),2.0,1.0],
                                    [float('nan')]*4])
def test_numeric_group_labels_first_last_and_missing(native,labels):
    frame=pd.DataFrame({'flow':[1.0,np.nan,3.0,4.0],
                        'opening':[np.nan,2.0,3.0,4.0],
                        'closing':[1.0,2.0,np.nan,4.0]}, index=pd.Index([9,4,8,2],name='t'))
    frame.columns.name='field';frame.attrs={'basis':['original']}
    source=inputs.table(frame,report=True)
    expected=reference(frame,labels,('opening',),('closing',))
    for result in [group_sum(source,labels,first=('opening',),last=('closing',)),
                   native.grouped(group_sum,source,labels,('opening',),('closing',)),
                   native.grouped_keywords(group_sum,source,labels,('opening',),('closing',))]:
        for owned in [result,pickle.loads(pickle.dumps(result))]:
            actual=to_pandas(owned)
            pd.testing.assert_frame_equal(actual,expected,check_exact=True)
            assert actual.attrs==expected.attrs
            assert actual.flags.allows_duplicate_labels==expected.flags.allows_duplicate_labels


@pytest.mark.parametrize('values', [[], [1e16,1.0,-1e16,3.0],
    [1e16, *([1.0] * 128), -1e16],
    [float('inf'),1.0,2.0], [float('inf'),float('-inf'),2.0],
    [float('nan')]*3, [0.0,-0.0,0.0]])
def test_empty_cancellation_infinities_and_all_missing(native,values):
    frame=pd.DataFrame({'flow':pd.Series(values,dtype='float64')})
    labels=[1]*len(frame);source=inputs.table(frame,report=True)
    expected=reference(frame,labels)
    actual=to_pandas(native.grouped(group_sum,source,labels))
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)


def test_dropped_group_with_missing_sum_matches_pandas(native):
    frame = pd.DataFrame({'flow': [np.nan, 1e16, 1.0, -1e16, 3.0]})
    labels = [np.nan, 1.0, 1.0, 1.0, 1.0]
    source = inputs.table(frame, report=True)
    actual = to_pandas(native.grouped(group_sum, source, labels))
    pd.testing.assert_frame_equal(actual, reference(frame, labels), check_exact=True)


def test_random_order_kahan_matches_pandas_exactly(native):
    rng=np.random.default_rng(1109)
    frame=pd.DataFrame(rng.normal(size=(2000,4))*10.0**rng.integers(-4,7,size=(2000,4)),columns=['a','b','c','d'])
    labels=[int(x)for x in rng.integers(0,17,len(frame))]
    actual=to_pandas(native.grouped(group_sum,inputs.table(frame,report=True),labels,('b',),('c',)))
    pd.testing.assert_frame_equal(actual,reference(frame,labels,('b',),('c',)),check_exact=True)


@pytest.mark.parametrize('labels', [[2,1,2],[1,1,2]])
def test_admitted_group_kernel_has_no_python_or_pandas_calls(native,labels):
    source=inputs.table(pd.DataFrame({'flow':[1.0,2.0,3.0], 'state':[np.nan,2.0,3.0]}),report=True)
    forbidden={group_sum.__code__,to_pandas.__code__,inputs.table.__code__,Table.__init__.__code__}
    calls=[]
    def profile(frame,event,arg):
        if event=='call' and (frame.f_code in forbidden or frame.f_globals.get('__name__','').startswith('pandas')):
            calls.append(frame.f_code.co_name)
    sys.setprofile(profile)
    try:owned=native.grouped(group_sum,source,labels,('state',))
    finally:sys.setprofile(None)
    assert calls==[]
    pd.testing.assert_frame_equal(to_pandas(owned),reference(to_pandas(source),labels,('state',)),check_exact=True)


def test_compatibility_dtype_and_alignment_are_preserved_not_native_proof(native):
    frame=pd.DataFrame({'flow':pd.array([1,pd.NA,3],dtype='Int64'), 'state':[True,False,True]},index=[9,2,4])
    labels=pd.Series(['b','a','b'],index=[4,9,2],name='year')
    source=inputs.table(frame,report=True)
    actual=to_pandas(native.grouped(group_sum,source,labels,('state',)))
    pd.testing.assert_frame_equal(actual,reference(frame,labels,('state',)),check_exact=True)


def test_group_errors_and_current_keyword_callable(native):
    source=inputs.table(pd.DataFrame({'flow':[1.0,2.0]}),report=True)
    for function in [group_sum,lambda t,l,**kw:native.grouped(group_sum,t,l,**kw)]:
        with pytest.raises(ValueError,match='length'):function(source,[1])
        with pytest.raises(ValueError,match='distinct'):function(source,[1,1],first=('flow',),last=('flow',))
        with pytest.raises(KeyError)as error:function(source,[1,1],first=('missing',))
        assert error.value.args==('missing',)
    observed=[]
    def current(*,labels,table,last,first):
        observed.append((labels,table,last,first));return 'replaced'
    assert native.grouped_keywords(current,source,[1,1])=='replaced'
    assert observed==[([1,1],source,(),())]


def test_single_state_column_names_are_not_split_into_characters(native):
    frame=pd.DataFrame({'flow':[1.0,2.0], 'opening':[3.0,4.0], 'closing':[5.0,6.0]})
    source=inputs.table(frame,report=True)
    expected=reference(frame,[1,1],('opening',),('closing',))
    for actual in [group_sum(source,[1,1],first='opening',last='closing'),
                   native.grouped(group_sum,source,[1,1],'opening','closing')]:
        pd.testing.assert_frame_equal(to_pandas(actual),expected,check_exact=True)
