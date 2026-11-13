"""Full-key differential checks against the canonical packaged provider."""
import pickle
import sys
import pytest
from modelx_helpers import typed_inputs as h
from modelx_helpers.immutable_inputs import Record, Table, freeze

from .test_keyed_float_lookup_provider import native

@pytest.mark.parametrize('keys,query',[
    ([('M',42),('F',43)],('M',42)),
    ([('M',42),('F',43)],('M',43)),
    ([('M',42),('F',43)],('X',42)),
    ([('M',1),('F',2)],('M',True)),
    ([('M',True),('F',False)],('M',1)),
    ([('M',True),('F',False)],('M',True)),
])
def test_at_full_key_selection_and_boolean_semantics(native,keys,query):
    import pandas as pd
    from modelx_helpers import pandas_inputs
    frame=pd.DataFrame({'rate':[1.5,2.5]},index=pd.MultiIndex.from_tuples(keys))
    for empty in [False,True]:
        source=frame.iloc[:0] if empty else frame
        table=pandas_inputs.table(source,indexing='at')
        for restored in [table,pickle.loads(pickle.dumps(table))]:
            try:expected=float(source.at[query,'rate'])
            except KeyError:
                with pytest.raises(KeyError) as actual:native.invoke(float,h.lookup,restored,query,'rate')
                assert actual.value.args == (query,)
                with pytest.raises(KeyError) as actual:float(h.lookup(restored,query,'rate'))
                assert actual.value.args == (query,)
            else:
                assert native.invoke(float,h.lookup,restored,query,'rate')==expected
                assert float(h.lookup(restored,query,'rate'))==expected


def test_at_native_error_path_has_no_python_lookup(native):
    import pandas as pd
    from modelx_helpers import pandas_inputs
    frame=pd.DataFrame({'rate':[1.5,2.5]},index=pd.MultiIndex.from_tuples([('M',42),('F',43)]))
    table=pandas_inputs.table(frame,indexing='at');calls=[]
    forbidden = {h.lookup.__code__, Table.__getitem__.__code__, Table.__init__.__code__,
                 Record.__getitem__.__code__, freeze.__code__}
    def profile(frame,event,arg):
        if event=='call' and frame.f_code in forbidden:calls.append(frame.f_code.co_name)
    sys.setprofile(profile)
    try:
        with pytest.raises(KeyError) as error:native.invoke(float,h.lookup,table,('M',43),'rate')
        assert error.value.args == (('M', 43),)
    finally:sys.setprofile(None)
    assert calls==[]
