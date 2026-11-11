"""Actual generator protocol and temporary ownership after native cursor wrapping."""
import gc
import pytest
from modelx_cython.tests.helper_contracts.test_expression_streams import api, native


def test_send_throw_close_and_prestart(api):
    for before in [True, False]:
        stream=api.over(range(3),lambda value:value+1)
        if not before:
            assert next(stream)==1
            assert stream.send('ignored sent value')==2
        failure=LookupError('injected')
        with pytest.raises(LookupError) as error:stream.throw(failure)
        assert error.value is failure
        with pytest.raises(StopIteration):next(stream)
    stream=api.over(range(3),lambda value:value)
    with pytest.raises(TypeError):stream.send(1)
    assert stream.send(None)==0
    failure=StopIteration('injected stop')
    with pytest.raises(RuntimeError) as error:stream.throw(failure)
    assert error.value.__cause__ is failure and error.value.__context__ is failure
    stream=api.over(range(3),lambda value:value)
    stream.close()
    with pytest.raises(StopIteration):next(stream)


def test_yielded_payload_is_not_retained_while_suspended(api):
    events=[]
    class Value:
        def __del__(self):events.append('released')
    stream=api.over(range(3),lambda value:Value())
    value=next(stream)
    del value
    assert events==['released']
    stream.close()


def test_unpacked_bindings_survive_mutating_raw_item(api):
    events=[]
    class Value:
        def __init__(self,name):self.name=name
        def __del__(self):events.append(self.name)
    raw=[Value('first'),Value('second')]
    stream=api.over(iter([raw]),lambda a,b:1,unpack=2)
    assert next(stream)==1
    raw[:]=[None,None]
    assert events==[]
    stream.close()
    assert set(events)=={'first','second'}


def test_closing_outer_generator_does_not_forward_to_source(api):
    calls=[]
    class Source:
        def __iter__(self):return self
        def __next__(self):return 1
        def close(self):calls.append('wrong close')
    source=Source();stream=api.over(source,lambda value:value)
    assert next(stream)==1
    stream.close()
    assert calls==[] and next(source)==1


@pytest.mark.parametrize('phase', ['new', 'started', 'closed'])
def test_throw_validation_keeps_generator_state_and_warning_order(api,phase):
    import warnings
    cases=[((),{}),((123,),{}),((ValueError('one'),ValueError('two')),{}),
           ((ValueError,None,123),{}),((ValueError,None,None,None),{}),
           ((),{'typ':ValueError}),((ValueError,),{}),((ValueError('existing'),),{}),
           ((ValueError,'message'),{}),((ValueError,None,None),{})]
    def check(generator,args,keywords):
        if phase=='started':assert next(generator)==0
        elif phase=='closed':generator.close()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            try:
                value=generator.throw(*args,**keywords)
                result=('value',value)
            except BaseException as error:
                result=('error',type(error),error.args)
        return result,[(w.category,str(w.message)) for w in observed],list(generator)
    for args,keywords in cases:
        expected=check((value for value in range(3)),args,keywords)
        actual=check(api.over(range(3),lambda value:value),args,keywords)
        assert actual==expected,(phase,args,keywords,expected,actual)


def test_invalid_throw_does_not_read_user_metaclass_names(api):
    events=[]
    failure=LookupError('metaclass name hook')
    class Meta(type):
        def __getattribute__(cls,name):
            if name=='__name__':events.append(name);raise failure
            return super().__getattribute__(name)
    class Wrong(metaclass=Meta):pass
    stream=api.over(range(3),lambda value:value)
    with pytest.raises(TypeError) as error:stream.throw(Wrong())
    assert error.value.args==('exceptions must be classes or instances deriving from BaseException, not Wrong',)
    assert events==[] and list(stream)==[0,1,2]


def test_legacy_throw_warning_filters_see_consumer_module(api):
    import warnings
    def check(generator):
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            warnings.filterwarnings('error',category=DeprecationWarning,module=r'modelx_helpers\.expression_streams')
            try:generator.throw(ValueError,'message')
            except BaseException as error:result=(type(error),error.args)
        return result,[(w.category,str(w.message),w.filename,w.lineno) for w in observed],list(generator)
    expected=check((value for value in range(3)))
    actual=check(api.over(range(3),lambda value:value))
    assert actual==expected
