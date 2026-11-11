from modelx_cython import runtime as native_runtime
"""Differential contracts for explicit live/Cython iterator helpers."""
from decimal import Decimal
import importlib
import inspect
import itertools
import ast  # Test-only: parser entry is monkeypatched to fail if called.
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from modelx_helpers import expression_streams as live


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp('explicit-streams')
    provider = Path(native_runtime.__file__).parent
    for stem, name in [('_native_streams', 'stream_native')]:
        for suffix in ['pyx', 'pxd']:
            source = (provider / f'{stem}.{suffix}').read_text()
            (root / f'{name}.{suffix}').write_text(source)
    (root / 'stream_probe.pyx').write_text('''cimport stream_native as native
from modelx_helpers.expression_streams import _PERIODS, _OVER, _FLATTEN, _SUM_VALUES, _SUM_FLOAT, _ZIP_VALUES, _NO_CALLBACK

def periods(*args):
    if len(args) == 1: return native.periods(_PERIODS, (), args[0])
    if len(args) == 2: return native.periods(_PERIODS, (), args[0], args[1])
    if len(args) == 3: return native.periods(_PERIODS, (), args[0], args[1], args[2])
    return _PERIODS(*args)
def over(source, function=_NO_CALLBACK, where=None, *, unpack=None):
    return native.over(_OVER, (), source, function, where, unpack)
def flatten(sources): return native.flatten(_FLATTEN, (), sources)
def zip_values(sources, *, strict=False): return native.zip_values(_ZIP_VALUES, ('strict',), sources, strict=strict)
def replaced_zip(function, sources, strict):
    return native.zip_values(function, ('strict','sources'), strict=strict, sources=sources)
def zipped_internal_map():
    cdef native.Stream source = native.Stream()
    source.source = iter(range(3))
    source.function = lambda value: value + 1
    return list(native.zip_values(_ZIP_VALUES, (), (source,)))
def sum_values(source): return native.sum_values(_SUM_VALUES, (), source)
def sum_float(source, *args, **kwargs):
    if args:
        if kwargs: return native.sum_float(_SUM_FLOAT, ('where',), source, args[0], where=kwargs['where'])
        return native.sum_float(_SUM_FLOAT, (), source, args[0])
    if kwargs: return native.sum_float(_SUM_FLOAT, ('where',), source, native.omitted_argument, where=kwargs['where'])
    return native.sum_float(_SUM_FLOAT, (), source)
def replaced_sum(function, source, callback, predicate):
    return native.sum_float(function, ('where', 'function', 'source'), source=source,
                            function=callback, where=predicate)
def replaced_over(function, source, callback, predicate):
    return native.over(function, ('function', 'where'), source, function=callback, where=predicate)
def argument_order(function, read):
    return native.sum_float(function, ('where','source','function'),
        where=read('where'), source=read('source'), function=read('function'))
''')
    (root / 'setup.py').write_text('''from setuptools import setup, Extension
from Cython.Build import cythonize
from pathlib import Path
import modelx_cython
setup(ext_modules=cythonize([Extension(name,[name+'.pyx'],language='c++')
 for name in ['stream_native','stream_probe']],
 include_path=[str(Path(modelx_cython.__file__).resolve().parent.parent)],compiler_directives={'language_level':3,'freethreading_compatible':True}))
''')
    run = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=root,
                         capture_output=True, text=True)
    (root / 'build.log').write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    sys.path.insert(0, str(root))
    try:
        yield importlib.import_module('stream_probe')
    finally:
        sys.path.remove(str(root))
        for name in ['stream_probe', 'stream_native']:
            sys.modules.pop(name, None)


@pytest.fixture(params=['live', 'native'])
def api(request):
    return live if request.param == 'live' else request.getfixturevalue('native')


def test_creation_filter_payload_and_one_shot(api):
    events = []
    class Source:
        def __iter__(self):
            events.append('iter')
            return iter(range(4))
    def predicate(value):
        events.append(('where', value))
        return value % 2
    def payload(value):
        events.append(('body', value))
        return value * 2
    stream = api.over(Source(), payload, where=predicate)
    assert events == ['iter']
    assert iter(stream) is stream
    assert next(stream) == 2
    assert events == ['iter', ('where', 0), ('where', 1), ('body', 1)]
    assert list(stream) == [6]
    assert list(stream) == []


@pytest.mark.parametrize('location', ['source', 'payload', 'predicate'])
def test_exception_identity_and_closed_producer(api, location):
    error = LookupError('original')
    def fail(*args):
        raise error
    class Source:
        def __iter__(self): return self
        def __next__(self): raise error
    source = Source() if location == 'source' else range(2)
    stream = api.over(source, fail if location == 'payload' else lambda x: x,
                      where=fail if location == 'predicate' else None)
    with pytest.raises(LookupError) as raised:
        next(stream)
    assert raised.value is error
    assert list(stream) == []


def test_natural_exhaustion_and_body_stopiteration(api):
    class Finished(StopIteration): pass
    class Source:
        def __iter__(self): return self
        def __next__(self): raise Finished('source payload')
    with pytest.raises(StopIteration) as raised:
        next(api.over(Source(), lambda x: x))
    assert type(raised.value) is StopIteration
    assert raised.value.value is None and raised.value.__context__ is None
    original = Finished('body payload')
    def fail(value): raise original
    stream = api.over(range(1), fail)
    with pytest.raises(RuntimeError, match='generator raised StopIteration') as raised:
        next(stream)
    assert raised.value.__cause__ is original and raised.value.__context__ is original
    assert list(stream) == []


def test_acquisition_is_eager_but_callbacks_are_lazy(api):
    original = TypeError('acquire')
    class Source:
        def __iter__(self): raise original
    with pytest.raises(TypeError) as raised:
        api.over(Source(), None)
    assert raised.value is original
    assert list(api.over([], None, where=None)) == []
    with pytest.raises(TypeError): next(api.over([1], None))


def test_consumer_failure_and_short_circuit_leave_source_resumable(api):
    error = ValueError('truth')
    class BadTruth:
        def __bool__(self): raise error
    stream = api.over([BadTruth(), 2, 3], lambda value: value)
    with pytest.raises(ValueError) as raised: all(stream)
    assert raised.value is error
    assert next(stream) == 2
    stream = api.over([False, True, True], lambda value: value)
    assert all(stream) is False
    assert next(stream) is True
    stream = api.over([True, False], lambda value: value)
    assert any(stream) is True and next(stream) is False


class CustomArithmetic:
    def __radd__(self, value): return 0
    def __float__(self): return 42.0


@pytest.mark.parametrize('values', [[], [1, 2, 3], [True, False, True],
    [1e16, 1.0, -1e16], [10**18, 1.0, -1e18],
    [2**63, -2**63, 1e16, 1.0, -1e16],
    [CustomArithmetic(), 1e16, 1.0, -1e16],
    [np.float64(1e16), np.float64(1), np.float64(-1e16)],
    [Decimal('0.1'), Decimal('0.2')], ['0.1', '0.2'],
    [float('inf'), float('-inf')], [float('nan')], [-0.0]])
def test_typed_sum_and_ordinary_sum_have_distinct_exact_contracts(api, values):
    actual = api.sum_float(iter(values))
    expected = sum((float(value) for value in values), 0.0)
    assert type(actual) is float
    if math.isnan(expected): assert math.isnan(actual)
    else: assert struct.pack('d', actual) == struct.pack('d', expected)
    try:
        ordinary = sum(values)
    except TypeError:
        with pytest.raises(TypeError): api.sum_values(iter(values))
    else:
        actual = api.sum_values(iter(values))
        assert type(actual) is type(ordinary)
        if isinstance(ordinary, float) and math.isnan(ordinary): assert math.isnan(actual)
        else: assert actual == ordinary


def test_typed_conversion_failure_preserves_retained_upstream(api):
    error = StopIteration('conversion')
    class BadFloat:
        def __float__(self): raise error
    source = api.over([BadFloat(), 2.0], lambda value: value)
    with pytest.raises(RuntimeError) as raised: api.sum_float(source)
    assert raised.value.__cause__ is error
    assert next(source) == 2.0


def test_shorthand_filters_before_mapping_and_conversion(api, monkeypatch):
    events = []
    def value(t):
        events.append(('body', t))
        return t
    def eligible(t):
        events.append(('where', t))
        return t % 2
    assert api.sum_float(api.periods(4), value, where=eligible) == 4.0
    assert events == [('where', 0), ('where', 1), ('body', 1), ('where', 2), ('where', 3), ('body', 3)]
    monkeypatch.setattr(live, 'over', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('mutable public over')))
    assert api.sum_float(range(2), float) == 1.0


@pytest.mark.parametrize('bounds', [(0,), (4,), (3, -4, -2), (-4, 4, 3),
    (-(2**63), -(2**63)+3), (2**63-1, -(2**63), -(2**63)),
    (2**70, 2**70+3)])
def test_period_domains_preserve_range_limits_and_one_shot_streams(api, bounds):
    domain = api.periods(*bounds)
    assert type(domain) is range
    assert list(api.over(domain, lambda value: value)) == list(range(*bounds))
    assert list(api.over(domain, lambda value: value)) == list(range(*bounds))


def test_range_cursor_endpoint_step_cross_product(api):
    endpoints = [-(2**63), -(2**63)+1, -7, -1, 0, 1, 7, 2**63-2, 2**63-1]
    steps = [-(2**63), -(2**63)+1, -7, -1, 0, 1, 7, 2**63-2, 2**63-1]
    for start, stop, step in itertools.product(endpoints, endpoints, steps):
        if step == 0:
            with pytest.raises(ValueError, match='arg 3 must not be zero'):
                api.periods(start, stop, step)
            continue
        domain = range(start, stop, step)
        count = (max(0, (stop-start+step-1)//step) if step > 0
                 else max(0, (start-stop-step-1)//(-step)))
        stream = api.over(api.periods(start, stop, step), lambda value: value)
        expected = list(itertools.islice(domain, 9))
        assert list(itertools.islice(stream, 9)) == expected
        if count <= 9:
            assert list(stream) == []


def test_unpack_precedes_filter_and_consumes_only_required_excess(api):
    events = []
    def fields():
        for value in range(5):
            events.append(value)
            yield value
    stream = api.over([fields()], lambda a, b: a + b,
                      where=lambda a, b: events.append('filter') or True, unpack=2)
    with pytest.raises(ValueError, match='too many values to unpack'): next(stream)
    assert events == [0, 1, 2]
    assert list(stream) == []
    assert list(api.over([(1, 2), (3, 4)], lambda a, b: a*b, unpack=2)) == [2, 12]
    for size in [0, 1]:
        with pytest.raises(ValueError, match=f'expected 2, got {size}'):
            next(api.over([range(size)], lambda a, b: None, unpack=2))


def test_generic_unpack_arity_and_index_conversion_order(api):
    assert list(api.over([()], lambda: 7, unpack=0)) == [7]
    assert list(api.over([(1, 2, 3, 4)], lambda a, b, c, d: a+b+c+d, unpack=4)) == [10]
    huge = 2**100
    with pytest.raises(ValueError, match=f'expected {huge}, got 0'):
        next(api.over([iter(())], lambda *args: None, unpack=huge))
    events = []
    class Index:
        def __init__(self, name, value): self.name, self.value = name, value
        def __index__(self):
            events.append(self.name)
            return self.value
    expected = range(Index('start', 1), Index('stop', 5), Index('step', 2))
    order = list(events)
    events.clear()
    actual = api.periods(Index('start', 1), Index('stop', 5), Index('step', 2))
    assert actual == expected and events == order


def test_explicit_domain_defaults_and_noniterable_unpack_messages(api):
    missing = inspect.signature(live.periods).parameters['stop'].default
    assert api.periods(5, missing) == range(5)
    assert api.periods(1, 5, missing) == range(1, 5)
    for count in [2, 3]:
        with pytest.raises(TypeError, match='cannot unpack non-iterable int object') as raised:
            next(api.over([7], lambda *args: None, unpack=count))
        assert raised.value.__context__ is None
    error = TypeError('original iter error')
    class Source:
        def __iter__(self): raise error
    with pytest.raises(TypeError) as raised:
        next(api.over([Source()], lambda *args: None, unpack=2))
    assert raised.value is error


def test_flatten_is_lazy_and_preserves_flat_reduction_order(api):
    events = []
    def inner(g):
        events.append(g)
        return api.over([1e16, 1.0, -1e16] if g == 0 else [2.0], lambda x: x)
    stream = api.flatten(api.over(range(2), inner))
    assert events == []
    assert next(stream) == 1e16 and events == [0]
    assert list(stream) == [1.0, -1e16, 2.0] and events == [0, 1]
    assert list(stream) == []
    assert api.sum_values(api.flatten(api.over(range(2), inner))) == 3.0


def test_flatten_inner_acquisition_stop_is_body_error(api):
    error = StopIteration('inner iter')
    class Inner:
        def __iter__(self): raise error
    stream = api.flatten([Inner()])
    with pytest.raises(RuntimeError) as raised: next(stream)
    assert raised.value.__cause__ is error
    assert list(stream) == []


def test_owned_sequence_cursor_preserves_identity_and_list_mutation(api):
    marker = object()
    assert next(api.over((marker,), lambda value: value)) is marker
    values = [1, 2]
    stream = api.over(values, lambda value: value)
    assert next(stream) == 1
    values.append(3)
    assert next(stream) == 2
    del values[0]
    assert list(stream) == []
    values.append(4)
    assert list(stream) == []
    assert list(api.flatten([('死亡', '基金'), [1.5, -0.0]])) == ['死亡', '基金', 1.5, -0.0]


def test_native_replacements_preserve_result_identity_and_keyword_layout(native):
    marker, events = object(), []
    source, callback, predicate = range(2), float, bool
    def replacement(*args, **kwargs):
        events.append((args, kwargs))
        return marker
    assert native.replaced_sum(replacement, source, callback, predicate) is marker
    assert events == [((), {'where': predicate, 'function': callback, 'source': source})]
    assert list(events[0][1]) == ['where', 'function', 'source']
    events.clear()
    assert native.replaced_over(replacement, source, callback, predicate) is marker
    assert events == [((source,), {'function': callback, 'where': predicate})]
    events.clear()
    def read(name):
        events.append(name)
        return name
    assert native.argument_order(replacement, read) is marker
    assert events[:3] == ['where', 'source', 'function']


def test_native_execution_does_not_parse_an_ast(native, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('AST parsing in stream execution')
    with monkeypatch.context() as patch:
        patch.setattr(ast, 'parse', forbidden)
        total = native.sum_float(native.periods(5), float, where=bool)
        flattened = list(native.flatten(native.over(range(2), lambda n: range(n))))
    assert total == 10.0 and flattened == [0]


@pytest.mark.parametrize('filtered', [False, True])
def test_identity_traversal_creation_identity_and_release(api, filtered):
    events = []
    class Payload:
        def __del__(self): events.append('release')
    class Source:
        position = 0
        def __iter__(self): events.append('iter'); return self
        def __next__(self):
            events.append(('next', self.position))
            if self.position:
                raise StopIteration('source result')
            self.position += 1
            return Payload()
    predicate = (lambda value: events.append('where') or True) if filtered else None
    stream = api.over(Source(), where=predicate)
    assert events == ['iter']
    value = next(stream)
    assert type(value) is Payload
    del value
    assert events[-1] == 'release'  # No identity mapper or yielded local retains it.
    with pytest.raises(StopIteration) as raised: next(stream)
    assert type(raised.value) is StopIteration and raised.value.value is None
    with pytest.raises(StopIteration): next(stream)


def test_identity_traversal_owns_fresh_protocol_not_source_delegation(api):
    sent = []
    def values():
        sent.append((yield 0))
        sent.append((yield 1))
        sent.append((yield 2))
        yield 3
    source = values()
    assert next(source) == 0
    stream = api.over(source)
    with pytest.raises(TypeError, match="can't send non-None"):
        stream.send(99)
    assert next(stream) == 1
    assert stream.send(99) == 2
    assert sent == [None, None]  # Wrapper send does not forward to the source.
    stream.close()
    assert next(source) == 3
    source.close()


def test_identity_filter_order_and_exception_identity(api):
    events = []
    failure = LookupError('identity predicate')
    def predicate(value):
        events.append(value)
        if value == 3: raise failure
        return value % 2
    stream = api.over(range(5), where=predicate)
    assert events == [] and next(stream) == 1 and events == [0, 1]
    with pytest.raises(LookupError) as raised: next(stream)
    assert raised.value is failure and events == [0, 1, 2, 3]
    with pytest.raises(StopIteration): next(stream)


def test_identity_predicate_stopiteration_is_body_error(api):
    failure = StopIteration('predicate')
    def predicate(value): raise failure
    stream = api.over((1,), where=predicate)
    with pytest.raises(RuntimeError, match='generator raised StopIteration') as raised:
        next(stream)
    assert raised.value.__cause__ is failure and raised.value.__context__ is failure


def test_identity_explicit_none_stays_noncallable_and_unpack_requires_mapper(api):
    stream = api.over((1,), None)
    with pytest.raises(TypeError): next(stream)
    events = []
    class Source:
        def __iter__(self): events.append('iter'); return iter(())
    with pytest.raises(TypeError, match='unpack requires a mapping function'):
        api.over(Source(), unpack=2)
    assert events == []
    assert list(api.over(((1, 2),))) == [(1, 2)]
    assert list(api.over((1, 2), unpack=None)) == [1, 2]


def test_identity_where_only_keeps_discarded_values_lazy(api):
    events = []
    class Payload:
        def __del__(self): events.append('release')
    def source():
        yield Payload()
        assert events == ['release']
    assert list(api.over(source(), where=lambda value: False)) == []
    assert events == ['release']


def test_identity_explicit_default_marker_is_still_identity(api):
    marker = inspect.signature(live.over).parameters['function'].default
    value = object()
    assert list(api.over((value,), function=marker)) == [value]


def test_identity_failure_leaves_retained_source_resumable(api):
    source = iter((1, 2, 3))
    failure = LookupError('stop predicate')
    def predicate(value): raise failure
    stream = api.over(source, where=predicate)
    with pytest.raises(LookupError) as raised: next(stream)
    assert raised.value is failure
    assert next(source) == 2
    with pytest.raises(StopIteration): next(stream)
