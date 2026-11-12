"""A zipped producer is a resumable iterator, never a sticky generator frame."""
import gc
import itertools
import pytest
from modelx_cython.tests.helper_contracts.test_expression_streams import api, native


@pytest.mark.parametrize('strict', [False, True])
def test_all_small_lengths_and_arities(api, strict):
    for count in range(5):
        for lengths in itertools.product(range(3), repeat=count):
            sources = tuple(list(range(n)) for n in lengths)
            def collect(stream):
                values = []
                for _ in range(5):
                    try: values.append(('value', next(stream)))
                    except BaseException as error:
                        values.append(('error', type(error), error.args, error.__context__))
                return values
            assert collect(api.zip_values(sources, strict=strict)) == collect(zip(*sources, strict=strict)), lengths


def test_eager_acquisition_strict_truth_and_lazy_left_to_right_steps(api):
    events = []
    class Strict:
        def __bool__(self): events.append('strict'); return False
    class Source:
        def __init__(self, name): self.name = name
        def __iter__(self): events.append(('iter', self.name)); return self
        def __next__(self): events.append(('next', self.name)); return self.name
    stream = api.zip_values((Source('a'), Source('b')), strict=Strict())
    assert events == ['strict', ('iter', 'a'), ('iter', 'b')]
    assert iter(stream) is stream and next(stream) == ('a', 'b')
    assert events[-2:] == [('next', 'a'), ('next', 'b')]


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('failure_kind', ['stop', 'lookup'])
@pytest.mark.parametrize('position', [0, 1, 2])
def test_source_errors_and_exhaustion_can_resume(api, strict, failure_kind, position):
    def run(factory):
        events = []
        failure = StopIteration('transient') if failure_kind == 'stop' else LookupError('transient')
        class Source:
            def __init__(self, name): self.name, self.n = name, 0
            def __iter__(self): return self
            def __next__(self):
                self.n += 1; events.append((self.name, self.n))
                if self.name == position and self.n == 1: raise failure
                return (self.name, self.n)
        stream = factory(tuple(Source(n) for n in range(3)), strict=strict)
        result = []
        for _ in range(3):
            try: result.append(('value', next(stream)))
            except BaseException as error:
                result.append(('error', type(error), error.args, error is failure,
                               error.__cause__, type(error.__context__) if error.__context__ else None))
        return result, events
    assert run(api.zip_values) == run(lambda sources, **kw: zip(*sources, **kw))


def test_mutating_list_stays_exhausted_and_range_bounds_are_exact(api):
    values = [1]
    stream = api.zip_values((values, range(5)))
    assert next(stream) == (1, 0)
    with pytest.raises(StopIteration): next(stream)
    values.append(2)
    with pytest.raises(StopIteration): next(stream)
    assert list(api.zip_values((range(2**63 - 2, 2**63), range(-2, 0)))) == [(2**63 - 2, -2), (2**63 - 1, -1)]
    assert list(api.zip_values((range(2**70, 2**70 + 2), range(2)))) == [(2**70, 0), (2**70 + 1, 1)]


def test_replacement_keeps_authored_keywords(native):
    seen = []
    def replacement(*, strict, sources):
        seen.append((strict, sources)); return sources
    sources = ([1], [2])
    assert native.replaced_zip(replacement, sources, True) is sources
    assert seen == [(True, sources)]
    assert native.zipped_internal_map() == [(1,), (2,), (3,)]


def test_unsupported_source_group_is_explicit(api):
    with pytest.raises(TypeError, match='tuple of source iterables'):
        api.zip_values([[1], [2]])


@pytest.mark.parametrize('retain_first', [False, True])
@pytest.mark.parametrize('lengths', [(2, 2), (2, 1)])
@pytest.mark.parametrize('strict', [False, True])
def test_retained_row_value_lifetimes(api, retain_first, lengths, strict):
    def run(factory):
        events = []
        class Value:
            def __init__(self, label): self.label = label
            def __del__(self): events.append(self.label)
        class Source:
            def __init__(self, name, count): self.name, self.count, self.n = name, count, 0
            def __iter__(self): return self
            def __next__(self):
                if self.n == self.count: raise StopIteration
                self.n += 1
                return Value((self.name, self.n))
        stream = factory(tuple(Source(i, n) for i, n in enumerate(lengths)), strict=strict)
        snapshots = []
        held = next(stream)
        if not retain_first: del held
        snapshots.append(events.copy())
        try:
            row = next(stream)
            del row
        except (StopIteration, ValueError): pass
        snapshots.append(events.copy())
        try: next(stream)
        except (StopIteration, ValueError): pass
        snapshots.append(events.copy())
        del stream
        snapshots.append(events.copy())
        if retain_first: del held
        snapshots.append(events.copy())
        return snapshots
    assert run(api.zip_values) == run(lambda sources, **kw: zip(*sources, **kw))


def test_does_not_retain_source_group_or_distinct_iterables(api):
    events = []
    class Source:
        def __iter__(self): return iter([1, 2])
        def __del__(self): events.append('source released')
    stream = api.zip_values((Source(),))
    gc.collect()
    assert events == ['source released']
    assert list(stream) == [(1,), (2,)]


def test_iterator_aliases_and_separate_sequence_cursors(api):
    iterator = iter(range(7))
    assert list(api.zip_values((iterator, iterator))) == [(0, 1), (2, 3), (4, 5)]
    values = [1, 2]
    assert list(api.zip_values((values, values))) == [(1, 1), (2, 2)]


def test_failed_iterator_acquisition_stops_before_later_sources(api):
    events = []
    failure = LookupError('acquire')
    class Source:
        def __init__(self, name): self.name = name
        def __iter__(self):
            events.append(self.name)
            if self.name == 1: raise failure
            return iter(range(2))
    with pytest.raises(LookupError) as raised:
        api.zip_values(tuple(Source(i) for i in range(3)))
    assert raised.value is failure and events == [0, 1]
