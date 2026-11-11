# cython: freethreading_compatible=True
"""Direct Cython helper ABI. No source, AST, expression tree or formula parser."""
import builtins
from types import BuiltinFunctionType
from cython cimport critical_section
from libc.limits cimport LLONG_MIN, LLONG_MAX
from cpython.long cimport PyLong_AsLongLong
from cpython.exc cimport PyErr_Format, PyErr_WarnEx
from cpython.version cimport PY_VERSION_HEX

cdef extern from "Python.h":
    bint PyExceptionClass_Check(object value) noexcept
    bint PyExceptionInstance_Check(object value) noexcept
    bint PyTraceBack_Check(object value) noexcept
from cpython.object cimport Py_TYPE
from modelx_helpers.expression_streams import _PERIODS, _OVER, _FLATTEN, _SUM_VALUES, _SUM_FLOAT, _FLOAT, _NO_CALLBACK, _DOMAIN_MISSING, _ZIP_VALUES
from modelx_cython._float_sum_state cimport F64Sum, f64_init, f64_add, f64_result
from modelx_cython._range_state cimport I64Range, range_init, range_next
from modelx_cython._call_layout cimport forward_call

cdef object omitted_argument = object()
cdef object _missing = omitted_argument
cdef object _sum_values = _SUM_VALUES
cdef object _builtin_sum = builtins.sum
if (type(_builtin_sum) is not BuiltinFunctionType or _builtin_sum.__name__ != 'sum'
        or _builtin_sum.__self__ is not builtins):
    _builtin_sum = None

cdef bint f64_accepts(object function):
    return (function is _sum_values or function is _builtin_sum and _builtin_sum is not None
            ) and PY_VERSION_HEX >= 0x030c0000

cdef extern from *:
    """
    static int mx_unpack_noniterable(PyObject *value) {
        return Py_TYPE(value)->tp_iter == NULL && !PySequence_Check(value);
    }
    static void mx_zip_replace(PyObject *row, Py_ssize_t position, PyObject *value) {
        PyObject *previous = PyTuple_GET_ITEM(row, position);
        Py_INCREF(value);
        PyTuple_SET_ITEM(row, position, value);
        Py_DECREF(previous);
    }
    static void mx_zip_track(PyObject *row) {
        if (!PyObject_GC_IsTracked(row)) PyObject_GC_Track(row);
    }
    static int mx_zip_unique(void *row) {
        /* Borrowed pointer is deliberate: an object-typed Cython argument
           adds a temporary INCREF and changes zip's row retention behavior. */
        #if PY_VERSION_HEX >= 0x030e0000
        return PyUnstable_Object_IsUniquelyReferenced((PyObject *)row);
        #elif defined(Py_GIL_DISABLED)
        #error "Native zip_values on free-threaded Python requires CPython 3.14 or newer"
        #else
        return Py_REFCNT((PyObject *)row) == 1;
        #endif
    }
    """
    bint mx_unpack_noniterable(object value) noexcept
    void mx_zip_replace(object row, Py_ssize_t position, object value) noexcept
    void mx_zip_track(object row) noexcept
    bint mx_zip_unique(void *row) noexcept


cdef object _replacement(object function, tuple keywords, tuple names, tuple values):
    return forward_call(function, keywords, names, values, _missing)

cdef object _unpack(object value, object count):
    cdef object first, second, third, iterator
    cdef list result
    cdef object position
    cdef bint exhausted
    cdef bint noniterable = False
    if type(value) is tuple:
        if len(value) < count:
            raise ValueError(f'not enough values to unpack (expected {count}, got {len(value)})')
        if len(value) > count:
            raise ValueError(f'too many values to unpack (expected {count})')
        return value
    try:
        iterator = iter(value)
    except TypeError:
        if count in (2, 3, 4) and mx_unpack_noniterable(value):
            noniterable = True
        else:
            raise
    if noniterable:
        # Match live Python destructuring without retaining get-iter's error
        # as exception context or invoking metaclass attribute hooks.
        PyErr_Format(TypeError, "cannot unpack non-iterable %.200s object", Py_TYPE(value).tp_name)
    result = []
    for position in range(count):
        exhausted = False
        try:
            result.append(next(iterator))
        except StopIteration:
            exhausted = True
        if exhausted:
            raise ValueError(f'not enough values to unpack (expected {count}, got {position})')
    exhausted = False
    try:
        next(iterator)
    except StopIteration:
        exhausted = True
    if not exhausted:
        raise ValueError(f'too many values to unpack (expected {count})')
    return tuple(result)


cdef object _advance(object source):
    if type(source) is Stream:
        return (<Stream>source).next_value()
    return next(source)


cdef void _capture(Stream result, object source) except *:
    cdef long long start, stop, step
    result.source = source
    if type(source) is range:
        try:
            start = PyLong_AsLongLong(source.start)
            stop = PyLong_AsLongLong(source.stop)
            step = PyLong_AsLongLong(source.step)
        except OverflowError:
            result.source = iter(source)
        else:
            range_init(&result.counter, start, stop, step)
            result.bounded = True
            result.source = None
    elif type(source) is tuple:
        result.source_kind = 1
    elif type(source) is list:
        result.source_kind = 2
    else:
        result.source = iter(source)


cdef Stream _cursor(object source):
    cdef Stream result
    if type(source) is Stream:
        return source
    result = Stream()
    result.identity = True
    _capture(result, source)
    return result


cdef class Stream:
    def __iter__(self):
        return self

    def close(self):
        self.closed = True
        self.source = self.function = self.predicate = self.current = self.inner = None

    def __next__(self):
        return self.next_value()

    cdef bint prepare_next(self) except *:
        try:
            self.pending = self.next_value()
        except StopIteration:
            return False
        return True

    cdef object take_pending(self):
        cdef object value = self.pending
        self.pending = None
        return value

    cdef object next_input(self):
        cdef long long index
        cdef object value
        if self.closed:
            raise StopIteration
        if self.bounded:
            if not range_next(&self.counter, &index):
                self.closed = True
                raise StopIteration
            return index
        if self.source_kind == 1:
            if self.position >= len(<tuple>self.source):
                self.closed = True
                self.source = None
                raise StopIteration
            value = (<tuple>self.source)[self.position]
            self.position += 1
            return value
        if self.source_kind == 2:
            if self.position >= len(<list>self.source):
                self.closed = True
                self.source = None
                raise StopIteration
            value = (<list>self.source)[self.position]
            self.position += 1
            return value
        return _advance(self.source)

    cdef object next_value(self):
        cdef long long index
        cdef object value
        cdef bint exhausted
        if self.closed:
            raise StopIteration
        if self.flattened:
            return self.next_flattened()
        while True:
            exhausted = False
            try:
                value = self.next_input()
            except StopIteration:
                exhausted = True
            except BaseException:
                self.close()
                raise
            if exhausted:
                self.close()
                raise StopIteration
            try:
                if self.identity:
                    return value if self.unpack is None else _unpack(value, self.unpack)
                self.current = value
                if self.unpack is None:
                    if self.predicate is not None and not self.predicate(value):
                        continue
                    return self.function(value)
                arguments = _unpack(value, self.unpack)
                self.current = arguments
                if self.predicate is not None and not self.predicate(*arguments):
                    continue
                return self.function(*arguments)
            except StopIteration as error:
                self.close()
                raise RuntimeError('generator raised StopIteration') from error
            except BaseException:
                self.close()
                raise

    cdef object next_flattened(self):
        cdef object source
        cdef bint exhausted
        while True:
            if self.inner is not None:
                exhausted = False
                try:
                    return _advance(self.inner)
                except StopIteration:
                    exhausted = True
                except BaseException:
                    self.close()
                    raise
                if exhausted:
                    self.inner = None
            exhausted = False
            try:
                source = self.next_input()
            except StopIteration:
                exhausted = True
            except BaseException:
                self.close()
                raise
            if exhausted:
                self.close()
                raise StopIteration
            try:
                self.inner = _cursor(source)
            except StopIteration as error:
                self.close()
                raise RuntimeError('generator raised StopIteration') from error
            except BaseException:
                self.close()
                raise




cdef class ZipValues:
    def __iter__(self):
        return self

    def __next__(self):
        with critical_section(self):
            return self.next_row()

    cdef object next_row(self):
        cdef Py_ssize_t position, other, count = len(self.cursors)
        cdef Stream cursor
        cdef tuple row
        cdef object value
        cdef bint exhausted
        if not count:
            raise StopIteration
        # Match zip's retained reusable row. Reuse only when no consumer holds
        # it; otherwise the new row is independent and the retained row stays.
        if mx_zip_unique(<void *>self.result):
            row = self.result
        else:
            row = tuple([None] * count)
        for position in range(count):
            cursor = <Stream>self.cursors[position]
            if not self.strict:
                # A source failure is not a generator body failure. Preserve
                # its identity and permit the next call to retry the iterator.
                value = cursor.next_input()
                mx_zip_replace(row, position, value)
                value = None
                continue
            exhausted = False
            try:
                value = cursor.next_input()
            except StopIteration:
                exhausted = True
            if not exhausted:
                mx_zip_replace(row, position, value)
                value = None
                continue
            if position:
                if position == 1:
                    raise ValueError('zip() argument 2 is shorter than argument 1')
                raise ValueError(f'zip() argument {position + 1} is shorter than arguments 1-{position}')
            for other in range(1, count):
                exhausted = False
                cursor = <Stream>self.cursors[other]
                try:
                    value = cursor.next_input()
                except StopIteration:
                    exhausted = True
                if not exhausted:
                    if other == 1:
                        raise ValueError('zip() argument 2 is longer than argument 1')
                    raise ValueError(f'zip() argument {other + 1} is longer than arguments 1-{other}')
            raise StopIteration
        mx_zip_track(row)
        return row


cdef object zip_values(object current_callable, tuple keywords, object sources, object strict=_missing):
    cdef ZipValues result
    cdef Stream cursor
    cdef list cursors
    cdef object source
    if current_callable is not _ZIP_VALUES:
        return _replacement(current_callable, keywords, ('sources', 'strict'), (sources, strict))
    if type(sources) is not tuple:
        raise TypeError('zip_values requires a tuple of source iterables')
    result = ZipValues()
    # zip evaluates strict truth before acquiring the first source iterator.
    result.strict = False if strict is _missing else bool(strict)
    cursors = []
    for source in <tuple>sources:
        # Always capture the public iterator. Reusing an internal mapped Stream
        # here would let raw next_input bypass its map/filter/flatten behavior.
        cursor = Stream()
        cursor.identity = True
        _capture(cursor, source)
        cursors.append(cursor)
    result.cursors = tuple(cursors)
    result.result = tuple([None] * len(cursors))
    return result



def _stream_generator(Stream cursor):
    try:
        while True:
            if not cursor.prepare_next():
                return
            yield cursor.take_pending()
    finally:
        cursor.close()




def _identity_generator(Stream cursor, object where):
    try:
        while cursor.prepare_next():
            if where(cursor.pending):
                yield cursor.take_pending()
            else:
                cursor.take_pending()
    finally:
        cursor.close()


cdef class ProtocolGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.started = False

    def __iter__(self):
        return self

    def __next__(self):
        self.started = True
        return next(self.generator)

    def send(self, value, /):
        if not self.started and value is not None and self.generator.gi_frame is not None:
            raise TypeError("can't send non-None value to a just-started generator")
        self.started = True
        return self.generator.send(value)

    def throw(self, *args, **kwargs):
        if kwargs:
            raise TypeError('generator.throw() takes no keyword arguments')
        if not args:
            raise TypeError('throw expected at least 1 argument, got 0')
        if len(args) > 3:
            raise TypeError(f'throw expected at most 3 arguments, got {len(args)}')
        if len(args) > 1:
            if PY_VERSION_HEX >= 0x030C0000:
                PyErr_WarnEx(DeprecationWarning, 'the (type, exc, tb) signature of throw() is deprecated, use the single-arg signature instead.', 1)
        if len(args) > 2 and args[2] is not None:
            if not PyTraceBack_Check(args[2]):
                raise TypeError('throw() third argument must be a traceback object')
        kind = args[0]
        if PyExceptionClass_Check(kind):
            pass
        elif PyExceptionInstance_Check(kind):
            if len(args) > 1 and args[1] is not None:
                raise TypeError('instance exception may not have a separate value')
        else:
            PyErr_Format(TypeError, "exceptions must be classes or instances deriving from BaseException, not %.200s", Py_TYPE(kind).tp_name)
        return self.generator.throw(*args)

    def close(self):
        return self.generator.close()


def _mapped_generator(Stream cursor, object function, object where, object unpack):
    if unpack is None:
        return (function(value) for value in cursor if where is None or where(value))
    if unpack == 2:
        return (function(a, b) for a, b in cursor if where is None or where(a, b))
    if unpack == 3:
        return (function(a, b, c) for a, b, c in cursor if where is None or where(a, b, c))
    if unpack == 4:
        return (function(a, b, c, d) for a, b, c, d in cursor if where is None or where(a, b, c, d))
    return (function(*arguments) for arguments in cursor if where is None or where(*arguments))


cdef object periods(object current_callable, tuple keywords, object start, object stop=_missing, object step=_missing):
    if current_callable is not _PERIODS:
        return _replacement(current_callable, keywords, ('start', 'stop', 'step'), (start, stop, step))
    if stop is _DOMAIN_MISSING:
        stop = _missing
    if step is _DOMAIN_MISSING:
        step = _missing
    if step is not _missing:
        return range(start, stop, step)
    if stop is not _missing:
        return range(start, stop)
    return range(start)


cdef object over(object current_callable, tuple keywords, object source, object function=_missing, object where=_missing, object unpack=_missing):
    cdef Stream result
    if current_callable is not _OVER:
        return _replacement(current_callable, keywords, ('source', 'function', 'where', 'unpack'),
                            (source, function, where, unpack))
    if where is _missing:
        where = None
    if unpack is _missing:
        unpack = None
    if unpack is not None and (type(unpack) is not int or unpack < 0):
        raise ValueError('unpack must be None or a nonnegative integer')
    if function is _NO_CALLBACK:
        function = _missing
    if function is _missing and unpack is not None:
        raise TypeError('unpack requires a mapping function')
    result = Stream()
    result.identity = True
    result.unpack = unpack
    _capture(result, source)
    if function is _missing:
        if where is None:
            return ProtocolGenerator(_stream_generator(result))
        return ProtocolGenerator(_identity_generator(result, where))
    return ProtocolGenerator(_mapped_generator(result, function, where, unpack))


cdef Stream _over_cursor(object source, object function, object where, object unpack=None):
    cdef Stream result
    result = Stream()
    result.function = function
    result.predicate = where
    result.unpack = unpack
    result.closed = False
    result.bounded = False
    _capture(result, source)
    return result


cdef object flatten(object current_callable, tuple keywords, object sources):
    cdef Stream result
    if current_callable is not _FLATTEN:
        return _replacement(current_callable, keywords, ('sources',), (sources,))
    result = Stream()
    _capture(result, sources)
    result.flattened = True
    return ProtocolGenerator(_stream_generator(result))


cdef object sum_values(object current_callable, tuple keywords, object source):
    if current_callable is not _sum_values:
        return _replacement(current_callable, keywords, ('source',), (source,))
    # CPython's C consumer preserves integer-prefix transitions, big integers,
    # NumPy/custom scalar dispatch and release-specific summation behavior.
    # A native Stream supplies C iteration/callbacks without a generator frame.
    return builtins.sum(source)


def provider_version():
    return 2



cdef double sum_float_kernel(Stream stream) except *:
    cdef F64Sum state
    cdef object value, converter = _FLOAT
    cdef double contribution
    f64_init(&state)
    while True:
        try:
            value = stream.next_value()
        except StopIteration:
            break
        try:
            contribution = converter(value)
        except StopIteration as error:
            # Conversion is the body of a private lazy normalization producer.
            # Do not close an externally retained upstream source on failure.
            raise RuntimeError('generator raised StopIteration') from error
        f64_add(&state, contribution)
    return f64_result(&state)


def _identity(value):
    return value


cdef object sum_float(object current_callable, tuple keywords, object source, object function=_missing, object where=_missing):
    cdef Stream stream
    if current_callable is not _SUM_FLOAT:
        return _replacement(current_callable, keywords, ('source', 'function', 'where'),
                            (source, function, where))
    if function is not _missing and function is not _NO_CALLBACK:
        stream = _over_cursor(source, function, None if where is _missing else where)
    elif where is not _missing and where is not None:
        stream = _over_cursor(source, _identity, where)
    elif type(source) is Stream:
        stream = source
    else:
        stream = _cursor(source)
    return sum_float_kernel(stream)
