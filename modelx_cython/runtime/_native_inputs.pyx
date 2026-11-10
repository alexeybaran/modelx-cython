# cython: freethreading_compatible=True
"""Cython implementation of explicit immutable helpers, copied by mx2cy.

Only exact immutable Table/Record instances are specialized. Mutable inputs,
subclasses and replaced function References call the original supplied function.
Views belong to their Table and contain no formula or policy state.
"""
from array import array
from enum import IntEnum
import math
from libc.math cimport fabs, isfinite, floor
from libc.limits cimport LONG_MIN, LONG_MAX
from cpython.float cimport PyFloat_AS_DOUBLE, PyFloat_AsDouble, PyFloat_CheckExact
from cpython.long cimport PyLong_AsLong, PyLong_AsLongLongAndOverflow, PyLong_Check
from cpython.number cimport PyNumber_Index
from cpython.object cimport PyObject_IsTrue
from cpython.version cimport PY_VERSION_HEX
from cpython.ref cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport cython
from modelx_helpers.immutable_inputs import Table, Record
from modelx_helpers.typed_inputs import _FLOOR, _native_operations
from modelx_cython._call_layout cimport forward_call

cdef object _missing = object()
cdef object omitted_argument = _missing
cdef object _lookup = _native_operations['lookup']
cdef object _previous = _native_operations['previous']
cdef object _contains = _native_operations['contains']
cdef object _key_count = _native_operations['key_count']
cdef object _key_at = _native_operations['key_at']
cdef object _first_key = _native_operations['first_key']
cdef object _last_key = _native_operations['last_key']
cdef object _subtable = _native_operations['subtable']
cdef object _has_group = _native_operations['has_group']
cdef object _sum_f64 = _native_operations['sum_f64']
cdef object _sum_helper = _native_operations['sum_helper']
cdef object _any_true = _native_operations['any_true']
cdef object _all_true = _native_operations['all_true']
cdef object _max_value = _native_operations['max_value']
cdef object _prepared_interpolation = _native_operations['prepared_interpolation']


cdef class TableView:
    cdef bint pandas_at
    cdef tuple keys, values, columns, ordered
    cdef dict positions, column_positions
    cdef long long[::1] integers
    cdef long long[::1] sorted_positions
    cdef object minimum, maximum
    cdef long long first_integer, last_integer
    cdef bint integral
    cdef bint numeric
    cdef bint contiguous
    cdef Py_ssize_t width
    cdef double[::1] numbers

    def __init__(self, table):
        self.pandas_at = table._index_error is not None
        self.keys = table._keys
        self.values = table._values
        self.columns = table._columns
        self.positions = dict(zip(self.keys, range(len(self.keys))))
        self.column_positions = dict(zip(self.columns, range(len(self.columns))))
        self.ordered = table._ordered
        if self.ordered:
            self.minimum, self.maximum = min(self.keys), max(self.keys)
            self.integral = all(type(key) is int and -(1 << 63) <= key < (1 << 63)
                                for key in self.keys)
            if self.integral:
                self.integers = array('q', self.ordered)
                self.sorted_positions = array('q', [self.positions[key] for key in self.ordered])
                self.first_integer = self.minimum
                self.last_integer = self.maximum
                # Table guarantees unique keys. Compute the span as Python ints.
                self.contiguous = self.maximum - self.minimum == len(self.ordered) - 1
                self.numeric = all(type(value) is float for row in self.values for value in row)
                if self.numeric:
                    self.width = len(self.columns)
                    self.numbers = array('d', [value for row in self.values for value in row])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline Py_ssize_t right(self, long long key) noexcept:
        cdef Py_ssize_t lo = 0, hi = self.integers.shape[0], mid
        if self.contiguous:
            if key < self.first_integer:
                return 0
            if key >= self.last_integer:
                return hi
            return <Py_ssize_t>(key - self.first_integer) + 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if key < self.integers[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    cdef object value(self, Py_ssize_t row, object column):
        cdef Py_ssize_t col
        if type(column) is str:
            found = self.column_positions.get(column)
            if found is None:
                # Preserve tuple.index's exception, including its message.
                col = self.columns.index(column)
            else:
                col = found
            return self.values[row][col]
        if isinstance(column, str):
            return self.values[row][self.columns.index(column)]
        return self.values[row][column]


cdef TableView view(object table):
    cdef dict cache = table._native_cache
    result = cache.get(TableView)
    if result is None:
        result = cache.setdefault(TableView, TableView(table))
    return <TableView>result


cdef object lookup(object function, object table, object key, object column, object default=_missing):
    cdef TableView data
    if function is _lookup and type(table) is Table and table._index_error is not None:
        return indexed_lookup(table,key,column,default)
    if (function is _lookup and type(table) is Table
            and (type(column) in (str, int) or isinstance(column, IntEnum))):
        data = view(table)
        position = data.positions.get(key)
        if position is not None:
            return data.value(position, column)
        if default is not _missing:
            return default
        raise KeyError(key)
    return function(table, key, column) if default is _missing else function(table, key, column, default)


cdef object previous(object function, object table, object key, object column, object default=_missing):
    cdef TableView data
    cdef Py_ssize_t index
    if function is _previous and type(table) is Table:
        data = view(table)
        if data.integral and type(key) is int and -(1 << 63) <= key < (1 << 63):
            index = data.right(key) - 1
            if index >= 0:
                return data.value(data.sorted_positions[index], column)
            if default is not _missing:
                return default
            raise ValueError(f'No key at or before {key!r}')
    return function(table, key, column) if default is _missing else function(table, key, column, default)


cdef object contains(object function, object table, object key):
    if function is _contains and type(table) is Table and table._index_error is not None:
        return indexed_contains(table,key)
    if function is _contains and type(table) is Table:
        return key in view(table).positions
    return function(table, key)


cdef object key_count(object function, object table):
    if function is _key_count and type(table) is Table:
        return len(view(table).keys)
    return function(table)


cdef object key_at(object function, object table, object index):
    if function is _key_at and type(table) is Table:
        if index < 0:
            raise IndexError(index)
        return view(table).keys[index]
    return function(table, index)


cdef object first_key(object function, object table):
    cdef TableView data
    if function is _first_key and type(table) is Table:
        data = view(table)
        if data.ordered:
            return data.minimum
    return function(table)


cdef object last_key(object function, object table):
    cdef TableView data
    if function is _last_key and type(table) is Table:
        data = view(table)
        if data.ordered:
            return data.maximum
    return function(table)


cdef object subtable(object function, object table, object key):
    if function is _subtable and type(table) is Record:
        return table._index[key]
    return function(table, key)


cdef object has_group(object function, object table, object key):
    if function is _has_group and type(table) is Record:
        return key in table._index
    return function(table, key)


cdef object sum_f64(object function, object values):
    if function is _sum_f64:
        return sum(values, 0.0)
    return function(values)


cdef object any_true(object function, object values):
    if function is _any_true:
        return any(values)
    return function(values)


cdef bint any_true_accepts(object function):
    return function is _any_true


cdef object all_true(object function, object values):
    if function is _all_true:
        return all(values)
    return function(values)


cdef bint all_true_accepts(object function):
    return function is _all_true


cdef object max_value(object function, object values):
    if function is _max_value:
        return max(values)
    return function(values)


cdef bint max_value_accepts(object function):
    return function is _max_value


cdef class LookupContext:
    def __reduce__(self):
        # Prepared resources are disposable and may contain memoryviews. Cell
        # results remain in their Space caches; rebuild resources on demand.
        return (LookupContext, ())

    def _mx_enable_private_lookup(self):
        # Called before the worker-exclusive collect receiver starts evaluating.
        self._private_owner = True

    cdef object get(self, object table, object column, Py_ssize_t* index):
        cdef TableView data
        cdef object retired_table, retired_data
        if self._private_owner:
            if table is self.first and column is self.column_first:
                index[0] = self.index_first
                return self.data_first
            if table is self.second and column is self.column_second:
                index[0] = self.index_second
                return self.data_second
        with cython.critical_section(self):
            if table is self.first and column is self.column_first:
                index[0] = self.index_first
                return self.data_first
            if table is self.second and column is self.column_second:
                index[0] = self.index_second
                return self.data_second
            if table is self.first:
                data = self.data_first
            elif table is self.second:
                data = self.data_second
            elif type(table) is Table:
                data = view(table)
            else:
                return None
            if data is not None and data.numeric:
                index[0] = numeric_column(data, column)
                if index[0] >= 0:
                    # Drop evicted references after the critical section.
                    retired_table, retired_data = self.second, self.data_second
                    self.second, self.data_second = self.first, self.data_first
                    self.column_second, self.index_second = self.column_first, self.index_first
                    self.first, self.data_first = table, data
                    self.column_first, self.index_first = column, index[0]
            return data


cdef Py_ssize_t numeric_column(TableView data, object column) except? -1:
    cdef Py_ssize_t index
    if type(column) is str:
        found = data.column_positions.get(column)
        if found is None:
            return -1
        return found
    if type(column) is int and -data.width <= column < data.width:
        index = column
        return index if index >= 0 else data.width + index
    return -1


cdef object prepared_interpolation(object function, object table, object column, object years,
                                   object lookup_fn, object last_fn, object math_module):
    cdef TableView data
    cdef long long col, lower, upper
    cdef Py_ssize_t lo, hi
    cdef int overflow
    cdef double position, base, end
    if function is not _prepared_interpolation:
        return function(table, column, years, lookup_fn, last_fn, math_module)
    if (lookup_fn is not _lookup or last_fn is not _last_key
            or math_module is not math or math.floor is not _FLOOR
            or type(table) is not Table or not PyLong_Check(column) or not PyFloat_CheckExact(years)):
        return None
    position = PyFloat_AS_DOUBLE(years)
    if not -9223372036854775808.0 <= position < 9223372036854775808.0:
        return None
    col = PyLong_AsLongLongAndOverflow(column, &overflow)
    data = view(table)
    if overflow or not data.numeric or not 0 <= col < data.width:
        return None
    lower = <long long>floor(position)
    upper = data.maximum
    if lower >= upper:
        hi = data.sorted_positions[data.integers.shape[0] - 1]
        return data.numbers[hi * data.width + col]
    lo = data.right(lower) - 1
    if lo < 0 or data.integers[lo] != lower:
        return None
    hi = lo + 1
    if data.integers[hi] != lower + 1:
        return None
    base = data.numbers[data.sorted_positions[lo] * data.width + col]
    end = data.numbers[data.sorted_positions[hi] * data.width + col]
    return base + (end - base) * (position - lower)


cdef double lookup_f64(object function, LookupContext context, object table,
                       long long key, object column, object default=_missing) except? -1:
    cdef TableView data
    cdef Py_ssize_t index, col
    if function is _lookup:
        col = -1
        data = context.get(table, column, &col)
        if data is not None and data.numeric:
            index = data.right(key) - 1
            if index < 0 or data.integers[index] != key:
                if default is _missing:
                    raise KeyError(key)
                return default
            if index >= 0 and data.integers[index] == key:
                if col >= 0:
                    return data.numbers[data.sorted_positions[index] * data.width + col]
    return lookup(function, table, key, column) if default is _missing else lookup(function, table, key, column, default)


cdef double previous_f64(object function, LookupContext context, object table,
                         long long key, object column, object default=_missing) except? -1:
    cdef TableView data
    cdef Py_ssize_t index, col
    if function is _previous:
        col = -1
        data = context.get(table, column, &col)
        if data is not None and data.numeric:
            index = data.right(key) - 1
            if index < 0:
                if default is _missing:
                    raise ValueError(f'No key at or before {key!r}')
                return default
            if index >= 0:
                if col >= 0:
                    return data.numbers[data.sorted_positions[index] * data.width + col]
    return previous(function, table, key, column) if default is _missing else previous(function, table, key, column, default)


cdef inline double _lookup_clamped_f64(object function, LookupContext context, object table,
                                       long long key, object last_function, object column) except? -1:
    cdef TableView data = None
    cdef Py_ssize_t index, col
    cdef long long upper
    if function is _lookup and last_function is _last_key:
        col = -1
        if context is not None:
            data = context.get(table, column, &col)
        elif type(table) is Table:
            data = view(table)
        if data is not None and data.numeric:
            upper = data.last_integer
            if upper < key:
                key = upper
            index = data.right(key) - 1
            if index < 0 or data.integers[index] != key:
                raise KeyError(key)
            if context is None:
                col = numeric_column(data, column)
            if col >= 0:
                return data.numbers[data.sorted_positions[index] * data.width + col]
    return lookup(function, table, min(key, last_key(last_function, table)), column)


cdef double lookup_clamped_f64(object function, object table, long long key,
                               object last_function, object column) except? -1:
    return _lookup_clamped_f64(function, None, table, key, last_function, column)


cdef double lookup_clamped_context_f64(object function, LookupContext context, object table,
                                       long long key, object last_function, object column) except? -1:
    return _lookup_clamped_f64(function, context, table, key, last_function, column)


cdef bint bounded_lookup_f64(object function, object first_fn, object last_fn,
                             LookupContext context, object table, long long key,
                             str column, double below, double* value) except? -1:
    cdef TableView data
    cdef Py_ssize_t col = -1, row, count
    if function is not _lookup or first_fn is not _first_key or last_fn is not _last_key:
        return False
    data = context.get(table, column, &col)
    if data is None or col < 0:
        return False
    count = data.integers.shape[0]
    if not count:
        return False
    if context._private_owner and context.bounded_data is None:
        context.bounded_data = data
        context.bounded_column = col
    if key < data.integers[0]:
        value[0] = below
        return True
    if key >= data.integers[count - 1]:
        row = count - 1
    else:
        row = data.right(key) - 1
        if row < 0 or data.integers[row] != key:
            return False
    value[0] = data.numbers[data.sorted_positions[row] * data.width + col]
    return True


cdef int optional_lookup_f64(object contains_fn, object lookup_fn, object table,
                             long long key, object column, double* result) except? -1:
    """Return missing, found, or unsupported without invoking a replacement."""
    cdef TableView data
    cdef Py_ssize_t index, col
    if contains_fn is _contains and lookup_fn is _lookup and type(table) is Table:
        data = view(table)
        if data.numeric:
            index = data.right(key) - 1
            if index < 0 or data.integers[index] != key:
                return 0
            col = numeric_column(data, column)
            if col >= 0:
                result[0] = data.numbers[data.sorted_positions[index] * data.width + col]
                return 1
    return 2


cdef class GroupedView:
    cdef long long[::1] groups, keys, offsets
    cdef double[::1] values

    def __init__(self, groups, keys, offsets, values):
        self.groups = array('q', groups)
        self.keys = array('q', keys)
        self.offsets = array('q', offsets)
        self.values = array('d', values)


cdef object prepare_group(object record, str column):
    groups, keys, offsets, values = [], [], [0], []
    for group, table in record._items:
        kind = type(group)
        if kind is not int and not (
                type(kind) is type(IntEnum) and issubclass(kind, IntEnum)
                and kind.__eq__ is int.__eq__ and kind.__hash__ is int.__hash__
                and kind.__index__ is int.__index__):
            return None
        number = int.__index__(group)
        if not -(1 << 63) <= number < (1 << 63) or type(table) is not Table:
            return None
        if column not in table._columns or not all(
                type(key) is int and -(1 << 63) <= key < (1 << 63) for key in table._keys):
            return None
        col = table._columns.index(column)
        if not all(type(row[col]) is float for row in table._values):
            return None
        groups.append(number)
        for key, row in sorted(zip(table._keys, table._values)):
            keys.append(key)
            values.append(row[col])
        offsets.append(len(keys))
    return GroupedView(groups, keys, offsets, values)


cdef object group_view(object record, str column):
    cdef dict cache = record._native_groups
    prepared = cache.get(column, _missing)
    if prepared is _missing:
        prepared = cache.setdefault(column, prepare_group(record, column))
    return prepared


cdef Py_ssize_t group_slot(GroupedView data, long long group) except? -2:
    cdef Py_ssize_t i
    for i in range(data.groups.shape[0]):
        if data.groups[i] == group:
            return i
    return -1


cdef inline object probe_group(object prepared, long long group, Py_ssize_t* slot):
    slot[0] = -2
    if prepared is None:
        return None
    slot[0] = group_slot(<GroupedView>prepared, group)
    return prepared


cdef object group_probe(object has_fn, object record, long long group, str column,
                        Py_ssize_t* slot):
    if has_fn is not _has_group or type(record) is not Record:
        slot[0] = -2
        return None
    return probe_group(group_view(record, column), group, slot)


cdef object group_context_probe(LookupContext context, object has_fn, object record,
                                long long group, str column, Py_ssize_t* slot):
    # The compiler emits this only on the exact, unlocked Space path.
    # Keep the captured view alive across year evaluation and Reference changes.
    cdef tuple binding
    cdef object prepared
    if has_fn is not _has_group:
        slot[0] = -2
        return None
    binding = context.grouped
    if binding is not None and binding[0] is record and binding[1] is column:
        return probe_group(binding[2], group, slot)
    if type(record) is not Record:
        slot[0] = -2
        return None
    prepared = group_view(record, column)
    context.grouped = (record, column, prepared)
    return probe_group(prepared, group, slot)


cdef int group_data_value(GroupedView data, Py_ssize_t slot,
                           long long key, double* value) except -1:
    cdef Py_ssize_t lo, hi, mid
    lo, hi = data.offsets[slot], data.offsets[slot + 1]
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if key == data.keys[mid]:
            value[0] = data.values[mid]
            return 1
        if key < data.keys[mid]:
            hi = mid
        else:
            lo = mid + 1
    return 0


cdef int group_value(object sub_fn, object contains_fn, object lookup_fn, object prepared,
                     Py_ssize_t slot, long long key, double* value) except -1:
    if sub_fn is not _subtable or contains_fn is not _contains or lookup_fn is not _lookup:
        return 2
    return group_data_value(<GroupedView>prepared, slot, key, value)


cdef int group_context_value(LookupContext context, object sub_fn, object contains_fn,
    object lookup_fn, object prepared, Py_ssize_t slot, long long key, double* value) except -1:
    if sub_fn is not _subtable or contains_fn is not _contains or lookup_fn is not _lookup:
        return 2
    if (context._private_owner and context.group_data is None
            and context.grouped is not None and context.grouped[2] is prepared):
        context.group_data = prepared
    return group_data_value(<GroupedView>prepared, slot, key, value)


cdef bint bound_group_probe(LookupContext context, long long group, Py_ssize_t* slot) except? -1:
    cdef PyObject* data = <PyObject*>context.group_data
    if data == <PyObject*>None:
        return False
    slot[0] = group_slot(<GroupedView>data, group)
    return True


cdef int bound_group_value(LookupContext context, Py_ssize_t slot,
                           long long key, double* value) except -1:
    # Caller obtained this slot from the same admitted context before key evaluation.
    cdef PyObject* data = <PyObject*>context.group_data
    return group_data_value(<GroupedView>data, slot, key, value)


cdef bint f64_accepts(object function):
    return function is _sum_f64 and PY_VERSION_HEX >= 0x030c0000


cdef object sum_helper(object function, object values):
    cdef double total = 0.0, correction = 0.0, value, following
    if function is not _sum_helper:
        return function(values)
    # This test implementation qualifies the 3.13 float-start sum contract.
    # Other interpreter versions retain their own builtin's behavior.
    if PY_VERSION_HEX >> 16 != 0x030d:
        return sum(values, 0.0)
    iterator = iter(values)
    for item in iterator:
        if type(item) is float:
            value = item
            following = total + value
            if fabs(total) >= fabs(value):
                correction += (total - following) + value
            else:
                correction += (value - following) + total
            total = following
        elif isinstance(item, int) and LONG_MIN <= item <= LONG_MAX:
            total += <double>PyLong_AsLong(item)
        else:
            if correction != 0.0 and isfinite(correction):
                total += correction
            result = total + item
            for item in iterator:
                result = result + item
            return result
    if correction != 0.0 and isfinite(correction):
        total += correction
    return total


cdef inline void remember_numeric_resource(LookupContext context, object source,
    object column, TableView data, Py_ssize_t index):
    if context._private_owner and data is not None and data.numeric and index >= 0:
        context.numeric_source = source
        context.numeric_column_source = column
        context.numeric_index = index
        context.numeric_data = data


cdef bint numeric_resource_ready(LookupContext context, const void* source, const void* column) noexcept:
    return (context._private_owner and context.numeric_data is not None
            and source == <const void*>context.numeric_source
            and (column == NULL or column == <const void*>context.numeric_column_source))


cdef inline double numeric_resource_value(LookupContext context, long long key, long long column,
    bint named, bint preceding, bint has_default, double default) except? -1:
    # The ready predicate proves lifetime/admission. Context retains the source
    # and immutable view; this borrowed pointer does not own another shared ref.
    cdef PyObject* data = <PyObject*>context.numeric_data
    cdef Py_ssize_t row = (<TableView>data).right(key) - 1
    cdef long long col
    if row < 0 or (not preceding and (<TableView>data).integers[row] != key):
        if has_default:
            return default
        if preceding:
            raise ValueError(f'No key at or before {key!r}')
        raise KeyError(key)
    col = context.numeric_index if named else column
    if not named and col < 0:
        col += (<TableView>data).width
    if col < 0 or col >= (<TableView>data).width:
        raise IndexError('tuple index out of range')
    return (<TableView>data).numbers[(<TableView>data).sorted_positions[row] * (<TableView>data).width + col]


cdef double bound_lookup_f64(LookupContext context, long long key, long long column,
                             bint named, bint has_default, double default) except? -1:
    return numeric_resource_value(context,key,column,named,False,has_default,default)


cdef double bound_previous_f64(LookupContext context, long long key, long long column,
                               bint named, bint has_default, double default) except? -1:
    return numeric_resource_value(context,key,column,named,True,has_default,default)

cdef bint bound_bounded_f64(LookupContext context, long long key,
                            double below, double* value) except? -1:
    # Context owns this immutable snapshot. No owning local view is needed.
    cdef PyObject* data = <PyObject*>context.bounded_data
    cdef Py_ssize_t row, count
    if data == <PyObject*>None:
        return False
    count = (<TableView>data).integers.shape[0]
    if key < (<TableView>data).integers[0]:
        value[0] = below
        return True
    if key >= (<TableView>data).integers[count - 1]:
        row = count - 1
    else:
        row = (<TableView>data).right(key) - 1
        if row < 0 or (<TableView>data).integers[row] != key:
            return False
    value[0] = (<TableView>data).numbers[
        (<TableView>data).sorted_positions[row] * (<TableView>data).width + context.bounded_column]
    return True



cdef double prepare_lookup_f64(object function, LookupContext context, object table,
    long long key, object column, object default=_missing) except? -1:
    cdef double value
    cdef TableView data
    cdef Py_ssize_t col = -1
    if context._private_owner:
        context.numeric_data = None
    value = lookup_f64(function,context,table,key,column,default)
    if context._private_owner and function is _lookup and type(table) is Table:
        data = context.get(table,column,&col)
        remember_numeric_resource(context,table,column,data,col)
    return value



cdef double prepare_previous_f64(object function, LookupContext context, object table,
    long long key, object column, object default=_missing) except? -1:
    cdef double value
    cdef TableView data
    cdef Py_ssize_t col = -1
    if context._private_owner:
        context.numeric_data = None
    value = previous_f64(function,context,table,key,column,default)
    if context._private_owner and function is _previous and type(table) is Table:
        data = context.get(table,column,&col)
        remember_numeric_resource(context,table,column,data,col)
    return value

# Preserve explicit conversion semantics, including replaced callables.
from builtins import float as _float_type

cdef double lookup_float_f64(object conversion, object function, LookupContext context,
                             object table, long long key, object column,
                             object default=_missing) except? -1:
    cdef TableView data
    cdef Py_ssize_t index, col = -1
    cdef KeyPart part
    if conversion is _float_type and function is _lookup:
        data = context.get(table, column, &col)
        if data is not None and data.numeric and not data.pandas_at:
            index = data.right(key) - 1
            if index < 0 or data.integers[index] != key:
                if default is _missing:
                    raise KeyError(key)
                # float(default) must parse strings, unlike implicit C conversion.
                return conversion(default)
            if col >= 0:
                return data.numbers[data.sorted_positions[index] * data.width + col]
    key_int(&part, key, False)
    return lookup_keyed_float_f64(conversion, function, context, table, &part,
                                  1, False, column, default)

# Primitive full-key lookup over the existing owned Table.
from cpython.object cimport PyObject_Hash
from cpython.unicode cimport PyUnicode_Compare
from builtins import float as _key_float
import sys as _key_sys

cdef enum:
    _KEY_INT = 1
    _KEY_BOOL = 2
    _KEY_TEXT = 3
    _KEY_FLOAT_INT = 4
    _KEY_FLOAT = 5
    _KEY_OBJECT = 6

cdef union FloatKeyBits:
    double value
    unsigned long long bits


import numpy as _key_numpy
cdef tuple _numpy_integer_keys = tuple(set((
    _key_numpy.int8, _key_numpy.int16, _key_numpy.int32, _key_numpy.int64,
    _key_numpy.uint8, _key_numpy.uint16, _key_numpy.uint32, _key_numpy.uint64,
    _key_numpy.intp, _key_numpy.uintp, _key_numpy.longlong, _key_numpy.ulonglong)))
cdef tuple _numpy_float_keys = (_key_numpy.float16, _key_numpy.float32, _key_numpy.float64)
cdef object _numpy_bool_key = _key_numpy.bool_


cdef void _key_real(KeyPart* part, double number, object original) noexcept:
    cdef FloatKeyBits storage
    part.number = 0
    part.text = <PyObject*>original
    if not isfinite(number):
        part.kind = _KEY_OBJECT
    elif -9223372036854775808.0 <= number < 9223372036854775808.0 and floor(number) == number:
        part.kind = _KEY_FLOAT_INT
        part.number = <long long>number
    else:
        part.kind = _KEY_FLOAT
        storage.value = number
        part.number = <long long>storage.bits


cdef void key_int(KeyPart* part, long long value, bint boolean) noexcept:
    part.kind = _KEY_BOOL if boolean else _KEY_INT
    part.number = value
    part.text = NULL

cdef void key_text(KeyPart* part, object value) noexcept:
    part.number = 0
    part.text = <PyObject*>value
    if type(value) is str:
        part.kind = _KEY_TEXT
    elif type(value) is float:
        _key_real(part, PyFloat_AS_DOUBLE(value), value)
    else:
        part.kind = _KEY_OBJECT

cdef bint _native_parts(const KeyPart* parts, Py_ssize_t count) noexcept:
    cdef Py_ssize_t i
    for i in range(count):
        if parts[i].kind not in (_KEY_INT, _KEY_BOOL, _KEY_TEXT, _KEY_FLOAT_INT, _KEY_FLOAT):
            return False
    return True

cdef unsigned long long _parts_hash(const KeyPart* parts, Py_ssize_t count, bint tuple_key) except? 0:
    cdef unsigned long long value = 1469598103934665603
    cdef Py_ssize_t i
    value = (value ^ <unsigned long long>tuple_key) * <unsigned long long>1099511628211
    value = (value ^ <unsigned long long>count) * <unsigned long long>1099511628211
    for i in range(count):
        if parts[i].kind == _KEY_TEXT:
            value = (value ^ 3) * <unsigned long long>1099511628211
            value = (value ^ <unsigned long long>PyObject_Hash(<object>parts[i].text)) * <unsigned long long>1099511628211
        elif parts[i].kind == _KEY_FLOAT:
            value = (value ^ 4) * <unsigned long long>1099511628211
            value = (value ^ <unsigned long long>parts[i].number) * <unsigned long long>1099511628211
        else:
            # bool and int have equal-key semantics in the existing Table.
            value = (value ^ 1) * <unsigned long long>1099511628211
            value = (value ^ <unsigned long long>parts[i].number) * <unsigned long long>1099511628211
    return value

cdef object _original_key(const KeyPart* parts, Py_ssize_t count, bint tuple_key):
    values = []
    cdef Py_ssize_t i
    for i in range(count):
        if parts[i].text != NULL:
            values.append(<object>parts[i].text)
        elif parts[i].kind == _KEY_INT:
            values.append(parts[i].number)
        elif parts[i].kind == _KEY_BOOL:
            values.append(bool(parts[i].number))
        else:
            if parts[i].text == NULL:
                raise SystemError('uninitialized native key component')
            values.append(<object>parts[i].text)
    return tuple(values) if tuple_key else values[0]

cdef class KeyedView:
    cdef tuple at_levels, at_offsets
    cdef signed char[::1] boolean_modes
    cdef long long at_bias
    cdef bint at_multi, encoded_errors
    cdef object owner
    cdef tuple text, columns, raw_values
    cdef long long[::1] numbers, kinds, slots
    cdef unsigned long long[::1] hashes
    cdef double[::1] values
    cdef unsigned char[::1] numeric
    cdef Py_ssize_t width, value_width, rows, mask
    cdef bint tuple_keys, admitted, pandas_columns

    def __init__(self, table):
        cdef KeyPart packed
        self.admitted = False
        self.owner = table
        self.at_levels = ()
        self.at_offsets = ()
        self.at_multi = False
        self.encoded_errors = True
        self.at_bias = 0
        spec = table._index_error
        if spec is not None:
            spec = spec._index
        self.pandas_columns = False if spec is None else spec.get('pandas_columns', False)
        self.boolean_modes = array('b', () if spec is None else spec['boolean_modes'])
        if spec is not None:
            self.at_multi = spec['multi']
            self.encoded_errors = spec.get('encoded_errors', True)
            self.at_bias = spec['bias']
            self.at_offsets = spec['offsets']
            self.at_levels = tuple(KeyedView(level) for level in spec['level_tables'])
        keys = table._keys
        self.rows = len(keys)
        self.columns = table._columns
        self.raw_values = table._values
        self.value_width = len(self.columns)
        self.width = 0
        self.tuple_keys = False
        if not keys:
            self.admitted = True
            return
        self.tuple_keys = type(keys[0]) is tuple
        self.width = len(keys[0]) if self.tuple_keys else 1
        kinds, numbers, texts, row_hashes = [], [], [], []
        for key in keys:
            if (type(key) is tuple) != self.tuple_keys:
                return
            parts = key if self.tuple_keys else (key,)
            if len(parts) != self.width:
                return
            for part in parts:
                if type(part) is str:
                    kinds.append(_KEY_TEXT); numbers.append(0); texts.append(part)
                elif type(part) in (int, bool) and -(1 << 63) <= part < (1 << 63):
                    kinds.append(_KEY_BOOL if type(part) is bool else _KEY_INT)
                    numbers.append(part); texts.append(None)
                elif type(part) is float:
                    key_text(&packed, part)
                    if packed.kind == _KEY_OBJECT:
                        return
                    kinds.append(packed.kind); numbers.append(packed.number); texts.append(part)
                else:
                    return
        self.kinds = array('q', kinds)
        self.numbers = array('q', numbers)
        self.text = tuple(texts)
        # Conversion belongs to this explicit float-context resource. Exact
        # immutable builtins have no user conversion effects. Defer errors and
        # other objects until the selected value is actually requested.
        numbers, valid = [], []
        for input_row in self.raw_values:
            for value in input_row:
                if type(value) in (float, int, bool, str):
                    try:
                        converted = _key_float(value)
                    except (ValueError, OverflowError):
                        pass
                    else:
                        numbers.append(converted); valid.append(1)
                        continue
                numbers.append(0.0); valid.append(0)
        self.values = array('d', numbers)
        self.numeric = array('B', valid)
        capacity = 8
        while capacity < self.rows * 2:
            capacity *= 2
        if capacity > _key_sys.maxsize // 8:
            return
        self.slots = array('q', [-1] * capacity)
        self.hashes = array('Q', [0] * capacity)
        self.mask = capacity - 1
        cdef Py_ssize_t row, slot
        cdef unsigned long long code
        for row in range(self.rows):
            code = self.row_hash(row)
            slot = <Py_ssize_t>(code & <unsigned long long>self.mask)
            while self.slots[slot] >= 0:
                slot = (slot + 1) & self.mask
            self.slots[slot] = row
            self.hashes[slot] = code
        self.admitted = True

    cdef unsigned long long row_hash(self, Py_ssize_t row) except? 0:
        cdef unsigned long long value = 1469598103934665603
        cdef Py_ssize_t i, position
        value = (value ^ <unsigned long long>self.tuple_keys) * <unsigned long long>1099511628211
        value = (value ^ <unsigned long long>self.width) * <unsigned long long>1099511628211
        for i in range(self.width):
            position = row * self.width + i
            if self.kinds[position] == _KEY_TEXT:
                value = (value ^ 3) * <unsigned long long>1099511628211
                value = (value ^ <unsigned long long>PyObject_Hash(self.text[position])) * <unsigned long long>1099511628211
            elif self.kinds[position] == _KEY_FLOAT:
                value = (value ^ 4) * <unsigned long long>1099511628211
                value = (value ^ <unsigned long long>self.numbers[position]) * <unsigned long long>1099511628211
            else:
                value = (value ^ 1) * <unsigned long long>1099511628211
                value = (value ^ <unsigned long long>self.numbers[position]) * <unsigned long long>1099511628211
        return value

    cdef bint equal(self, Py_ssize_t row, const KeyPart* parts) except -1:
        cdef Py_ssize_t i, position
        for i in range(self.width):
            position = row * self.width + i
            if self.boolean_modes.shape[0] and self.boolean_modes[i] and (
                    (parts[i].kind == _KEY_BOOL) != (self.kinds[position] == _KEY_BOOL)):
                return False
            if parts[i].kind == _KEY_TEXT:
                if self.kinds[position] != _KEY_TEXT:
                    return False
                if PyUnicode_Compare(<object>parts[i].text, self.text[position]) != 0:
                    return False
            elif parts[i].kind == _KEY_FLOAT:
                if self.kinds[position] != _KEY_FLOAT or parts[i].number != self.numbers[position]:
                    return False
            else:
                if self.kinds[position] not in (_KEY_INT, _KEY_BOOL, _KEY_FLOAT_INT) or parts[i].number != self.numbers[position]:
                    return False
        return True

    cdef Py_ssize_t find(self, const KeyPart* parts, Py_ssize_t count, bint tuple_key) except? -2:
        cdef unsigned long long code
        cdef Py_ssize_t slot, row
        if not self.rows or count != self.width or tuple_key != self.tuple_keys:
            return -1
        code = _parts_hash(parts, count, tuple_key)
        slot = <Py_ssize_t>(code & <unsigned long long>self.mask)
        while self.slots[slot] >= 0:
            row = self.slots[slot]
            if code == self.hashes[slot] and self.equal(row, parts):
                return row
            slot = (slot + 1) & self.mask
        return -1

    cdef object error_key(self, const KeyPart* parts, Py_ssize_t count, bint tuple_key):
        cdef Py_ssize_t i, position
        cdef KeyedView level
        if not self.encoded_errors or not self.at_multi or not tuple_key or count != len(self.at_levels):
            return _original_key(parts, count, tuple_key)
        encoded = 0
        for i in range(count):
            if (self.boolean_modes[i] == 1 and parts[i].kind != _KEY_BOOL or
                    self.boolean_modes[i] == 2 and parts[i].kind == _KEY_BOOL):
                return _original_key(parts, count, tuple_key)
            level = self.at_levels[i]
            position = level.find(&parts[i], 1, False)
            if position < 0:
                return _original_key(parts, count, tuple_key)
            encoded |= (position + self.at_bias) << self.at_offsets[i]
        return encoded

cdef class KeyedCache:
    cdef object source, column
    cdef KeyedView data
    cdef Py_ssize_t index
    cdef int column_kind

cdef KeyedCache _keyed_cache(LookupContext context, object table, object column):
    cdef KeyedCache cache
    cdef KeyedView data
    cdef object retired
    if context._private_owner:
        cache = context.keyed_cache
        if cache is not None and cache.source is table and cache.column is column:
            return cache
    with cython.critical_section(context):
        cache = context.keyed_cache
        if cache is not None and cache.source is table and cache.column is column:
            return cache
        cache = KeyedCache()
        cache.source = table
        cache.column = column
        cache.index = -1
        cache.column_kind = 0
        if type(table) is Table:
            data = table._native_cache.get(KeyedView)
            if data is None:
                data = KeyedView(table)
                table._native_cache[KeyedView] = data
            if data.admitted:
                cache.data = data
                if type(column) is str:
                    cache.column_kind = 1
                    try: cache.index = data.columns.index(column)
                    except ValueError:
                        if data.pandas_columns:
                            raise KeyError(column) from None
                elif type(column) is int or type(column) is bool:
                    cache.column_kind = 2 if -_key_sys.maxsize - 1 <= column <= _key_sys.maxsize else 3
                    if -data.value_width <= column < data.value_width:
                        cache.index = column if column >= 0 else column + data.value_width
        retired = context.keyed_cache
        context.keyed_cache = cache
    return cache

cdef double lookup_keyed_float_f64(object conversion, object function, LookupContext context,
                                   object table, const KeyPart* parts, Py_ssize_t count,
                                   bint tuple_key, object column, object default=_missing) except? -1:
    cdef KeyedCache cache
    cdef KeyedView data
    cdef Py_ssize_t row, position
    if conversion is _key_float and function is _lookup and _native_parts(parts, count):
        cache = _keyed_cache(context, table, column)
        data = cache.data
        if data is not None and cache.column_kind:
            row = data.find(parts, count, tuple_key)
            if row < 0:
                if default is _missing:
                    raise KeyError(data.error_key(parts, count, tuple_key))
                return conversion(default)
            if cache.index < 0:
                if cache.column_kind == 1:
                    raise ValueError('tuple.index(x): x not in tuple')
                if cache.column_kind == 3:
                    raise IndexError("cannot fit 'int' into an index-sized integer")
                raise IndexError('tuple index out of range')
            position = row * data.value_width + cache.index
            if data.numeric[position]:
                return data.values[position]
            # The row/column interaction is already native. Invoke conversion
            # on precisely the selected original scalar, preserving its error
            # or user-defined conversion without calling Python lookup.
            return conversion(data.raw_values[row][cache.index])
    key = _original_key(parts, count, tuple_key)
    return conversion(function(table, key, column)) if default is _missing else conversion(function(table, key, column, default))

cdef double lookup_key0_float_f64(object conversion, object function, LookupContext context, object table, bint tuple_key, object column, object default=_missing) except? -1:
    cdef KeyPart[1] parts
    key_int(&parts[0], 0, False)
    return lookup_keyed_float_f64(conversion, function, context, table, &parts[0], 0, tuple_key, column, default)

cdef double lookup_key1_float_f64(object conversion, object function, LookupContext context, object table, KeyArg0 key0, bint tuple_key, object column, object default=_missing) except? -1:
    cdef KeyPart[1] parts
    if KeyArg0 is KeyInteger:
        key_int(&parts[0], key0, False)
    elif KeyArg0 is bint:
        key_int(&parts[0], key0, True)
    else:
        key_text(&parts[0], key0)
    return lookup_keyed_float_f64(conversion, function, context, table, &parts[0], 1, tuple_key, column, default)

cdef double lookup_key2_float_f64(object conversion, object function, LookupContext context, object table, KeyArg0 key0, KeyArg1 key1, bint tuple_key, object column, object default=_missing) except? -1:
    cdef KeyPart[2] parts
    if KeyArg0 is KeyInteger:
        key_int(&parts[0], key0, False)
    elif KeyArg0 is bint:
        key_int(&parts[0], key0, True)
    else:
        key_text(&parts[0], key0)
    if KeyArg1 is KeyInteger:
        key_int(&parts[1], key1, False)
    elif KeyArg1 is bint:
        key_int(&parts[1], key1, True)
    else:
        key_text(&parts[1], key1)
    return lookup_keyed_float_f64(conversion, function, context, table, &parts[0], 2, tuple_key, column, default)

cdef double lookup_key3_float_f64(object conversion, object function, LookupContext context, object table, KeyArg0 key0, KeyArg1 key1, KeyArg2 key2, bint tuple_key, object column, object default=_missing) except? -1:
    cdef KeyPart[3] parts
    if KeyArg0 is KeyInteger:
        key_int(&parts[0], key0, False)
    elif KeyArg0 is bint:
        key_int(&parts[0], key0, True)
    else:
        key_text(&parts[0], key0)
    if KeyArg1 is KeyInteger:
        key_int(&parts[1], key1, False)
    elif KeyArg1 is bint:
        key_int(&parts[1], key1, True)
    else:
        key_text(&parts[1], key1)
    if KeyArg2 is KeyInteger:
        key_int(&parts[2], key2, False)
    elif KeyArg2 is bint:
        key_int(&parts[2], key2, True)
    else:
        key_text(&parts[2], key2)
    return lookup_keyed_float_f64(conversion, function, context, table, &parts[0], 3, tuple_key, column, default)

cdef double lookup_key4_float_f64(object conversion, object function, LookupContext context, object table, KeyArg0 key0, KeyArg1 key1, KeyArg2 key2, KeyArg3 key3, bint tuple_key, object column, object default=_missing) except? -1:
    cdef KeyPart[4] parts
    if KeyArg0 is KeyInteger:
        key_int(&parts[0], key0, False)
    elif KeyArg0 is bint:
        key_int(&parts[0], key0, True)
    else:
        key_text(&parts[0], key0)
    if KeyArg1 is KeyInteger:
        key_int(&parts[1], key1, False)
    elif KeyArg1 is bint:
        key_int(&parts[1], key1, True)
    else:
        key_text(&parts[1], key1)
    if KeyArg2 is KeyInteger:
        key_int(&parts[2], key2, False)
    elif KeyArg2 is bint:
        key_int(&parts[2], key2, True)
    else:
        key_text(&parts[2], key2)
    if KeyArg3 is KeyInteger:
        key_int(&parts[3], key3, False)
    elif KeyArg3 is bint:
        key_int(&parts[3], key3, True)
    else:
        key_text(&parts[3], key3)
    return lookup_keyed_float_f64(conversion, function, context, table, &parts[0], 4, tuple_key, column, default)


cdef class IndexedContext:
    pass


cdef LookupContext _indexed_context(object table):
    cdef LookupContext context
    context = table._native_cache.get(IndexedContext)
    if context is None:
        context = LookupContext()
        table._native_cache[IndexedContext] = context
    return context


cdef void _pack_part(object value, KeyPart* part) except *:
    cdef int overflow = 0
    cdef long long number
    if type(value) in (int, bool) and -(1 << 63) <= value < (1 << 63):
        key_int(part, value, type(value) is bool)
    elif type(value) is _numpy_bool_key:
        key_int(part, PyObject_IsTrue(value), True)
        part.text = <PyObject*>value
    elif type(value) in _numpy_integer_keys:
        number = PyLong_AsLongLongAndOverflow(PyNumber_Index(value), &overflow)
        if overflow:
            key_text(part, value)
        else:
            key_int(part, number, False)
            part.text = <PyObject*>value
    elif type(value) in _numpy_float_keys:
        _key_real(part, PyFloat_AsDouble(value), value)
    else:
        key_text(part, value)


cdef bint _pack_key(object key, KeyPart* parts, Py_ssize_t* count, bint* tuple_key) except -1:
    cdef Py_ssize_t i
    tuple_key[0] = type(key) is tuple
    if not tuple_key[0]:
        count[0] = 1
        _pack_part(key, &parts[0])
        return _native_parts(parts, 1)
    count[0] = len(key)
    if count[0] > 4:
        return False
    for i in range(count[0]):
        _pack_part(key[i], &parts[i])
    return _native_parts(parts, count[0])


cdef KeyPart* _pack_wide_key(object key, Py_ssize_t* count) except? NULL:
    """Spill only wider tuples; reuse the existing primitive key contract."""
    cdef KeyPart* parts
    cdef Py_ssize_t i
    if type(key) is not tuple or len(key) <= 4:
        return NULL
    count[0] = len(key)
    if count[0] > _key_sys.maxsize // sizeof(KeyPart):
        raise MemoryError()
    parts = <KeyPart*>PyMem_Malloc(count[0] * sizeof(KeyPart))
    if parts == NULL:
        raise MemoryError()
    try:
        for i in range(count[0]):
            _pack_part(key[i], &parts[i])
        if _native_parts(parts, count[0]):
            return parts
    except:
        PyMem_Free(parts)
        raise
    PyMem_Free(parts)
    return NULL


cdef object _wide_indexed_lookup(object table, object key, object column, object default):
    # Own the tuple until every borrowed Unicode component has been consumed.
    cdef object owner = key
    cdef Py_ssize_t count, row
    cdef KeyPart* parts = _pack_wide_key(owner, &count)
    cdef KeyedCache cache
    cdef KeyedView data
    try:
        if parts != NULL:
            cache = _keyed_cache(_indexed_context(table), table, column)
            data = cache.data
            if data is not None and cache.index >= 0:
                row = data.find(parts, count, True)
                if row < 0:
                    if default is _missing:
                        raise KeyError(data.error_key(parts, count, True))
                    return default
                return data.raw_values[row][cache.index]
        return _lookup(table, key, column) if default is _missing else _lookup(table, key, column, default)
    finally:
        PyMem_Free(parts)


cdef Py_ssize_t _wide_indexed_position(object table, object key) except? -2:
    cdef object owner = key
    cdef Py_ssize_t count, position
    cdef KeyPart* parts = _pack_wide_key(owner, &count)
    cdef KeyedCache cache
    cdef KeyedView data
    try:
        if parts != NULL:
            cache = _keyed_cache(_indexed_context(table), table, 0)
            data = cache.data
            if data is not None:
                position = data.find(parts, count, True)
                if position < 0:
                    raise KeyError(data.error_key(parts, count, True))
                return position
        return -1
    finally:
        PyMem_Free(parts)


cdef object _wide_indexed_contains(object table, object key):
    cdef object owner = key
    cdef Py_ssize_t count
    cdef KeyPart* parts = _pack_wide_key(owner, &count)
    cdef KeyedCache cache
    cdef KeyedView data
    try:
        if parts != NULL:
            cache = _keyed_cache(_indexed_context(table), table, 0)
            data = cache.data
            if data is not None:
                return data.find(parts, count, True) >= 0
        return _contains(table, key)
    finally:
        PyMem_Free(parts)

cdef object indexed_lookup(object table, object key, object column, object default=_missing):
    cdef KeyPart[4] parts
    cdef Py_ssize_t count, row
    cdef bint tuple_key
    cdef KeyedCache cache
    cdef KeyedView data
    if _pack_key(key, &parts[0], &count, &tuple_key):
        cache = _keyed_cache(_indexed_context(table), table, column)
        data = cache.data
        if data is not None and cache.index >= 0:
            row = data.find(&parts[0], count, tuple_key)
            if row < 0:
                if default is _missing:
                    raise KeyError(data.error_key(&parts[0], count, tuple_key))
                return default
            return data.raw_values[row][cache.index]
    return _wide_indexed_lookup(table, key, column, default)

cdef Py_ssize_t indexed_position(object table, object key) except? -2:
    """Native unique-row position, or -1 when the key domain is not admitted."""
    cdef KeyPart[4] parts
    cdef Py_ssize_t count, position
    cdef bint tuple_key
    cdef KeyedCache cache
    cdef KeyedView data
    if _pack_key(key, &parts[0], &count, &tuple_key):
        cache = _keyed_cache(_indexed_context(table), table, 0)
        data = cache.data
        if data is not None:
            position = data.find(&parts[0], count, tuple_key)
            if position < 0:
                raise KeyError(data.error_key(&parts[0], count, tuple_key))
            return position
    return _wide_indexed_position(table, key)


cdef object indexed_contains(object table, object key):
    cdef KeyPart[4] parts
    cdef Py_ssize_t count
    cdef bint tuple_key
    cdef KeyedCache cache
    cdef KeyedView data
    if _pack_key(key, &parts[0], &count, &tuple_key):
        cache = _keyed_cache(_indexed_context(table), table, 0)
        data = cache.data
        if data is not None:
            return data.find(&parts[0], count, tuple_key) >= 0
    return _wide_indexed_contains(table, key)


from builtins import int as _typed_int, bool as _typed_bool, str as _typed_str
from modelx_helpers.typed_inputs import _typed_value as _typed_python_value, _MISSING as _typed_missing
cdef object _lookup_float_fn = _native_operations['lookup_float']
cdef object _lookup_int_fn = _native_operations['lookup_int']
cdef object _lookup_bool_fn = _native_operations['lookup_bool']
cdef object _lookup_str_fn = _native_operations['lookup_str']


cdef object _explicit_value(object source, object key, object column, object default):
    # The public Python sentinel is meaningful when explicitly supplied. The
    # provider token denotes an omitted native slot; neither means None.
    if column is _typed_missing:
        column = _missing
    if default is _typed_missing:
        default = _missing
    if type(source) is Record:
        if column is not _missing:
            raise TypeError('Record lookup does not accept a column; use default=')
        value = source._index.get(key, _missing)
        if value is _missing:
            if default is _missing:
                raise KeyError(key)
            return default
        return value
    if type(source) is Table:
        if column is _missing:
            raise TypeError('Table lookup requires a column')
        return indexed_lookup(source, key, column, default)
    if column is _missing:
        if default is _missing:
            return _typed_python_value(source, key)
        return _typed_python_value(source, key, default=default)
    if default is _missing:
        return _typed_python_value(source, key, column)
    return _typed_python_value(source, key, column, default)


cdef double lookup_float(object function, tuple keywords, object source, object key, object column=_missing, object default=_missing) except? -1:
    cdef KeyPart[4] parts
    cdef Py_ssize_t count
    cdef bint tuple_key
    if function is not _lookup_float_fn:
        return forward_call(function, keywords, ('source', 'key', 'column', 'default'),
                            (source, key, column, default), omitted_argument)
    if column is _typed_missing:
        column = _missing
    if default is _typed_missing:
        default = _missing
    if type(source) is Table and column is not _missing and _pack_key(key,&parts[0],&count,&tuple_key):
        return lookup_keyed_float_f64(_float_type,_lookup,_indexed_context(source),source,
                                      &parts[0],count,tuple_key,column,default)
    return _float_type(_explicit_value(source,key,column,default))


cdef long long lookup_int(object function, tuple keywords, object source, object key, object column=_missing, object default=_missing) except? -1:
    if function is not _lookup_int_fn:
        return forward_call(function, keywords, ('source', 'key', 'column', 'default'),
                            (source, key, column, default), omitted_argument)
    result = _typed_int(_explicit_value(source,key,column,default))
    if not -(1 << 63) <= result < (1 << 63):
        raise OverflowError('lookup_int result outside signed 64-bit range')
    return result


cdef bint lookup_bool(object function, tuple keywords, object source, object key, object column=_missing, object default=_missing) except? -1:
    if function is not _lookup_bool_fn:
        return forward_call(function, keywords, ('source', 'key', 'column', 'default'),
                            (source, key, column, default), omitted_argument)
    return _typed_bool(_explicit_value(source,key,column,default))


cdef str lookup_str(object function, tuple keywords, object source, object key, object column=_missing, object default=_missing):
    if function is not _lookup_str_fn:
        return forward_call(function, keywords, ('source', 'key', 'column', 'default'),
                            (source, key, column, default), omitted_argument)
    return _typed_str(_explicit_value(source,key,column,default))


cdef object _lookup_object_fn = _native_operations['lookup_object']

cdef object lookup_object(object function, tuple keywords, object source, object key, object column=_missing, object default=_missing):
    if function is not _lookup_object_fn:
        return forward_call(function, keywords, ('source', 'key', 'column', 'default'),
                            (source, key, column, default), omitted_argument)
    return _explicit_value(source,key,column,default)


from pandas.core.dtypes.missing import isna as _pandas_isna
from pandas._libs.missing import NA as _pandas_NA
from pandas._libs.tslibs.nattype import NaT as _pandas_NaT
cdef object _is_missing_fn = _native_operations['is_missing']


cdef bint is_missing(object function, tuple keywords, object predicate, object value) except? -1:
    cdef double number
    if function is not _is_missing_fn:
        return forward_call(function, keywords, ('predicate', 'value'),
                            (predicate, value), omitted_argument)
    if predicate is _pandas_isna:
        if value is None or value is _pandas_NA or value is _pandas_NaT:
            return True
        if type(value) in (int, bool) or type(value) in _numpy_integer_keys or type(value) is _numpy_bool_key:
            return False
        if isinstance(value, (str, bytes)):
            return False
        if type(value) is float or type(value) in _numpy_float_keys:
            number = PyFloat_AsDouble(value)
            return number != number
    return _typed_bool(predicate(value))


from types import MappingProxyType
from cpython.object cimport PyObject_GenericSetAttr, PyTypeObject
from modelx_helpers.table_operations import row as _row, column as _column

# Cython 3.2.9's bundled declaration takes PyObject*, whereas Python 3.12's
# actual API takes PyTypeObject*. Keep the real signature for GCC14 as well.
cdef extern from "Python.h":
    object _record_alloc "PyType_GenericAlloc"(PyTypeObject*, Py_ssize_t)

cdef object row(object function, tuple keywords, object table, object key):
    cdef Py_ssize_t position
    cdef tuple metadata, values, columns, items
    cdef object result, spec
    if function is not _row or type(table) is not Table:
        return forward_call(function, keywords, ('table', 'key'), (table, key), omitted_argument)
    metadata = table._metadata
    if metadata is None or metadata[0] != 'frame':
        raise ValueError('row requires table(..., rows=True) with pandas row metadata')
    if metadata[4] is None:
        raise ValueError('row requires table(..., rows=True)')
    if type(key) is bool and table._index_error is not None:
        spec = table._index_error._index
        if not spec.get('row_bool_labels', spec['boolean_modes'][0] == 1):
            raise KeyError(f'{key}: boolean label can not be used without a boolean index')
    position = indexed_position(table, key)
    if position < 0:
        return forward_call(function, keywords, ('table', 'key'), (table, key), omitted_argument)
    values = metadata[4][position]
    columns = table._columns
    items = tuple((columns[i], values[i]) for i in range(len(columns)))
    result = _record_alloc(<PyTypeObject*>Record, 0)
    PyObject_GenericSetAttr(result, '_items', items)
    PyObject_GenericSetAttr(result, '_index', MappingProxyType(dict(items)))
    PyObject_GenericSetAttr(result, '_native_groups', {})
    PyObject_GenericSetAttr(result, '_preserve_scalar_types', getattr(table, '_preserve_scalar_types', False))
    PyObject_GenericSetAttr(result, '_metadata', ('row', metadata[2], metadata[5][position],
                                                metadata[8][position], metadata[6], metadata[7]))
    return result


cdef object column(object function, tuple keywords, object table, object name, object report=_missing):
    cdef Py_ssize_t position, i
    cdef tuple columns, values
    cdef list result
    cdef object metadata, carrier, items
    if (function is not _column or type(table) is not Table
            or not (report is _missing or report is True or report is False)):
        return forward_call(function, keywords, ('table', 'name', 'report'), (table, name, report), omitted_argument)
    columns = table._columns
    try:
        position = columns.index(name)
    except ValueError:
        raise KeyError(name) from None
    values = table._values
    result = [None] * len(values)
    for i in range(len(values)):
        result[i] = values[i][position]
    if report is not True:
        return tuple(result)
    metadata = table._metadata
    if metadata is None or metadata[0] != 'frame':
        raise ValueError('column report requires table(..., report=True) metadata')
    carrier = _record_alloc(<PyTypeObject*>Record, 0)
    items = (('values', tuple(result)),)
    PyObject_GenericSetAttr(carrier, '_items', items)
    PyObject_GenericSetAttr(carrier, '_index', MappingProxyType(dict(items)))
    PyObject_GenericSetAttr(carrier, '_native_groups', {})
    PyObject_GenericSetAttr(carrier, '_preserve_scalar_types', getattr(table, '_preserve_scalar_types', False))
    PyObject_GenericSetAttr(carrier, '_metadata', ('series', metadata[1], metadata[3][position],
                                                 name, metadata[6], metadata[7]))
    return carrier


from modelx_helpers.table_operations import group_sum as _group_sum
import numpy as _group_numpy
from pandas.core.groupby import ops as _group_pandas_ops
cdef object _group_float64 = _group_numpy.dtype('float64')
cdef object _group_sum_ufunc = getattr(_group_pandas_ops, '_REDUCEAT_UFUNCS', {}).get('sum')
cdef object _group_reduceat = None if _group_sum_ufunc is None else _group_sum_ufunc.reduceat


cdef object group_sum(object function, tuple keywords, object table, object labels,
                      object first=_missing, object last=_missing):
    cdef tuple metadata, columns, requested, ordered_columns, group_keys, source_values
    cdef tuple output_values, output_dtypes
    cdef object key, value, result, index_dtype
    cdef dict groups, positions
    cdef list source_columns, modes, rows
    cdef list summed_columns, starts
    cdef Py_ssize_t i, j, group, width, size, slot, previous_group = -2, offset = 0
    cdef double number, y, updated
    cdef bint integral = True, floating = True, monotonic = True, missing_sum = False
    cdef double[::1] totals, compensation
    cdef double[:, ::1] work_values
    cdef object work_array, reduced
    cdef unsigned char[::1] seen
    if function is not _group_sum or type(table) is not Table:
        return forward_call(function, keywords, ('table', 'labels', 'first', 'last'),
                            (table, labels, first, last), omitted_argument)
    metadata = table._metadata
    if (metadata is None or len(metadata) < 10 or metadata[0] != 'frame'
            or type(labels) not in (tuple, list)):
        return forward_call(function, keywords, ('table', 'labels', 'first', 'last'),
                            (table, labels, first, last), omitted_argument)
    for value in metadata[3]:
        if value != _group_float64:
            return forward_call(function, keywords, ('table', 'labels', 'first', 'last'),
                                (table, labels, first, last), omitted_argument)
    for key in labels:
        integral = integral and type(key) is int and -(1 << 63) <= key < (1 << 63)
        floating = floating and type(key) is float
    if not integral and not floating:
        return forward_call(function, keywords, ('table', 'labels', 'first', 'last'),
                            (table, labels, first, last), omitted_argument)
    if first is _missing:
        first = ()
    if last is _missing:
        last = ()
    first = (first,) if isinstance(first, str) else tuple(first)
    last = (last,) if isinstance(last, str) else tuple(last)
    requested = first + last
    if len(set(requested)) != len(requested):
        raise ValueError('first and last columns must be distinct')
    columns = table._columns
    for key in requested:
        if key not in columns:
            raise KeyError(key)
    if len(labels) != len(table._keys):
        raise ValueError('group labels must match table length')
    ordered_columns = requested + tuple(key for key in columns if key not in requested)
    source_columns = [columns.index(key) for key in ordered_columns]
    modes = [1 if key in first else 2 if key in last else 0 for key in ordered_columns]
    # Missing group labels are dropped, matching pandas' default dropna=True.
    group_keys = tuple(sorted(set(key for key in labels if key == key)))
    positions = {key: i for i, key in enumerate(group_keys)}
    width, size = len(ordered_columns), len(group_keys)
    totals = array('d', [0.0]) * (width * size)
    compensation = array('d', [0.0]) * (width * size)
    seen = array('B', [0]) * (width * size)
    source_values = table._values
    for i in range(len(labels)):
        key = labels[i]
        group = -1 if key != key else positions[key]
        if group < previous_group:
            monotonic = False
        previous_group = group
        if key != key:
            offset += 1
            continue
        for j in range(width):
            number = source_values[i][source_columns[j]]
            if number != number:
                if modes[j] == 0:
                    missing_sum = True
                continue
            slot = group * width + j
            if modes[j] == 1:
                if not seen[slot]:
                    totals[slot] = number
            elif modes[j] == 2:
                totals[slot] = number
            else:
                y = number - compensation[slot]
                updated = totals[slot] + y
                compensation[slot] = (updated - totals[slot]) - y
                if compensation[slot] != compensation[slot]:
                    compensation[slot] = 0.0
                totals[slot] = updated
            seen[slot] = 1
    # pandas 3.1 admits NumPy's segmented C reduction for monotonic group ids
    # without NaNs in the floating sum block. Other installed pandas versions
    # use the compensated C loop above. The donor's explicit ufunc capability
    # is read once at provider import; no formulas or helper bodies are parsed.
    summed_columns = [j for j in range(width) if modes[j] == 0]
    if (_group_reduceat is not None and monotonic and not missing_sum
            and size and summed_columns):
        starts = []
        previous_group = -1
        for i in range(offset, len(labels)):
            group = positions[labels[i]]
            if group != previous_group:
                starts.append(i - offset)
            previous_group = group
        work_array = _group_numpy.empty((len(summed_columns), len(labels) - offset), dtype='float64')
        work_values = work_array
        for j in range(len(summed_columns)):
            for i in range(offset, len(labels)):
                work_values[j, i - offset] = source_values[i][source_columns[summed_columns[j]]]
        reduced = _group_reduceat(work_array, _group_numpy.array(starts, dtype='intp'), axis=1)
        for j in range(len(summed_columns)):
            for i in range(size):
                totals[i * width + summed_columns[j]] = reduced[j, i]
    rows = []
    for i in range(size):
        rows.append(tuple(totals[i * width + j] if modes[j] == 0 or seen[i * width + j]
                          else float('nan') for j in range(width)))
    output_values = tuple(rows)
    output_dtypes = (_group_float64,) * width
    if not labels:
        index_dtype = 'object'
    else:
        index_dtype = 'int64' if integral else 'float64'
    result = _record_alloc(<PyTypeObject*>Table, 0)
    PyObject_GenericSetAttr(result, '_keys', group_keys)
    PyObject_GenericSetAttr(result, '_values', output_values)
    PyObject_GenericSetAttr(result, '_columns', ordered_columns)
    PyObject_GenericSetAttr(result, '_positions', MappingProxyType(positions))
    PyObject_GenericSetAttr(result, '_ordered', group_keys)
    PyObject_GenericSetAttr(result, '_native_cache', {})
    PyObject_GenericSetAttr(result, '_index_error', None)
    PyObject_GenericSetAttr(result, '_metadata', (
        'frame', ('labels', group_keys, None, index_dtype),
        ('labels', ordered_columns, metadata[9][1], metadata[9][0]),
        output_dtypes, None, (), metadata[6], metadata[7], (), metadata[9]))
    return result


from pandas import DataFrame as _report_frame_type, Index as _report_index_type, RangeIndex as _report_range_type
from pandas.core.arrays.string_ import StringArray as _report_string_array_type
from pandas.core.flags import Flags as _report_flags_type
from modelx_helpers.pandas_inputs import table as _table_constructor
cdef object _report_int64 = _group_numpy.dtype('int64')


cdef object _native_report_table(object frame, bint with_rows, bint encoded_errors,
                                 bint preserve_scalar_types):
    """Copy admitted physical report storage without pandas/Python operations."""
    cdef object manager, block, data, index, column_axis, column_data, column_storage
    cdef object index_data, index_spec, index_metadata, column_dtype, flags, result, spec
    cdef object attrs, value, interval
    cdef tuple columns, keys, row_names, values, row_values, items, metadata
    cdef list copied, copied_rows
    cdef Py_ssize_t i, j, height, width
    cdef const double[:, :] numbers
    if type(frame) is not _report_frame_type:
        return None
    attrs = frame._attrs
    flags = frame._flags
    if type(attrs) is not dict or attrs or type(flags) is not _report_flags_type:
        return None
    manager = frame._mgr
    if len(manager.blocks) != 1:
        return None
    block = manager.blocks[0]
    data = block.values
    if type(data) is not _group_numpy.ndarray or data.dtype != _group_float64 or data.ndim != 2:
        return None
    numbers = data
    width, height = numbers.shape[0], numbers.shape[1]
    if not width:
        return None
    value = block._mgr_locs.as_array
    for i in range(width):
        if value[i] != i:
            return None
    column_axis, index = manager.axes[0], manager.axes[1]
    if type(column_axis) is not _report_index_type:
        return None
    column_data = column_axis._data
    if type(column_data) is _group_numpy.ndarray:
        column_storage, column_dtype = column_data, column_data.dtype
    elif type(column_data) is _report_string_array_type:
        column_storage, column_dtype = column_data._ndarray, column_data._dtype
    else:
        return None
    columns = tuple(column_storage)
    if len(columns) != width or len(set(columns)) != width:
        return None
    for value in columns:
        if type(value) is not str:
            return None
    if type(index) is _report_range_type:
        interval = index._range
        keys = tuple(interval)
        row_names = keys
        index_metadata = ('range', interval.start, interval.stop, interval.step, index._name)
    elif type(index) is _report_index_type:
        index_data = index._data
        if (type(index_data) is not _group_numpy.ndarray
                or index_data.dtype.kind == 'b'
                or height and index_data.dtype != _report_int64):
            return None
        keys = tuple(int(value) for value in index_data)
        row_names = tuple(index_data)
        index_metadata = ('labels', keys, index._name, index_data.dtype)
    else:
        return None
    if len(keys) != height:
        return None
    if len(set(keys)) != height:
        raise ValueError('duplicate table key')
    copied, copied_rows = [], []
    for i in range(height):
        if preserve_scalar_types:
            copied.append(tuple(data[j, i] for j in range(width)))
        else:
            copied.append(tuple(numbers[j, i] for j in range(width)))
        if with_rows:
            copied_rows.append(tuple(data[j, i] for j in range(width)))
    values = tuple(copied)
    row_values = tuple(copied_rows) if with_rows else None
    # Scalar integer axes need no MultiIndex engine encoding or level tables.
    items = (('multi', False), ('encoded_errors', encoded_errors), ('pandas_columns', True),
             ('row_bool_labels', False), ('levels', (keys,)),
             ('level_tables', ()), ('offsets', (0,)), ('boolean_modes', (2,)), ('bias', 0))
    spec = _record_alloc(<PyTypeObject*>Record, 0)
    PyObject_GenericSetAttr(spec, '_items', items)
    PyObject_GenericSetAttr(spec, '_index', MappingProxyType(dict(items)))
    PyObject_GenericSetAttr(spec, '_native_groups', {})
    PyObject_GenericSetAttr(spec, '_metadata', None)
    metadata = ('frame', index_metadata, ('labels', columns, column_axis._name, column_dtype),
                (_group_float64,) * width, row_values,
                (_group_float64,) * height if with_rows else (), {},
                flags._allows_duplicate_labels, row_names if with_rows else (),
                (column_dtype, column_axis._name))
    result = _record_alloc(<PyTypeObject*>Table, 0)
    PyObject_GenericSetAttr(result, '_keys', keys)
    PyObject_GenericSetAttr(result, '_values', values)
    PyObject_GenericSetAttr(result, '_columns', columns)
    PyObject_GenericSetAttr(result, '_positions', MappingProxyType({key: i for i, key in enumerate(keys)}))
    PyObject_GenericSetAttr(result, '_ordered', tuple(sorted(keys)))
    PyObject_GenericSetAttr(result, '_native_cache', {})
    PyObject_GenericSetAttr(result, '_index_error', spec)
    PyObject_GenericSetAttr(result, '_metadata', metadata)
    PyObject_GenericSetAttr(result, '_preserve_scalar_types', preserve_scalar_types)
    return result



# Physical mixed-column preparation for the existing inputs.table contract.
from cpython.object cimport PyObject_TypeCheck
from cpython.long cimport PyLong_Check
from cpython.float cimport PyFloat_Check
from cpython.unicode cimport PyUnicode_Check
from cpython.bytes cimport PyBytes_Check
from pandas import StringDtype as _frame_string_dtype_type
cdef object _frame_object_dtype = _group_numpy.dtype('object')
cdef tuple _frame_numeric_scalar_types = (_numpy_integer_keys + _numpy_float_keys
    + (_group_numpy.longdouble, _numpy_bool_key, _group_numpy.str_, _group_numpy.bytes_))


cdef object _frame_normalize(object value):
    if type(value) in _frame_numeric_scalar_types:
        return value.item()
    if PyObject_TypeCheck(value, <PyTypeObject*>_group_numpy.generic):
        return _missing
    if value is None or value is _pandas_NA or value is _pandas_NaT:
        return value
    if PyLong_Check(value) or PyFloat_Check(value) or PyUnicode_Check(value) or PyBytes_Check(value):
        return value
    return _missing


cdef object _frame_axes(object manager):
    cdef object column_axis = manager.axes[0]
    cdef object index = manager.axes[1]
    cdef object data, storage, column_dtype, interval, index_metadata, value
    cdef tuple columns, keys, row_names
    if type(column_axis) is not _report_index_type:
        return None
    data = column_axis._data
    if type(data) is _group_numpy.ndarray:
        storage, column_dtype = data, data.dtype
    elif type(data) is _report_string_array_type:
        storage, column_dtype = data._ndarray, data._dtype
    else:
        return None
    columns = tuple(storage)
    if not columns or len(set(columns)) != len(columns):
        return None
    for value in columns:
        if type(value) is not str:
            return None
    if type(index) is _report_range_type:
        interval = index._range
        keys = tuple(interval)
        row_names = keys
        index_metadata = ('range', interval.start, interval.stop, interval.step, index._name)
    elif type(index) is _report_index_type:
        data = index._data
        if (type(data) is not _group_numpy.ndarray or data.dtype.kind == 'b'
                or len(data) and data.dtype != _report_int64):
            return None
        keys = tuple(int(value) for value in data)
        row_names = tuple(data)
        index_metadata = ('labels', keys, index._name, data.dtype)
    else:
        return None
    if len(set(keys)) != len(keys):
        return None  # Reference preparation preserves conversion/error ordering.
    return (columns, keys, row_names, index_metadata,
            ('labels', columns, column_axis._name, column_dtype),
            (column_dtype, column_axis._name))


cdef object _frame_row_dtype(tuple dtypes):
    cdef object first = dtypes[0]
    cdef object dtype, left_na, right_na
    cdef bint same = True
    cdef bint all_numpy = True
    cdef bint any_boolean = False
    cdef bint any_other = False
    for dtype in dtypes:
        if isinstance(dtype, _group_numpy.dtype):
            if dtype.kind == 'b':
                any_boolean = True
            else:
                any_other = True
        else:
            all_numpy = False
        if dtype is first:
            continue
        if isinstance(dtype, _group_numpy.dtype) and isinstance(first, _group_numpy.dtype):
            if dtype != first:
                same = False
        elif type(dtype) is _frame_string_dtype_type and type(first) is _frame_string_dtype_type:
            left_na, right_na = dtype._na_value, first._na_value
            if dtype._storage != first._storage:
                return _missing
            if left_na is not right_na:
                if not (type(left_na) is float and type(right_na) is float
                        and left_na != left_na and right_na != right_na):
                    return _missing
        else:
            same = False
    if same:
        return first
    if not all_numpy or any_boolean and any_other:
        return _frame_object_dtype
    return _group_numpy.result_type(*dtypes)


cdef object _frame_owner(tuple keys, tuple values, tuple columns, object metadata,
                         bint encoded_errors, bint preserve_scalar_types):
    cdef object spec, result
    cdef tuple items = (('multi', False), ('encoded_errors', encoded_errors),
        ('pandas_columns', True), ('row_bool_labels', False), ('levels', (keys,)),
        ('level_tables', ()), ('offsets', (0,)), ('boolean_modes', (2,)), ('bias', 0))
    spec = _record_alloc(<PyTypeObject*>Record, 0)
    PyObject_GenericSetAttr(spec, '_items', items)
    PyObject_GenericSetAttr(spec, '_index', MappingProxyType(dict(items)))
    PyObject_GenericSetAttr(spec, '_native_groups', {})
    PyObject_GenericSetAttr(spec, '_metadata', None)
    PyObject_GenericSetAttr(spec, '_preserve_scalar_types', False)
    result = _record_alloc(<PyTypeObject*>Table, 0)
    PyObject_GenericSetAttr(result, '_keys', keys)
    PyObject_GenericSetAttr(result, '_values', values)
    PyObject_GenericSetAttr(result, '_columns', columns)
    PyObject_GenericSetAttr(result, '_positions', MappingProxyType({key: i for i, key in enumerate(keys)}))
    PyObject_GenericSetAttr(result, '_ordered', tuple(sorted(keys)))
    PyObject_GenericSetAttr(result, '_native_cache', {})
    PyObject_GenericSetAttr(result, '_index_error', spec)
    PyObject_GenericSetAttr(result, '_metadata', metadata)
    PyObject_GenericSetAttr(result, '_preserve_scalar_types', preserve_scalar_types)
    return result


cdef object _native_frame_table(object frame, bint with_rows, bint with_metadata,
                                bint encoded_errors, bint preserve_scalar_types):
    cdef object attrs, flags, manager, axes, block, data, places, dtype, row_dtype
    cdef object raw, value, metadata
    cdef tuple columns, keys, row_names, dtypes
    cdef list descriptors, column_dtypes, values, row_values, copied, selected
    cdef Py_ssize_t height, width, i, j, offset, position
    if type(frame) is not _report_frame_type:
        return None
    attrs, flags = frame._attrs, frame._flags
    if type(attrs) is not dict or attrs or type(flags) is not _report_flags_type:
        return None
    manager = frame._mgr
    axes = _frame_axes(manager)
    if axes is None:
        return None
    columns, keys, row_names = axes[0], axes[1], axes[2]
    width, height = len(columns), len(keys)
    descriptors, column_dtypes = [None] * width, [None] * width
    for block in manager.blocks:
        data = block.values
        places = block._mgr_locs.as_array
        if type(data) is _group_numpy.ndarray:
            if (data.ndim != 2 or data.shape[1] != height or data.shape[0] != len(places)
                    or data.dtype.kind not in ('b', 'i', 'u', 'f', 'c', 'O')):
                return None
            for offset in range(len(places)):
                position = places[offset]
                if not 0 <= position < width or descriptors[position] is not None:
                    return None
                descriptors[position] = (data, offset, False)
                column_dtypes[position] = data.dtype
        elif type(data) is _report_string_array_type:
            raw = data._ndarray
            if type(raw) is not _group_numpy.ndarray or raw.ndim != 1 or raw.shape[0] != height or len(places) != 1:
                return None
            position = places[0]
            if not 0 <= position < width or descriptors[position] is not None:
                return None
            descriptors[position] = (raw, 0, True)
            column_dtypes[position] = data._dtype
        else:
            return None
    if any(value is None for value in descriptors):
        return None
    dtypes = tuple(column_dtypes)
    row_dtype = _frame_row_dtype(dtypes) if with_rows else None
    if row_dtype is _missing:
        return None
    values, row_values = [], []
    for i in range(height):
        copied, selected = [], []
        for j in range(width):
            data, offset, value = descriptors[j]
            raw = data[i] if value else data[offset, i]
            if with_rows:
                if row_dtype is _frame_object_dtype or not isinstance(row_dtype, _group_numpy.dtype):
                    selected.append(raw)
                else:
                    selected.append(row_dtype.type(raw))
            value = raw if preserve_scalar_types else _frame_normalize(raw)
            if value is _missing:
                return None
            copied.append(value)
        values.append(tuple(copied))
        if with_rows:
            row_values.append(tuple(selected))
    metadata = None
    if with_metadata:
        metadata = ('frame', axes[3], axes[4], dtypes,
            tuple(row_values) if with_rows else None,
            (row_dtype,) * height if with_rows else (), {},
            flags._allows_duplicate_labels, row_names if with_rows else (), axes[5])
    return _frame_owner(keys, tuple(values), columns, metadata,
                        encoded_errors, preserve_scalar_types)



cdef object table(object function, tuple keywords, object frame, object columns=_missing,
                  object groups=_missing, object indexing=_missing, object rows=_missing,
                  object report=_missing, object preserve_scalar_types=_missing):
    cdef object prepared
    if (function is _table_constructor
            and (preserve_scalar_types is _missing or preserve_scalar_types is False
                 or preserve_scalar_types is True)
            and (rows is _missing or rows is True or rows is False)
            and (report is _missing or report is True or report is False)
            and (columns is _missing or columns is None)
            and (groups is _missing or type(groups) is tuple and not groups)
            and (indexing is _missing or type(indexing) is str and indexing in ('loc', 'at'))):
        if rows is True or report is True:
            prepared = _native_report_table(frame, rows is True, indexing == 'at',
                                             preserve_scalar_types is True)
            if prepared is not None:
                return prepared
        prepared = _native_frame_table(frame, rows is True, rows is True or report is True,
                                        indexing == 'at', preserve_scalar_types is True)
        if prepared is not None:
            return prepared
    return forward_call(function, keywords, ('frame', 'columns', 'groups', 'indexing', 'rows', 'report', 'preserve_scalar_types'),
                        (frame, columns, groups, indexing, rows, report, preserve_scalar_types), omitted_argument)
