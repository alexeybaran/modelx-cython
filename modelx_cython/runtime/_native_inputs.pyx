# cython: freethreading_compatible=True
"""Cython implementation of explicit immutable helpers, copied by mx2cy.

Only exact immutable Table/Record instances are specialized. Mutable inputs,
subclasses and replaced function References call the original supplied function.
Views belong to their Table and contain no formula or policy state.
"""
from array import array
from cpython.array cimport array as Array, clone
from enum import IntEnum
import math
from libc.math cimport fabs, isfinite, floor
from libc.limits cimport LONG_MIN, LONG_MAX
from cpython.float cimport PyFloat_AS_DOUBLE, PyFloat_AsDouble, PyFloat_CheckExact
from cpython.long cimport PyLong_AsLong, PyLong_AsLongLongAndOverflow, PyLong_Check
from cpython.number cimport PyNumber_Index
from cpython.object cimport PyObject_IsTrue
from cpython.tuple cimport PyTuple_GET_ITEM
from cpython.version cimport PY_VERSION_HEX
from cpython.ref cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport cython
from modelx_helpers.immutable_inputs import Table, Record
from modelx_helpers.typed_inputs import _FLOOR, _native_operations
from modelx_cython._call_layout cimport forward_call

__modelx_cython_provider__ = (1, 'modelx_cython.runtime._native_inputs', 'owned-table-1')

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


cdef inline bint _table_view_part(object value) noexcept:
    kind = type(value)
    return (kind is int or kind is bool or kind is str or kind is float
            or kind is bytes or kind is complex or value is None)


cdef inline bint _table_view_key(object value) noexcept:
    if type(value) is tuple:
        for part in <tuple>value:
            if not _table_view_part(part):
                return False
        return True
    return _table_view_part(value)


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
        cdef object key
        cdef bint all_integers = True
        self.pandas_at = table._index_error is not None
        self.keys = table._keys
        self.values = table._values
        self.columns = table._columns
        # Hashing or ordering an unsupported label would run user code during
        # view construction, even when the eventual lookup reads another row.
        for key in self.keys:
            if type(key) is not int:
                all_integers = False
                if not _table_view_key(key):
                    return
        self.column_positions = dict(zip(self.columns, range(len(self.columns))))
        self.positions = dict(zip(self.keys, range(len(self.keys))))
        self.ordered = table._ordered
        if self.ordered:
            if all_integers:
                self.minimum, self.maximum = self.ordered[0], self.ordered[-1]
                self.integral = -(1 << 63) <= self.minimum and self.maximum < (1 << 63)
            else:
                self.minimum, self.maximum = min(self.keys), max(self.keys)
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
            and (type(column) in (str, int) or isinstance(column, IntEnum))
            and _table_view_key(key)):
        data = view(table)
        if data.positions is not None:
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
    if function is _previous and type(table) is NativeTable:
        table = (<NativeTable>table)._owner
    if (function is _previous and type(table) is Table
            and type(key) is int and -(1 << 63) <= key < (1 << 63)):
        data = view(table)
        if data.integral:
            index = data.right(key) - 1
            if index >= 0:
                return data.value(data.sorted_positions[index], column)
            if default is not _missing:
                return default
            raise ValueError(f'No key at or before {key!r}')
    return function(table, key, column) if default is _missing else function(table, key, column, default)


cdef object contains(object function, object table, object key):
    cdef TableView data
    if function is _contains and type(table) is NativeTable:
        table = (<NativeTable>table)._owner
    if function is _contains and type(table) is Table and table._index_error is not None:
        return indexed_contains(table,key)
    if function is _contains and type(table) is Table:
        data = view(table)
        if data.positions is not None:
            return key in data.positions
        return key in table._positions
    return function(table, key)


cdef object key_count(object function, object table):
    if function is _key_count and type(table) is Table:
        return len(table._keys)
    return function(table)


cdef object key_at(object function, object table, object index):
    if function is _key_at and type(table) is Table:
        if index < 0:
            raise IndexError(index)
        return table._keys[index]
    return function(table, index)


cdef object first_key(object function, object table):
    cdef TableView data
    if function is _first_key and type(table) is NativeTable:
        table = (<NativeTable>table)._owner
    if function is _first_key and type(table) is Table:
        data = view(table)
        if data.ordered:
            return data.minimum
    return function(table)


cdef object last_key(object function, object table):
    cdef TableView data
    if function is _last_key and type(table) is NativeTable:
        table = (<NativeTable>table)._owner
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


cdef inline Py_ssize_t endpoint_column(const PyObject* data, object column) noexcept:
    cdef Py_ssize_t last
    if <PyObject*>column == PyTuple_GET_ITEM((<TableView>data).columns, 0):
        return 0
    last = (<TableView>data).width - 1
    if <PyObject*>column == PyTuple_GET_ITEM((<TableView>data).columns, last):
        return last
    return -1


cdef object prepared_interpolation(object function, object table, object column, object years,
                                   object lookup_fn, object last_fn, object math_module):
    cdef long long col
    cdef int overflow
    if function is not _prepared_interpolation:
        return function(table, column, years, lookup_fn, last_fn, math_module)
    if (lookup_fn is not _lookup or last_fn is not _last_key
            or math_module is not math or math.floor is not _FLOOR
            or type(table) is not Table or not PyLong_Check(column) or not PyFloat_CheckExact(years)):
        return None
    col = PyLong_AsLongLongAndOverflow(column, &overflow)
    if overflow:
        return None
    return interpolate_numeric(view(table), col, PyFloat_AS_DOUBLE(years))


cdef object interpolate_numeric(TableView data, long long col, double position):
    cdef long long lower, upper
    cdef Py_ssize_t lo, hi
    cdef double base, end
    if (not data.numeric or not 0 <= col < data.width
            or not -9223372036854775808.0 <= position < 9223372036854775808.0):
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
            if col >= 0:
                return data.numbers[data.sorted_positions[index] * data.width + col]
    return lookup(function, table, key, column) if default is _missing else lookup(function, table, key, column, default)


cdef double previous_f64(object function, LookupContext context, object table,
                         long long key, object column, object default=_missing) except? -1:
    cdef TableView data = None
    cdef Py_ssize_t index, col
    if function is _previous:
        col = -1
        if type(table) is NativeTable:
            data = (<NativeTable>table)._numeric_view
            table = (<NativeTable>table)._owner
            if data is not None:
                col = endpoint_column(<PyObject*>data, column)
        if col < 0:
            data = context.get(table, column, &col)
        if data is not None and data.numeric:
            index = data.right(key) - 1
            if index < 0:
                if default is _missing:
                    raise ValueError(f'No key at or before {key!r}')
                return default
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


cdef inline bint bounded_table_value(const PyObject* data, Py_ssize_t column, long long key,
                                     double below, double* value) except? -1:
    cdef Py_ssize_t row, count = (<TableView>data).integers.shape[0]
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
        (<TableView>data).sorted_positions[row] * (<TableView>data).width + column]
    return True


cdef bint bounded_lookup_f64(object function, object first_fn, object last_fn,
                             LookupContext context, object table, long long key,
                             str column, double below, double* value) except? -1:
    cdef TableView data
    cdef Py_ssize_t col = -1
    if function is not _lookup or first_fn is not _first_key or last_fn is not _last_key:
        return False
    data = context.get(table, column, &col)
    if data is None or col < 0:
        return False
    if not data.integers.shape[0]:
        return False
    if context._private_owner and context.bounded_data is None:
        context.bounded_data = data
        context.bounded_column = col
    return bounded_table_value(<PyObject*>data, col, key, below, value)


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


cdef object group_slice_context_probe(LookupContext context, object record, long long group,
                                      str column, Py_ssize_t* slot):
    return group_context_probe(context, _has_group, record, group, column, slot)


cdef int group_slice_context_value(LookupContext context, object prepared, Py_ssize_t slot,
                                   long long key, double* value) except -1:
    return group_context_value(context, _subtable, _contains, _lookup, prepared, slot, key, value)


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
    object column, TableView data, Py_ssize_t index, object function=None):
    if context._private_owner and data is not None and data.numeric and index >= 0:
        context.numeric_source = source
        context.numeric_column_source = column
        context.numeric_index = index
        context.numeric_data = data
        context.numeric_function = function


cdef bint numeric_resource_ready(LookupContext context, const void* source, const void* column) noexcept:
    return (context._private_owner and context.numeric_data is not None
            and source == <const void*>context.numeric_source
            and (column == NULL or column == <const void*>context.numeric_column_source))


cdef bint numeric_binding_ready(LookupContext context, const void* source, const void* column,
                                object function) noexcept:
    return function is context.numeric_function and numeric_resource_ready(context, source, column)


cdef inline double numeric_table_value(const PyObject* data, long long key, long long column,
    bint preceding, bint has_default, double default) except? -1:
    cdef Py_ssize_t row = (<TableView>data).right(key) - 1
    if row < 0 or (not preceding and (<TableView>data).integers[row] != key):
        if has_default:
            return default
        if preceding:
            raise ValueError(f'No key at or before {key!r}')
        raise KeyError(key)
    if column < 0:
        column += (<TableView>data).width
    if column < 0 or column >= (<TableView>data).width:
        raise IndexError('tuple index out of range')
    return (<TableView>data).numbers[(<TableView>data).sorted_positions[row] * (<TableView>data).width + column]


cdef inline double numeric_resource_value(LookupContext context, long long key, long long column,
    bint named, bint preceding, bint has_default, double default) except? -1:
    # The ready predicate proves lifetime/admission. Context retains the source
    # and immutable view; this borrowed pointer does not own another shared ref.
    return numeric_table_value(<PyObject*>context.numeric_data, key,
                               context.numeric_index if named else column, preceding, has_default, default)


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
    if data == <PyObject*>None:
        return False
    return bounded_table_value(data, context.bounded_column, key, below, value)



cdef double prepare_lookup_f64(object function, LookupContext context, object table,
    long long key, object column, object default=_missing) except? -1:
    cdef double value
    cdef TableView data
    cdef Py_ssize_t col = -1
    if context._private_owner:
        context.numeric_data = None
    if function is _lookup and type(table) is NativeTable:
        table = (<NativeTable>table)._owner
    value = lookup_f64(function,context,table,key,column,default)
    if context._private_owner and function is _lookup and type(table) is Table:
        data = context.get(table,column,&col)
        remember_numeric_resource(context,table,column,data,col,function)
    return value



cdef double prepare_previous_f64(object function, LookupContext context, object table,
    long long key, object column, object default=_missing) except? -1:
    cdef double value
    cdef TableView data
    cdef Py_ssize_t col = -1
    if context._private_owner:
        context.numeric_data = None
    if function is _previous and type(table) is NativeTable:
        table = (<NativeTable>table)._owner
    value = previous_f64(function,context,table,key,column,default)
    if context._private_owner and function is _previous and type(table) is Table:
        data = context.get(table,column,&col)
        remember_numeric_resource(context,table,column,data,col,function)
    return value

# Preserve explicit conversion semantics, including replaced callables.
from builtins import float as _float_type

cdef inline double _numeric_float_read(const PyObject* data, Py_ssize_t column,
                                       long long key, object default) except? -1:
    cdef Py_ssize_t index = (<TableView>data).right(key) - 1
    if index < 0 or (<TableView>data).integers[index] != key:
        if default is _missing:
            raise KeyError(key)
        if PyFloat_CheckExact(default):
            return PyFloat_AS_DOUBLE(default)
        return _float_type(default)
    return (<TableView>data).numbers[(<TableView>data).sorted_positions[index]
                                      * (<TableView>data).width + column]

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
    cdef signed char[::1] boolean_modes
    # The original tuple keeps every object referenced by parts alive.
    cdef tuple keys, columns, raw_values
    cdef KeyPart* parts
    cdef long long[::1] slots
    cdef unsigned long long[::1] hashes
    cdef double[::1] values
    cdef unsigned char[::1] numeric
    cdef Py_ssize_t width, value_width, rows, mask
    cdef bint tuple_keys, admitted, pandas_columns

    def __cinit__(self, table):
        cdef KeyPart* parts
        cdef Py_ssize_t row, slot, offset, i, capacity
        cdef unsigned long long code
        cdef int overflow
        cdef object key, part, kind, value
        cdef tuple input_row
        spec = table._index_error
        if spec is not None:
            spec = spec._index
        self.pandas_columns = False if spec is None else spec.get('pandas_columns', False)
        self.boolean_modes = array('b', () if spec is None else spec['boolean_modes'])
        self.keys = table._keys
        self.rows = len(self.keys)
        self.columns = table._columns
        self.raw_values = table._values
        self.value_width = len(self.columns)
        self.width = 0
        self.tuple_keys = False
        if not self.keys:
            self.admitted = True
            return
        self.tuple_keys = type(self.keys[0]) is tuple
        self.width = len(self.keys[0]) if self.tuple_keys else 1
        if self.tuple_keys and spec is not None and not spec['multi']:
            # A plain object Index can contain tuple labels. Pandas treats
            # these as scalar keys and does not validate their components.
            self.boolean_modes = array('b')
        if (self.width and self.rows > _key_sys.maxsize // sizeof(KeyPart) // self.width
                or self.rows > _key_sys.maxsize // sizeof(double) // self.value_width):
            raise MemoryError()
        self.parts = <KeyPart*>PyMem_Malloc(max(1, self.rows * self.width) * sizeof(KeyPart))
        if self.parts == NULL:
            raise MemoryError()
        capacity = 8
        while capacity < self.rows * 2:
            capacity *= 2
        if capacity > _key_sys.maxsize // 8:
            return
        self.slots = array('q', [-1] * capacity)
        self.hashes = array('Q', [0] * capacity)
        self.mask = capacity - 1
        for row in range(self.rows):
            key = self.keys[row]
            if (type(key) is tuple) != self.tuple_keys:
                return
            if self.tuple_keys and len(key) != self.width:
                return
            parts = self.parts + row * self.width
            for i in range(self.width):
                part = (<tuple>key)[i] if self.tuple_keys else key
                kind = type(part)
                if kind is int or kind is bool:
                    key_int(&parts[i], PyLong_AsLongLongAndOverflow(part, &overflow), kind is bool)
                    if overflow:
                        return
                else:
                    key_text(&parts[i], part)
                    if parts[i].kind == _KEY_OBJECT:
                        return
            code = _parts_hash(parts, self.width, self.tuple_keys)
            slot = <Py_ssize_t>(code & <unsigned long long>self.mask)
            while self.slots[slot] >= 0:
                slot = (slot + 1) & self.mask
            self.slots[slot] = row
            self.hashes[slot] = code
        # Conversion belongs to this explicit float-context resource. Exact
        # immutable builtins have no user conversion effects. Defer errors and
        # other objects until the selected value is actually requested.
        cdef Array numbers = clone(array('d'), self.rows * self.value_width, False)
        cdef Array valid = clone(array('B'), self.rows * self.value_width, True)
        offset = 0
        for input_row in self.raw_values:
            for value in input_row:
                kind = type(value)
                if kind is float:
                    numbers.data.as_doubles[offset] = PyFloat_AS_DOUBLE(value)
                    valid.data.as_uchars[offset] = 1
                elif kind is int or kind is bool or kind is str:
                    try:
                        numbers.data.as_doubles[offset] = _key_float(value)
                    except (ValueError, OverflowError):
                        pass
                    else:
                        valid.data.as_uchars[offset] = 1
                offset += 1
        self.values = numbers
        self.numeric = valid
        self.admitted = True

    def __dealloc__(self):
        PyMem_Free(self.parts)

    cdef bint equal(self, Py_ssize_t row, const KeyPart* parts) except -1:
        cdef Py_ssize_t i
        cdef const KeyPart* stored = self.parts + row * self.width
        for i in range(self.width):
            if self.boolean_modes.shape[0] and self.boolean_modes[i] and (
                    (parts[i].kind == _KEY_BOOL) != (stored[i].kind == _KEY_BOOL)):
                return False
            if parts[i].kind == _KEY_TEXT:
                if (stored[i].kind != _KEY_TEXT
                        or (parts[i].text != stored[i].text
                            and PyUnicode_Compare(<object>parts[i].text, <object>stored[i].text) != 0)):
                    return False
            elif parts[i].kind == _KEY_FLOAT:
                if stored[i].kind != _KEY_FLOAT or parts[i].number != stored[i].number:
                    return False
            else:
                if stored[i].kind not in (_KEY_INT, _KEY_BOOL, _KEY_FLOAT_INT) or parts[i].number != stored[i].number:
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
                    raise KeyError(_original_key(parts, count, tuple_key))
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

cdef class IndexedContext:
    pass


cdef LookupContext _indexed_context(object table):
    cdef LookupContext context
    context = table._native_cache.get(IndexedContext)
    if context is None:
        context = LookupContext()
        table._native_cache[IndexedContext] = context
    return context


cdef inline bint _key_type_is_one_of(object kind, tuple supported) noexcept:
    for candidate in supported:
        if kind is candidate:
            return True
    return False


cdef void _pack_part(object value, KeyPart* part) except *:
    cdef int overflow = 0
    cdef long long number
    kind = type(value)
    if (kind is int or kind is bool) and -(1 << 63) <= value < (1 << 63):
        key_int(part, value, kind is bool)
    elif kind is str or kind is float:
        key_text(part, value)
    elif kind is _numpy_bool_key:
        key_int(part, PyObject_IsTrue(value), True)
        part.text = <PyObject*>value
    elif _key_type_is_one_of(kind, _numpy_integer_keys):
        number = PyLong_AsLongLongAndOverflow(PyNumber_Index(value), &overflow)
        if overflow:
            key_text(part, value)
        else:
            key_int(part, number, False)
            part.text = <PyObject*>value
    elif _key_type_is_one_of(kind, _numpy_float_keys):
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


cdef tuple _wide_indexed_row(object table, object key, object column, bint needs_column):
    """Search with borrowed key parts while the key argument remains alive."""
    cdef Py_ssize_t count
    cdef KeyPart* parts = _pack_wide_key(key, &count)
    cdef KeyedCache cache
    if parts == NULL:
        return None
    try:
        cache = _keyed_cache(_indexed_context(table), table, column)
        if cache.data is not None and (not needs_column or cache.index >= 0):
            return cache, cache.data.find(parts, count, True)
    finally:
        PyMem_Free(parts)
    return None


cdef object _wide_indexed_lookup(object table, object key, object column, object default):
    cdef KeyedCache cache
    cdef Py_ssize_t row
    cdef tuple found = _wide_indexed_row(table, key, column, True)
    if found is not None:
        cache, row = found
        if row >= 0:
            return cache.data.raw_values[row][cache.index]
        if default is _missing:
            raise KeyError(key)
        return default
    return _lookup(table, key, column) if default is _missing else _lookup(table, key, column, default)


cdef Py_ssize_t _wide_indexed_position(object table, object key) except? -2:
    cdef tuple found = _wide_indexed_row(table, key, 0, False)
    if found is None:
        return -1
    if found[1] < 0:
        raise KeyError(key)
    return found[1]

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
                    raise KeyError(key)
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
                raise KeyError(key)
            return position
    return _wide_indexed_position(table, key)


cdef object indexed_contains(object table, object key):
    cdef KeyPart[4] parts
    cdef Py_ssize_t count
    cdef bint tuple_key
    cdef KeyedCache cache
    cdef KeyedView data
    cdef tuple found
    if _pack_key(key, &parts[0], &count, &tuple_key):
        cache = _keyed_cache(_indexed_context(table), table, 0)
        data = cache.data
        if data is not None:
            return data.find(&parts[0], count, tuple_key) >= 0
    if type(key) is tuple and len(key) > 4:
        found = _wide_indexed_row(table, key, 0, False)
        if found is not None:
            return found[1] >= 0
    return _contains(table, key)


from builtins import int as _typed_int, bool as _typed_bool, str as _typed_str
from modelx_helpers import typed_inputs as _typed_inputs_module
from modelx_helpers import pdn as _pdn_module
from modelx_helpers.typed_inputs import _typed_value as _typed_python_value, _MISSING as _typed_missing
cdef dict _typed_inputs_bindings = vars(_typed_inputs_module)
cdef dict _pdn_bindings = vars(_pdn_module)
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
    if type(source) is NativeTable and type((<NativeTable>source)._owner) is Table:
        source = (<NativeTable>source)._owner
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
    if type(source) is NativeTable and type((<NativeTable>source)._owner) is Table:
        source = (<NativeTable>source)._owner
    if type(source) is Table and column is not _missing and _pack_key(key,&parts[0],&count,&tuple_key):
        return lookup_keyed_float_f64(_float_type,_lookup,_indexed_context(source),source,
                                      &parts[0],count,tuple_key,column,default)
    return _float_type(_explicit_value(source,key,column,default))


cdef class _InputFloat:
    pass


cdef double record_field_float(object function, LookupContext context, object record,
                               object key, object namespace=None) except? -1:
    """Bind the exact immutable scalar used by an ordinary typed row read."""
    cdef _InputFloat binding, retired
    if function is _lookup_float_fn and type(record) is Record:
        if context._private_owner:
            binding = context.record_float
        else:
            with cython.critical_section(context):
                binding = context.record_float
        if binding is not None and binding.source is record and binding.key is key:
            return binding.value
        value = record._index.get(key, _missing)
        if type(value) in (float, int) or type(value) in _numpy_float_keys or type(value) in _numpy_integer_keys:
            binding = _InputFloat()
            binding.source, binding.key = record, key
            binding.function = function
            binding.value = _float_type(value)
            if context._private_owner:
                context.record_float = binding
                context.record_namespace = namespace
            else:
                with cython.critical_section(context):
                    retired = context.record_float
                    context.record_float = binding
            return binding.value
    return lookup_float(function, (), record, key)


cdef object read_record_item(object source, object key):
    if type(source) is Record:
        return source._index[key]
    return source[key]


cdef double read_cast_record_field_float(object conversion, LookupContext context,
                                        object source, object key) except? -1:
    if conversion is _float_type and type(source) is Record:
        return read_record_field_float(_lookup_float_fn, context, source, key)
    return conversion(source[key])


cdef bint record_cast_ready(LookupContext context, const void* source, const void* key,
                            object conversion) noexcept:
    return conversion is _float_type and record_field_ready(context, source, key, None)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t _input_column(KeyedView data, object column) except? -2:
    cdef str label
    if type(column) is not str:
        return -1
    label = <str>column
    if label == <str>data.columns[0]:
        return 0
    if label == <str>data.columns[data.value_width - 1]:
        return data.value_width - 1
    try:
        return data.columns.index(label)
    except ValueError:
        return -1


cdef inline bint _input_value_at(NativeTable table, Py_ssize_t row, Py_ssize_t index,
                                 double* result) noexcept:
    cdef Py_ssize_t position = row * table._input_width + index
    if not table._input_numeric[position]:
        return False
    result[0] = table._input_values[position]
    return True


cdef inline bint _input_value(NativeTable table, Py_ssize_t row, object column, double* result) except -1:
    cdef Py_ssize_t index = _input_column(<KeyedView>table._indexed_view, column)
    return index >= 0 and _input_value_at(table, row, index, result)


cdef class _NativeTableIndex:
    cdef object _owner

    def __init__(self, owner):
        self._owner = owner

    def __contains__(self, key):
        return contains(_contains, self._owner, key)

    def __iter__(self):
        return iter(self._owner._keys)

    def __len__(self):
        return len(self._owner._keys)

    def max_int(self):
        return _typed_int(last_key(_last_key, self._owner))

    def max_float(self):
        return _float_type(last_key(_last_key, self._owner))


@cython.final
cdef class NativeTable:
    """A Cython receiver for a Table owned by an ordinary Cell cache."""

    def __init__(self, owner):
        cdef TableView data
        cdef KeyedView indexed
        if not isinstance(owner, Table):
            raise TypeError('declared Table Cell returned a different type')
        self._owner = owner
        if type(owner) is Table:
            self._index_view = _NativeTableIndex(owner)
        if type(owner) is Table and owner._index_error is None:
            data = view(owner)
            if data.numeric:
                self._numeric_view = data
        elif type(owner) is Table:
            keys = owner._keys
            if keys and (type(keys[0]) is int or len(keys) == 1 and type(keys[0]) is str):
                data = view(owner)
                if type(keys[0]) is int and not (data.integral and data.contiguous):
                    return
                cached = owner._native_cache.get(KeyedView)
                if cached is None:
                    cached = owner._native_cache.setdefault(KeyedView, KeyedView(owner))
                indexed = <KeyedView>cached
                if indexed.admitted and not owner._index_error['multi']:
                    self._indexed_view = indexed
                    self._input_values = &indexed.values[0]
                    self._input_numeric = &indexed.numeric[0]
                    self._input_width = indexed.value_width
                    if data.integral and data.contiguous:
                        self._input_keys = data
                        self._input_positions = &data.sorted_positions[0]
                        self._input_low = data.first_integer
                        self._input_high = data.last_integer
                    else:
                        self._single_label = keys[0]

    cdef double lookup_float(self, KeyArg0 key, object column, object default=_typed_missing) except? -1:
        cdef PyObject* data = <PyObject*>self._numeric_view
        cdef double value
        cdef Py_ssize_t col, last
        if KeyArg0 is KeyInteger:
            if data != <PyObject*>None:
                # The receiver owns the view throughout the call, as in c8's
                # bound lookup. Exact labels at either end already have a slot.
                last = (<TableView>data).width - 1
                if <PyObject*>column == PyTuple_GET_ITEM((<TableView>data).columns, 0):
                    col = 0
                elif <PyObject*>column == PyTuple_GET_ITEM((<TableView>data).columns, last):
                    col = last
                else:
                    col = numeric_column(<TableView>data, column)
                if col >= 0:
                    return _numeric_float_read(data, col, key,
                                               _missing if default is _typed_missing else default)
            if self._input_keys is not None:
                if self._input_low <= key <= self._input_high and _input_value(
                        self, self._input_positions[key - self._input_low],
                        column, &value):
                    return value
        elif KeyArg0 is object:
            if self._single_label is not None and type(key) is str and <str>key == <str>self._single_label:
                if _input_value(self, 0, column, &value):
                    return value
        if type(self._owner) is Table:
            return lookup_float(_lookup_float_fn, (), self._owner, <object>key, column, default)
        if default is _typed_missing:
            return self._owner.lookup_float(<object>key, column)
        return self._owner.lookup_float(<object>key, column, default)

    cdef object previous(self, KeyArg0 key, object column):
        cdef double value
        if KeyArg0 is KeyInteger:
            if table_previous_value(self, key, column, &value):
                return value
        return self._owner.previous(<object>key, column)

    cdef double interpolate_float(self, object key, object column) except? -1:
        cdef TableView data
        if type(self._owner) is Table and PyFloat_CheckExact(key) and math.floor is _FLOOR:
            data = view(self._owner)
            value = interpolate_numeric(data, numeric_column(data, column), PyFloat_AS_DOUBLE(key))
            if value is not None:
                return value
        return self._owner.interpolate_float(key, column)

    cpdef object row(self, object key):
        if type(self._owner) is Table:
            return row(_row, (), self._owner, key)
        return self._owner.row(key)

    cpdef object slice(self, object key):
        if type(self._owner) is Table:
            return row(_row, (), self._owner, key)
        return self._owner.slice(key)

    @property
    def index(self):
        if self._index_view is None:
            return self._owner.index
        return self._index_view

    def __getitem__(self, key):
        return self._owner[key]

    def __iter__(self):
        return iter(self._owner)

    def __len__(self):
        return len(self._owner)

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def __modelx_cython_owner__(self):
        """Expose the semantic owner when passing a view to another package."""
        return self._owner

    def __modelx_table_owner__(self):
        return self._owner

    def __modelx_table_owner_abi__(self):
        return 1

    def __eq__(self, other):
        return self._owner == other

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return NativeTable, (self._owner,)


Table.register(NativeTable)


cdef NativeTable bind_native_table(object value, str cell):
    """Bind a Table or keep a local wrapper already owned by another Cell."""
    value_type = type(value)
    if value_type is NativeTable:
        return <NativeTable>value
    if value_type is Table or Table in value_type.__mro__:
        return NativeTable(value)
    if isinstance(value, Table):
        abi = getattr(value_type, '__modelx_table_owner_abi__', None)
        owner = (getattr(value_type, '__modelx_cython_owner__', None)
                 or getattr(value_type, '__modelx_table_owner__', None))
        if callable(abi) and callable(owner) and abi(value) == 1:
            value = owner(value)
            if Table in type(value).__mro__:
                return NativeTable(value)
    raise TypeError(f'{cell} expected a Table result, got {value_type.__name__}. '
                    "Rebuild with a representative sample or set return_type='object'.")


cdef double table_field_float(LookupContext context, NativeTable table, object key,
                              object column) except? -1:
    cdef _InputFloat first, second, binding, retired
    cdef double value
    if table._single_label is not None:
        if context._private_owner:
            first, second = context.input_first, context.input_second
        else:
            with cython.critical_section(context):
                first, second = context.input_first, context.input_second
        if first is not None and first.source is table and first.key is key and first.column is column:
            return first.value
        if second is not None and second.source is table and second.key is key and second.column is column:
            return second.value
        if type(key) is str and <str>key == <str>table._single_label and _input_value(
                table, 0, column, &value):
            binding = _InputFloat()
            binding.source, binding.key, binding.column = table, key, column
            binding.value = value
            if context._private_owner:
                context.input_second = context.input_first
                context.input_first = binding
            else:
                with cython.critical_section(context):
                    retired = context.input_second
                    context.input_second = context.input_first
                    context.input_first = binding
            return value
    return table.lookup_float(<object>key, column)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint try_clamped_table_float(NativeTable table, LookupContext context, bint lower, object namespace,
                                  long long key, object column, double* result) except -1:
    cdef Py_ssize_t index, row
    cdef PyObject* snapshot = <PyObject*>table._numeric_view
    cdef TableView data
    cdef unsigned char mask = 2 if lower else 1
    cdef dict bindings
    if table._input_keys is None and table._numeric_view is None:
        return False
    if not (context._private_owner and context.clamped_bindings_ready & mask
            and context.clamped_namespace is namespace):
        if namespace is None and not lower:
            bindings = None
        elif namespace is _typed_inputs_module:
            bindings = _typed_inputs_bindings
        elif namespace is _pdn_module:
            bindings = _pdn_bindings
        else:
            return False
        if bindings is not None:
            if (bindings.get('last_key') is not _last_key
                    or (lower and bindings.get('first_key') is not _first_key)):
                return False
        if context._private_owner:
            context.clamped_bindings_ready |= mask
    if snapshot != <PyObject*>None:
        if namespace is None and not lower:
            index = endpoint_column(snapshot, column)
            if index >= 0:
                if key < (<TableView>snapshot).first_integer:
                    return False
                return bounded_table_value(snapshot, index, key, 0.0, result)
        if context._private_owner:
            data = <TableView>table._numeric_view
        else:
            index = -1
            data = context.get(table._owner, column, &index)
            if data is None or index < 0:
                return False
        if key < data.first_integer and not lower:
            return False
        key = min(max(key, data.first_integer), data.last_integer)
        row = data.right(key) - 1
        if row < 0 or data.integers[row] != key:
            return False
        if context._private_owner:
            if (context.clamped_source is table and context.clamped_column is column
                    and context.clamped_namespace is namespace):
                index = context.clamped_index
            else:
                index = numeric_column(data, column)
                if index < 0:
                    return False
                context.clamped_source = table
                context.clamped_column = column
                context.clamped_namespace = namespace
                context.clamped_index = index
        result[0] = data.numbers[data.sorted_positions[row] * data.width + index]
        return True
    if key < table._input_low:
        if not lower:
            return False
        key = table._input_low
    elif key > table._input_high:
        key = table._input_high
    if (context._private_owner and context.clamped_source is table
            and context.clamped_column is column and context.clamped_namespace is namespace):
        index = context.clamped_index
    else:
        index = _input_column(<KeyedView>table._indexed_view, column)
        if index < 0:
            return False
        if context._private_owner:
            context.clamped_source = table
            context.clamped_column = column
            context.clamped_namespace = namespace
            context.clamped_index = index
    return _input_value_at(table, table._input_positions[key - table._input_low], index, result)


cdef double prepare_table_method(LookupContext context, NativeTable table,
                                 long long key, object column, object default=_typed_missing) except? -1:
    cdef TableView data
    cdef Py_ssize_t col
    cdef object owner = table._owner
    if (context._private_owner and context.numeric_data is not None
            and owner is context.numeric_source and column is context.numeric_column_source):
        return _numeric_float_read(<PyObject*>context.numeric_data, context.numeric_index, key,
                                   _missing if default is _typed_missing else default)
    cdef double value = table.lookup_float(<KeyInteger>key, column, default)
    if context._private_owner:
        context.numeric_data = None
        if table._numeric_view is not None:
            data = <TableView>table._numeric_view
            col = numeric_column(data, column)
            remember_numeric_resource(context, owner, column, data, col)
    return value


cdef inline bint table_previous_value(NativeTable table, long long key, object column,
                                     double* value) except -1:
    cdef PyObject* data = <PyObject*>table._numeric_view
    cdef Py_ssize_t row, col
    if data == <PyObject*>None:
        return False
    col = endpoint_column(data, column)
    if col < 0:
        col = numeric_column(<TableView>data, column)
    row = (<TableView>data).right(key) - 1
    if row < 0 or col < 0:
        return False
    value[0] = (<TableView>data).numbers[(<TableView>data).sorted_positions[row]
                                        * (<TableView>data).width + col]
    return True


cdef double read_previous_table_float(NativeTable table, long long key, object column) except? -1:
    cdef double value
    if table_previous_value(table, key, column, &value):
        return value
    return table._owner.previous(key, column)


cdef double read_previous_table_int(NativeTable table, long long key, long long column) except? -1:
    cdef PyObject* data = <PyObject*>table._numeric_view
    if data != <PyObject*>None:
        return numeric_table_value(data, key, column, True, False, 0.0)
    return table._owner.previous(key, column)


cdef double prepare_previous_table_method(LookupContext context, NativeTable table,
                                          long long key, object column) except? -1:
    if table._numeric_view is not None:
        if context._private_owner:
            return prepare_previous_f64(_previous, context, table._owner, key, column)
        return previous_f64(_previous, context, table._owner, key, column)
    if context._private_owner:
        context.numeric_data = None
    return table._owner.previous(key, column)


cdef bint read_bounded_table_float(NativeTable table, LookupContext context, object first_fn,
                                   long long key, str column, double below, double* value,
                                   bint owned_index=False) except? -1:
    cdef PyObject* data = <PyObject*>table._numeric_view
    cdef Py_ssize_t last
    if (table._index_view is None if owned_index else first_fn is not _first_key):
        return False
    if data != <PyObject*>None:
        if <PyObject*>column == PyTuple_GET_ITEM((<TableView>data).columns, 0):
            return bounded_table_value(data, 0, key, below, value)
        last = (<TableView>data).width - 1
        if <PyObject*>column == PyTuple_GET_ITEM((<TableView>data).columns, last):
            return bounded_table_value(data, last, key, below, value)
    if context._private_owner:
        if context.clamped_source is table and context.clamped_column is column:
            if bound_bounded_f64(context, key, below, value):
                return True
        else:
            context.bounded_data = None
    if not bounded_lookup_f64(_lookup, _first_key, _last_key, context, table._owner, key, column, below, value):
        return False
    if context._private_owner:
        context.clamped_source = table
        context.clamped_column = column
    return True


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


import pandas as _pandas
from pandas import NA as _pandas_NA, NaT as _pandas_NaT
from modelx_helpers.pandas_inputs import isna as _prepared_isna
cdef object _is_missing_fn = _native_operations['is_missing']
cdef object _pandas_isna = _pandas.isna
_isna_origin = getattr(_pandas_isna, '__module__', None)
if (_pandas_isna is not getattr(_pandas, 'isnull', None)
        or not isinstance(_isna_origin, str) or _isna_origin.partition('.')[0] != 'pandas'):
    # A replacement installed before import must retain its own semantics.
    _pandas_isna = object()


cdef inline int _scalar_isna(object value) except -2:
    cdef double number
    if value is None or value is _pandas_NA or value is _pandas_NaT:
        return 1
    if type(value) in (int, bool) or type(value) in _numpy_integer_keys or type(value) is _numpy_bool_key:
        return 0
    if isinstance(value, (str, bytes)):
        return 0
    if type(value) is float or type(value) in _numpy_float_keys:
        number = PyFloat_AsDouble(value)
        return number != number
    return -1


cdef object isna(object function, object value):
    cdef int missing
    if function is _prepared_isna and _pandas.isna is _pandas_isna:
        missing = _scalar_isna(value)
        if missing >= 0:
            return missing != 0
    return function(value)


cdef bint is_missing(object function, tuple keywords, object predicate, object value) except? -1:
    cdef int missing
    if function is not _is_missing_fn:
        return forward_call(function, keywords, ('predicate', 'value'),
                            (predicate, value), omitted_argument)
    if predicate is _pandas_isna:
        missing = _scalar_isna(value)
        if missing >= 0:
            return missing
    return _typed_bool(predicate(value))


from types import MappingProxyType
from cpython.object cimport PyObject_GenericSetAttr, PyTypeObject
from modelx_helpers.table_operations import row as _row, column as _column

# Cython 3.2.9's bundled declaration takes PyObject*, whereas Python 3.12's
# actual API takes PyTypeObject*. Keep the real signature for GCC14 as well.
cdef extern from "Python.h":
    object _record_alloc "PyType_GenericAlloc"(PyTypeObject*, Py_ssize_t)

cdef tuple _prepared_row_metadata(tuple metadata, tuple columns, bint preserve_scalar_types):
    cdef object selected
    cdef tuple items
    cdef list records = []
    cdef Py_ssize_t i
    for i, values in enumerate(metadata[4]):
        items = tuple(zip(columns, values))
        selected = _record_alloc(<PyTypeObject*>Record, 0)
        PyObject_GenericSetAttr(selected, '_items', items)
        PyObject_GenericSetAttr(selected, '_index', MappingProxyType(dict(items)))
        PyObject_GenericSetAttr(selected, '_native_groups', {})
        PyObject_GenericSetAttr(selected, '_preserve_scalar_types', preserve_scalar_types)
        PyObject_GenericSetAttr(selected, '_metadata', ('row', metadata[2], metadata[5][i],
                                                     metadata[8][i], metadata[6], metadata[7]))
        records.append(selected)
    # Keep the legacy scalar tuples readable by previously compiled models.
    return (*metadata, tuple(records))


cdef object row(object function, tuple keywords, object table, object key):
    cdef Py_ssize_t position
    cdef tuple metadata
    cdef object prepared, index, spec
    if function is not _row or type(table) is not Table:
        return forward_call(function, keywords, ('table', 'key'), (table, key), omitted_argument)
    metadata = table._metadata
    if metadata is None or metadata[0] != 'frame':
        raise ValueError('row requires table(..., rows=True) with pandas row metadata')
    if metadata[4] is None:
        raise ValueError('row requires table(..., rows=True)')
    prepared = metadata[10] if len(metadata) > 10 else None
    if prepared is not None and type(key) is int:
        # A matching exact integer label can read the prepared row through
        # the existing key index. Other domains keep their pandas key rules.
        index = table._positions.get(key)
        if index is not None and type(table._keys[index]) is int:
            return prepared[index]
    if type(key) is bool and table._index_error is not None:
        spec = table._index_error._index
        if not spec.get('row_bool_labels', spec['boolean_modes'][0] == 1):
            raise KeyError(f'{key}: boolean label can not be used without a boolean index')
    position = indexed_position(table, key)
    if position < 0:
        return forward_call(function, keywords, ('table', 'key'), (table, key), omitted_argument)
    if prepared is not None:
        return prepared[position]
    return _row(table, key)


cdef object column(object function, tuple keywords, object table, object name, object report=_missing):
    cdef Py_ssize_t position, i
    cdef tuple columns, values
    cdef list result
    cdef object metadata, carrier, items, series_metadata, containers
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
    series_metadata = ('series', metadata[1], metadata[3][position], name, metadata[6], metadata[7])
    if len(metadata) > 11:
        containers = {i: row[position] for i, row in metadata[11].items() if position in row}
        if containers:
            series_metadata = (*series_metadata, containers)
    PyObject_GenericSetAttr(carrier, '_metadata', series_metadata)
    return carrier


from modelx_helpers.table_operations import group_sum as _group_sum
import numpy as _group_numpy
cdef object _group_float64 = _group_numpy.dtype('float64')


cdef object _group_reduction():
    # A monotonic float64 group may use NumPy summation. Observe the installed
    # pandas behavior on a cancellation case that distinguishes it from the
    # compensated loop, using the same public groupby API as the Python helper.
    reduceat = _group_numpy.add.reduceat
    cases = (([1e16, 1.0, -1e16, 3.0], 3.0),
             ([1e16, *([1.0] * 128), -1e16], 128.0))
    for terms, compensated in cases:
        values = _group_numpy.array(terms, dtype='float64')
        numpy_total = reduceat(values, [0])[0]
        if numpy_total == compensated:
            continue
        frame = _pandas.DataFrame({'value': values})
        pandas_total = frame.groupby(_pandas.Index([0] * len(values))).sum().iat[0, 0]
        return reduceat if pandas_total == numpy_total else None
    return None


cdef object _group_reduceat = _group_reduction()


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
            if _group_reduceat is not None and not missing_sum:
                # Pandas checks the whole summed block before dropping rows.
                for j in range(width):
                    if modes[j] == 0:
                        number = source_values[i][source_columns[j]]
                        if number != number:
                            missing_sum = True
                            break
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
    # For the monotonic nonmissing domain, retain the installed pandas
    # reduction mode observed during provider initialization.
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
from modelx_helpers.pandas_inputs import table as _table_constructor
cdef object _report_int64 = _group_numpy.dtype('int64')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _float_frame_rows(object frame, bint with_rows, bint preserve_scalar_types):
    """Copy the homogeneous float matrix through pandas' public array boundary."""
    cdef object data = frame.to_numpy(copy=False)
    cdef const double[:, :] numbers = data
    cdef Py_ssize_t i, j, height = numbers.shape[0], width = numbers.shape[1]
    cdef list values = [], row_values = []
    cdef tuple row = ()
    for i in range(height):
        if with_rows or preserve_scalar_types:
            row = tuple([data[i, j] for j in range(width)])
            if with_rows:
                row_values.append(row)
        if preserve_scalar_types:
            values.append(row)
        else:
            values.append(tuple([numbers[i, j] for j in range(width)]))
    return tuple(values), tuple(row_values) if with_rows else None



# Physical mixed-column preparation for the existing inputs.table contract.
from cpython.object cimport PyObject_TypeCheck
from cpython.long cimport PyLong_Check, PyLong_CheckExact
from cpython.float cimport PyFloat_Check
from cpython.unicode cimport PyUnicode_Check, PyUnicode_CheckExact
from cpython.bytes cimport PyBytes_Check, PyBytes_CheckExact
from pandas import StringDtype as _frame_string_dtype_type
cdef object _frame_object_dtype = _group_numpy.dtype('object')
cdef tuple _frame_numeric_scalar_types = (_numpy_integer_keys + _numpy_float_keys
    + (_group_numpy.longdouble, _numpy_bool_key, _group_numpy.str_, _group_numpy.bytes_))


cdef object _frame_normalize(object value):
    # The public bulk object conversion supplies Python scalars for ordinary
    # NumPy columns. Avoid checking the NumPy scalar registry for each one.
    if (value is None or value is True or value is False
            or PyLong_CheckExact(value) or PyFloat_CheckExact(value)
            or PyUnicode_CheckExact(value) or PyBytes_CheckExact(value)):
        return value
    if type(value) in _frame_numeric_scalar_types:
        return value.item()
    if PyObject_TypeCheck(value, <PyTypeObject*>_group_numpy.generic):
        return _missing
    if value is None or value is _pandas_NA or value is _pandas_NaT:
        return value
    if PyLong_Check(value) or PyFloat_Check(value) or PyUnicode_Check(value) or PyBytes_Check(value):
        return value
    return _missing


cdef object _frame_axes(object frame, bint with_rows):
    cdef object column_axis = frame.columns
    cdef object index = frame.index
    cdef object data, column_dtype, column_name, index_dtype, index_metadata, value
    cdef tuple columns, keys, row_names
    cdef dict positions
    if type(column_axis) is not _report_index_type:
        return None
    columns = tuple(column_axis)
    column_dtype = column_axis.dtype
    column_name = column_axis.name
    if not columns or len(set(columns)) != len(columns):
        return None
    for value in columns:
        if type(value) is not str:
            return None
    if type(index) is _report_range_type:
        keys = tuple(index)
        row_names = keys if with_rows else ()
        index_metadata = ('range', index.start, index.stop, index.step, index.name)
    elif type(index) is _report_index_type:
        index_dtype = index.dtype
        if (not isinstance(index_dtype, _group_numpy.dtype) or index_dtype.kind == 'b'
                or len(index) and index_dtype != _report_int64):
            return None
        data = index.to_numpy(copy=False)
        keys = tuple(int(value) for value in data)
        row_names = tuple(data) if with_rows else ()
        index_metadata = ('labels', keys, index.name, index_dtype)
    else:
        return None
    positions = {key: i for i, key in enumerate(keys)}
    if len(positions) != len(keys):
        return None  # Reference preparation preserves conversion/error ordering.
    return (columns, keys, row_names, index_metadata,
            ('labels', columns, column_name, column_dtype),
            (column_dtype, column_name), positions,
            keys if type(index) is _report_range_type and index.step > 0 else tuple(sorted(keys)))


cdef object _frame_row_dtype(tuple dtypes):
    cdef object first = dtypes[0]
    cdef object dtype
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
            if dtype != first:
                return _missing
        else:
            same = False
    if same:
        return first
    if not all_numpy or any_boolean and any_other:
        return _frame_object_dtype
    return _group_numpy.result_type(*dtypes)


cdef object _frame_owner(tuple keys, tuple values, tuple columns, object metadata,
                         dict positions, tuple ordered, bint preserve_scalar_types):
    cdef object spec, result
    cdef tuple items = (('multi', False), ('pandas_columns', True),
        ('row_bool_labels', False), ('boolean_modes', (2,)))
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
    PyObject_GenericSetAttr(result, '_positions', MappingProxyType(positions))
    PyObject_GenericSetAttr(result, '_ordered', ordered)
    PyObject_GenericSetAttr(result, '_native_cache', {})
    PyObject_GenericSetAttr(result, '_index_error', spec)
    PyObject_GenericSetAttr(result, '_metadata', metadata)
    PyObject_GenericSetAttr(result, '_preserve_scalar_types', preserve_scalar_types)
    return result


cdef object _frame_columns(object frame):
    """Read the dtype and array from each public column in one pass."""
    cdef object column, dtype
    cdef list arrays = [], dtypes = []
    for _, column in frame.items():
        dtype = column.dtype
        if isinstance(dtype, _group_numpy.dtype) and dtype.kind in ('b', 'i', 'u', 'f', 'c', 'O'):
            arrays.append(column.to_numpy(copy=False))
        elif type(dtype) is _frame_string_dtype_type:
            arrays.append(column.array)
        else:
            return None
        dtypes.append(dtype)
    return arrays, tuple(dtypes)


cdef object _native_frame_table(object frame, bint with_rows, bint with_metadata,
                                bint preserve_scalar_types):
    cdef object axes, dtype, row_dtype, row_type, raw, value, metadata, row_values_tuple, row, column_data
    cdef tuple columns, keys, row_names, dtypes, values, row_casts
    cdef list arrays = [], copied_rows, row_values, copied, selected
    cdef Py_ssize_t height, width, i, j
    cdef bint homogeneous_float
    if type(frame) is not _report_frame_type or (with_metadata and frame.attrs):
        return None
    axes = _frame_axes(frame, with_rows)
    if axes is None:
        return None
    columns, keys, row_names = axes[0], axes[1], axes[2]
    width, height = len(columns), len(keys)
    if with_rows and preserve_scalar_types:
        # Raw rows already need each column array. Avoid building and
        # iterating a second pandas Series solely for its dtype metadata.
        column_data = _frame_columns(frame)
        if column_data is None:
            return None
        arrays, dtypes = column_data
    else:
        dtypes = tuple(frame.dtypes)
    homogeneous_float = not arrays and all(dtype == _group_float64 for dtype in dtypes)
    row_dtype = _group_float64 if homogeneous_float else (_frame_row_dtype(dtypes) if with_rows else None)
    if row_dtype is _missing:
        return None
    if homogeneous_float:
        # Reading a homogeneous frame as one matrix keeps its native
        # value-copy loop and avoids constructing a Series for every column.
        values, row_values_tuple = _float_frame_rows(frame, with_rows, preserve_scalar_types)
    elif (not preserve_scalar_types and (not with_rows or row_dtype is _frame_object_dtype)
          and all(type(dtype) is _frame_string_dtype_type
                  or isinstance(dtype, _group_numpy.dtype) and dtype.kind in ('b', 'i', 'u', 'f', 'O')
                  for dtype in dtypes)):
        # Object conversion keeps mixed integers and floats exact. Read the
        # values in one public call without constructing a Series per column.
        copied_rows, row_values = [], []
        row_casts = (tuple(dtype.type if isinstance(dtype, _group_numpy.dtype)
                           and dtype.kind != 'O' else None for dtype in dtypes)
                     if with_rows else ())
        for row in frame.to_numpy(dtype=object, copy=False).tolist():
            copied = []
            for raw in row:
                value = _frame_normalize(raw)
                if value is _missing:
                    return None
                copied.append(value)
            copied_rows.append(tuple(copied))
            if with_rows:
                row_values.append(tuple(raw if cast is None else cast(raw)
                                        for cast, raw in zip(row_casts, row)))
        values = tuple(copied_rows)
        row_values_tuple = tuple(row_values) if with_rows else None
    else:
        if not arrays:
            column_data = _frame_columns(frame)
            if column_data is None:
                return None
            arrays, dtypes = column_data
        row_type = (row_dtype.type if with_rows and isinstance(row_dtype, _group_numpy.dtype)
                    and row_dtype != _frame_object_dtype else None)
        copied_rows, row_values = [], []
        for i in range(height):
            copied, selected = [], []
            for j in range(width):
                raw = arrays[j][i]
                if with_rows:
                    selected.append(raw if row_type is None or type(raw) is row_type else row_type(raw))
                value = raw if preserve_scalar_types else _frame_normalize(raw)
                if value is _missing:
                    return None
                copied.append(value)
            copied_rows.append(tuple(copied))
            if with_rows:
                row_values.append(tuple(selected))
        values = tuple(copied_rows)
        row_values_tuple = tuple(row_values) if with_rows else None
    if with_metadata:
        metadata = ('frame', axes[3], axes[4], dtypes, row_values_tuple,
            (row_dtype,) * height if with_rows else (), {},
            frame.flags.allows_duplicate_labels, row_names, axes[5])
        if with_rows:
            metadata = _prepared_row_metadata(metadata, columns, preserve_scalar_types)
    else:
        metadata = ('axes', axes[3], axes[4], dtypes)
    return _frame_owner(keys, values, columns, metadata, axes[6], axes[7], preserve_scalar_types)



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
        prepared = _native_frame_table(frame, rows is True, rows is True or report is True,
                                        preserve_scalar_types is True)
        if prepared is not None:
            return prepared
    return forward_call(function, keywords, ('frame', 'columns', 'groups', 'indexing', 'rows', 'report', 'preserve_scalar_types'),
                        (frame, columns, groups, indexing, rows, report, preserve_scalar_types), omitted_argument)
