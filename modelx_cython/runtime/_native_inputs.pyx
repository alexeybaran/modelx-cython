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
from modelx_helpers.immutable_inputs import Table, Record, _index_key, _valid_index_key
from modelx_helpers import immutable_inputs as _immutable_inputs
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
cdef object _rectangular_storage = getattr(_immutable_inputs, '_RECTANGULAR_STORAGE', object())
cdef object _contiguous_storage = getattr(_immutable_inputs, '_CONTIGUOUS_STORAGE', object())


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
    cdef tuple values, columns
    cdef dict column_positions
    cdef long long[::1] integers
    cdef long long[::1] sorted_positions
    cdef long long first_integer, last_integer
    cdef bint integral
    cdef bint numeric
    cdef bint contiguous
    cdef Py_ssize_t width
    cdef double[::1] numbers

    def __init__(self, table):
        cdef object key
        cdef tuple keys = table._keys
        cdef tuple ordered
        if not keys:
            return
        for key in keys:
            if type(key) is not int:
                return
        ordered = table._ordered
        low, high = ordered[0], ordered[-1]
        if not -(1 << 63) <= low <= high < (1 << 63):
            return
        self.integral = True
        self.values = table._values
        self.columns = table._columns
        self.column_positions = dict(zip(self.columns, range(len(self.columns))))
        self.integers = array('q', ordered)
        self.sorted_positions = array('q', [table._positions[key] for key in ordered])
        self.first_integer = low
        self.last_integer = high
        # Table guarantees unique keys. Compute the span as Python ints.
        self.contiguous = high - low == len(ordered) - 1
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

    cdef inline Py_ssize_t position(self, long long key, bint preceding, bint clamp) noexcept:
        cdef Py_ssize_t row
        if clamp and key >= self.last_integer:
            return self.integers.shape[0] - 1
        row = self.right(key) - 1
        if row >= 0 and (preceding or self.integers[row] == key):
            return row
        return -1

    cdef inline double number_at(self, Py_ssize_t row, Py_ssize_t column) noexcept:
        return self.numbers[self.sorted_positions[row] * self.width + column]

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
    return <TableView>table._native_cache.get(TableView)


cdef object lookup(object function, object table, object key, object column, object default=_missing):
    if function is _lookup and type(table) is Table and table._index_error is not None:
        return indexed_lookup(table,key,column,default)
    if (function is _lookup and type(table) is Table
            and (type(column) in (str, int) or isinstance(column, IntEnum))
            and _table_view_key(key)):
        return _read_owned_table(table, key, column, default)
    return function(table, key, column) if default is _missing else function(table, key, column, default)


cdef object previous(object function, object table, object key, object column, object default=_missing):
    cdef TableView data
    cdef Py_ssize_t index
    if function is _previous and type(table) is NativeTable:
        table = (<NativeTable>table)._owner
    if (function is _previous and type(table) is Table
            and type(key) is int and -(1 << 63) <= key < (1 << 63)):
        data = view(table)
        if data is not None and data.integral:
            index = data.position(key, True, False)
            if index >= 0:
                return data.value(data.sorted_positions[index], column)
            if default is not _missing:
                return default
            raise ValueError(f'No key at or before {key!r}')
    return function(table, key, column) if default is _missing else function(table, key, column, default)


cdef object contains(object function, object table, object key):
    if function is _contains and type(table) is NativeTable:
        table = (<NativeTable>table)._owner
    if function is _contains and type(table) is Table and table._index_error is not None:
        return indexed_contains(table,key)
    if function is _contains and type(table) is Table:
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
        if data is not None and data.integral:
            return data.first_integer
        return min(table._keys)
    return function(table)


cdef object last_key(object function, object table):
    cdef TableView data
    if function is _last_key and type(table) is NativeTable:
        table = (<NativeTable>table)._owner
    if function is _last_key and type(table) is Table:
        data = view(table)
        if data is not None and data.integral:
            return data.last_integer
        return max(table._keys)
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
    cdef TableView data
    if function is not _prepared_interpolation:
        return function(table, column, years, lookup_fn, last_fn, math_module)
    if (lookup_fn is not _lookup or last_fn is not _last_key
            or math_module is not math or math.floor is not _FLOOR
            or type(table) is not Table or not PyLong_Check(column) or not PyFloat_CheckExact(years)):
        return None
    col = PyLong_AsLongLongAndOverflow(column, &overflow)
    if overflow:
        return None
    data = view(table)
    if data is None:
        return function(table, column, years, lookup_fn, last_fn, math_module)
    return interpolate_numeric(data, col, PyFloat_AS_DOUBLE(years))


cdef object interpolate_numeric(TableView data, long long col, double position):
    cdef long long lower, upper
    cdef Py_ssize_t lo, hi
    cdef double base, end
    if (data is None or not data.numeric or not 0 <= col < data.width
            or not -9223372036854775808.0 <= position < 9223372036854775808.0):
        return None
    lower = <long long>floor(position)
    upper = data.last_integer
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
            index = data.position(key, False, False)
            if index < 0:
                if default is _missing:
                    raise KeyError(key)
                return default
            if col >= 0:
                return data.number_at(index, col)
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
            index = data.position(key, True, False)
            if index < 0:
                if default is _missing:
                    raise ValueError(f'No key at or before {key!r}')
                return default
            if col >= 0:
                return data.number_at(index, col)
    return previous(function, table, key, column) if default is _missing else previous(function, table, key, column, default)


cdef inline double _lookup_clamped_f64(object function, LookupContext context, object table,
                                       long long key, object last_function, object column) except? -1:
    cdef TableView data = None
    cdef Py_ssize_t index, col
    if function is _lookup and last_function is _last_key:
        col = -1
        if context is not None:
            data = context.get(table, column, &col)
        elif type(table) is Table:
            data = view(table)
        if data is not None and data.numeric:
            index = data.position(key, False, True)
            if index < 0:
                raise KeyError(key)
            if context is None:
                col = numeric_column(data, column)
            if col >= 0:
                return data.number_at(index, col)
    return lookup(function, table, min(key, last_key(last_function, table)), column)


cdef double lookup_clamped_f64(object function, object table, long long key,
                               object last_function, object column) except? -1:
    return _lookup_clamped_f64(function, None, table, key, last_function, column)


cdef double lookup_clamped_context_f64(object function, LookupContext context, object table,
                                       long long key, object last_function, object column) except? -1:
    return _lookup_clamped_f64(function, context, table, key, last_function, column)


cdef inline bint bounded_table_value(const PyObject* data, Py_ssize_t column, long long key,
                                     double below, double* value) except? -1:
    cdef Py_ssize_t row
    if key < (<TableView>data).integers[0]:
        value[0] = below
        return True
    row = (<TableView>data).position(key, False, True)
    if row < 0:
        return False
    value[0] = (<TableView>data).number_at(row, column)
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
        if data is not None and data.numeric:
            index = data.position(key, False, False)
            if index < 0:
                return 0
            col = numeric_column(data, column)
            if col >= 0:
                result[0] = data.number_at(index, col)
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
    cdef object key, scalar, ordered, positions
    cdef tuple rows
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
        ordered = table._ordered
        if ordered is None or column not in table._columns:
            return None
        col = table._columns.index(column)
        positions, rows = table._positions, table._values
        groups.append(number)
        for key in ordered:
            if not PyLong_CheckExact(key) or not -(1 << 63) <= key < (1 << 63):
                return None
            scalar = rows[positions[key]][col]
            if not PyFloat_CheckExact(scalar):
                return None
            keys.append(key)
            values.append(scalar)
        offsets.append(len(keys))
    return GroupedView(groups, keys, offsets, values)


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
    return probe_group(record._native_groups.get(column), group, slot)


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
    prepared = record._native_groups.get(column)
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
    cdef Py_ssize_t row = (<TableView>data).position(key, preceding, False)
    if row < 0:
        if has_default:
            return default
        if preceding:
            raise ValueError(f'No key at or before {key!r}')
        raise KeyError(key)
    if column < 0:
        column += (<TableView>data).width
    if column < 0 or column >= (<TableView>data).width:
        raise IndexError('tuple index out of range')
    return (<TableView>data).number_at(row, column)


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
    cdef Py_ssize_t index = (<TableView>data).position(key, False, False)
    if index < 0:
        if default is _missing:
            raise KeyError(key)
        if PyFloat_CheckExact(default):
            return PyFloat_AS_DOUBLE(default)
        return _float_type(default)
    return (<TableView>data).number_at(index, column)

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
        # These slots serve explicit float reads. Keep the original objects
        # for ordinary selection and materialization.
        cdef Array numbers = clone(array('d'), self.rows * self.value_width, False)
        cdef Array valid = clone(array('B'), self.rows * self.value_width, True)
        offset = 0
        for input_row in self.raw_values:
            for value in input_row:
                if PyFloat_CheckExact(value):
                    numbers.data.as_doubles[offset] = PyFloat_AS_DOUBLE(value)
                    valid.data.as_uchars[offset] = 1
                elif type(value) in _numpy_integer_keys:
                    numbers.data.as_doubles[offset] = PyFloat_AsDouble(value)
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

cdef Py_ssize_t _owned_position(object table, object key) except? -2:
    cdef object spec = table._index_error
    cdef object position
    cache = table._native_cache
    rectangular = cache.get(_rectangular_storage)
    contiguous = cache.get(_contiguous_storage)
    if rectangular is not None or contiguous is not None:
        try:
            return table._compact_position(key, rectangular, contiguous)
        except KeyError:
            return -1
    if spec is not None:
        spec = spec._index
        if not _valid_index_key(key, spec):
            return -1
        key = _index_key(key, spec)
    position = table._positions.get(key)
    return -1 if position is None else position


cdef object _read_owned_table(object table, object key, object column, object default):
    """Select one row through the index already owned by the Python table."""
    cdef object spec = table._index_error
    cdef object name
    cdef Py_ssize_t position
    cdef tuple columns = table._columns
    cache = table._native_cache
    if _rectangular_storage in cache or _contiguous_storage in cache:
        return _lookup(table, key, column) if default is _missing else _lookup(table, key, column, default)
    if spec is not None and type(column) is str and spec._index.get('pandas_columns', False):
        name = column
        try:
            column = columns.index(name)
        except ValueError:
            raise KeyError(name) from None
    try:
        position = _owned_position(table, key)
        if position < 0:
            raise KeyError(key)
    except KeyError:
        if default is _missing:
            raise
        return default
    if type(column) is str:
        column = columns.index(column)
    return table._values[position][column]


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
            if data is not None and data.admitted:
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
        if type(table) is Table and (type(column) is str or type(column) is int or type(column) is bool):
            return conversion(_read_owned_table(table, _original_key(parts, count, tuple_key), column, default))
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


cdef KeyPart* _pack_key(object key, KeyPart* small, Py_ssize_t* count,
                         bint* tuple_key) except? NULL:
    cdef KeyPart* parts = small
    cdef Py_ssize_t i
    tuple_key[0] = type(key) is tuple
    if not tuple_key[0]:
        count[0] = 1
        _pack_part(key, small)
        return small if _native_parts(small, 1) else NULL
    count[0] = len(key)
    if count[0] > 4:
        if count[0] > _key_sys.maxsize // sizeof(KeyPart):
            raise MemoryError()
        parts = <KeyPart*>PyMem_Malloc(count[0] * sizeof(KeyPart))
        if parts == NULL:
            raise MemoryError()
    try:
        for i in range(count[0]):
            _pack_part((<tuple>key)[i], &parts[i])
        if _native_parts(parts, count[0]):
            return parts
    except:
        if parts != small:
            PyMem_Free(parts)
        raise
    if parts != small:
        PyMem_Free(parts)
    return NULL


cdef object indexed_lookup(object table, object key, object column, object default=_missing):
    cdef KeyPart[4] small
    cdef Py_ssize_t count, position
    cdef bint tuple_key
    cdef KeyPart* parts = _pack_key(key, &small[0], &count, &tuple_key)
    cdef bint packed = parts != NULL
    cdef KeyedCache cache
    cdef KeyedView data
    if packed:
        try:
            cache = _keyed_cache(_indexed_context(table), table, column)
            data = cache.data
            if data is not None and cache.index >= 0:
                position = data.find(parts, count, tuple_key)
                if position >= 0:
                    return data.raw_values[position][cache.index]
                if default is _missing:
                    raise KeyError(key)
                return default
        finally:
            if parts != &small[0]:
                PyMem_Free(parts)
    if (packed or count <= 4) and (type(column) is str or type(column) is int or type(column) is bool):
        return _read_owned_table(table, key, column, default)
    return _lookup(table, key, column) if default is _missing else _lookup(table, key, column, default)


cdef Py_ssize_t _indexed_key_position(object table, object key) except? -3:
    cdef KeyPart[4] small
    cdef Py_ssize_t count
    cdef bint tuple_key
    cdef KeyPart* parts = _pack_key(key, &small[0], &count, &tuple_key)
    cdef KeyedView data
    if parts == NULL:
        return -2
    try:
        data = table._native_cache.get(KeyedView)
        if data is not None and data.admitted:
            return data.find(parts, count, tuple_key)
        return _owned_position(table, key)
    finally:
        if parts != &small[0]:
            PyMem_Free(parts)


cdef Py_ssize_t indexed_position(object table, object key) except? -2:
    """Native unique-row position, or -1 when the key domain is not admitted."""
    cdef Py_ssize_t position = _indexed_key_position(table, key)
    if position == -1:
        raise KeyError(key)
    return -1 if position == -2 else position


cdef object indexed_contains(object table, object key):
    cdef Py_ssize_t position = _indexed_key_position(table, key)
    return _contains(table, key) if position == -2 else position >= 0


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


cdef class RectangularView:
    cdef const double[::1] numbers
    cdef tuple positions, widths
    cdef object column

    def __init__(self, tuple storage, column):
        axes, data = storage[:2]
        self.numbers = data
        self.column = column
        self.widths = tuple(map(len, axes))
        self.positions = (storage[2] if len(storage) == 3 else
                          tuple(dict(zip(labels, range(len(labels)))) for labels in axes))

    cdef inline bint read(self, object key, object column, double* value) except -1:
        cdef Py_ssize_t row = 0, axis, width = len(self.widths)
        cdef object part, digit
        if ((column is not self.column and (type(column) is not str or column != self.column))
                or type(key) is not tuple or len(key) != width):
            return False
        for axis in range(width):
            part = (<tuple>key)[axis]
            if not PyLong_CheckExact(part) and not PyUnicode_CheckExact(part):
                return False
            digit = self.positions[axis].get(part)
            if digit is None:
                return False
            row = row * <Py_ssize_t>self.widths[axis] + <Py_ssize_t>digit
        value[0] = self.numbers[row]
        return True


cdef class ContiguousView:
    cdef long long low, high
    cdef const long long[:, ::1] integers
    cdef const double[:, ::1] floats
    cdef tuple columns
    cdef bint integral

    def __init__(self, long long low, data, tuple columns):
        self.low = low
        self.high = low + (data.shape[0] - 1)
        self.columns = columns
        self.integral = data.dtype == _key_numpy.dtype('int64')
        if self.integral:
            self.integers = data
        else:
            self.floats = data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline bint read(self, long long key, object column, double* value) except -1:
        cdef Py_ssize_t pos, row
        if key < self.low or key > self.high or type(column) is not str:
            return False
        if <PyObject*>column == PyTuple_GET_ITEM(self.columns, 0):
            pos = 0
        else:
            try:
                pos = self.columns.index(column)
            except ValueError:
                return False
        row = <Py_ssize_t>(key - self.low)
        value[0] = <double>self.integers[row, pos] if self.integral else self.floats[row, pos]
        return True


cdef double lookup_float(object function, tuple keywords, object source, object key, object column=_missing, object default=_missing) except? -1:
    cdef KeyPart[4] small
    cdef KeyPart* parts
    cdef Py_ssize_t count
    cdef bint tuple_key
    cdef RectangularView rectangular
    cdef ContiguousView contiguous
    cdef double value
    cdef int overflow
    cdef long long integer
    if function is not _lookup_float_fn:
        return forward_call(function, keywords, ('source', 'key', 'column', 'default'),
                            (source, key, column, default), omitted_argument)
    if column is _typed_missing:
        column = _missing
    if default is _typed_missing:
        default = _missing
    if type(source) is NativeTable and type((<NativeTable>source)._owner) is Table:
        source = (<NativeTable>source)._owner
    if type(source) is Table and column is not _missing:
        rectangular = source._native_cache.get(RectangularView)
        if rectangular is not None and rectangular.read(key, column, &value):
            return value
        contiguous = source._native_cache.get(ContiguousView)
        if contiguous is not None and PyLong_CheckExact(key):
            integer = PyLong_AsLongLongAndOverflow(key, &overflow)
            if not overflow and contiguous.read(integer, column, &value):
                return value
        parts = _pack_key(key, &small[0], &count, &tuple_key)
        if parts != NULL:
            try:
                return lookup_keyed_float_f64(_float_type, _lookup, _indexed_context(source), source,
                                              parts, count, tuple_key, column, default)
            finally:
                if parts != &small[0]:
                    PyMem_Free(parts)
        if count <= 4 and (type(column) is str or type(column) is int or type(column) is bool):
            return _float_type(_read_owned_table(source, key, column, default))
        return _float_type(_lookup(source, key, column) if default is _missing
                           else _lookup(source, key, column, default))
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
                                 double* result) except -1:
    cdef Py_ssize_t position = row * table._input_width + index
    if table._input_numeric[position]:
        result[0] = table._input_values[position]
        return True
    cdef object scalar = (<KeyedView>table._indexed_view).raw_values[row][index]
    if type(scalar) is not int and type(scalar) is not bool and type(scalar) is not str:
        return False
    try:
        result[0] = _key_float(scalar)
    except (ValueError, OverflowError):
        return False
    return True


cdef inline bint _input_value(NativeTable table, Py_ssize_t row, object column, double* result) except -1:
    cdef Py_ssize_t index = _input_column(<KeyedView>table._indexed_view, column)
    return index >= 0 and _input_value_at(table, row, index, result)


cdef class _NativeTableIndex:
    """The index of a NativeTable, answering exactly as the live helper's index (D54).

    Compiled code reaches it only for index reads the lowering left to Python (W1.3 lowers
    the target-syntax reads to the typed handle). Every member asks the helper's own index,
    so a Python read gives the live answer: ``k in index`` is a prefix test on a multi-level
    table (D27a), iteration and ``len`` read the table's storage, ``max_int``/``max_float``
    and the pandas vocabulary (``min``, ``max``, ``asof``, ``searchsorted``, ``index[i]``)
    are the helper's.
    """
    cdef object _owner

    def __init__(self, owner):
        self._owner = owner

    def __contains__(self, key):
        return key in self._owner.index

    def __iter__(self):
        return iter(self._owner.index)

    def __len__(self):
        return len(self._owner.index)

    def max_int(self):
        return self._owner.index.max_int()

    def max_float(self):
        return self._owner.index.max_float()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._owner.index, name)

    def __getitem__(self, key):
        return self._owner.index[key]


@cython.final
cdef class NativeTable:
    """A Cython receiver for a Table owned by an ordinary Cell cache."""

    def __init__(self, owner):
        cdef TableView data
        cdef KeyedView indexed
        if not isinstance(owner, Table):
            raise TypeError('declared Table Cell returned a different type')
        self._owner = owner
        # M1 (W1.2): the typed handle built at preparation (compiled pdn.table) or by the first
        # table_handle() request; a bind only finds it and never writes the table's cache
        self.handle = owner._native_cache.get(TableHandle) if type(owner) is Table else None
        if type(owner) is Table:
            self._index_view = _NativeTableIndex(owner)
            self._numeric_view = None
            self._contiguous_view = owner._native_cache.get(ContiguousView)
            if _contiguous_storage in owner._native_cache:
                return
            if _rectangular_storage in owner._native_cache:
                return
        if type(owner) is Table:
            keys = owner._keys
            if keys and (type(keys[0]) is int or len(keys) == 1 and type(keys[0]) is str):
                data = view(owner)
                spec = owner._index_error
                if spec is None and data is not None and data.numeric:
                    self._numeric_view = data
                    return
                if spec is not None and spec._index['multi']:
                    return
                cached = owner._native_cache.get(KeyedView)
                if data is not None and data.integral and data.contiguous:
                    self._input_keys = data
                    self._input_positions = &data.sorted_positions[0]
                    self._input_low = data.first_integer
                    self._input_high = data.last_integer
                if cached is not None:
                    indexed = <KeyedView>cached
                    if not indexed.admitted:
                        return
                    self._indexed_view = indexed
                    self._input_values = &indexed.values[0]
                    self._input_numeric = &indexed.numeric[0]
                    self._input_width = indexed.value_width
                    if self._input_keys is None and len(keys) == 1 and type(keys[0]) is str:
                        self._single_label = keys[0]

    cdef double lookup_float(self, KeyArg0 key, object column, object default=_typed_missing) except? -1:
        cdef PyObject* data = <PyObject*>self._numeric_view
        cdef double value
        cdef Py_ssize_t col, last
        if KeyArg0 is KeyInteger:
            if self._contiguous_view is not None and (<ContiguousView>self._contiguous_view).read(key, column, &value):
                return value
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
            if self._indexed_view is not None and self._input_keys is not None:
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
            if data is not None:
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
        return _semantic_owner(self._owner)

    def __modelx_table_owner__(self):
        return _semantic_owner(self._owner)

    def __modelx_table_owner_abi__(self):
        return 1

    def __eq__(self, other):
        return _semantic_owner(self._owner) == other

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return NativeTable, (_semantic_owner(self._owner),)


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
    cdef long long low, high
    cdef PyObject* snapshot = <PyObject*>table._numeric_view
    cdef TableView data
    cdef unsigned char mask = 2 if lower else 1
    cdef dict bindings
    if table._input_keys is not None:
        low, high = table._input_low, table._input_high
    elif snapshot == <PyObject*>None:
        if type(table._owner) is not Table:
            return False
        # A pandas integer axis establishes the bounds without ordering an
        # unrelated user-defined label during a scalar read.
        metadata = table._owner._metadata
        if metadata is None or metadata[0] not in ('axes', 'frame'):
            return False
        axis = metadata[1]
        if type(axis) is _report_index_type:
            if not isinstance(axis.dtype, _group_numpy.dtype) or axis.dtype.kind not in ('i', 'u'):
                return False
        elif type(axis) is tuple:
            if not (axis[0] == 'range' or axis[0] == 'labels'
                    and isinstance(axis[3], _group_numpy.dtype) and axis[3].kind in ('i', 'u')):
                return False
        elif type(axis) is not _report_range_type:
            return False
        bounds = table._owner._ordered
        if not bounds:
            return False
        first, last = bounds[0], bounds[len(bounds) - 1]
        if type(first) is not int or type(last) is not int or not -(1 << 63) <= first <= last < (1 << 63):
            return False
        low, high = first, last
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
        row = data.position(key, False, False)
        if row < 0:
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
        result[0] = data.number_at(row, index)
        return True
    if key < low:
        if not lower:
            return False
        key = low
    elif key > high:
        key = high
    if table._input_keys is None or table._indexed_view is None:
        if type(column) is str:
            if column not in table._owner._columns:
                return False
        elif (type(column) is not int and type(column) is not bool
                or not -len(table._owner._columns) <= column < len(table._owner._columns)):
            return False
        scalar = _read_owned_table(table._owner, key, column, None)
        if type(scalar) is float:
            result[0] = PyFloat_AS_DOUBLE(scalar)
            return True
        if type(scalar) is int or type(scalar) is bool or type(scalar) is str:
            try:
                result[0] = _float_type(scalar)
            except (ValueError, OverflowError):
                return False
            return True
        return False
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
    row = (<TableView>data).position(key, True, False)
    if row < 0 or col < 0:
        return False
    value[0] = (<TableView>data).number_at(row, col)
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
# Older modelx installs can still supply the existing table helpers.
from modelx_helpers import pandas_inputs as _pandas_inputs
cdef object _group_constructor = getattr(_pandas_inputs, 'group_record', None)
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



cdef void _prepare_group_record(object value) except *:
    children = value._items
    if children and type(children[0][1]) is Table:
        for column in children[0][1]._columns:
            prepared = prepare_group(value, column)
            if prepared is not None:
                value._native_groups[column] = prepared


cdef void _prepare_tables(object value) except *:
    """Install table resources while executing an explicit table constructor."""
    if type(value) is Table:
        _try_bound_handle(value)        # M1 (W1.2): the typed handle, before the table is shared
        cache = value._native_cache
        contiguous = cache.get(_contiguous_storage)
        if contiguous is not None:
            first, numbers = contiguous
            if ContiguousView not in cache and -(1 << 63) <= first <= first + len(numbers) - 1 < (1 << 63):
                cache.setdefault(ContiguousView, ContiguousView(first, numbers, value._columns))
            return
        rectangular = cache.get(_rectangular_storage)
        if rectangular is not None:
            if RectangularView not in cache:
                cache.setdefault(RectangularView, RectangularView(rectangular, value._columns[0]))
            return
        if value._keys and type(value._keys[0]) is int and TableView not in cache:
            cache.setdefault(TableView, TableView(value))
        if KeyedView not in cache:
            cache.setdefault(KeyedView, KeyedView(value))
    elif type(value) is Record:
        for _, child in value._items:
            _prepare_tables(child)
        if value._items and not all(type(child) is Table and _contiguous_storage in child._native_cache
                                   for _, child in value._items):
            _prepare_group_record(value)


cdef object group_record(object function, object tables):
    cdef object prepared = function(tables)
    if function is _group_constructor and type(prepared) is Record:
        _prepare_group_record(prepared)
    return prepared


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
            _prepare_tables(prepared)
            return prepared
    prepared = forward_call(function, keywords, ('frame', 'columns', 'groups', 'indexing', 'rows', 'report', 'preserve_scalar_types'),
                            (frame, columns, groups, indexing, rows, report, preserve_scalar_types), omitted_argument)
    if function is _table_constructor:
        _prepare_tables(prepared)
    return prepared


# ---- M1 typed table handles (W1.2; interface in _native_inputs.pxd, M1-INTERFACE.md) -----
# A TableHandle is built once per helper Table object from the helper's storage contract
# (`table_storage`, W1.H), with every sub-table of every key prefix built eagerly as a view of
# the same arrays (D25 Q2). Nothing is written after construction, so reads take no lock, and
# no read touches a Python attribute, method or dict of a shared object: keys are normalised
# into C parts and looked up in C indexes, values are read from C arrays. The few reads that
# must run Python code (exotic label objects, asof on an index not in increasing order, cells
# of object arrays, row_record) are counted in `handle_fallbacks()`.

from cpython.bool cimport PyBool_Check
from cpython.float cimport PyFloat_Check, PyFloat_FromDouble
from cpython.long cimport PyLong_FromLongLong, PyLong_AsDouble
from cpython.unicode cimport PyUnicode_Check
from cpython.tuple cimport PyTuple_Check, PyTuple_GET_SIZE
from cpython.object cimport PyObject_TypeCheck, PyTypeObject
from cpython.number cimport PyNumber_Float, PyNumber_Long
from cpython.bytearray cimport PyByteArray_AS_STRING
from cpython.sequence cimport PySequence_Fast_ITEMS
from libc.math cimport isnan, isinf
from libc.string cimport memcpy

cdef extern from *:
    """
    #if defined(_MSC_VER)
    #include <intrin.h>
    static volatile __int64 mx_handle_fallbacks_ = 0;
    static void mx_handle_fallback(void) { _InterlockedIncrement64(&mx_handle_fallbacks_); }
    static long long mx_handle_fallback_count(void) { return (long long)mx_handle_fallbacks_; }
    #else
    static long long mx_handle_fallbacks_ = 0;
    static void mx_handle_fallback(void) { __atomic_fetch_add(&mx_handle_fallbacks_, 1, __ATOMIC_RELAXED); }
    static long long mx_handle_fallback_count(void) { return __atomic_load_n(&mx_handle_fallbacks_, __ATOMIC_RELAXED); }
    #endif
    static inline Py_hash_t mx_mix64(unsigned long long x) {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27; x *= 0x94d049bb133111ebULL; x ^= x >> 31;
        return (Py_hash_t)(x >> 1);
    }
    static inline Py_hash_t mx_hash_double(double d) {
        unsigned long long u; memcpy(&u, &d, sizeof(u)); return mx_mix64(u ^ 0x9e3779b97f4a7c15ULL);
    }
    static inline Py_hash_t mx_hash_step(Py_hash_t acc, Py_hash_t h) {
        return (Py_hash_t)(((unsigned long long)acc ^ (unsigned long long)h) * 1000003ULL);
    }
    """
    void mx_handle_fallback() noexcept nogil
    long long mx_handle_fallback_count() noexcept nogil
    Py_hash_t mx_mix64(unsigned long long x) noexcept nogil
    Py_hash_t mx_hash_double(double d) noexcept nogil
    Py_hash_t mx_hash_step(Py_hash_t acc, Py_hash_t h) noexcept nogil

cdef extern from "Python.h":
    # PyObject* spellings: no reference counting on shared labels during a read.
    int mx_unicode_compare "PyUnicode_Compare"(PyObject* left, PyObject* right)
    int mx_rich_equal "PyObject_RichCompareBool"(PyObject* left, PyObject* right, int op) except -1
    PyObject* mx_tuple_item "PyTuple_GET_ITEM"(PyObject* tuple, Py_ssize_t i) noexcept nogil
    Py_ssize_t mx_tuple_size "PyTuple_GET_SIZE"(PyObject* tuple) noexcept nogil
    ctypedef struct PyMemberDef:
        Py_ssize_t offset
    ctypedef struct PyMemberDescrObject:
        PyMemberDef* d_member

cdef enum:
    HK_NONE = 0         # never matches (an int beyond 64 bits, an unhashable object)
    HK_INT = 1
    HK_FLOAT = 2
    HK_STR = 3
    HK_MISSING = 4      # NaN, None, pd.NA, pd.NaT under a pandas index spec
    HK_OBJ = 5          # any other label object: == and hash (counted fallback)

cdef enum:
    HC_F64 = 1
    HC_I64 = 2
    HC_BOOL = 3
    HC_OBJ = 4

cdef enum:
    HB_NO = 0
    HB_BOOL = 1
    HB_CINT = 2         # a C integer part: a bool on a bool-only level when 0 or 1

cdef object _table_storage_fn = getattr(_immutable_inputs, 'table_storage', None)
cdef object _no_extreme = object()
cdef PyTypeObject* _np_integer_type = <PyTypeObject*>_group_numpy.integer
cdef PyTypeObject* _np_floating_type = <PyTypeObject*>_group_numpy.floating
cdef PyTypeObject* _np_bool_type = <PyTypeObject*>_group_numpy.bool_
cdef object _np_float64 = _group_numpy.dtype('float64')
cdef object _np_int64 = _group_numpy.dtype('int64')
cdef object _pd_na = getattr(_pandas, 'NA', _missing)
cdef object _pd_nat = getattr(_pandas, 'NaT', _missing)
cdef Py_ssize_t _record_items_offset = -1


cdef Py_ssize_t _slot_offset(object owner_type, str name):
    """Byte offset of a __slots__ member, so a hot read takes the immutable field directly."""
    descr = owner_type.__dict__.get(name)
    if descr is None or type(descr).__name__ != 'member_descriptor':
        return -1
    return (<PyMemberDescrObject*><PyObject*>descr).d_member.offset


_record_items_offset = _slot_offset(Record, '_items')


def handle_fallbacks():
    """Number of handle reads that ran Python code (W1.2 debug counter, process-wide)."""
    return mx_handle_fallback_count()


# ---- key parts ------------------------------------------------------------------------

cdef inline int _part_object(HandlePart* q, object k) except -1:
    q.kind = HK_OBJ
    q.boolean = HB_NO
    q.o = <PyObject*>k
    try:
        q.h = hash(k)
    except TypeError:           # unhashable: never a label (the helper raises KeyError too)
        q.kind = HK_NONE
    return 0


cdef inline int _part_double(HandlePart* q, double d, object original, bint spec) except -1:
    q.boolean = HB_NO
    if isnan(d):
        if spec:
            q.kind = HK_MISSING
            q.h = 0x5bd1e995
            return 0
        return _part_object(q, original)
    if d == floor(d) and -9223372036854775808.0 <= d < 9223372036854775808.0:
        q.kind = HK_INT
        q.i = <long long>d
        q.h = mx_mix64(<unsigned long long>q.i)
    else:
        q.kind = HK_FLOAT
        q.f = d
        q.h = mx_hash_double(d)
    return 0


cdef int _part_from(HandlePart* q, object k, bint spec) except -1:
    """Normalise a key part as pandas compares labels: equal numbers match whatever their type
    (int, IntEnum, numpy scalar, integral float), bools carry a flag for the boolean label
    modes, NaN/None/NA/NaT are one missing label under an index spec."""
    cdef int overflow = 0
    cdef long long v
    q.boolean = HB_NO
    if PyLong_Check(k):
        if PyBool_Check(k):
            q.kind = HK_INT
            q.boolean = HB_BOOL
            q.i = 1 if k is True else 0
            q.h = mx_mix64(<unsigned long long>q.i)
            return 0
        v = PyLong_AsLongLongAndOverflow(k, &overflow)
        if overflow:
            q.kind = HK_NONE
            return 0
        q.kind = HK_INT
        q.i = v
        q.h = mx_mix64(<unsigned long long>v)
        return 0
    if PyFloat_Check(k):
        return _part_double(q, PyFloat_AS_DOUBLE(k), k, spec)
    if PyUnicode_Check(k):
        q.kind = HK_STR
        q.o = <PyObject*>k
        q.h = hash(k)
        return 0
    if k is None:
        if spec:
            q.kind = HK_MISSING
            q.h = 0x5bd1e995
            return 0
        return _part_object(q, k)
    if PyObject_TypeCheck(k, _np_bool_type):
        q.kind = HK_INT
        q.boolean = HB_BOOL
        q.i = 1 if PyObject_IsTrue(k) else 0
        q.h = mx_mix64(<unsigned long long>q.i)
        return 0
    if PyObject_TypeCheck(k, _np_integer_type):
        return _part_from(q, PyNumber_Index(k), spec)
    if PyObject_TypeCheck(k, _np_floating_type):
        return _part_double(q, PyFloat_AsDouble(k), k, spec)
    if spec and (k is _pd_na or k is _pd_nat):
        q.kind = HK_MISSING
        q.h = 0x5bd1e995
        return 0
    return _part_object(q, k)


cdef inline void _part_cint(HandlePart* q, long long v) noexcept:
    q.kind = HK_INT
    q.boolean = HB_CINT
    q.i = v
    q.h = mx_mix64(<unsigned long long>v)


cdef inline int _set_part(HandlePart* q, TableKey p, bint spec) except -1:
    if TableKey is object:
        return _part_from(q, p, spec)
    else:
        _part_cint(q, p)
        return 0


cdef inline object _original(TableKey p):
    if TableKey is object:
        return p
    else:
        return PyLong_FromLongLong(p)


cdef inline bint _mode_ok(const HandlePart* q, unsigned char mode) noexcept:
    if mode == 0 or q.kind == HK_MISSING:
        return True
    if q.boolean == HB_CINT:
        return mode == 2 or q.i == 0 or q.i == 1
    if mode == 1:
        return q.boolean == HB_BOOL
    return q.boolean == HB_NO


cdef inline int _part_eq(const HandlePart* a, const HandlePart* q) except -1:
    if a.kind != q.kind or a.h != q.h:
        return 0
    if a.kind == HK_INT:
        return a.i == q.i
    if a.kind == HK_FLOAT:
        return a.f == q.f
    if a.kind == HK_STR:
        return a.o == q.o or mx_unicode_compare(a.o, q.o) == 0
    if a.kind == HK_MISSING:
        return 1
    if a.kind == HK_OBJ:
        if a.o == q.o:
            return 1
        mx_handle_fallback()
        return mx_rich_equal(a.o, q.o, 2)       # Py_EQ
    return 0


cdef inline Py_hash_t _hparts_hash(const HandlePart* parts, Py_ssize_t n) noexcept nogil:
    cdef Py_hash_t h = 0x345678
    cdef Py_ssize_t j
    for j in range(n):
        h = mx_hash_step(h, parts[j].h)
    return mx_mix64(<unsigned long long>h)


cdef Py_ssize_t _keys_find(const HandleKeys* k, const HandlePart* q) except -2:
    """Row (or axis position) of the key q (k.nparts parts) in k; -1 when absent."""
    cdef Py_ssize_t j, lo, hi, mid, r
    cdef unsigned long long offset
    cdef size_t i
    cdef const HandlePart* stored
    cdef long long v
    cdef int same
    for j in range(k.nparts):
        if q[j].kind == HK_NONE or not _mode_ok(&q[j], k.modes[j]):
            return -1
    if k.scheme == TABLE_DENSE:
        if q[0].kind != HK_INT:
            return -1
        # unsigned difference: no signed overflow at +-2**63 or below a negative first key (D15)
        offset = <unsigned long long>q[0].i - <unsigned long long>k.first
        return <Py_ssize_t>offset if offset < <unsigned long long>k.count else -1
    if k.scheme == TABLE_SORTED:
        if q[0].kind != HK_INT:
            return -1
        v = q[0].i
        lo = 0
        hi = k.count
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if k.skeys[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        if lo < k.count and k.skeys[lo] == v:
            return k.srows[lo]
        return -1
    if k.count == 0:
        return -1
    i = <size_t>_hparts_hash(q, k.nparts) & k.mask
    while True:
        r = k.slots[i]
        if r == 0:
            return -1
        r -= 1
        stored = &k.parts[r * k.pstride + k.poff]
        same = 1
        for j in range(k.nparts):
            if not _part_eq(&stored[j], &q[j]):
                same = 0
                break
        if same:
            return r
        i = (i + 1) & k.mask


# ---- construction ------------------------------------------------------------------------

cdef void* _alloc(list hold, Py_ssize_t nbytes) except NULL:
    """Zeroed memory owned by a bytearray in `hold` (never resized, so the pointer stays valid)."""
    cdef object buffer = bytearray(nbytes if nbytes > 0 else 1)
    hold.append(buffer)
    return <void*>PyByteArray_AS_STRING(buffer)


cdef const unsigned char* _modes(list hold, object spec, Py_ssize_t levels, unsigned char default) except NULL:
    cdef unsigned char* modes = <unsigned char*>_alloc(hold, levels)
    cdef Py_ssize_t j
    found = spec['boolean_modes'] if spec is not None else ()
    for j in range(levels):
        modes[j] = <unsigned char>found[j] if j < len(found) else default
    return modes


cdef void _hash_index(HandleKeys* k, list hold) except *:
    """Open-addressing slots over k's stored keys (unique, as table keys are)."""
    cdef size_t size = 8, i
    cdef Py_ssize_t r
    cdef Py_ssize_t* slots
    while size < <size_t>(2 * k.count):
        size <<= 1
    slots = <Py_ssize_t*>_alloc(hold, <Py_ssize_t>(size * sizeof(Py_ssize_t)))
    for r in range(k.count):
        i = <size_t>_hparts_hash(&k.parts[r * k.pstride + k.poff], k.nparts) & (size - 1)
        while slots[i] != 0:
            i = (i + 1) & (size - 1)
        slots[i] = r + 1
    k.slots = slots
    k.mask = size - 1


cdef void _index_single(TableHandle t, HandleKeys* k, list hold, const HandlePart* parts,
                        Py_ssize_t count, Py_ssize_t pstride, Py_ssize_t poff,
                        const unsigned char* modes, object labels) except *:
    """A one-level key index: DENSE when the keys are consecutive ascending ints, SORTED for
    other ints, HASHED otherwise. Sets t's int-key facts and index extremes (as the helper's
    index.min()/max(): the smallest and largest label)."""
    cdef Py_ssize_t r
    cdef bint ints = True, increasing = True
    cdef long long* okeys
    cdef long long* skeys
    cdef Py_ssize_t* srows
    k.count = count
    k.nparts = 1
    k.pstride = pstride
    k.poff = poff
    k.parts = parts
    k.modes = modes
    for r in range(count):
        if parts[r * pstride + poff].kind != HK_INT or parts[r * pstride + poff].boolean != HB_NO:
            ints = False
            break
    t._int_keys = ints and count > 0
    if t._int_keys:
        okeys = <long long*>_alloc(hold, count * sizeof(long long))
        for r in range(count):
            okeys[r] = parts[r * pstride + poff].i
            if r and okeys[r] <= okeys[r - 1]:
                increasing = False
        t._okeys = okeys
        t._increasing = increasing
        if increasing and <unsigned long long>okeys[count - 1] - <unsigned long long>okeys[0] == <unsigned long long>(count - 1):
            k.scheme = TABLE_DENSE
            k.first = okeys[0]
            k.last = okeys[count - 1]
        else:
            k.scheme = TABLE_SORTED
            values = [okeys[r] for r in range(count)]
            order = sorted(range(count), key=values.__getitem__)
            skeys = <long long*>_alloc(hold, count * sizeof(long long))
            srows = <Py_ssize_t*>_alloc(hold, count * sizeof(Py_ssize_t))
            for r in range(count):
                srows[r] = order[r]
                skeys[r] = okeys[srows[r]]
            k.skeys = skeys
            k.srows = srows
            k.first = skeys[0]
            k.last = skeys[count - 1]
        t._first_obj = PyLong_FromLongLong(k.first)
        t._last_obj = PyLong_FromLongLong(k.last)
    else:
        k.scheme = TABLE_HASHED
        _hash_index(k, hold)
        t._first_obj, t._last_obj = _extremes(labels)


cdef tuple _extremes(object labels):
    """(min, max) of the labels as the helper's index.min()/max(); NaN when empty."""
    if not len(labels):
        return float('nan'), float('nan')
    try:
        return min(labels), max(labels)
    except TypeError:
        return _no_extreme, _no_extreme


cdef HandlePart* _label_parts(list hold, object labels, bint spec) except NULL:
    cdef Py_ssize_t r, count = len(labels)
    cdef HandlePart* parts = <HandlePart*>_alloc(hold, count * sizeof(HandlePart))
    hold.append(labels)                 # the parts borrow str and object labels
    for r in range(count):
        _part_from(&parts[r], labels[r], spec)
    return parts


cdef TableHandle _empty_like(TableHandle source):
    """A new handle sharing source's data arrays and identity (keys and children unset)."""
    cdef TableHandle t = TableHandle.__new__(TableHandle)
    t.owner = source.owner
    t._hold = source._hold
    t._prefix = source._prefix
    t._columns = source._columns
    t.width = source.width
    t._ckind = source._ckind
    t._dbl = source._dbl
    t._int = source._int
    t._obj = source._obj
    t._array = source._array
    t._stride = source._stride
    t._colbase = source._colbase
    t._spec_missing = source._spec_missing
    t._preserve = source._preserve
    t._first_obj = _no_extreme
    t._last_obj = _no_extreme
    t._children = ()
    return t


cdef TableHandle _clone(TableHandle source):
    cdef TableHandle t = _empty_like(source)
    t.scheme = source.scheme
    t.rows = source.rows
    t.levels = source.levels
    t.base = source.base
    t._children = source._children
    t._labels = source._labels
    t._first_obj = source._first_obj
    t._last_obj = source._last_obj
    t._keys = source._keys
    t._ckeys = source._ckeys
    t._axes = source._axes
    t._spans = source._spans
    t._int_keys = source._int_keys
    t._increasing = source._increasing
    t._okeys = source._okeys
    return t


cdef void _product_children(TableHandle t, list hold, tuple axes, list axis_handles) except *:
    """Eager depth-1 children of a PRODUCT handle: by position on its first axis."""
    cdef Py_ssize_t position
    cdef TableHandle child, axis
    cdef list children = []
    for position in range(len(axes[0])):
        child = _empty_like(t)
        child._prefix = t._prefix + (axes[0][position],)
        child.base = t.base + position * t._spans[0]
        child.rows = t._spans[0]
        child.levels = t.levels - 1
        if child.levels > 1:
            child.scheme = TABLE_PRODUCT
            child._axes = t._axes + 1
            child._spans = t._spans + 1
            child._labels = axes[1:]
            child._first_obj, child._last_obj = _product_extremes(axes[1:])
            _product_children(child, hold, axes[1:], axis_handles[1:])
        else:
            axis = <TableHandle>axis_handles[1]
            child.scheme = axis._keys.scheme
            child._keys = axis._keys
            child._labels = axis._labels
            child._int_keys = axis._int_keys
            child._increasing = axis._increasing
            child._okeys = axis._okeys
            child._first_obj = axis._first_obj
            child._last_obj = axis._last_obj
        children.append(child)
    t._children = tuple(children)


cdef tuple _product_extremes(tuple axes):
    try:
        return tuple(min(axis) for axis in axes), tuple(max(axis) for axis in axes)
    except (TypeError, ValueError):
        return _no_extreme, _no_extreme


cdef void _rows_children(TableHandle t, list hold, HandlePart* parts, Py_ssize_t levels,
                         Py_ssize_t depth, tuple labels, const unsigned char* modes) except *:
    """Eager children of a multi-level row-stored handle at key depth `depth`: rows are laid out
    so that each prefix is one run, so a child is the run [start, start + count) of t."""
    cdef Py_ssize_t start = 0, end, r, count
    cdef TableHandle child
    cdef list children = []
    cdef list firsts = []
    cdef HandlePart* heads
    cdef const HandlePart* row_parts = parts + t.base * levels
    while start < t.rows:
        end = start + 1
        while end < t.rows and _part_eq(&row_parts[end * levels + depth],
                                         &row_parts[start * levels + depth]):
            end += 1
        count = end - start
        child = _empty_like(t)
        child._prefix = t._prefix + (labels[t.base + start][depth],)
        child.base = t.base + start
        child.rows = count
        child.levels = t.levels - 1
        if child.levels > 1:
            child.scheme = TABLE_HASHED
            child._keys.scheme = TABLE_HASHED
            child._keys.count = count
            child._keys.nparts = child.levels
            child._keys.pstride = levels
            child._keys.poff = depth + 1
            child._keys.parts = parts + child.base * levels
            child._keys.modes = modes + depth + 1
            _hash_index(&child._keys, hold)
            child._labels = tuple(labels[child.base + r][depth + 1:] for r in range(count))
            child._first_obj, child._last_obj = _extremes(child._labels)
            _rows_children(child, hold, parts, levels, depth + 1, labels, modes)
        else:
            child._labels = tuple(labels[child.base + r][depth + 1] for r in range(count))
            _index_single(child, &child._keys, hold, parts + child.base * levels, count,
                          levels, depth + 1, modes + depth + 1, child._labels)
            child.scheme = child._keys.scheme
        children.append(child)
        firsts.append(start)
        start = end
    t._children = tuple(children)
    heads = <HandlePart*>_alloc(hold, len(firsts) * sizeof(HandlePart))
    for r in range(len(firsts)):
        heads[r] = row_parts[<Py_ssize_t>firsts[r] * levels + depth]
    t._ckeys.scheme = TABLE_HASHED
    t._ckeys.count = len(firsts)
    t._ckeys.nparts = 1
    t._ckeys.pstride = 1
    t._ckeys.poff = 0
    t._ckeys.parts = heads
    t._ckeys.modes = modes + depth
    _hash_index(&t._ckeys, hold)


cdef object _canonical(const HandlePart* q, object label):
    """A Python value equal for exactly the parts _part_eq calls equal (for grouping)."""
    if q.kind == HK_INT:
        return (1, q.i)
    if q.kind == HK_FLOAT:
        return (2, q.f)
    if q.kind == HK_STR:
        return (3, <object>q.o)
    if q.kind == HK_MISSING:
        return (4,)
    return (5, label)


cdef list _hierarchical_order(list rows, Py_ssize_t depth, Py_ssize_t levels, list canon):
    """Rows grouped by part `depth` in first-appearance order, recursively: every key prefix
    becomes one run; rows keep their relative order within a run."""
    cdef dict groups
    cdef list result
    if depth >= levels - 1 or len(rows) < 2:
        return rows
    groups = {}
    for r in rows:
        groups.setdefault(canon[r][depth], []).append(r)
    if len(groups) == 1:
        return _hierarchical_order(rows, depth + 1, levels, canon)
    result = []
    for group in groups.values():
        result.extend(_hierarchical_order(group, depth + 1, levels, canon))
    return result


cdef void _row_columns(TableHandle t, list hold, tuple values, list order, Py_ssize_t width) except *:
    """Typed column arrays of a row-stored table in handle row order: a float, int or bool
    column gets a C array; every cell also stays available as the stored object."""
    cdef Py_ssize_t rows = len(order), r, c, i
    cdef unsigned char* kinds = <unsigned char*>_alloc(hold, width)
    cdef double* numbers = NULL
    cdef long long* integers = NULL
    cdef int overflow = 0
    cdef bint any_float = False, any_int = False
    cdef unsigned char kind, this
    for c in range(width):
        kind = HC_F64
        for r in range(rows):
            value = values[<Py_ssize_t>order[r]][c]
            if type(value) is float:
                this = HC_F64
            elif type(value) is bool:
                this = HC_BOOL
            elif type(value) is int:
                PyLong_AsLongLongAndOverflow(value, &overflow)
                this = HC_OBJ if overflow else HC_I64
            else:
                this = HC_OBJ
            if r == 0:
                kind = this
            elif this != kind:
                kind = HC_OBJ
            if kind == HC_OBJ:
                break
        kinds[c] = kind
        any_float = any_float or kind == HC_F64
        any_int = any_int or kind == HC_I64 or kind == HC_BOOL
    if any_float:
        numbers = <double*>_alloc(hold, rows * width * sizeof(double))
    if any_int:
        integers = <long long*>_alloc(hold, rows * width * sizeof(long long))
    flat = tuple([values[<Py_ssize_t>order[r]][c] for r in range(rows) for c in range(width)])
    hold.append(flat)
    for r in range(rows):
        for c in range(width):
            i = r * width + c
            value = flat[i]
            if kinds[c] == HC_F64:
                numbers[i] = PyFloat_AS_DOUBLE(value)
            elif kinds[c] == HC_I64 or kinds[c] == HC_BOOL:
                integers[i] = PyLong_AsLongLongAndOverflow(value, &overflow)
    t._ckind = kinds
    t._dbl = numbers
    t._int = integers
    t._obj = PySequence_Fast_ITEMS(flat) if rows and width else NULL
    t._stride = width


cdef void _init_contiguous(TableHandle t, list hold, tuple storage) except *:
    cdef Py_ssize_t rows, c
    cdef const double[:, ::1] numbers
    cdef const long long[:, ::1] integers
    cdef unsigned char* kinds
    cdef HandlePart* parts
    start, data, spec = storage[1], storage[2], storage[4]
    rows = len(data)
    if not -(1 << 63) <= start <= start + rows - 1 < (1 << 63):
        raise OverflowError('contiguous table keys exceed 64-bit integers')
    t.levels = 1
    t.rows = rows
    kinds = <unsigned char*>_alloc(hold, t.width)
    if data.dtype == _np_float64 and data.ndim == 2:
        array = _group_numpy.ascontiguousarray(data)
        hold.append(array)
        numbers = array
        t._dbl = &numbers[0, 0] if rows else NULL
        kind = HC_F64
    elif data.dtype == _np_int64 and data.ndim == 2:
        array = _group_numpy.ascontiguousarray(data)
        hold.append(array)
        integers = array
        t._int = &integers[0, 0] if rows else NULL
        kind = HC_I64
    else:
        kind = HC_OBJ
        t._array = data
    for c in range(t.width):
        kinds[c] = kind
    if t._preserve:
        t._array = data             # obj() returns the stored numpy scalar, as the helper
    t._ckind = kinds
    t._stride = t.width
    # DENSE keys start .. start + rows - 1; contiguous tables reject bool keys (helper _find_row)
    t.scheme = TABLE_DENSE
    t._keys.scheme = TABLE_DENSE
    t._keys.count = rows
    t._keys.nparts = 1
    t._keys.first = start
    t._keys.last = start + rows - 1
    t._keys.modes = _modes(hold, spec, 1, 2)
    t._int_keys = rows > 0
    t._increasing = True
    t._first_obj = start
    t._last_obj = start + rows - 1


cdef void _init_rectangular(TableHandle t, list hold, tuple storage) except *:
    cdef Py_ssize_t levels, j
    cdef const double[::1] numbers
    cdef unsigned char* kinds
    cdef HandleKeys* axes_keys
    cdef Py_ssize_t* spans
    cdef TableHandle axis
    cdef const unsigned char* modes
    axes, data, spec = tuple(storage[1]), storage[2], storage[4]
    levels = len(axes)
    t.levels = levels
    array = _group_numpy.ascontiguousarray(data, dtype=_np_float64).reshape(-1)
    hold.append(array)
    numbers = array
    t._dbl = &numbers[0] if len(array) else NULL
    kinds = <unsigned char*>_alloc(hold, 1)
    kinds[0] = HC_F64
    t._ckind = kinds
    t._stride = 1
    t.rows = len(array)
    modes = _modes(hold, spec, levels, 0)
    axes_keys = <HandleKeys*>_alloc(hold, levels * sizeof(HandleKeys))
    spans = <Py_ssize_t*>_alloc(hold, levels * sizeof(Py_ssize_t))
    axis_handles = []
    for j in range(levels):
        labels = tuple(axes[j])
        axis = TableHandle.__new__(TableHandle)
        axis._labels = labels
        _index_single(axis, &axis._keys, hold, _label_parts(hold, labels, t._spec_missing),
                      len(labels), 1, 0, modes + j, labels)
        axes_keys[j] = axis._keys
        axis_handles.append(axis)
    hold.append(tuple(axis_handles))
    spans[levels - 1] = 1
    for j in range(levels - 2, -1, -1):
        spans[j] = spans[j + 1] * len(axes[j + 1])
    t.scheme = TABLE_PRODUCT
    t._axes = axes_keys
    t._spans = spans
    t._labels = axes
    t._first_obj, t._last_obj = _product_extremes(axes)
    _product_children(t, hold, axes, axis_handles)


cdef void _init_rows(TableHandle t, list hold, tuple storage) except *:
    cdef Py_ssize_t levels, rows, r, j
    cdef HandlePart* original
    cdef HandlePart* parts
    cdef const unsigned char* modes
    keys, values, spec = storage[1], storage[2], storage[5]
    rows = len(keys)
    if spec is not None:
        levels = len(spec['boolean_modes']) if spec['multi'] else 1
    else:
        levels = len(keys[0]) if keys and type(keys[0]) is tuple else 1
    t.levels = levels
    t.rows = rows
    modes = _modes(hold, spec, levels, 0)
    if levels == 1:
        _row_columns(t, hold, values, list(range(rows)), t.width)
        t._labels = tuple(keys)
        _index_single(t, &t._keys, hold, _label_parts(hold, t._labels, t._spec_missing),
                      rows, 1, 0, modes, t._labels)
        t.scheme = t._keys.scheme
        return
    hold.append(keys)
    original = <HandlePart*>_alloc(hold, rows * levels * sizeof(HandlePart))
    canon = []
    for r in range(rows):
        key = keys[r]
        if type(key) is not tuple or len(key) != levels:
            raise TypeError('multi-level table key is not a tuple of one part per level')
        for j in range(levels):
            _part_from(&original[r * levels + j], key[j], t._spec_missing)
        canon.append(tuple(_canonical(&original[r * levels + j], key[j]) for j in range(levels)))
    order = _hierarchical_order(list(range(rows)), 0, levels, canon)
    parts = <HandlePart*>_alloc(hold, rows * levels * sizeof(HandlePart))
    for r in range(rows):
        memcpy(&parts[r * levels], &original[<Py_ssize_t>order[r] * levels], levels * sizeof(HandlePart))
    _row_columns(t, hold, values, order, t.width)
    t._labels = tuple([keys[<Py_ssize_t>order[r]] for r in range(rows)])
    t.scheme = TABLE_HASHED
    t._keys.scheme = TABLE_HASHED
    t._keys.count = rows
    t._keys.nparts = levels
    t._keys.pstride = levels
    t._keys.poff = 0
    t._keys.parts = parts
    t._keys.modes = modes
    _hash_index(&t._keys, hold)
    t._first_obj, t._last_obj = _extremes(keys)
    _rows_children(t, hold, parts, levels, 0, t._labels, modes)


cdef double _object_f64(PyObject* value) except? -1:
    """float(value) for a stored cell object."""
    if PyFloat_Check(<object>value):
        return PyFloat_AS_DOUBLE(<object>value)
    if PyLong_Check(<object>value):
        return PyLong_AsDouble(<object>value)
    mx_handle_fallback()
    return PyFloat_AS_DOUBLE(PyNumber_Float(<object>value))


cdef long long _double_i64(double value) except? -1:
    """int(value) for a float cell, in a C long long."""
    if isnan(value):
        raise ValueError('cannot convert float NaN to integer')
    if isinf(value):
        raise OverflowError('cannot convert float infinity to integer')
    if not -9223372036854775808.0 <= value < 9223372036854775808.0:
        raise OverflowError('Python int too large to convert to C long long')
    return <long long>value


cdef long long _object_i64(PyObject* value) except? -1:
    cdef int overflow = 0
    cdef long long result
    if PyLong_Check(<object>value):
        result = PyLong_AsLongLongAndOverflow(<object>value, &overflow)
    elif PyFloat_Check(<object>value):
        return _double_i64(PyFloat_AS_DOUBLE(<object>value))
    else:
        mx_handle_fallback()
        result = PyLong_AsLongLongAndOverflow(PyNumber_Long(<object>value), &overflow)
    if overflow:
        raise OverflowError('Python int too large to convert to C long long')
    return result


cdef inline Py_ssize_t _h_cell(TableHandle self, Py_ssize_t row, Py_ssize_t col) except -1:
    if <size_t>row >= <size_t>self.rows or <size_t>col >= <size_t>self.width:
        raise IndexError(f'table cell ({row}, {col}) is out of range')
    return (self.base + row) * self._stride + self._colbase + col

cdef Py_ssize_t _h_row(TableHandle self, const HandlePart* q, Py_ssize_t n) except -2:
    cdef Py_ssize_t j, position, row = 0
    if n != self.levels:
        return -1
    if self.scheme == TABLE_PRODUCT:
        for j in range(n):
            position = _keys_find(&self._axes[j], &q[j])
            if position < 0:
                return -1
            row += position * self._spans[j]
        return row
    return _keys_find(&self._keys, q)

cdef PyObject* _h_child(PyObject* parent, const HandlePart* q) except? NULL:
    """Borrowed child of `parent` for one prefix part (NULL when absent): the walk takes no
    reference on the shared handles; the root keeps every child alive."""
    cdef Py_ssize_t position
    if (<TableHandle>parent).levels < 2:
        return NULL
    if (<TableHandle>parent).scheme == TABLE_PRODUCT:
        position = _keys_find(&(<TableHandle>parent)._axes[0], q)
    else:
        position = _keys_find(&(<TableHandle>parent)._ckeys, q)
    if position < 0:
        return NULL
    return mx_tuple_item(<PyObject*>(<TableHandle>parent)._children, position)


cdef TableHandle _h_sub(TableHandle self, const HandlePart* q, Py_ssize_t n):
    cdef PyObject* handle = <PyObject*>self
    cdef Py_ssize_t j
    if n >= self.levels:
        raise TypeError(f'sub_{n} needs fewer key parts than the {self.levels} levels of the table')
    for j in range(n):
        handle = _h_child(handle, &q[j])
        if handle == NULL:
            return None
    if (<TableHandle>handle)._colbase != self._colbase or (<TableHandle>handle).width != self.width:
        return _column_of(<TableHandle>handle, self._colbase, self._columns)
    return <TableHandle>handle        # the one reference a sub-table read takes

cdef bint _h_has(TableHandle self, const HandlePart* q, Py_ssize_t n) except -1:
    cdef PyObject* handle = <PyObject*>self
    cdef Py_ssize_t j
    if n == self.levels:
        return _h_row(self, q, n) >= 0
    if n > self.levels or n == 0:
        return False
    if self.scheme == TABLE_PRODUCT:
        for j in range(n):
            if _keys_find(&self._axes[j], &q[j]) < 0:
                return False
        return True
    for j in range(n):
        handle = _h_child(handle, &q[j])
        if handle == NULL:
            return False
    return True

cdef int _h_unpack(TableHandle self, object key, HandlePart* q) except -1:
    """A whole tuple key passed as one part of a multi-level table: its parts, else 0."""
    cdef Py_ssize_t j, n
    if not (PyTuple_Check(key) and self.levels > 1):
        return 0
    n = PyTuple_GET_SIZE(key)
    if n == 0 or n > 6:
        return 0
    for j in range(n):
        _part_from(&q[j], <object>PyTuple_GET_ITEM(key, j), self._spec_missing)
    return <int>n

cdef int _h_int_index(TableHandle self, str what) except -1:
    if self.levels != 1 or not self._int_keys:
        raise TypeError(f'{what} needs a one-level table with integer keys')
    return 0

cdef object _h_label(TableHandle self, Py_ssize_t row):
    cdef Py_ssize_t j, position
    if self.scheme == TABLE_PRODUCT:
        parts = []
        for j in range(self.levels):
            position = (row // self._spans[j]) % self._axes[j].count
            parts.append(self._labels[j][position])
        return tuple(parts)
    if self._labels is None:
        return self._keys.first + row
    return self._labels[row]

cdef inline PyObject* _h_value(RecordHandle self, Py_ssize_t slot) except NULL:
    cdef PyObject* items = <PyObject*>self._items     # borrowed: no reference on the shared pairs
    if <size_t>slot >= <size_t>mx_tuple_size(items):
        raise IndexError(f'record slot {slot} is out of range')
    return mx_tuple_item(mx_tuple_item(items, slot), 1)


@cython.final
cdef class TableHandle:
    """Typed read handle of one helper Table (or of one of its sub-tables): see the pxd."""

    def __init__(self, owner):
        cdef list hold = []
        if type(owner) is NativeTable:
            owner = (<NativeTable>owner)._owner
        if _table_storage_fn is None:
            raise TypeError('typed table handles need modelx_helpers.immutable_inputs.table_storage')
        storage = tuple(_table_storage_fn(owner))
        self.owner = owner
        self._hold = hold
        self._prefix = ()
        self._columns = tuple(storage[-2])
        self.width = len(self._columns)
        self._spec_missing = storage[-1] is not None
        self._preserve = bool(getattr(owner, '_preserve_scalar_types', False))
        self._first_obj = _no_extreme
        self._last_obj = _no_extreme
        self._children = ()
        if storage[0] == 'contiguous':
            _init_contiguous(self, hold, storage)
        elif storage[0] == 'rectangular':
            _init_rectangular(self, hold, storage)
        elif storage[0] == 'rows':
            _init_rows(self, hold, storage)
        else:
            raise TypeError(f'unknown table storage {storage[0]!r}')
        _attach_table_objects(self)

    # -- columns and cells ------------------------------------------------------------------

    cdef Py_ssize_t col(self, object name) except -2:
        cdef Py_ssize_t j
        for j in range(self.width):
            if <PyObject*>name == PyTuple_GET_ITEM(self._columns, j):
                return j
        if PyUnicode_Check(name):
            for j in range(self.width):
                if mx_unicode_compare(<PyObject*>name, PyTuple_GET_ITEM(self._columns, j)) == 0:
                    return j
        raise KeyError(name)


    cdef double f64(self, Py_ssize_t row, Py_ssize_t col) except? -1:
        cdef Py_ssize_t i = _h_cell(self, row, col)
        cdef unsigned char kind = self._ckind[self._colbase + col]
        if kind == HC_F64:
            return self._dbl[i]
        if kind == HC_I64 or kind == HC_BOOL:
            return <double>self._int[i]
        if self._obj != NULL:
            return _object_f64(self._obj[i])
        mx_handle_fallback()
        return float(self._array[self.base + row, self._colbase + col])

    cdef long long i64(self, Py_ssize_t row, Py_ssize_t col) except? -1:
        cdef Py_ssize_t i = _h_cell(self, row, col)
        cdef unsigned char kind = self._ckind[self._colbase + col]
        if kind == HC_I64 or kind == HC_BOOL:
            return self._int[i]
        if kind == HC_F64:
            return _double_i64(self._dbl[i])
        if self._obj != NULL:
            return _object_i64(self._obj[i])
        mx_handle_fallback()
        return int(self._array[self.base + row, self._colbase + col])

    cdef object obj(self, Py_ssize_t row, Py_ssize_t col):
        cdef Py_ssize_t i = _h_cell(self, row, col)
        cdef unsigned char kind = self._ckind[self._colbase + col]
        if self._obj != NULL:
            return <object>self._obj[i]
        if self._array is not None:
            mx_handle_fallback()
            if self._preserve:
                return self._array[self.base + row, self._colbase + col]
            return self._array.item(self.base + row, self._colbase + col)
        if kind == HC_F64:
            return PyFloat_FromDouble(self._dbl[i])
        if kind == HC_BOOL:
            return self._int[i] != 0
        return PyLong_FromLongLong(self._int[i])

    # -- keys -----------------------------------------------------------------------------






    cdef Py_ssize_t row_1(self, TableKey p0) except -2:
        cdef HandlePart q[6]
        cdef Py_ssize_t row, n = 0
        if TableKey is object:
            n = _h_unpack(self, p0, q)
        if n == 0:
            _set_part(&q[0], p0, self._spec_missing)
            n = 1
        row = _h_row(self, q, n)
        if row < 0:
            raise KeyError(_original(p0))
        return row

    cdef Py_ssize_t row_2(self, TableKey p0, TableKey p1) except -2:
        cdef HandlePart q[2]
        cdef Py_ssize_t row
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        row = _h_row(self, q, 2)
        if row < 0:
            raise KeyError((_original(p0), _original(p1)))
        return row

    cdef Py_ssize_t row_3(self, TableKey p0, TableKey p1, TableKey p2) except -2:
        cdef HandlePart q[3]
        cdef Py_ssize_t row
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        row = _h_row(self, q, 3)
        if row < 0:
            raise KeyError((_original(p0), _original(p1), _original(p2)))
        return row

    cdef Py_ssize_t row_4(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3) except -2:
        cdef HandlePart q[4]
        cdef Py_ssize_t row
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        row = _h_row(self, q, 4)
        if row < 0:
            raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3)))
        return row

    cdef Py_ssize_t row_5(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4) except -2:
        cdef HandlePart q[5]
        cdef Py_ssize_t row
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        _set_part(&q[4], p4, self._spec_missing)
        row = _h_row(self, q, 5)
        if row < 0:
            raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3),
                            _original(p4)))
        return row

    cdef Py_ssize_t row_6(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4, TableKey p5) except -2:
        cdef HandlePart q[6]
        cdef Py_ssize_t row
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        _set_part(&q[4], p4, self._spec_missing)
        _set_part(&q[5], p5, self._spec_missing)
        row = _h_row(self, q, 6)
        if row < 0:
            raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3),
                            _original(p4), _original(p5)))
        return row

    cdef bint has_1(self, TableKey p0) except -1:
        cdef HandlePart q[6]
        cdef Py_ssize_t n = 0
        if TableKey is object:
            n = _h_unpack(self, p0, q)
        if n == 0:
            _set_part(&q[0], p0, self._spec_missing)
            n = 1
        return _h_has(self, q, n)

    cdef bint has_2(self, TableKey p0, TableKey p1) except -1:
        cdef HandlePart q[2]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        return _h_has(self, q, 2)

    cdef bint has_3(self, TableKey p0, TableKey p1, TableKey p2) except -1:
        cdef HandlePart q[3]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        return _h_has(self, q, 3)

    cdef bint has_4(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3) except -1:
        cdef HandlePart q[4]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        return _h_has(self, q, 4)

    cdef bint has_5(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4) except -1:
        cdef HandlePart q[5]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        _set_part(&q[4], p4, self._spec_missing)
        return _h_has(self, q, 5)

    cdef bint has_6(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4, TableKey p5) except -1:
        cdef HandlePart q[6]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        _set_part(&q[4], p4, self._spec_missing)
        _set_part(&q[5], p5, self._spec_missing)
        return _h_has(self, q, 6)

    cdef TableHandle sub_1(self, TableKey p0):
        cdef HandlePart q[1]
        _set_part(&q[0], p0, self._spec_missing)
        child = _h_sub(self, q, 1)
        if child is None:
            raise KeyError(_original(p0))
        return child

    cdef TableHandle sub_2(self, TableKey p0, TableKey p1):
        cdef HandlePart q[2]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        child = _h_sub(self, q, 2)
        if child is None:
            raise KeyError((_original(p0), _original(p1)))
        return child

    cdef TableHandle sub_3(self, TableKey p0, TableKey p1, TableKey p2):
        cdef HandlePart q[3]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        child = _h_sub(self, q, 3)
        if child is None:
            raise KeyError((_original(p0), _original(p1), _original(p2)))
        return child

    cdef TableHandle sub_4(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3):
        cdef HandlePart q[4]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        child = _h_sub(self, q, 4)
        if child is None:
            raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3)))
        return child

    cdef TableHandle sub_5(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4):
        cdef HandlePart q[5]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        _set_part(&q[4], p4, self._spec_missing)
        child = _h_sub(self, q, 5)
        if child is None:
            raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3),
                            _original(p4)))
        return child

    # -- index ------------------------------------------------------------------------------


    cdef long long first_int(self) except? -1:
        if self.rows == 0:
            raise ValueError('cannot convert float NaN to integer')
        _h_int_index(self, 'index.min()')
        return self._keys.first

    cdef long long last_int(self) except? -1:
        if self.rows == 0:
            raise ValueError('cannot convert float NaN to integer')
        _h_int_index(self, 'index.max()')
        return self._keys.last

    cdef object first_key(self):
        if self._first_obj is _no_extreme:
            mx_handle_fallback()
            return self.helper().index.min()
        return self._first_obj

    cdef object last_key(self):
        if self._last_obj is _no_extreme:
            mx_handle_fallback()
            return self.helper().index.max()
        return self._last_obj

    cdef Py_ssize_t asof_row(self, long long k) except -2:
        """Row of index.asof(k): the last key <= k; KeyError(k) below the first key (D25a)."""
        cdef Py_ssize_t lo = 0, hi, mid
        if self.levels == 1 and self._int_keys:
            if self._keys.scheme == TABLE_DENSE:
                if k >= self._keys.last:
                    return self.rows - 1
                if k < self._keys.first:
                    raise KeyError(k)
                return <Py_ssize_t>(<unsigned long long>k - <unsigned long long>self._keys.first)
            if self._increasing:
                hi = self.rows
                while lo < hi:
                    mid = lo + (hi - lo) // 2
                    if k < self._okeys[mid]:
                        hi = mid
                    else:
                        lo = mid + 1
                if lo == 0:
                    raise KeyError(k)
                return lo - 1
        mx_handle_fallback()        # labels not in increasing order: the helper (pandas) answers
        return self.row_1(<object>self.helper().index.asof(k))

    cdef long long asof_int(self, long long k) except? -1:
        cdef Py_ssize_t row
        if self.levels == 1 and self._int_keys:
            if self._keys.scheme == TABLE_DENSE:
                if k >= self._keys.last:
                    return self._keys.last
                if k < self._keys.first:
                    raise KeyError(k)
                return k
            if self._increasing:
                return self._okeys[self.asof_row(k)]
        mx_handle_fallback()
        return self.helper().index.asof(k)

    cdef Py_ssize_t searchsorted_right(self, long long k) noexcept:
        """index.searchsorted(k, side="right") on a one-level int index; -1 when not applicable
        (the read is then not typed). Bisects the keys in row order, as the helper."""
        cdef Py_ssize_t lo = 0, hi, mid
        if self.levels != 1 or not self._int_keys:
            return -1
        if self._keys.scheme == TABLE_DENSE:
            if k < self._keys.first:
                return 0
            if k >= self._keys.last:
                return self.rows
            return <Py_ssize_t>(<unsigned long long>k - <unsigned long long>self._keys.first) + 1
        hi = self.rows
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if k < self._okeys[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    cdef long long key_int(self, Py_ssize_t i) except? -1:
        cdef Py_ssize_t position = i + self.rows if i < 0 else i
        _h_int_index(self, 'index[i]')
        if not 0 <= position < self.rows:
            raise IndexError(f'index {i} is out of bounds for axis 0 with size {self.rows}')
        if self._keys.scheme == TABLE_DENSE:
            return self._keys.first + position
        return self._okeys[position]

    # -- escapes ----------------------------------------------------------------------------


    cdef object row_record(self, Py_ssize_t row):
        """The row as the helper's Record (T.loc[k]): a Python object, built by the helper."""
        if <size_t>row >= <size_t>self.rows:
            raise IndexError(f'table row {row} is out of range')
        mx_handle_fallback()
        record = self.helper().loc[_h_label(self, row)]
        if self._colbase or self.width != len(record):
            return record[self._columns[0]] if self.width == 1 else record
        return record

    cdef TableHandle column_view(self, Py_ssize_t col):
        if <size_t>col >= <size_t>self.width:
            raise IndexError(f'table column {col} is out of range')
        return _column_of(self, self._colbase + col, (self._columns[col],))

    cdef object table_object(self):
        """The bound table of this handle (D45): a NativeTable whose handle is this handle, the
        same object on every call; Python reads on it go to the helper's table (a sub-table is
        materialised by the helper's per-prefix memo on first Python use only)."""
        if self._table_obj is None:
            raise TypeError('a column view is not a table')
        return self._table_obj

    # -- Python side (tests, receipts) -------------------------------------------------------

    def helper(self):
        """The helper Table this handle reads (a sub-table is materialised by the helper's memo)."""
        if not self._prefix:
            return self.owner
        return self.owner.loc[self._prefix if len(self._prefix) > 1 else self._prefix[0]]

    @property
    def scheme_name(self):
        return {TABLE_DENSE: 'dense', TABLE_SORTED: 'sorted', TABLE_HASHED: 'hashed',
                TABLE_PRODUCT: 'product'}.get(self.scheme, '?')

    @property
    def shape(self):
        return (self.rows, self.width, self.levels, self.base)

    @property
    def prefix(self):
        return self._prefix

    @property
    def children(self):
        return self._children

    def memory(self):
        """Bytes of this handle tree. `children`: what the eager sub-tables add (their handle
        objects, own key indexes and label tuples); `shared`: everything else (typed columns,
        key parts, the root's indexes, read by every handle of the table); `handles`: count."""
        import sys as _sys
        cdef TableHandle handle
        arrays = 0
        for item in self._hold:
            if type(item) is bytearray:
                arrays += len(<bytearray>item)
            elif type(item) is tuple:
                arrays += _sys.getsizeof(item)
            elif hasattr(item, 'nbytes'):
                arrays += item.nbytes
        count, objects, owned, labels = 0, 0, 0, 0
        stack = [(child, self.scheme) for child in self._children]
        while stack:
            handle, parent = stack.pop()
            count += 1
            objects += _sys.getsizeof(handle)
            if parent != TABLE_PRODUCT:             # PRODUCT children share their axes' arrays
                owned += _own_key_bytes(handle)
                labels += _sys.getsizeof(handle._labels)
            stack.extend((child, handle.scheme) for child in handle._children)
        children = objects + owned + labels
        total = arrays + _sys.getsizeof(self) + objects + labels
        return {'handles': 1 + count, 'children': children, 'shared': total - children, 'total': total}


cdef Py_ssize_t _own_key_bytes(TableHandle h):
    """Bytes of the key arrays a row-stored sub-table owns (its full-key and child indexes)."""
    cdef Py_ssize_t size = 0
    if h.scheme == TABLE_HASHED:
        size += (h._keys.mask + 1) * sizeof(Py_ssize_t)
        if h.levels > 1:
            size += (h._ckeys.mask + 1) * sizeof(Py_ssize_t) + h._ckeys.count * sizeof(HandlePart)
    elif h.scheme == TABLE_SORTED:
        size += h.rows * (sizeof(long long) + sizeof(Py_ssize_t))
    if h._okeys != NULL:
        size += h.rows * sizeof(long long)
    return size


@cython.final
cdef class _HandleTable:
    """Owner of the table object of a sub-table handle: the helper's sub-table, asked from the
    helper (memoised per prefix there) when Python code first reads it, then held here.

    Holding it matters: a per-policy table-valued cell (``return T.loc[prefix]``, D53) is read
    by per-step formulas (``sub["c"]``, ``sub.index``); asking the helper each time ran
    ``owner.loc[prefix]`` (its memo lookup) on every read, twice to three times the cost of
    the read itself (diag-uc-fr). The write is benign when two threads race: both store the
    helper's memoised object, the same one."""
    cdef TableHandle handle
    cdef object _sub

    cdef inline object _helper(self):
        cdef object sub = self._sub
        if sub is None:
            sub = self.handle.helper()
            self._sub = sub
        return sub

    def _table(self):
        return self._helper()

    def __getattr__(self, name):
        return getattr(self._helper(), name)

    def __getitem__(self, key):
        return self._helper()[key]

    def __iter__(self):
        return iter(self._helper())

    def __len__(self):
        return len(self._helper())

    def __contains__(self, key):
        return key in self._helper()

    def __eq__(self, other):
        return self._helper() == other

    def __hash__(self):
        return id(self)

    def __reduce__(self):
        return self._helper().__reduce__()


Table.register(_HandleTable)


cdef inline object _semantic_owner(object owner):
    if type(owner) is _HandleTable:
        return (<_HandleTable>owner)._helper()
    return owner


cdef void _attach_table_objects(TableHandle root) except *:
    """Give every handle of a table its NativeTable, while the tree is still private."""
    cdef TableHandle handle
    cdef NativeTable table
    cdef _HandleTable lazy
    stack = [root]
    while stack:
        handle = <TableHandle>stack.pop()
        table = NativeTable.__new__(NativeTable)
        if handle._prefix:
            lazy = _HandleTable.__new__(_HandleTable)
            lazy.handle = handle
            table._owner = lazy
        else:
            table._owner = handle.owner
        table.handle = handle
        handle._table_obj = table
        stack.extend(handle._children)


cdef TableHandle _column_of(TableHandle source, Py_ssize_t colbase, tuple columns):
    cdef TableHandle view = _clone(source)
    view._colbase = colbase
    view._columns = columns
    view.width = len(columns)
    view._table_obj = None
    return view


@cython.final
cdef class RecordHandle:
    """Typed field reads of one helper Record: field(name) once, then f64/i64/obj by slot."""

    def __init__(self, owner):
        _record_init(self, owner)

    cdef Py_ssize_t field(self, object name) except -2:
        cdef Py_ssize_t j, n = mx_tuple_size(<PyObject*>self._items)
        cdef PyObject* key
        for j in range(n):
            if <PyObject*>name == mx_tuple_item(mx_tuple_item(<PyObject*>self._items, j), 0):
                return j
        for j in range(n):
            key = mx_tuple_item(mx_tuple_item(<PyObject*>self._items, j), 0)
            if PyUnicode_Check(name) and PyUnicode_Check(<object>key):
                if mx_unicode_compare(<PyObject*>name, key) == 0:
                    return j
            elif not PyUnicode_Check(name) and not PyUnicode_Check(<object>key):
                if mx_rich_equal(<PyObject*>name, key, 2):
                    mx_handle_fallback()
                    return j
        raise KeyError(name)


    cdef double f64(self, Py_ssize_t slot) except? -1:
        return _object_f64(_h_value(self, slot))

    cdef long long i64(self, Py_ssize_t slot) except? -1:
        return _object_i64(_h_value(self, slot))

    cdef object obj(self, Py_ssize_t slot):
        return <object>_h_value(self, slot)


cdef void _record_init(RecordHandle handle, object owner) except *:
    cdef PyObject* items
    handle.owner = owner
    if isinstance(owner, Record) and _record_items_offset >= 0:
        # the Record's own immutable pairs, read from its slot (no attribute lookup)
        items = (<PyObject**>(<char*><PyObject*>owner + _record_items_offset))[0]
        if items != NULL and PyTuple_Check(<object>items):
            handle._items = <tuple>items
            return
    if isinstance(owner, Record):
        handle._items = owner._items
        return
    mx_handle_fallback()
    handle._items = tuple(owner.items())


cdef object _bound_handle(object owner):
    """The handle of a helper Table, built once per table object and kept in its native cache
    (at preparation for tables built by compiled code, else at the first table_handle request)."""
    cdef dict cache = owner._native_cache
    handle = cache.get(TableHandle)
    if handle is None:
        handle = cache.setdefault(TableHandle, TableHandle(owner))
    return handle


cdef object _try_bound_handle(object owner):
    """_bound_handle, or None when this table has no handle form (a bind never fails on it)."""
    if _table_storage_fn is None:
        return None
    try:
        return _bound_handle(owner)
    except Exception:
        return None


cdef TableHandle table_handle(object value):
    """The handle of a bound table: NativeTable.handle, else the table's own handle."""
    if type(value) is NativeTable:
        if (<NativeTable>value).handle is not None:
            return (<NativeTable>value).handle
        value = (<NativeTable>value)._owner
    elif type(value) is TableHandle:
        return <TableHandle>value
    mx_handle_fallback()        # not a bound table: the handle is found through its cache
    if not isinstance(value, Table):
        raise TypeError(f'expected a Table, got {type(value).__name__}')
    return _bound_handle(value)


cdef RecordHandle record_handle(object value):
    cdef RecordHandle handle = RecordHandle.__new__(RecordHandle)
    _record_init(handle, value)
    return handle


# ---- W1.9: borrowed sub-tables and index labels ----------------------------------------------

cdef inline TableRef _sub_ref(TableHandle h, const HandlePart* q, Py_ssize_t n) except? NULL:
    """Borrowed shared child for n prefix parts, NULL when absent (no reference is taken)."""
    cdef PyObject* handle = <PyObject*>h
    cdef Py_ssize_t j
    if n >= h.levels:
        raise TypeError(f'sub_{n} needs fewer key parts than the {h.levels} levels of the table')
    for j in range(n):
        handle = _h_child(handle, &q[j])
        if handle == NULL:
            return NULL
    if (<TableHandle>handle)._colbase != h._colbase or (<TableHandle>handle).width != h.width:
        raise TypeError('a column view has no borrowed sub-table')    # never lowered on views
    return handle


cdef TableRef sub_ref_1(TableHandle h, TableKey p0) except NULL:
    cdef HandlePart q[1]
    _set_part(&q[0], p0, h._spec_missing)
    found = _sub_ref(h, q, 1)
    if found == NULL:
        raise KeyError(_original(p0))
    return found


cdef TableRef sub_ref_2(TableHandle h, TableKey p0, TableKey p1) except NULL:
    cdef HandlePart q[2]
    _set_part(&q[0], p0, h._spec_missing)
    _set_part(&q[1], p1, h._spec_missing)
    found = _sub_ref(h, q, 2)
    if found == NULL:
        raise KeyError((_original(p0), _original(p1)))
    return found


cdef TableRef sub_ref_3(TableHandle h, TableKey p0, TableKey p1, TableKey p2) except NULL:
    cdef HandlePart q[3]
    _set_part(&q[0], p0, h._spec_missing)
    _set_part(&q[1], p1, h._spec_missing)
    _set_part(&q[2], p2, h._spec_missing)
    found = _sub_ref(h, q, 3)
    if found == NULL:
        raise KeyError((_original(p0), _original(p1), _original(p2)))
    return found


cdef TableRef sub_ref_4(TableHandle h, TableKey p0, TableKey p1, TableKey p2, TableKey p3) except NULL:
    cdef HandlePart q[4]
    _set_part(&q[0], p0, h._spec_missing)
    _set_part(&q[1], p1, h._spec_missing)
    _set_part(&q[2], p2, h._spec_missing)
    _set_part(&q[3], p3, h._spec_missing)
    found = _sub_ref(h, q, 4)
    if found == NULL:
        raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3)))
    return found


cdef TableRef sub_ref_5(TableHandle h, TableKey p0, TableKey p1, TableKey p2, TableKey p3,
                        TableKey p4) except NULL:
    cdef HandlePart q[5]
    _set_part(&q[0], p0, h._spec_missing)
    _set_part(&q[1], p1, h._spec_missing)
    _set_part(&q[2], p2, h._spec_missing)
    _set_part(&q[3], p3, h._spec_missing)
    _set_part(&q[4], p4, h._spec_missing)
    found = _sub_ref(h, q, 5)
    if found == NULL:
        raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3), _original(p4)))
    return found


cdef inline Py_ssize_t _key_position(TableHandle h, Py_ssize_t i) except -1:
    cdef Py_ssize_t position = i + h.rows if i < 0 else i
    if not 0 <= position < h.rows:
        raise IndexError(f'index {i} is out of bounds for axis 0 with size {h.rows}')
    return position


cdef object key_label(TableHandle h, Py_ssize_t i):
    """The i-th label of the index in row order (a tuple on several levels), as the helper's
    ``index[i]``; negative i counts from the end."""
    return _h_label(h, _key_position(h, i))


cdef double key_f64(TableHandle h, Py_ssize_t i) except? -1:
    """``float(index[i])``: a float label is read without a reference on it."""
    cdef Py_ssize_t position = _key_position(h, i)
    cdef PyObject* label
    if h.scheme != TABLE_PRODUCT and h._labels is not None and h.levels == 1:
        label = mx_tuple_item(<PyObject*>h._labels, position)
        if PyFloat_CheckExact(<object>label):
            return PyFloat_AS_DOUBLE(<object>label)
    return float(_h_label(h, position))


def key_label_of(TableHandle h, Py_ssize_t i):
    """Python access to key_label (tests)."""
    return key_label(h, i)


def key_f64_of(TableHandle h, Py_ssize_t i):
    """Python access to key_f64 (tests)."""
    return key_f64(h, i)


def sub_ref_of(TableHandle h, *parts):
    """Python access to sub_ref_n (tests): the shared child handle it borrows."""
    cdef TableRef found
    n = len(parts)
    if n == 1:
        found = sub_ref_1[object](h, parts[0])
    elif n == 2:
        found = sub_ref_2[object](h, parts[0], parts[1])
    elif n == 3:
        found = sub_ref_3[object](h, parts[0], parts[1], parts[2])
    else:
        raise TypeError('sub_ref_of takes 1 to 3 parts')
    return <object>found


def table_handle_of(value):
    """Python access to table_handle (tests and receipts)."""
    return table_handle(value)
