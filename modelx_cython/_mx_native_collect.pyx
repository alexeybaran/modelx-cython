# cython: freethreading_compatible=True
"""Owned numeric storage for compiled Collect results."""

import sys

from libc.stdlib cimport calloc, free, malloc
from libc.limits cimport LLONG_MIN, LLONG_MAX


cdef enum:
    FLOAT = 0
    INTEGER = 1
    BOOL = 2
    PYTHON = 3


cdef NumericValue* allocate(Py_ssize_t width, Py_ssize_t count, bint zero) except? NULL:
    cdef NumericValue* values
    if width == 0:
        return NULL
    if count > sys.maxsize // sizeof(NumericValue) // width:
        raise MemoryError('Collect result is too large')
    if zero:
        values = <NumericValue*>calloc(count * width, sizeof(NumericValue))
    else:
        values = <NumericValue*>malloc(count * width * sizeof(NumericValue))
    if values == NULL:
        raise MemoryError()
    return values


cdef Field* make_fields(tuple schema, Py_ssize_t* width, Py_ssize_t* object_width) except? NULL:
    cdef Field* fields
    cdef Py_ssize_t index, k, length
    cdef long long key
    if not schema:
        return NULL
    fields = <Field*>calloc(len(schema), sizeof(Field))
    if fields == NULL:
        raise MemoryError()
    try:
        for index, (_, domain, kind) in enumerate(schema):
            fields[index].kind = kind
            if kind == PYTHON:
                fields[index].offset = object_width[0]
                object_width[0] += 1
                continue
            length = 1 if domain is None else len(domain)
            if length > sys.maxsize - width[0]:
                raise MemoryError('Collect result is too wide')
            fields[index].offset = width[0]
            fields[index].width = length
            width[0] += length
            if domain is None or not domain:
                continue
            fields[index].start = domain[0]
            if all(key - domain[0] == k for k, key in enumerate(domain)):
                continue
            fields[index].keys = <long long*>malloc(length * sizeof(long long))
            if fields[index].keys == NULL:
                raise MemoryError()
            for k, key in enumerate(domain):
                fields[index].keys[k] = key
    except BaseException:
        release_fields(fields, len(schema))
        raise
    return fields


cdef void release_fields(Field* fields, Py_ssize_t count) noexcept:
    cdef Py_ssize_t i
    if fields != NULL:
        for i in range(count):
            free(fields[i].keys)
        free(fields)


cpdef object integer_keys(object value):
    """Accept plain integer axes. None asks the compiler hook to use its general path."""
    if type(value) not in (tuple, range, list):
        return None
    keys = tuple(value)
    for key in keys:
        if type(key) is not int or not -(1 << 63) <= key < (1 << 63):
            return None
    if len(keys) != len(set(keys)):
        return None
    return keys


cdef class NativeCollect:
    def __init__(self, tuple coordinates, tuple individual, tuple sums,
                 Py_ssize_t workers, result_type):
        self.coordinates = coordinates
        self.sections = tuple(name for name, schema in (('individual', individual), ('sum', sums)) if schema)
        self.schemas = individual, sums
        self.individual_count, self.sum_count = len(individual), len(sums)
        self.worker_count = workers
        self.result_type = result_type
        self.python_rows = {}
        self.large_sums = {}
        if individual:
            first = coordinates[0]
            if (type(first) is int and LLONG_MIN <= first <= LLONG_MAX - len(coordinates) + 1
                    and all(type(key) is int and key == first + i for i, key in enumerate(coordinates))):
                self.dense_coordinates = True
                self.first_coordinate = first
            else:
                self.positions = {coordinate: i for i, coordinate in enumerate(coordinates)}
        self.individual_fields = make_fields(individual, &self.row_width, &self.object_width)
        self.sum_fields = make_fields(sums, &self.sum_width, &self.object_width)
        self.rows = allocate(self.row_width, len(coordinates), False)
        self.sums = allocate(self.sum_width, workers, True)
        if self.sum_width and any(kind in (INTEGER, BOOL) for _, _, kind in sums):
            self.wide = <unsigned char*>calloc(workers * self.sum_width, 1)
            if self.wide == NULL:
                raise MemoryError()
        if self.object_width:
            if len(coordinates) > sys.maxsize // self.object_width:
                raise MemoryError('Collect result is too large')
            self.object_rows = [None] * (len(coordinates) * self.object_width)

    def __dealloc__(self):
        release_fields(self.individual_fields, self.individual_count)
        release_fields(self.sum_fields, self.sum_count)
        free(self.rows)
        free(self.sums)
        free(self.wide)

    cdef void write(self, bint individual, Py_ssize_t row, Py_ssize_t worker,
                    Py_ssize_t field, Py_ssize_t index, double value) noexcept nogil:
        if individual:
            self.rows[row * self.row_width + self.individual_fields[field].offset + index].floating = value
        else:
            self.sums[worker * self.sum_width + self.sum_fields[field].offset + index].floating += value

    cdef void write_int(self, bint individual, Py_ssize_t row, Py_ssize_t worker,
                        Py_ssize_t field, Py_ssize_t index, long long value) except *:
        if individual:
            self.rows[row * self.row_width + self.individual_fields[field].offset + index].integer = value
        else:
            self.add_int(worker * self.sum_width + self.sum_fields[field].offset + index, value)

    cdef void add_int(self, Py_ssize_t offset, long long value) except *:
        cdef long long old = self.sums[offset].integer
        if self.wide[offset]:
            self.large_sums[offset] += value
        elif ((value > 0 and old > LLONG_MAX - value) or (value < 0 and old < LLONG_MIN - value)):
            self.large_sums[offset] = int(old) + int(value)
            self.wide[offset] = 1
        else:
            self.sums[offset].integer = old + value

    cdef void write_object(self, Py_ssize_t row, Py_ssize_t field, object value) except *:
        self.object_rows[row * self.object_width + self.individual_fields[field].offset] = value

    def finish(self):
        cdef Py_ssize_t worker, field, i, partial
        cdef Field column
        for worker in range(1, self.worker_count):
            for field in range(self.sum_count):
                column = self.sum_fields[field]
                for i in range(column.offset, column.offset + column.width):
                    partial = worker * self.sum_width + i
                    if column.kind == FLOAT:
                        self.sums[i].floating += self.sums[partial].floating
                    elif self.wide[partial]:
                        self.large_sums[i] = self.box(False, 0, INTEGER, i) + self.large_sums[partial]
                        self.wide[i] = 1
                    else:
                        self.add_int(i, self.sums[partial].integer)
        result = self.result_type._from_native(self)
        self.python_data = result._data
        return result

    @property
    def fields(self):
        return self.sections

    cpdef object keys(self, bint individual, Py_ssize_t field):
        return self.schemas[0 if individual else 1][field][1]

    def materialize(self, str section):
        cdef Py_ssize_t row
        if section == 'individual':
            return {key: self.row_as_python(row) for row, key in enumerate(self.coordinates)}
        return self.result_type(self._as_fields(False, 0))

    def _as_fields(self, bint individual, Py_ssize_t row):
        cdef Py_ssize_t field, index, offset
        cdef Field* layout = self.individual_fields if individual else self.sum_fields
        schema = self.schemas[0 if individual else 1]
        result = {}
        for field, (name, domain, kind) in enumerate(schema):
            offset = layout[field].offset
            if domain is None:
                result[name] = self.box(individual, row, kind, offset)
            else:
                result[name] = {key: self.box(individual, row, kind, offset + index)
                                for index, key in enumerate(domain)}
        return result

    cdef object box(self, bint individual, Py_ssize_t row, int kind, Py_ssize_t offset):
        cdef NumericValue* values
        if kind == PYTHON:
            return self.object_rows[row * self.object_width + offset]
        values = self.rows + row * self.row_width if individual else self.sums
        if kind == FLOAT:
            return values[offset].floating
        if not individual and self.wide[offset]:
            return self.large_sums[offset]
        if kind == BOOL and (individual or len(self.coordinates) == 1):
            return bool(values[offset].integer)
        return values[offset].integer

    cdef object row_as_python(self, Py_ssize_t row):
        record = self.python_rows.get(row)
        if record is None:
            record = self.python_rows.setdefault(row, self.result_type(self._as_fields(True, row)))
        return record

    cdef Py_ssize_t coordinate_position(self, object coordinate) except -1:
        cdef long long key
        cdef unsigned long long offset
        if not self.dense_coordinates:
            return self.positions[coordinate]
        if type(coordinate) is int:
            if LLONG_MIN <= coordinate <= LLONG_MAX:
                key = coordinate
                offset = <unsigned long long>key - <unsigned long long>self.first_coordinate
                if key >= self.first_coordinate and offset < <unsigned long long>len(self.coordinates):
                    return <Py_ssize_t>offset
        else:
            hash(coordinate)
            try:
                return self.coordinates.index(coordinate)
            except ValueError:
                pass
        raise KeyError(coordinate)


cdef class NativeRow:
    def __getattr__(self, str name):
        return getattr(self.owner.row_as_python(self.row), name)

    def __reduce__(self):
        return self.owner.row_as_python(self.row).__reduce__()


cdef object individual_at(object result, object coordinate):
    cdef NativeCollect owner
    cdef NativeRow row
    native = getattr(result, '_mx_native', None)
    if type(native) is NativeCollect:
        owner = <NativeCollect>native
        if owner.individual_fields != NULL and 'individual' not in owner.python_data:
            row = NativeRow.__new__(NativeRow)
            row.owner = owner
            row.row = owner.coordinate_position(coordinate)
            return row
    return result.individual[coordinate]


cdef Py_ssize_t field_offset(Field* column, long long key, bint series) except -1:
    cdef unsigned long long offset
    cdef Py_ssize_t index
    if not series:
        return column.offset
    if column.keys == NULL:
        offset = <unsigned long long>key - <unsigned long long>column.start
        if key >= column.start and offset < <unsigned long long>column.width:
            return column.offset + <Py_ssize_t>offset
    else:
        for index in range(column.width):
            if column.keys[index] == key:
                return column.offset + index
    raise KeyError(key)


cdef NativeRow numeric_row(object record, Py_ssize_t field, str name, bint series, bint floating):
    cdef NativeRow row
    if type(record) is NativeRow:
        row = <NativeRow>record
        if (field < row.owner.individual_count and row.owner.schemas[0][field][0] == name
                and (row.owner.schemas[0][field][1] is not None) == series
                and row.owner.individual_fields[field].kind != PYTHON
                and (row.owner.individual_fields[field].kind == FLOAT) == floating
                and row.row not in row.owner.python_rows):
            return row
    return None


cdef object row_value(object record, str name, long long key, bint series, bint index_read):
    value = getattr(record, name)
    return (value[key] if index_read else value(key)) if series else value


cdef NativeCollect numeric_sum(object result, Py_ssize_t field, str name, bint series, bint floating):
    cdef NativeCollect owner
    native = getattr(result, '_mx_native', None)
    if type(native) is NativeCollect:
        owner = <NativeCollect>native
        if (field < owner.sum_count and owner.schemas[1][field][0] == name
                and (owner.schemas[1][field][1] is not None) == series
                and (owner.sum_fields[field].kind == FLOAT) == floating and 'sum' not in owner.python_data):
            return owner
    return None


cdef object sum_value(object result, str name, long long key, bint series, bint index_read):
    value = getattr(result.sum, name)
    return (value[key] if index_read else value(key)) if series else value


cdef double read_individual(object record, Py_ssize_t field, str name, long long key,
                            bint series, bint index_read) except? -1:
    cdef NativeRow row = numeric_row(record, field, name, series, True)
    if row is None:
        return row_value(record, name, key, series, index_read)
    return row.owner.rows[row.row * row.owner.row_width + field_offset(&row.owner.individual_fields[field], key, series)].floating


cdef long long read_individual_int(object record, Py_ssize_t field, str name, long long key,
                                   bint series, bint index_read) except? -1:
    cdef NativeRow row = numeric_row(record, field, name, series, False)
    if row is None:
        return row_value(record, name, key, series, index_read)
    return row.owner.rows[row.row * row.owner.row_width + field_offset(&row.owner.individual_fields[field], key, series)].integer


cdef double read_sum(object result, Py_ssize_t field, str name, long long key,
                     bint series, bint index_read) except? -1:
    cdef NativeCollect owner = numeric_sum(result, field, name, series, True)
    if owner is None:
        return sum_value(result, name, key, series, index_read)
    return owner.sums[field_offset(&owner.sum_fields[field], key, series)].floating


cdef long long read_sum_int(object result, Py_ssize_t field, str name, long long key,
                            bint series, bint index_read) except? -1:
    cdef NativeCollect owner = numeric_sum(result, field, name, series, False)
    cdef Py_ssize_t offset
    if owner is None:
        return sum_value(result, name, key, series, index_read)
    offset = field_offset(&owner.sum_fields[field], key, series)
    if owner.wide[offset]:
        return owner.large_sums[offset]
    return owner.sums[offset].integer
