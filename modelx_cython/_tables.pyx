# cython: freethreading_compatible=True, boundscheck=False, wraparound=False, initializedcheck=False
"""Preparation of a pandas table for the keyed reads compiled code makes fast.

``prepare(value)`` turns the user's DataFrame or Series into a ``PreparedTable``: the rows stored sorted by their
level codes, one ascending array of combined codes (and a dense grid when small) for keyed lookups, and typed
column blocks copied from the frame. The user's object is never changed or replaced: it stays what every formula
sees. ``prepare`` returns None, and ``unprepared_reason`` says why, for a table whose reads must stay on pandas (a
non-unique index, NaN labels, labels that compare equal although pandas keeps them apart, duplicate column labels,
too many key combinations, an index level pandas looks up by rules of its own: dates, periods, intervals,
categories, bools, labels of mixed types).

A lookup equals ``value.loc[key, column]`` on the tables ``prepare`` accepts. The key lookups and value reads
compiled code uses are inline functions of the .pxd. This file is copied into each compiled model package that
reads a table, as the numeric caches are, so a compiled model needs no part of modelx_cython at run time.
"""
import weakref

import numpy as np
import pandas as pd

cdef Py_ssize_t _GRID_LIMIT = 65536
cdef long long _COMBINED_LIMIT = 1LL << 62
cdef int _MAX_LEVELS = 8


cdef extern from "Python.h":
    Py_hash_t _object_hash "PyObject_Hash"(PyObject* o) except -1
    int _object_equal "PyObject_RichCompareBool"(PyObject* a, PyObject* b, int op) except -1
    int _Py_EQ "Py_EQ"


cdef inline bint _plain_int(object label):
    # an int label (including IntEnum members and numpy integers), not a bool
    return isinstance(label, (int, np.integer)) and not isinstance(label, (bool, np.bool_))


cdef class PreparedTable:
    """A pandas table prepared for keyed reads. Immutable once built."""

    @property
    def source(self):
        """The user's frame or Series this table was prepared from (None once it is gone)."""
        return self._source()

    cdef Py_ssize_t row_search(self, long long combined) noexcept nogil:
        """The stored row of a combined code by binary search (a table without a grid), or -1. Out of line, so the
        inline reads stay small."""
        cdef Py_ssize_t lo = 0, hi = self.nrows, mid
        while lo < hi:
            mid = (lo + hi) >> 1
            if self.combined[mid] < combined:
                lo = mid + 1
            else:
                hi = mid
        if lo < self.nrows and self.combined[lo] == combined:
            return lo
        return -1

    cdef Py_ssize_t name_position(self, object label) except -2:
        """The position of column ``label`` (a name chosen at run time), or -1: the hash and equality of a dict
        lookup, through borrowed pointers only (no reference, no lock per read)."""
        cdef Py_hash_t h = _object_hash(<PyObject*>label)
        cdef Py_ssize_t i = <Py_ssize_t>(<size_t>h & <size_t>self.name_mask), position
        while True:
            position = self.name_pos[i]
            if position < 0:
                return -1
            if <PyObject*>self.name_obj[i] == <PyObject*>label or (
                    self.name_hash[i] == h and _object_equal(<PyObject*>self.name_obj[i], <PyObject*>label, _Py_EQ)):
                return position
            i = (i + 1) & self.name_mask

    cdef Py_ssize_t code_obj(self, int level, object key) except -2:
        """The code of any key part at ``level``, or -1 when absent."""
        cdef object code
        codes = self.level_codes[level]
        if codes is None:
            if type(key) is int:            # the common case first (no numpy type lookups)
                if not -0x8000000000000000 <= key <= 0x7fffffffffffffff:
                    return -1
                return self.code_ll(level, <long long>key)
            if isinstance(key, (float, np.floating)) and float(key).is_integer():
                key = int(key)          # pandas finds an integral float key on an int index
            if not _plain_int(key) or not -0x8000000000000000 <= key <= 0x7fffffffffffffff:
                return -1
            return self.code_ll(level, <long long>key)
        if type(key) is bool or isinstance(key, np.bool_):
            return -1               # no label of a text or float level, although True == 1.0 (pandas: KeyError)
        code = (<dict>codes).get(key)
        return -1 if code is None else <Py_ssize_t>code

    cdef Py_ssize_t row_key(self, object key) except -2:
        """The stored row of a full key given as Python objects (a scalar, or a tuple), or -1."""
        cdef Py_ssize_t[8] buffer
        cdef Py_ssize_t code
        cdef int level
        if self.nlevels == 1:
            return -1 if isinstance(key, tuple) else self.code_obj(0, key)
        if not isinstance(key, tuple) or len(<tuple>key) != self.nlevels:
            return -1
        for level in range(self.nlevels):
            code = self.code_obj(level, (<tuple>key)[level])
            if code < 0:
                return -1
            buffer[level] = code
        return self.row_codes(buffer, self.nlevels)

    cdef bint prefix_range(self, const Py_ssize_t* codes, int count, Py_ssize_t* start, Py_ssize_t* stop) noexcept nogil:
        """The stored rows [start, stop) whose first ``count`` level codes are ``codes``; False when none."""
        cdef long long low = 0, high
        cdef int i
        cdef Py_ssize_t lo, hi, mid
        if count < 1 or count > self.nlevels:
            return False
        for i in range(count):
            if codes[i] < 0 or codes[i] >= self.level_size[i]:
                return False
            low += codes[i] * self.radix[i]
        high = low + self.radix[count - 1]
        lo, hi = 0, self.nrows
        while lo < hi:
            mid = (lo + hi) >> 1
            if self.combined[mid] < low:
                lo = mid + 1
            else:
                hi = mid
        start[0] = lo
        hi = self.nrows
        while lo < hi:
            mid = (lo + hi) >> 1
            if self.combined[mid] < high:
                lo = mid + 1
            else:
                hi = mid
        stop[0] = lo
        return stop[0] > start[0]

    cdef object object_at(self, Py_ssize_t row, Py_ssize_t slot):
        return (<tuple>self.objects[slot])[row]


def unprepared_reason(value):
    """Why ``prepare(value)`` returns None (None when it does not)."""
    try:
        _build(value)
    except Exception as error:
        return str(error) or type(error).__name__
    return None


def prepare(value):
    """The PreparedTable of a DataFrame or Series, or None when its reads must stay on pandas."""
    try:
        return _build(value)
    except Exception:       # also whatever pandas raises on labels it cannot order or convert: the reads stay pandas
        return None


def _level(values):
    """(kind, labels in code order, first, dict or None) for the labels of one index level."""
    unique = list(dict.fromkeys(values))
    if values.hasnans:
        raise ValueError('NaN labels')
    if len(unique) != len(pd.unique(values)):
        raise ValueError('labels that compare equal but pandas keeps apart')
    if unique and all(_plain_int(label) for label in unique):
        ordered = sorted(unique)
        if ordered[len(ordered) - 1] - ordered[0] + 1 == len(ordered):
            return LEVEL_DENSE, ordered, ordered[0], None
        return LEVEL_SORTED, ordered, 0, None
    try:
        ordered = sorted(unique)
    except TypeError:
        ordered = unique                     # labels without an order: codes in first appearance
    codes = {label: code for code, label in enumerate(ordered)}
    if len(codes) != len(ordered):
        raise ValueError('labels that compare equal but pandas keeps apart')
    return LEVEL_OBJECT, ordered, 0, codes


cdef size_t _address(array):
    """The address of the first element of a C-contiguous numpy array (kept alive by its table)."""
    assert array.flags.c_contiguous and array.size > 0
    return <size_t>array.ctypes.data


cdef void _point(PreparedTable table, sorted_array, combined, grid, f64, i64, kinds, slots):
    """Point the table's raw pointers into its arrays and keep the arrays alive with it."""
    table._arrays = (sorted_array, combined, grid, f64, i64, kinds, slots)
    for array in table._arrays:
        array.flags.writeable = False
    table.sorted_labels = <const long long*>_address(sorted_array)
    table.combined = <const long long*>_address(combined)
    table.grid = <const Py_ssize_t*>_address(grid)
    table.f64 = <const double*>_address(f64)
    table.i64 = <const long long*>_address(i64)
    table.column_kind = <const int*>_address(kinds)
    table.column_slot = <const Py_ssize_t*>_address(slots)


cdef void _name_map(PreparedTable table):
    """The column name map of ``name_position`` (open addressing, at least twice as many entries as columns)."""
    cdef Py_ssize_t size = 8, i
    while size < 2 * len(table.column_positions):
        size *= 2
    objs = np.zeros(size, dtype=np.intp)
    hashes = np.zeros(size, dtype=np.intp)      # Py_hash_t has the width of Py_ssize_t
    positions = np.full(size, -1, dtype=np.intp)
    for label, position in table.column_positions.items():     # the dict's own key objects (kept alive by it)
        h = hash(label)
        i = h & (size - 1)
        while positions[i] >= 0:
            i = (i + 1) & (size - 1)
        objs[i], hashes[i], positions[i] = id(label), h, position
    table._name_arrays = (objs, hashes, positions)
    table.name_obj = <const Py_ssize_t*>_address(objs)
    table.name_hash = <const Py_hash_t*>_address(hashes)
    table.name_pos = <const Py_ssize_t*>_address(positions)
    table.name_mask = size - 1


cdef PreparedTable _build(object value):
    cdef PreparedTable table = PreparedTable.__new__(PreparedTable)
    cdef Py_ssize_t i, n
    cdef long long product = 1
    if isinstance(value, pd.Series):
        table.is_series = True
        frame = value.to_frame(name=0)
    elif isinstance(value, pd.DataFrame):
        table.is_series = False
        frame = value
    else:
        raise ValueError(f'not a pandas DataFrame or Series: {type(value).__name__}')
    index = frame.index
    if not index.is_unique:
        raise ValueError('the index is not unique')
    if not table.is_series and not frame.columns.is_unique:
        raise ValueError('duplicate column labels')
    if index.nlevels > _MAX_LEVELS:
        raise ValueError(f'more than {_MAX_LEVELS} index levels')
    for level_index in (index.levels if index.nlevels > 1 else (index,)):
        # a level is prepared only when pandas finds a label by its equality: not a date found by a text, a number
        # found inside an interval, a category, a bool
        if type(level_index) not in (pd.Index, pd.RangeIndex) or level_index.inferred_type not in (
                'integer', 'string', 'floating', 'mixed-integer-float', 'empty'):
            raise ValueError(f'an index level pandas looks up by rules of its own ({type(level_index).__name__} of '
                             f'{level_index.inferred_type} labels)')
    table._source = weakref.ref(value)
    table.nrows = n = len(index)
    table.nlevels = index.nlevels
    levels = [_level(index.get_level_values(i)) for i in range(table.nlevels)]
    sorted_labels, offset = [], 0
    for i in range(table.nlevels):
        kind, labels, first, codes = levels[i]
        table.level_kind[i] = kind
        table.level_first[i] = first
        table.level_size[i] = len(labels)
        table.level_offset[i] = offset
        if kind == LEVEL_SORTED:
            sorted_labels.extend(labels)
            offset += len(labels)
    table.level_labels = tuple(level[1] for level in levels)
    table.level_codes = tuple(level[3] for level in levels)
    table.level_names = tuple(index.names)
    sorted_array = np.ascontiguousarray(sorted_labels if sorted_labels else [0], dtype=np.int64)
    table._arrays = (sorted_array,)              # alive before code_ll reads it below
    table.sorted_labels = <const long long*>_address(sorted_array)
    for i in range(table.nlevels - 1, -1, -1):
        table.radix[i] = product
        product *= max(1, table.level_size[i])
        if product > _COMBINED_LIMIT:
            raise ValueError('too many key combinations')
    codes = np.zeros(n, dtype=np.int64)
    for i in range(table.nlevels):
        labels = index.get_level_values(i)
        level_codes = np.fromiter((table.code_obj(i, label) for label in labels), dtype=np.int64, count=n)
        codes += level_codes * table.radix[i]
    order = np.argsort(codes, kind='stable')
    # per prefix depth d (0: the whole table), whether the stored rows of every prefix of d levels are in pandas'
    # order, and the labels of level d exact as doubles: an ordered lookup then searches the stored rows
    within = []
    for d in range(table.nlevels):
        if d == 0:
            same = np.ones(max(n - 1, 0), dtype=bool)
        else:
            prefix = codes[order] // table.radix[d - 1]
            same = prefix[1:] == prefix[:-1]
        exact = all(_plain_int(label) and -9007199254740992 <= label <= 9007199254740992
                    for label in table.level_labels[d])
        within.append(bool(exact and (n < 2 or (order[1:][same] > order[:-1][same]).all())))
    table.ordered_within = tuple(within)
    combined = np.ascontiguousarray(codes[order] if n else np.zeros(1, dtype=np.int64), dtype=np.int64)
    if product <= _GRID_LIMIT:
        grid = np.full(max(1, product), -1, dtype=np.intp)
        grid[combined[:n]] = np.arange(n, dtype=np.intp)
        table.has_grid = True
    else:
        grid = np.zeros(1, dtype=np.intp)
    # columns: copies, stored in the sorted row order
    labels = [None] if table.is_series else list(frame.columns)
    kinds, slots, f64_cols, i64_cols, objects = [], [], [], [], []
    for position, label in enumerate(labels):
        column = frame.iloc[:, position]
        if column.dtype == np.float64:
            kinds.append(COLUMN_F64)
            slots.append(len(f64_cols))
            f64_cols.append(column.to_numpy(dtype=np.float64, copy=True)[order])
        elif column.dtype == np.int64:
            kinds.append(COLUMN_I64)
            slots.append(len(i64_cols))
            i64_cols.append(column.to_numpy(dtype=np.int64, copy=True)[order])
        else:
            kinds.append(COLUMN_OBJECT)
            slots.append(len(objects))
            if column.dtype == object or isinstance(column.dtype, pd.StringDtype):
                # the stored objects themselves, in one pass (not one pandas call per value)
                objects.append(tuple(column.to_numpy(dtype=object)[order]))
            else:
                # any other dtype (text, categories, dates): each value as pandas returns it
                objects.append(tuple(column.iat[int(row)] for row in order))
    table.column_positions = {label: position for position, label in enumerate(labels)}
    _name_map(table)
    kinds = np.ascontiguousarray(kinds if kinds else [0], dtype=np.intc)
    slots = np.ascontiguousarray(slots if slots else [0], dtype=np.intp)
    f64 = np.ascontiguousarray(np.column_stack(f64_cols) if f64_cols else np.zeros((max(1, n), 1)), dtype=np.float64)
    i64 = np.ascontiguousarray(np.column_stack(i64_cols) if i64_cols else np.zeros((max(1, n), 1), dtype=np.int64),
                               dtype=np.int64)
    table.f64_width, table.i64_width = f64.shape[1], i64.shape[1]
    _point(table, sorted_array, combined, grid, f64, i64, kinds, slots)
    table.objects = tuple(objects)
    # the dtype pandas gives a row (it widens number columns to one dtype: int32 beside float64 reads as float64)
    table.row_object = table.is_series or n == 0 or getattr(frame.iloc[0].dtype, 'kind', 'O') not in 'iufc'
    table.row_float = not table.row_object and frame.iloc[0].dtype.kind == 'f'     # a row of float64 fields
    table.max_label = index.max() if n else None        # index extremes, as pandas gives them
    table.min_label = index.min() if n else None
    return table


# ---- binding --------------------------------------------------------------------------------------------
# A read finds the table of a frame by the identity of the frame (``prepared``) within the generation it was
# prepared in: each new model of this package starts a generation (``renew``, called by its top-level Spaces), so a
# frame is prepared again for a new model and never read from a copy an earlier model made. The entry is dropped
# when the frame dies. Each operation on the map is one dict operation (no lock): two threads that miss prepare
# the same frame twice, and one table stays.
cdef dict _prepared = {}               # id(frame) -> (weak reference to the frame, PreparedTable or None, generation)
cdef Py_ssize_t _generation = 0
cdef PreparedTable _NOT_PREPARED = PreparedTable.__new__(PreparedTable)     # every slot -1: reads stay pandas


cdef void renew() noexcept:
    """A new model of this package: every frame is prepared again at its first read in it."""
    global _generation
    _generation += 1


def _forget(key, ref):
    entry = _prepared.get(key)
    if entry is not None and entry[0] is ref:       # not the entry of a later frame at the same address
        _prepared.pop(key, None)


cdef object _prepare_new(object frame):
    """Prepare ``frame`` as it is now and make it the table its reads find (None: they stay pandas)."""
    key = id(frame)
    signatures = model_signatures()
    table = prepare(frame)
    if table is not None:
        _decide(table, signatures)
    _prepared[key] = (weakref.ref(frame, lambda ref, key=key: _forget(key, ref)), table, _generation)
    return table


cdef object prepared(object frame):
    """The PreparedTable of ``frame`` (None when its reads must stay pandas): the one prepared at an earlier read
    in this model's generation, else prepared now."""
    entry = _prepared.get(id(frame))
    if entry is not None and <Py_ssize_t>entry[2] == _generation and entry[0]() is frame:
        return entry[1]
    return _prepare_new(frame)


cdef object bind_ref(object frame):
    """What an item holds for one frame receiver: its PreparedTable, or a table whose every slot is -1 when the
    frame is not prepared (its reads then run on pandas). Taken once per item and receiver."""
    table = prepared(frame) if frame is not None else None
    return _NOT_PREPARED if table is None else table


cdef object _signatures = None


cdef object model_signatures():
    """The read signatures of the compiled model this copy belongs to (``_mx_table_signatures.SIGNATURES``, written
    at build), loaded once."""
    global _signatures
    if _signatures is None:
        import importlib
        package = __name__.rpartition('.')[0]
        try:        # the runtime used on its own (no model package) has no compiled reads
            _signatures = (tuple(importlib.import_module(package + '._mx_table_signatures').SIGNATURES)
                           if package else ())
        except ImportError:
            _signatures = ()
    return _signatures


cdef void _decide(PreparedTable table, object signatures):
    """Decide, once per prepared table, the slot of every read signature of the model:
    (levels, columns, label, kind) ->
    - ``levels``: one letter per index level, ``i`` for a level a read takes integer keys on (it must still be an
      int level), ``o`` for any other; the number of levels must be the same;
    - ``columns``: for a row read by field, every (label, kind) of the traced frame (the row's dtype depends on all
      of them); else None;
    - ``label``, ``kind``: the column read and its block kind (0 float64, 1 int64, 2 object; -1: the index; -4:
      a field chosen at run time, ``columns`` given; -2: the extremes of a non-empty index; -6: the extremes of an
      int level, ``label`` its (name or None, position); -3: an ordered lookup within prefixes of ``label`` levels).
    The slot is one plus the column's slot in its block (1 for a read of the index), or -1."""
    cdef int level
    cdef Py_ssize_t position
    slots = []
    for levels, columns, label, kind in signatures:
        result = len(levels) == table.nlevels
        if result:
            for level in range(table.nlevels):
                if levels[level] == 'i' and table.level_kind[level] == LEVEL_OBJECT:
                    result = False
                    break
        if result and columns is not None:
            if len(columns) != len(table.column_positions):
                result = False
            elif not table.row_object and any(expected == COLUMN_OBJECT for _, expected in columns):
                result = False      # a field kept as an object, but pandas widens this row: not the stored value
            else:
                for name, expected in columns:
                    position = table.column_positions.get(name, -1)
                    if position < 0 or table.column_kind[position] != expected:
                        result = False
                        break
        value = -1
        if result:
            if kind < 0:
                value = _index_slot(table, label, kind)
            else:
                position = table.column_positions.get(label, -1)
                if position >= 0 and table.column_kind[position] == kind:
                    value = table.column_slot[position] + 1
        slots.append(value)
    array = np.asarray(slots + [-1], dtype=np.intp)
    table._slot_array = array
    table.slots = <const Py_ssize_t*><size_t>array.ctypes.data
    table.n_slots = len(slots)


cdef Py_ssize_t _index_slot(PreparedTable table, object label, int kind):
    """The slot of a read of the index itself (kind < 0): 1 when this table answers it, else -1."""
    if kind == -2:      # the extremes of a non-empty index
        return 1 if table.nrows > 0 else -1
    if kind == -6:      # the extremes of level label[1], named label[0] (None: any name), an int level, not empty
        return 1 if (label[1] < table.nlevels and (label[0] is None or table.level_names[label[1]] == label[0])
                     and table.level_kind[label[1]] != LEVEL_OBJECT and table.nrows > 0) else -1
    if kind == -3:      # an ordered lookup within prefixes of ``label`` levels (None: the whole table)
        return 1 if table.ordered_within[label or 0] else -1
    return 1 if kind in (-1, -4) else -1


def native_slots(frame):
    """How many of the model's read signatures ``frame`` serves natively (0 when its reads run on pandas)."""
    cdef PreparedTable table = prepared(frame)
    if table is None:
        return 0
    return sum(1 for i in range(table.n_slots) if table.slots[i] > 0)


cdef Py_ssize_t hold_code(object table, int level, object value) except -2:
    """One plus the code of ``value`` at ``level`` of a held table (resolved once per item); -1 when the table is
    not prepared or the value is a label it does not have; -2 when the value is not one label (a list, a mask, a
    slice, an unhashable value). A read takes it only when it is positive, so pandas answers or raises otherwise."""
    cdef Py_ssize_t code
    if selects_rows(value):
        return -2
    if table is None or table is _NOT_PREPARED:
        return -1
    code = (<PreparedTable>table).code_obj(level, value)
    return code + 1 if code >= 0 else -1


cdef Py_ssize_t hold_row(object table, object key) except -2:
    """One plus the stored row of ``key`` in a held table, for the fields of a row whose key is fixed per item; -1
    when the table is not prepared or has no such row; -2 when a part of the key is not one label. A read takes
    it only when it is positive, so pandas answers or raises otherwise."""
    cdef Py_ssize_t row
    if selects_rows(key) and not (type(key) is tuple and not any(selects_rows(part) for part in key)):
        return -2
    if table is None or table is _NOT_PREPARED:
        return -1
    row = (<PreparedTable>table).row_key(key)
    return row + 1 if row >= 0 else -1


cdef bint selects_rows(object key):
    """A key part pandas does not look up as one label: it selects rows (a list, an array, a mask, a slice, an
    index, a table), is a key of several parts, or cannot be a label at all (unhashable)."""
    if type(key) is int or type(key) is float or type(key) is str:      # the common labels first
        return False
    if isinstance(key, (list, tuple, slice, set, dict, np.ndarray, pd.Series, pd.Index, pd.DataFrame)):
        return True
    try:
        hash(key)
    except TypeError:
        return True
    return False


cdef int missing_codes(PreparedRef ref, const Py_ssize_t* codes, int count) except -1:
    """KeyError for a key given as codes: the labels of the parts found, None for a part absent from its level
    (the type is pandas'; the key shown lists the parts found)."""
    cdef PreparedTable table = <PreparedTable>ref
    parts = tuple(table.level_labels[i][codes[i]] if i < table.nlevels and 0 <= codes[i] < table.level_size[i]
                  else None for i in range(count))
    raise KeyError(parts if count > 1 else parts[0])


cdef object object_field(PreparedRef ref, Py_ssize_t slot, Py_ssize_t row):
    return (<PreparedTable>ref).object_at(row - 1, slot - 1)


# The numpy scalar pandas returns for a cell value, for a read whose value meets numpy's own rules (``round`` of it,
# a method of it, a division by zero, an integer power): such a read is boxed, the others stay C numbers.
cdef object _float64 = np.float64
cdef object _int64 = np.int64


cdef object f64_box(double value):
    """``numpy.float64``, as pandas returns a value of a float64 column."""
    return _float64(value)


cdef object i64_box(PreparedRef ref, long long value, bint field):
    """``numpy.int64`` as pandas returns a value of an int64 column; a field of a row pandas widens to float64
    (``field``, the table's rows of number columns) is ``numpy.float64``."""
    if field and (<PreparedTable>ref).row_float:
        return _float64(value)
    return _int64(value)


cdef double field_f64(PreparedRef ref, Py_ssize_t row, object label) except? -1.0:
    """``float(T.loc[key, label])`` at a stored row (``row`` is one plus the row), the column chosen at run time;
    KeyError as pandas for a label that is not a column."""
    cdef Py_ssize_t position = (<PreparedTable>ref).name_position(label)
    if position < 0:
        raise KeyError(label)
    cdef int kind = (<PreparedTable>ref).column_kind[position]
    cdef Py_ssize_t slot = (<PreparedTable>ref).column_slot[position]
    if kind == COLUMN_F64:
        return (<PreparedTable>ref).f64_at(row - 1, slot)
    if kind == COLUMN_I64:
        return <double>(<PreparedTable>ref).i64_at(row - 1, slot)
    return float((<PreparedTable>ref).object_at(row - 1, slot))


cdef bint _is_number(object x):
    """An int or float (numpy's too) that is not a bool, not NaN, and that a double holds exactly."""
    if type(x) is float or isinstance(x, np.floating):
        return x == x
    if (type(x) is int or isinstance(x, np.integer)) and not isinstance(x, (bool, np.bool_)):
        return -9007199254740992 <= x <= 9007199254740992
    return False


cdef object max_obj(PreparedRef ref):
    return (<PreparedTable>ref).max_label


cdef object min_obj(PreparedRef ref):
    return (<PreparedTable>ref).min_label


cdef Py_ssize_t view_start(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2, Py_ssize_t c3,
                           Py_ssize_t c4) noexcept:
    """The first stored row of the prefix given as ``count`` level codes, or -1 when it has no row."""
    cdef Py_ssize_t[5] codes
    cdef Py_ssize_t start, stop
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4
    if not (<PreparedTable>ref).prefix_range(codes, count, &start, &stop):
        return -1
    return start


cdef Py_ssize_t view_stop(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2, Py_ssize_t c3,
                          Py_ssize_t c4) noexcept:
    """One past the last stored row of the prefix, or -1 when it has no row."""
    cdef Py_ssize_t[5] codes
    cdef Py_ssize_t start, stop
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4
    if not (<PreparedTable>ref).prefix_range(codes, count, &start, &stop):
        return -1
    return stop
