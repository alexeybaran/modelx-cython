# cython: freethreading_compatible=True, boundscheck=False, wraparound=False, initializedcheck=False
"""Bind-time preparation of a pandas table (T-2; design toolkit p2/tables/DESIGN.md).

``prepare(value)`` turns the user's DataFrame or Series into a ``PreparedTable`` the compiled reads use: the rows
stored sorted by their level codes, one ascending array of combined codes (and a dense grid when small) for keyed
lookups, typed column blocks copied from the frame, and the index extremes and asof keys. The user's object is
never changed or replaced: it stays what every formula sees (``source``). ``prepare`` returns None, and
``unprepared_reason`` says why, for a table whose reads must stay pandas (a non-unique index, NaN labels, labels
that compare equal although pandas keeps them apart, duplicate column labels, too many key combinations).

A lookup equals ``value.loc[key, column]`` on the tables ``prepare`` accepts: the tests compare them on random
keys, misses included (``tests/test_prepared_tables.py``). The key lookups and value reads compiled code uses are
inline methods of the .pxd.
"""
from libc.math cimport isnan

import threading
import weakref

import numpy as np
import pandas as pd

cdef Py_ssize_t _GRID_LIMIT = 65536
cdef long long _COMBINED_LIMIT = 1LL << 62
cdef int _MAX_LEVELS = 8


cdef inline bint _plain_int(object label):
    # an int label (including IntEnum members and numpy integers), not a bool
    return isinstance(label, (int, np.integer)) and not isinstance(label, (bool, np.bool_))


cdef class PreparedTable:
    """A pandas table prepared for keyed reads. Immutable once built."""

    @property
    def source(self):
        """The user's frame or Series this table was prepared from (None once it is gone)."""
        return self._source()

    def __reduce__(self):
        return (prepare, (self.source,))

    cdef Py_ssize_t code_obj(self, int level, object key) except -2:
        """The code of any key part at ``level``, or -1 when absent."""
        cdef object code
        codes = self.level_codes[level]
        if codes is None:
            if isinstance(key, (float, np.floating)) and float(key).is_integer():
                key = int(key)          # pandas finds an integral float key on an int index
            if not _plain_int(key):
                return -1
            return self.code_ll(level, <long long>key)
        code = (<dict>codes).get(key)
        return -1 if code is None else <Py_ssize_t>code

    cdef Py_ssize_t row_key(self, object key) except -2:
        """The stored row of a full key given as Python objects (a scalar, or a tuple), or -1."""
        cdef Py_ssize_t[8] buffer
        cdef Py_ssize_t code
        cdef int level
        if self.nlevels == 1:
            if isinstance(key, tuple):
                return -1
            code = self.code_obj(0, key)
            return code
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

    cdef Py_ssize_t asof_position(self, double x) noexcept nogil:
        """Position in asof_labels of the last label <= x, or -1 (x below the first label, or NaN)."""
        cdef Py_ssize_t lo = 0, hi = self.asof_keys.shape[0], mid
        if isnan(x):
            return -1
        while lo < hi:
            mid = (lo + hi) >> 1
            if self.asof_keys[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    cdef object object_at(self, Py_ssize_t row, Py_ssize_t slot):
        return (<tuple>self.objects[slot])[row]

    cpdef Py_ssize_t column(self, object label) except -2:
        """The position of column ``label`` (None for a Series); KeyError as pandas raises it."""
        position = self.column_positions.get(label)
        if position is None:
            raise KeyError(label)
        return position

    # ------------------------------------------------------------ Python access (tests, fallbacks)
    def key_codes(self, key):
        """Level codes of a full key (a scalar for one level, a tuple otherwise); None when a part is absent."""
        parts = key if self.nlevels > 1 else (key,)
        if not isinstance(parts, tuple) or len(parts) != self.nlevels:
            return None
        codes = []
        for level, part in enumerate(parts):
            code = self.code_obj(level, part)
            if code < 0:
                return None
            codes.append(code)
        return codes

    def row(self, key):
        """The stored row of ``key``, or -1."""
        cdef Py_ssize_t[8] buffer
        codes = self.key_codes(key)
        if codes is None:
            return -1
        for i, code in enumerate(codes):
            buffer[i] = code
        return self.row_codes(buffer, len(codes))

    def value(self, key, column=None):
        """``source.loc[key, column]`` (``source.loc[key]`` for a Series), as pandas returns it."""
        cdef Py_ssize_t row = self.row(key)
        cdef Py_ssize_t position
        if row < 0:
            raise KeyError(key)
        position = self.column(None if self.is_series else column)
        kind = self.column_kind[position]
        slot = self.column_slot[position]
        if kind == COLUMN_F64:
            return np.float64(self.f64_at(row, slot))
        if kind == COLUMN_I64:
            return np.int64(self.i64_at(row, slot))
        return self.object_at(row, slot)

    def contains(self, key):
        """``key in source.index``."""
        return self.row(key) >= 0

    def asof(self, x):
        """``source.index.asof(x)`` for a flat numeric index."""
        position = self.asof_position(float(x))
        return self.asof_labels[position] if position >= 0 else np.nan


def unprepared_reason(value):
    """Why ``prepare(value)`` returns None (None when it does not)."""
    try:
        _build(value)
    except ValueError as error:
        return str(error)
    return None


def prepare(value):
    """The PreparedTable of a DataFrame or Series, or None when its reads must stay pandas."""
    try:
        return _build(value)
    except ValueError:
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


cdef void _point(PreparedTable table, sorted_array, combined, grid, f64, i64):
    """Point the table's raw pointers into its arrays and keep the arrays alive with it."""
    cdef const long long[::1] s = sorted_array
    cdef const long long[::1] c = combined
    cdef const Py_ssize_t[::1] g = grid
    cdef const double[:, ::1] f = f64
    cdef const long long[:, ::1] i = i64
    table._arrays = (sorted_array, combined, grid, f64, i64)
    for array in table._arrays:
        array.flags.writeable = False
    table.sorted_labels = &s[0]
    table.combined = &c[0]
    table.grid = &g[0]
    table.f64 = &f[0, 0]
    table.i64 = &i[0, 0]


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
    sorted_array = np.ascontiguousarray(sorted_labels if sorted_labels else [0], dtype=np.int64)
    cdef const long long[::1] sorted_view = sorted_array
    table._arrays = (sorted_array,)              # alive before code_ll reads it below
    table.sorted_labels = &sorted_view[0]
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
            objects.append(tuple(column.iat[int(row)] for row in order))
    table.column_positions = {label: position for position, label in enumerate(labels)}
    table.column_kind = np.asarray(kinds, dtype=np.intc) if kinds else np.zeros(1, dtype=np.intc)
    table.column_slot = np.asarray(slots, dtype=np.intp) if slots else np.zeros(1, dtype=np.intp)
    f64 = np.ascontiguousarray(np.column_stack(f64_cols) if f64_cols else np.zeros((max(1, n), 1)), dtype=np.float64)
    i64 = np.ascontiguousarray(np.column_stack(i64_cols) if i64_cols else np.zeros((max(1, n), 1), dtype=np.int64),
                               dtype=np.int64)
    table.f64_width, table.i64_width = f64.shape[1], i64.shape[1]
    _point(table, sorted_array, combined, grid, f64, i64)
    table.objects = tuple(objects)
    # index extremes and asof keys, as pandas gives them
    table.max_label = index.max() if n else None
    table.min_label = index.min() if n else None
    if table.nlevels == 1 and n and index.dtype.kind in 'iuf':
        sorted_index = index.sort_values()
        table.asof_keys = np.ascontiguousarray(sorted_index.to_numpy(dtype=np.float64))
        table.asof_labels = [sorted_index[i] for i in range(n)]
    else:
        table.asof_keys = np.zeros(0)
        table.asof_labels = []
    return table



# ---- binding (T-3) ----------------------------------------------------------------------------------
# One PreparedTable per frame object, prepared on first use and dropped when the frame dies. A table
# assigned to a Data cell, or recomputed after a clear, is a new object: it is prepared afresh, and no
# prepared copy of the old frame is ever used for it.
cdef dict _prepared = {}               # id(frame) -> (weak reference to the frame, PreparedTable or None)
_lock = threading.RLock()


def _forget(key):
    with _lock:
        _prepared.pop(key, None)


def forget():
    """Drop every prepared copy (a Space was cleared: a frame edited in place is prepared again)."""
    with _lock:
        _prepared.clear()


try:
    from modelx_cython.runtime.table_cells import register_prepared as _register
except ImportError:         # the runtime used on its own (tests)
    pass
else:
    _register(forget)


cdef object prepared(object frame):
    """The PreparedTable of ``frame`` (None when its reads must stay pandas), prepared once per object."""
    key = id(frame)
    with _lock:
        entry = _prepared.get(key)
    if entry is not None and entry[0]() is frame:
        return entry[1]
    table = prepare(frame)
    ref = weakref.ref(frame, lambda _, key=key: _forget(key))
    with _lock:
        _prepared[key] = (ref, table)
    return table


cdef int missing(object key) except -1:
    raise KeyError(key)


cdef object hold(object frame, object label, int kind, Py_ssize_t* slot):
    """Resolve a hold: return the PreparedTable of ``frame`` (the holder keeps it) and set ``*slot`` to one plus
    the slot of column ``label`` within its ``kind`` block (one for a hold of index membership, ``kind`` < 0).
    ``*slot`` = -1 (and None returned) when the read stays pandas: the frame is not prepared, the column is
    absent, or its kind is another. 0 is left to mean unresolved: a new item's field starts at 0."""
    cdef PreparedTable table = prepared(frame)
    cdef Py_ssize_t position
    slot[0] = -1
    if table is None:
        return None
    if kind < 0:
        slot[0] = 1
        return table
    position = table.column_positions.get(label, -1)
    if position < 0 or table.column_kind[position] != kind:
        return None
    slot[0] = table.column_slot[position] + 1
    return table


cdef int missing2(object a, object b) except -1:
    raise KeyError((a, b))


cdef int missing3(object a, object b, object c) except -1:
    raise KeyError((a, b, c))
