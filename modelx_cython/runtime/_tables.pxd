# Bind-time preparation of a pandas table for the reads the compiler makes fast (design: toolkit p2/tables/DESIGN.md).
# A PreparedTable is built once from the user's DataFrame or Series and never changes afterwards. The key lookups
# and value reads below are inline, so the C compiler inlines them into the compiled model; they take no
# reference on a shared object (an object-column read returns a new reference to the value it reads).

from cpython.ref cimport PyObject

cdef enum:
    LEVEL_DENSE = 0      # consecutive int labels: code = key - first
    LEVEL_SORTED = 1     # other int labels: binary search over the sorted labels
    LEVEL_OBJECT = 2     # any other labels (text, enum members, floats): dict label -> code

cdef enum:
    COLUMN_F64 = 0
    COLUMN_I64 = 1
    COLUMN_OBJECT = 2

cdef class PreparedTable:
    cdef object _source                    # weak reference to the user's frame or Series (``source``): a prepared
                                           # copy never keeps a replaced frame alive
    cdef readonly bint is_series
    cdef readonly Py_ssize_t nrows, nlevels
    cdef readonly tuple level_labels       # per level: its labels in code order (a list)
    cdef tuple level_codes                 # per level: dict label -> code for LEVEL_OBJECT, else None
    cdef int[8] level_kind
    cdef long long[8] level_first          # LEVEL_DENSE
    cdef Py_ssize_t[8] level_size          # distinct labels
    cdef Py_ssize_t[8] level_offset        # LEVEL_SORTED: start of the level's labels in sorted_labels
    cdef const long long* sorted_labels    # raw pointers into the arrays held by `_arrays` (no bounds checks)
    cdef long long[8] radix                # combined code = sum(code[i] * radix[i]); radix[nlevels - 1] = 1
    cdef const long long* combined         # combined code of every stored row, ascending (rows are stored sorted)
    cdef const Py_ssize_t* grid            # combined code -> stored row, or -1 (when the product is small)
    cdef bint has_grid
    cdef readonly dict column_positions    # column label -> position
    cdef const int[::1] column_kind        # COLUMN_*
    cdef const Py_ssize_t[::1] column_slot # position within its block
    cdef const double* f64                 # stored row x float64 column, row-major
    cdef Py_ssize_t f64_width
    cdef const long long* i64              # stored row x int64 column, row-major
    cdef Py_ssize_t i64_width
    cdef tuple _arrays                     # the numpy arrays the pointers point into (kept alive, never written)
    cdef tuple objects                     # per object column: tuple of the values pandas returns, by stored row
    cdef readonly object max_label, min_label   # index.max(), index.min() as pandas returns them
    cdef const double[::1] asof_keys       # sorted labels of a flat numeric index (for asof)
    cdef readonly object asof_labels       # the labels themselves, by position, as pandas returns them
    cdef object _slot_array                # T-4.2: per read signature of the model, one plus the slot of its column
    cdef const Py_ssize_t* slots           # (1 for a read of the index), or -1 when this table is not of the kinds
    cdef Py_ssize_t n_slots                # the read was compiled for (decided once, when the table is prepared)

    cdef inline Py_ssize_t code_ll(self, int level, long long key) noexcept nogil:
        """The code of an int key part at ``level``, or -1 (absent, or an object level)."""
        cdef Py_ssize_t lo, hi, mid, end
        cdef int kind = self.level_kind[level]
        if kind == LEVEL_DENSE:
            if key < self.level_first[level] or key - self.level_first[level] >= self.level_size[level]:
                return -1
            return <Py_ssize_t>(key - self.level_first[level])
        if kind == LEVEL_SORTED:
            lo = self.level_offset[level]
            end = hi = lo + self.level_size[level]
            while lo < hi:
                mid = (lo + hi) >> 1
                if self.sorted_labels[mid] < key:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < end and self.sorted_labels[lo] == key:
                return lo - self.level_offset[level]
        return -1

    cdef inline Py_ssize_t row_codes(self, const Py_ssize_t* codes, int count) noexcept nogil:
        """The stored row of a full key given as level codes, or -1."""
        cdef long long combined = 0
        cdef int i
        cdef Py_ssize_t lo, hi, mid
        if count != self.nlevels:
            return -1
        for i in range(count):
            if codes[i] < 0 or codes[i] >= self.level_size[i]:
                return -1
            combined += codes[i] * self.radix[i]
        if count == 1:
            return <Py_ssize_t>combined          # one level: rows are stored in code order, one per code
        if self.has_grid:
            return self.grid[combined]
        lo, hi = 0, self.nrows
        while lo < hi:
            mid = (lo + hi) >> 1
            if self.combined[mid] < combined:
                lo = mid + 1
            else:
                hi = mid
        if lo < self.nrows and self.combined[lo] == combined:
            return lo
        return -1

    cdef inline Py_ssize_t row_ll(self, long long key) noexcept nogil:
        """The stored row of a one-level int key, or -1."""
        if self.nlevels != 1:
            return -1
        return self.code_ll(0, key)

    cdef inline double f64_at(self, Py_ssize_t row, Py_ssize_t slot) noexcept nogil:
        return self.f64[row * self.f64_width + slot]

    cdef inline long long i64_at(self, Py_ssize_t row, Py_ssize_t slot) noexcept nogil:
        return self.i64[row * self.i64_width + slot]

    cdef Py_ssize_t code_obj(self, int level, object key) except -2
    cdef Py_ssize_t row_key(self, object key) except -2
    cdef bint prefix_range(self, const Py_ssize_t* codes, int count, Py_ssize_t* start, Py_ssize_t* stop) noexcept nogil
    cdef Py_ssize_t asof_position(self, double x) noexcept nogil
    cdef object object_at(self, Py_ssize_t row, Py_ssize_t slot)
    cpdef Py_ssize_t column(self, object label) except -2


# ---- reads of compiled code (T-3) ---------------------------------------------------------------------
# A PreparedRef is a borrowed pointer to a PreparedTable held by the reading item (or a local of the formula):
# the holder keeps it alive, so a read through it takes no reference. ``slot`` is what ``hold`` set: one plus
# the column's slot. Key parts are a C long long (an integer part on an integer level: no Python object) or the
# Python value (looked up as pandas does).
ctypedef PyObject* PreparedRef
ctypedef long long IntKey

ctypedef fused KeyA:
    IntKey
    object

ctypedef fused KeyB:
    IntKey
    object

ctypedef fused KeyC:
    IntKey
    object

cdef object prepared(object frame)
cdef int missing(object key) except -1
cdef object bind_ref(object frame)
cdef object bind_ref_checked(object frame)
cdef Py_ssize_t hold_code(object table, int level, object value) except -2
cdef Py_ssize_t asof_row(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2,
                         Py_ssize_t c3, Py_ssize_t c4, double x) noexcept
cdef Py_ssize_t hold_row(object table, object key, bint assigned) except -2
cdef int missing_codes(PreparedRef ref, const Py_ssize_t* codes, int count) except -1
cdef object object_field(PreparedRef ref, Py_ssize_t slot, Py_ssize_t row)


cdef inline Py_ssize_t _code(PreparedTable table, int level, KeyA part) except -2:
    if KeyA is IntKey:
        return table.code_ll(level, part)
    else:
        return table.code_obj(level, part)


cdef inline Py_ssize_t row1(PreparedTable table, KeyA a) except -2:
    if KeyA is IntKey:
        return table.row_ll(a)
    else:
        return table.row_key(a)


cdef inline Py_ssize_t row2(PreparedTable table, KeyA a, KeyB b) except -2:
    cdef Py_ssize_t[2] codes
    codes[0] = _code(table, 0, a)
    codes[1] = _code(table, 1, b)
    return table.row_codes(codes, 2)


cdef inline Py_ssize_t row3(PreparedTable table, KeyA a, KeyB b, KeyC c) except -2:
    cdef Py_ssize_t[3] codes
    codes[0] = _code(table, 0, a)
    codes[1] = _code(table, 1, b)
    codes[2] = _code(table, 2, c)
    return table.row_codes(codes, 3)


cdef int missing2(object a, object b) except -1
cdef int missing3(object a, object b, object c) except -1


cdef inline double read_f64(PreparedRef ref, Py_ssize_t slot, KeyA a) except? -1.0:
    """``frame.loc[key, column]`` of a float64 column through a held table; KeyError as pandas."""
    cdef Py_ssize_t row = row1(<PreparedTable>ref, a)
    if row < 0:
        missing(a)
    return (<PreparedTable>ref).f64_at(row, slot - 1)


cdef inline double read_f64_2(PreparedRef ref, Py_ssize_t slot, KeyA a, KeyB b) except? -1.0:
    cdef Py_ssize_t row = row2(<PreparedTable>ref, a, b)
    if row < 0:
        missing2(a, b)
    return (<PreparedTable>ref).f64_at(row, slot - 1)


cdef inline double read_f64_3(PreparedRef ref, Py_ssize_t slot, KeyA a, KeyB b, KeyC c) except? -1.0:
    cdef Py_ssize_t row = row3(<PreparedTable>ref, a, b, c)
    if row < 0:
        missing3(a, b, c)
    return (<PreparedTable>ref).f64_at(row, slot - 1)


cdef inline long long read_i64(PreparedRef ref, Py_ssize_t slot, KeyA a) except? -1:
    """``frame.loc[key, column]`` of an int64 column through a held table; KeyError as pandas."""
    cdef Py_ssize_t row = row1(<PreparedTable>ref, a)
    if row < 0:
        missing(a)
    return (<PreparedTable>ref).i64_at(row, slot - 1)


cdef inline long long read_i64_2(PreparedRef ref, Py_ssize_t slot, KeyA a, KeyB b) except? -1:
    cdef Py_ssize_t row = row2(<PreparedTable>ref, a, b)
    if row < 0:
        missing2(a, b)
    return (<PreparedTable>ref).i64_at(row, slot - 1)


cdef inline long long read_i64_3(PreparedRef ref, Py_ssize_t slot, KeyA a, KeyB b, KeyC c) except? -1:
    cdef Py_ssize_t row = row3(<PreparedTable>ref, a, b, c)
    if row < 0:
        missing3(a, b, c)
    return (<PreparedTable>ref).i64_at(row, slot - 1)


cdef inline bint contains(PreparedRef ref, KeyA a) except -1:
    """``key in frame.index`` through a held table."""
    return row1(<PreparedTable>ref, a) >= 0


cdef inline bint contains_2(PreparedRef ref, KeyA a, KeyB b) except -1:
    return row2(<PreparedTable>ref, a, b) >= 0


cdef inline bint contains_3(PreparedRef ref, KeyA a, KeyB b, KeyC c) except -1:
    return row3(<PreparedTable>ref, a, b, c) >= 0


# ---- T-4: reads by level codes (held per item, or computed per read) and held rows ---------------------------
# A key of up to MAX_CODES parts is passed as one level code per part: ``held - 1`` for a part fixed per item
# (resolved once by ``hold_code``), ``lcode(ref, level, k)`` for a C integer part, ``ocode(ref, level, k)`` for
# any other part. No fused types: one function per part count.

cdef inline Py_ssize_t lcode(PreparedRef ref, int level, long long key) noexcept nogil:
    return (<PreparedTable>ref).code_ll(level, key)


cdef inline Py_ssize_t ocode(PreparedRef ref, int level, object key) except -2:
    return (<PreparedTable>ref).code_obj(level, key)


cdef inline Py_ssize_t _row_c(PreparedRef ref, const Py_ssize_t* codes, int count) except -2:
    cdef Py_ssize_t row = (<PreparedTable>ref).row_codes(codes, count)
    if row < 0:
        missing_codes(ref, codes, count)
    return row


cdef inline double f64_c(PreparedRef ref, Py_ssize_t slot, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2,
                         Py_ssize_t c3, Py_ssize_t c4, Py_ssize_t c5) except? -1.0:
    """``frame.loc[key, column]`` of a float64 column, the key given as level codes (unused codes are 0)."""
    cdef Py_ssize_t[6] codes
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4; codes[5] = c5
    return (<PreparedTable>ref).f64_at(_row_c(ref, codes, count), slot - 1)


cdef inline long long i64_c(PreparedRef ref, Py_ssize_t slot, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2,
                            Py_ssize_t c3, Py_ssize_t c4, Py_ssize_t c5) except? -1:
    cdef Py_ssize_t[6] codes
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4; codes[5] = c5
    return (<PreparedTable>ref).i64_at(_row_c(ref, codes, count), slot - 1)


cdef inline bint contains_c(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2,
                            Py_ssize_t c3, Py_ssize_t c4, Py_ssize_t c5) noexcept nogil:
    cdef Py_ssize_t[6] codes
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4; codes[5] = c5
    return (<PreparedTable>ref).row_codes(codes, count) >= 0


# a field of a row held per item (``row - 1`` from ``hold_row``)
cdef inline double f64_row(PreparedRef ref, Py_ssize_t slot, Py_ssize_t row) noexcept nogil:
    return (<PreparedTable>ref).f64_at(row - 1, slot - 1)


cdef inline long long i64_row(PreparedRef ref, Py_ssize_t slot, Py_ssize_t row) noexcept nogil:
    return (<PreparedTable>ref).i64_at(row - 1, slot - 1)


# T-4: extremes of a one-level int index (``T.index.max()`` / ``.min()``), as C integers (no reference taken)
cdef inline long long max_ll(PreparedRef ref) noexcept:
    if (<PreparedTable>ref).level_kind[0] == LEVEL_DENSE:
        return (<PreparedTable>ref).level_first[0] + (<PreparedTable>ref).level_size[0] - 1
    return (<PreparedTable>ref).sorted_labels[(<PreparedTable>ref).level_offset[0] + (<PreparedTable>ref).level_size[0] - 1]


cdef inline long long min_ll(PreparedRef ref) noexcept:
    if (<PreparedTable>ref).level_kind[0] == LEVEL_DENSE:
        return (<PreparedTable>ref).level_first[0]
    return (<PreparedTable>ref).sorted_labels[(<PreparedTable>ref).level_offset[0]]


# T-4.2: the slot of a read, through the table its item holds: one C load (a read stays pandas when it is <= 0)
cdef inline Py_ssize_t slot(PreparedRef ref, int signature) noexcept:
    if signature < (<PreparedTable>ref).n_slots:
        return (<PreparedTable>ref).slots[signature]
    return -1
