# Bind-time preparation of a pandas table for the reads the compiler makes fast (design: toolkit p2/tables/DESIGN.md).
# A PreparedTable is built once from the user's DataFrame or Series and never changes afterwards. The key lookups
# and value reads below are inline, so the C compiler inlines them into the compiled model; they take no
# reference on a shared object (an object-column read returns a new reference to the value it reads).

cdef enum:
    LEVEL_DENSE = 0      # consecutive int labels: code = key - first
    LEVEL_SORTED = 1     # other int labels: binary search over the sorted labels
    LEVEL_OBJECT = 2     # any other labels (text, enum members, floats): dict label -> code

cdef enum:
    COLUMN_F64 = 0
    COLUMN_I64 = 1
    COLUMN_OBJECT = 2

cdef class PreparedTable:
    cdef readonly object source            # the user's frame or Series (the object the formula sees)
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
    cdef bint prefix_range(self, const Py_ssize_t* codes, int count, Py_ssize_t* start, Py_ssize_t* stop) noexcept nogil
    cdef Py_ssize_t asof_position(self, double x) noexcept nogil
    cdef object object_at(self, Py_ssize_t row, Py_ssize_t slot)
    cpdef Py_ssize_t column(self, object label) except -2
