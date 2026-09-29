"""Immutable prepared lookup curves. The SAME source is used by Python and Cython.

Pandas stays in preparation; queries never use pandas. An accompanying .pxd gives
Cython native buffer and method types. Keys and values are finite float64; integer
keys must be exactly representable. Duplicate keys are errors, not silently merged.
"""
import math
import numpy as np
import cython
if cython.compiled:
    from cython.cimports.libc.math import isfinite, isnan, floor, pow as real_pow, NAN as _MISSING
else:
    from math import isfinite, isnan, floor, pow as real_pow, nan as _MISSING


class IndexedInt1:
    """Owned int64 snapshot with NumPy's ordinary one-axis index semantics."""
    def __init__(self, values):
        source = np.asarray(values)
        if source.ndim != 1 or source.dtype != np.dtype('int64'):
            raise TypeError('IndexedInt1 requires a one-dimensional int64 array')
        self._v = np.array(source, order='C', copy=True)
        self._n = self._v.shape[0]

    @property
    def arrays(self):
        return (np.array(self._v, copy=True),)

    def __reduce__(self):
        return type(self), self.arrays

    @cython.locals(i=cython.Py_ssize_t)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def get(self, i):
        if i < 0:
            i += self._n
        if i < 0 or i >= self._n:
            raise IndexError('array index out of bounds')
        return self._v[i]


class IndexedFloat1:
    """Owned float64 snapshot with NumPy's ordinary one-axis index semantics."""
    def __init__(self, values):
        source = np.asarray(values)
        if source.ndim != 1 or source.dtype != np.dtype('float64'):
            raise TypeError('IndexedFloat1 requires a one-dimensional float64 array')
        self._v = np.array(source, order='C', copy=True)
        self._n = self._v.shape[0]

    @property
    def arrays(self):
        return (np.array(self._v, copy=True),)

    def __reduce__(self):
        return type(self), self.arrays

    @cython.locals(i=cython.Py_ssize_t)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def get(self, i):
        if i < 0:
            i += self._n
        if i < 0 or i >= self._n:
            raise IndexError('array index out of bounds')
        return self._v[i]


class IndexedInt2:
    """Owned int64 snapshot with NumPy's two-axis index semantics."""
    def __init__(self, values):
        source = np.asarray(values)
        if source.ndim != 2 or source.dtype != np.dtype('int64'):
            raise TypeError('IndexedInt2 requires a two-dimensional int64 array')
        self._v = np.array(source, order='C', copy=True)
        self._n = self._v.shape[0]
        self._m = self._v.shape[1]

    @property
    def arrays(self):
        return (np.array(self._v, copy=True),)

    def __reduce__(self):
        return type(self), self.arrays

    @cython.locals(i=cython.Py_ssize_t, j=cython.Py_ssize_t)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def get(self, i, j):
        if i < 0:
            i += self._n
        if j < 0:
            j += self._m
        if i < 0 or i >= self._n or j < 0 or j >= self._m:
            raise IndexError('array index out of bounds')
        return self._v[i, j]


class IndexedFloat2:
    """Owned float64 snapshot with NumPy's two-axis index semantics."""
    def __init__(self, values):
        source = np.asarray(values)
        if source.ndim != 2 or source.dtype != np.dtype('float64'):
            raise TypeError('IndexedFloat2 requires a two-dimensional float64 array')
        self._v = np.array(source, order='C', copy=True)
        self._n = self._v.shape[0]
        self._m = self._v.shape[1]

    @property
    def arrays(self):
        return (np.array(self._v, copy=True),)

    def __reduce__(self):
        return type(self), self.arrays

    @cython.locals(i=cython.Py_ssize_t, j=cython.Py_ssize_t)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def get(self, i, j):
        if i < 0:
            i += self._n
        if j < 0:
            j += self._m
        if i < 0 or i >= self._n or j < 0 or j >= self._m:
            raise IndexError('array index out of bounds')
        return self._v[i, j]


class Curve:
    def __init__(self, keys, values):
        x = np.array(keys, dtype=np.float64, order='C', copy=True)
        y = np.array(values, dtype=np.float64, order='C', copy=True)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError('Curve requires equal-length one-dimensional keys and values')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Curve requires finite keys and values')
        order = np.argsort(x, kind='stable')
        x, y = x[order], y[order]
        if len(x) > 1 and np.any(x[1:] <= x[:-1]):
            raise ValueError('Curve keys must be unique')
        # Own and freeze inputs: editing the caller\'s DataFrame cannot stale caches.
        x.flags.writeable = False
        y.flags.writeable = False
        self._x, self._y = x, y
        self._n = len(x)
        self._dense = bool(len(x) and x[0] == floor(x[0]) and
                           np.array_equal(x, np.arange(len(x)) + x[0]))
        self._start = float(x[0]) if len(x) else 0.0

    @property
    def arrays(self):
        """Independent snapshots for transport/serialization, not writable aliases."""
        return np.array(self._x, copy=True), np.array(self._y, copy=True)

    def __reduce__(self):
        return type(self), self.arrays

    @cython.locals(lo=cython.Py_ssize_t, hi=cython.Py_ssize_t,
                   mid=cython.Py_ssize_t)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _right(self, x):
        lo, hi = 0, self._n
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if self._x[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @cython.locals(i=cython.Py_ssize_t, delta=cython.double)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _index(self, x):
        if self._dense:
            delta = x - self._start
            if 0 <= delta < self._n:
                i = int(delta)
                if delta == i:
                    return i
            return -1
        i = self._right(x) - 1
        if i >= 0 and self._x[i] == x:
            return i
        return -1

    @cython.locals(i=cython.Py_ssize_t)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def at(self, x):
        """Exact key; a missing key raises KeyError."""
        if not isfinite(x):
            raise ValueError('Lookup key must be finite')
        i = self._index(x)
        if i < 0:
            raise KeyError(x)
        return self._y[i]

    @cython.locals(i=cython.Py_ssize_t)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get(self, x, default=0.0):
        """Exact key or the explicitly supplied default."""
        if not isfinite(x):
            raise ValueError('Lookup key must be finite')
        i = self._index(x)
        return default if i < 0 else self._y[i]

    @cython.locals(i=cython.Py_ssize_t)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def previous(self, x, below=_MISSING):
        """Last key <= x; hold the final value above range.

        Below the first key (or for an empty curve), raise unless `below` is given.
        NaN is reserved to mean 'raise', and is not an admitted table value.
        """
        if not isfinite(x):
            raise ValueError('Lookup key must be finite')
        i = self._right(x) - 1
        if i < 0:
            if isnan(below):
                raise KeyError(x)
            return below
        return self._y[i]

    @cython.locals(i=cython.Py_ssize_t, weight=cython.double)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def linear(self, x):
        """Linear interpolation; raise below range, hold at/above final key."""
        if not isfinite(x):
            raise ValueError('Lookup key must be finite')
        i = self._right(x) - 1
        if i < 0:
            raise KeyError(x)
        if i >= self._n - 1:
            return self._y[i]
        weight = (x - self._x[i]) / (self._x[i + 1] - self._x[i])
        return self._y[i] + (self._y[i + 1] - self._y[i]) * weight


def curve(series):
    """Prepare a pandas Series indexed by a numeric key; labels remain explicit."""
    return Curve(series.index.to_numpy(), series.to_numpy())


def power(base, exponent):
    """Real-valued power, with the same error contract in both implementations."""
    if base < 0.0 and exponent != floor(exponent):
        raise ValueError('Fractional power of a negative base is outside the real kernel domain')
    if base == 0.0 and exponent < 0.0:
        raise ZeroDivisionError('Zero raised to a negative power')
    result = real_pow(base, exponent)
    if not isfinite(result):
        raise OverflowError('Non-finite power result')
    return result

class CurveTable:
    """All groups of a numeric curve table; group selection happens on each query.

    Public labels are strings or integers; native queries use exact int64 codes.
    No policy or scenario is selected while packing this representation.
    """
    def __init__(self, groups, offsets, keys, values, labels=None):
        raw = np.asarray(groups)
        if raw.ndim != 1 or raw.dtype.kind not in 'iu' or (raw.size and
                (int(raw.min()) < -(1 << 63) or int(raw.max()) >= (1 << 63))):
            raise ValueError('Group labels require exact int64 integers')
        g = np.array(raw, dtype=np.int64, copy=True, order='C')
        braw = np.asarray(offsets)
        if braw.ndim != 1 or braw.dtype.kind not in 'iu':
            raise ValueError('Offsets require integral boundaries')
        b = np.array(braw, dtype=np.int64, copy=True, order='C')
        x = np.array(keys, dtype=np.float64, copy=True, order='C')
        y = np.array(values, dtype=np.float64, copy=True, order='C')
        if len(b) != len(g) + 1 or b[0] != 0 or x.ndim != 1 or y.ndim != 1 or b[-1] != len(x) or len(x) != len(y):
            raise ValueError('Invalid ragged curve layout')
        if np.any(b[1:] < b[:-1]) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid curve boundaries or nonfinite values')
        if len(g) > 1 and np.any(g[1:] <= g[:-1]):
            raise ValueError('Group labels must be sorted and unique')
        dense = np.zeros(len(g), dtype=np.uint8)
        for i in range(len(g)):
            sub = x[b[i]:b[i+1]]
            if len(sub)>1 and np.any(sub[1:] <= sub[:-1]):
                raise ValueError('Curve keys must be sorted and unique within each group')
            dense[i] = bool(len(sub) and sub[0] == floor(sub[0]) and
                            np.array_equal(sub, np.arange(len(sub)) + sub[0]))
        for a in (g,b,x,y,dense): a.flags.writeable = False
        labels = tuple(int(v) for v in g) if labels is None else tuple(labels)
        if len(labels) != len(g) or len(set(labels)) != len(labels):
            raise ValueError('Group labels must match groups and be unique')
        if any(type(v) not in (str,int) for v in labels):
            raise ValueError('Group labels must be strings or exact integers')
        self._labels = labels
        self._codes = dict(zip(labels,map(int,g)))
        self._groups, self._offsets, self._x, self._y = g,b,x,y
        self._dense = dense
        self._ng = len(g)
        self._group_dense = bool(len(g) and int(g[-1])-int(g[0]) == len(g)-1)

    @property
    def arrays(self):
        return tuple(np.array(v, copy=True) for v in (self._groups,self._offsets,self._x,self._y))

    @property
    def labels(self):
        return self._labels

    def code(self, value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, int, np.integer)): raise KeyError(value)
        if value in self._codes: return self._codes[value]
        if isinstance(value,(int,np.integer)) and value in self._codes.values(): return int(value)
        raise KeyError(value)

    def __reduce__(self):
        return type(self), (*self.arrays, self.labels)

    @cython.locals(lo=cython.Py_ssize_t, hi=cython.Py_ssize_t, mid=cython.Py_ssize_t)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def _group_index(self, group):
        if not self._ng or group < self._groups[0] or group > self._groups[self._ng-1]:
            return -1
        if self._group_dense:
            return group - self._groups[0]
        lo, hi = 0, self._ng
        while lo < hi:
            mid = lo + (hi-lo)//2
            if self._groups[mid] < group: lo = mid+1
            else: hi = mid
        return lo if lo < self._ng and self._groups[lo] == group else -1

    @cython.locals(lo=cython.Py_ssize_t, hi=cython.Py_ssize_t, mid=cython.Py_ssize_t,
                   start=cython.Py_ssize_t, stop=cython.Py_ssize_t)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def _right(self, gi, x):
        start, stop = self._offsets[gi], self._offsets[gi+1]
        if start == stop or x < self._x[start]: return start
        if x >= self._x[stop-1]: return stop
        if self._dense[gi]:
            return start + cython.cast(cython.Py_ssize_t, floor(x-self._x[start])) + 1
        lo,hi = start,stop
        while lo < hi:
            mid = lo+(hi-lo)//2
            if self._x[mid] <= x: lo = mid+1
            else: hi = mid
        return lo

    @cython.locals(gi=cython.Py_ssize_t, i=cython.Py_ssize_t)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def at(self, group, x):
        if not isfinite(x): raise ValueError('Lookup key must be finite')
        gi = self._group_index(group)
        if gi < 0: raise KeyError(group)
        i = self._offsets[gi]
        if i == self._offsets[gi+1] or x < self._x[i] or x > self._x[self._offsets[gi+1]-1]:
            raise KeyError((group,x))
        if self._dense[gi]:
            i += cython.cast(cython.Py_ssize_t, x-self._x[i])
        else:
            i = self._right(gi,x)-1
        if self._x[i] != x: raise KeyError((group,x))
        return self._y[i]

    @cython.locals(gi=cython.Py_ssize_t, i=cython.Py_ssize_t)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def get(self, group, x, default=0.0):
        if not isfinite(x): raise ValueError('Lookup key must be finite')
        gi = self._group_index(group)
        if gi < 0: return default
        i = self._offsets[gi]
        if i == self._offsets[gi+1] or x < self._x[i] or x > self._x[self._offsets[gi+1]-1]: return default
        if self._dense[gi]:
            i += cython.cast(cython.Py_ssize_t, x-self._x[i])
        else:
            i = self._right(gi,x)-1
        if self._x[i] != x: return default
        return self._y[i]

    @cython.locals(gi=cython.Py_ssize_t, i=cython.Py_ssize_t)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def previous(self, group, x, below=_MISSING):
        if not isfinite(x): raise ValueError('Lookup key must be finite')
        gi = self._group_index(group)
        if gi < 0: raise KeyError(group)
        i = self._offsets[gi+1]-1
        if i >= self._offsets[gi] and x >= self._x[i]: return self._y[i]
        i = self._right(gi,x)-1
        if i < self._offsets[gi]:
            if isnan(below): raise KeyError((group,x))
            return below
        return self._y[i]

    @cython.locals(gi=cython.Py_ssize_t, i=cython.Py_ssize_t, weight=cython.double)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def linear(self, group, x):
        if not isfinite(x): raise ValueError('Lookup key must be finite')
        gi = self._group_index(group)
        if gi < 0: raise KeyError(group)
        i = self._right(gi,x)-1
        if i < self._offsets[gi]: raise KeyError((group,x))
        if i >= self._offsets[gi+1]-1: return self._y[i]
        weight = (x-self._x[i]) / (self._x[i+1]-self._x[i])
        return self._y[i] + (self._y[i+1]-self._y[i])*weight


def curve_table(series, *, group, codes=None):
    """Pack a complete pandas MultiIndex Series; never select a model input.

    `group` names one index level; the remaining numeric level is the curve key.
    Optional `codes` is a declared label encoding (e.g. {'M': Sex.M, 'F': Sex.F}).
    The resulting query API accepts a group expression at evaluation time.
    """
    if series.index.nlevels != 2 or group not in series.index.names:
        raise ValueError('Expected a two-level index and an explicit group level')
    if not series.index.is_unique:
        raise ValueError('Duplicate group/key pairs')
    mapping, labels_by_code = {}, {}
    source_labels = list(series.index.get_level_values(group).unique())
    if codes is None:
        if all(isinstance(v,(int,np.integer)) and not isinstance(v,(bool,np.bool_)) for v in source_labels):
            codes = {v:int(v) for v in source_labels}
        elif all(isinstance(v,str) for v in source_labels):
            codes = {v:i for i,v in enumerate(sorted(source_labels))}
        else: raise ValueError('Mixed group-label kinds require an explicit encoding')
    for label, values in series.groupby(level=group, sort=False):
        code = codes[label] if codes is not None else label
        if isinstance(code,(bool,np.bool_)) or not isinstance(code,(int,np.integer)):
            raise ValueError('Use an explicit encoding for non-integral group labels')
        if int(code) in mapping: raise ValueError('Category encoding must be one-to-one')
        mapping[int(code)] = curve(values.droplevel(group)).arrays
        labels_by_code[int(code)] = str(label) if isinstance(label,str) else int(label)
    groups = sorted(mapping)
    offsets,xx,yy = [0],[],[]
    for code in groups:
        x,y = mapping[code];xx.extend(x);yy.extend(y);offsets.append(len(xx))
    return CurveTable(np.array(groups,dtype=np.int64),np.array(offsets,dtype=np.int64),xx,yy,[labels_by_code[g] for g in groups])
