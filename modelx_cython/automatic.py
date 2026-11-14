"""Model-independent cache policy; observations never narrow logical domains."""

from .cell_entries import DenseAxis
import ctypes
import sys

# Shared resource policies, independent of model/Cell names. At double plus
# bint-readiness storage the dense ceiling is 768 KiB per Cell on Windows. Associative
# storage retains the existing growing overflow table beyond its primary cap.
MAX_DENSE_ENTRIES = 65536
MAX_PRIMARY_ENTRIES = 4096


def enabled(cell):
    module = getattr(getattr(cell, 'parent', None), 'module', None)
    return getattr(getattr(module, 'spec', None), 'automatic_optimizations', False)


def observed_axes(cell, logical):
    runtime = getattr(cell, '_rt', None)
    minimum = getattr(runtime, 'observed_min_args', {})
    maximum = getattr(runtime, 'observed_max_args', {})
    if not logical or any(axis.key not in minimum or axis.key not in maximum for axis in logical):
        return logical
    axes = tuple(DenseAxis(axis.key, max(axis.lower, minimum[axis.key]),
                           min(axis.stop, maximum[axis.key] + 1), axis.c_type) for axis in logical)
    if not all(axis.lower < axis.stop for axis in axes):
        return logical
    full, compact = 1, 1
    for complete, observed in zip(logical, axes):
        full *= complete.stop - complete.lower
        compact *= observed.stop - observed.lower
    value_size = ctypes.sizeof(ctypes.c_int if cell.get_rettype_expr(c_style=True) == 'bint'
                               else ctypes.c_longlong)
    saved = (full - compact) * (value_size + ctypes.sizeof(ctypes.c_int))
    # A compact cache adds a pointer and a cold dictionary. Do not trade a
    # near-full native array (and its range proofs) for less than that storage.
    spill_cost = ctypes.sizeof(ctypes.c_void_p) + sys.getsizeof({})
    return axes if saved >= spill_cost else logical


def max_dense_entries_set(cell):
    """Whether the spec sets max_dense_entries for the Cell or its Space (it then applies whether or
    not automatic optimizations are on)."""
    if 'max_dense_entries' in getattr(cell, '_spec', {}):
        return True
    parent = getattr(cell, 'parent', None)
    spec = getattr(getattr(parent, 'module', None), 'spec', None)
    return spec is not None and hasattr(spec, 'get_spec') and 'max_dense_entries' in spec.get_spec(parent.fqname)


def max_dense_entries(cell):
    """D74: the inline dense ceiling of a Cell: spec key ``max_dense_entries`` of the Cell, else of its
    Space, else MAX_DENSE_ENTRIES. Inline dense storage is part of every instance on every route."""
    value = getattr(cell, '_spec', {}).get('max_dense_entries')
    if value is None:
        parent = getattr(cell, 'parent', None)
        module = getattr(parent, 'module', None)
        spec = getattr(module, 'spec', None)
        if spec is not None and hasattr(spec, 'get_spec'):
            value = spec.get_spec(parent.fqname).get('max_dense_entries')
    if value is None:
        return MAX_DENSE_ENTRIES
    if type(value) is not int or value <= 0:
        raise ValueError(f'{getattr(cell, "fqname", cell)}: max_dense_entries must be a positive integer')
    return value


def dense_fits(cell):
    limit = max_dense_entries(cell)
    count = 1
    for axis in observed_axes(cell, cell.dense_logical_axes):
        count *= axis.stop - axis.lower
        if count > limit:
            return False
    return count > 0


# D74: on the collect route that reuses one root per worker, a hash-cached Cell with integer keys may hold
# its whole logical window dense on that root (heap storage allocated on the root only; other instances
# keep the hash store). Budget per Cell and per Space, largest windows first; 13 bytes per entry (value
# word, ready byte, written index for a reset that clears only what was written).
REUSE_DENSE_CELL_BYTES = 16 << 20
REUSE_DENSE_SPACE_BYTES = 64 << 20
REUSE_DENSE_ENTRY_BYTES = 13


def reuse_dense_windows(cls):
    """Return ({cell name: (axes, entries, bytes)}, {cell name: reason}) for the hash-cached Cells of a
    Space whose collect reuses one root. Spec key ``reuse_dense_windows`` (Space) false turns it off."""
    chosen, refused, candidates = {}, {}, []
    spec = cls.module.spec.get_spec(cls.fqname) if hasattr(getattr(cls, 'module', None), 'spec') else {}
    enabled_spec = spec.get('reuse_dense_windows', True)
    if type(enabled_spec) is not bool:
        raise ValueError(f'{cls.fqname}: reuse_dense_windows must be a boolean')
    for name, cell in cls.cells.items():
        if cell.is_special() or not cell.has_formula_def or not cell.has_args() or not cell.has_typeinfo():
            continue
        if not cell.uses_native_numeric_cache:
            continue
        if not enabled_spec:
            refused[name] = 'reuse_dense_windows is false for the Space'
            continue
        if len(cell.params) > 3:
            refused[name] = 'more than three parameters (wide key store)'
            continue
        if not cell.is_int_args:
            refused[name] = 'key type: the parameters are not all C integers'
            continue
        try:
            axes = tuple(cell.dense_logical_axes)
        except (KeyError, AttributeError, TypeError):
            refused[name] = 'no bounded logical window for the parameters'
            continue
        entries = 1
        for axis in axes:
            entries *= int(axis.stop) - int(axis.lower)
        size = entries * REUSE_DENSE_ENTRY_BYTES
        if entries <= 0:
            refused[name] = 'empty logical window'
        elif size > REUSE_DENSE_CELL_BYTES:
            refused[name] = (f'window {" x ".join(str(int(a.stop) - int(a.lower)) for a in axes)} = '
                             f'{entries:,} entries, {size:,} bytes exceeds the per-cell budget '
                             f'{REUSE_DENSE_CELL_BYTES:,}')
        else:
            candidates.append((size, name, axes, entries))
    used = 0
    for size, name, axes, entries in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if used + size > REUSE_DENSE_SPACE_BYTES:
            refused[name] = (f'Space budget {REUSE_DENSE_SPACE_BYTES:,} bytes exhausted '
                             f'({used:,} used by larger windows; this one {size:,})')
            continue
        used += size
        chosen[name] = (axes, entries, size)
    return chosen, refused


def primary_capacity(cell):
    # Formula evaluations are cache misses, counted per receiver rather than
    # summed across traced policies. Keep the table's existing <=50% load rule.
    evaluations = getattr(getattr(cell, '_rt', None), 'max_evaluations_per_receiver', 0)
    if evaluations <= 0:
        return 0
    return min(MAX_PRIMARY_ENTRIES, 1 << (2 * evaluations - 1).bit_length())
