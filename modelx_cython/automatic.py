"""Model-independent cache policy; observations never narrow logical domains."""

from .cell_entries import DenseAxis
import ctypes
import sys

# Pilot the September dense geometry with up to 24 MiB of double values and
# readiness per Cell. Associative caches still retain the growing overflow table.
MAX_DENSE_ENTRIES = 2097152
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


def dense_fits(cell):
    count = 1
    for axis in observed_axes(cell, cell.dense_logical_axes):
        count *= axis.stop - axis.lower
        if count > MAX_DENSE_ENTRIES:
            return False
    return count > 0


def primary_capacity(cell):
    # Formula evaluations are cache misses, counted per receiver rather than
    # summed across traced policies. Keep the table's existing <=50% load rule.
    evaluations = getattr(getattr(cell, '_rt', None), 'max_evaluations_per_receiver', 0)
    if evaluations <= 0:
        return 0
    return min(MAX_PRIMARY_ENTRIES, 1 << (2 * evaluations - 1).bit_length())
