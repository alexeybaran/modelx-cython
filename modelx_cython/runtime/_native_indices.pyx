# cython: freethreading_compatible=True
"""Native input-order table keys; no policy state or alternate tables."""
import cython
from modelx_helpers.immutable_inputs import Record, Table
from modelx_helpers.table_keys import keys as _keys
from modelx_cython._call_layout cimport forward_call

cdef object omitted_argument = object()


cdef object _bound_owner(object source):
    """The helper Table under a NativeTable bound by observed admission (D69), else ``source``.

    This provider cannot cimport the NativeTable class of ``_native_inputs``; a bound table
    names its helper owner through ``__modelx_table_owner__`` (ABI 1), the same unwrapping
    ``modelx_helpers`` does when the call falls back to it.
    """
    owner = getattr(type(source), '__modelx_table_owner__', None)
    if owner is not None and getattr(type(source), '__modelx_table_owner_abi__', None) is not None:
        unwrapped = owner(source)
        if type(unwrapped) is Table:
            return unwrapped
    return source


cdef tuple _owned_keys(object source):
    if type(source) is Table:
        return source._keys
    if type(source) is Record:
        if source._metadata == ('index',):
            return source._index['keys']
        return tuple(source._index)
    return None


cdef object keys(object function, tuple keywords, object table):
    if function is _keys:
        if type(table) is not Table and type(table) is not Record:
            table = _bound_owner(table)
        if type(table) is Table:
            return table._keys
        if type(table) is Record:
            return _owned_keys(table)
    return forward_call(function, keywords, ('table',), (table,), omitted_argument)


