"""Pass the standard mx2cy trace's observed Cell types to the native frame.

This supplies reference context. It does not rewrite formula expressions or
claim that a small trace proves every possible input type.
"""

import numbers
from enum import IntEnum

from .contracts import ContractError


def _scalar_type(observed):
    if observed is bool:
        return bool
    if isinstance(observed, type) and issubclass(observed, IntEnum):
        return observed
    if isinstance(observed, type) and issubclass(observed, numbers.Integral):
        return int
    if isinstance(observed, type) and issubclass(observed, numbers.Real):
        return float
    if isinstance(observed, type) and issubclass(observed, str):
        return str
    return object


def from_mx2cy_trace(logger, source_manifest, *, space='Projection'):
    """Return optional annotations for exactly the source-preserved Cells.

    Unobserved Cells keep the frontend's object fallback. The caller remains
    responsible for validating compiled results beyond the traced sample.
    """
    key = source_manifest['model'] + '.' + space
    try:
        source_cells = source_manifest['spaces'][key]['cells']
    except KeyError as exc:
        raise ContractError('Source manifest has no Space: ' + key) from exc
    suffix = '._mx_classes._c_' + space + '._f_'
    result = {}
    for fullname, observed in logger.cells_info.items():
        if suffix not in fullname:
            continue
        before, name = fullname.rsplit(suffix, 1)
        if not before or name not in source_cells:
            continue
        if name in result:
            raise ContractError('Ambiguous traced Cell: ' + name)
        row = {'return': (object if observed.ret_type.is_array else
                          _scalar_type(observed.ret_type.value_type))}
        row.update({arg: _scalar_type(typ)
                    for arg, typ in observed.arg_types.items()})
        result[name] = row
    return result
