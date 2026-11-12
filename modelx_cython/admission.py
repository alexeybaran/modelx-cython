"""Describe actual Cython representations without claiming nogil execution."""

from .consts import FORMULA_PREF, MODULE_PREF, SPACE_PREF


def _lock_reason(source, kind, resource='table'):
    _, *parents, _, cls, cell = source.split('.')
    space = '.'.join([*(name[len(MODULE_PREF):] for name in parents),
                      cls[len(SPACE_PREF):]])
    name = cell[len(FORMULA_PREF):]
    return f'native {kind} {resource} read from {space}.{name} requires {space} in export locked_spaces'


def describe(module_infos):
    from modelx_cython.numeric_cache import primary_capacity
    result = {}
    for module in module_infos.values():
        for cls in module.classes.values():
            for cell in cls.cells.values():
                if cell.is_special():
                    continue
                formula_python = cell.formula_is_generator or cell.formula_has_captured_default or (
                    not cell.has_formula_def and cell.body_has_closure)
                public_python = cell.body_has_closure or cell.called_with_kwargs
                if not cell.has_formula_def:
                    cache = 'none'
                elif not cell.has_args():
                    cache = 'scalar'
                elif cell.has_typeinfo() and cell.is_arrayable():
                    cache = 'dense'
                elif cell.uses_native_numeric_cache:
                    cache = 'native-numeric-keys'
                else:
                    cache = 'dictionary'
                reasons = []
                if not cell.has_typeinfo():
                    reasons.append('formula not observed by the sample')
                if cell.formula_is_generator:
                    reasons.append('yield requires a Python method')
                if cell.formula_has_captured_default:
                    reasons.append('captured default uses the compiled Python signature')
                if cell.body_has_closure:
                    reasons.append('closure requires a Python public method')
                if cell.called_with_kwargs:
                    reasons.append('keyword call retains the Python public boundary')
                if cell.is_locked:
                    reasons.append('exported shared-Space lock retained')
                if cell.observed_native_binding is not None and cell.native_provider_binding is None:
                    reasons.append('native provider result requires a matching return annotation or direct forwarding from a typed Cell')
                reasons.extend(_lock_reason(source, 'cached', resource)
                               for source, resource in getattr(cell, 'unlocked_provider_sources', ()))
                reasons.extend(_lock_reason(source, kind)
                               for source, kind in getattr(cell, 'unlocked_table_reads', ()))
                result[cell.fqname] = {
                    'formula': 'python-method' if formula_python else 'cython-cfunction',
                    'public_entry': 'python-method' if public_python else 'cython-cpdef',
                    'inline_entry': getattr(cell, 'inline_entry', None),
                    'return_type': cell.get_rettype_expr(c_style=True),
                    'arguments': {name: cell.get_argtype_expr(name, c_style=True)
                                  for name in cell.params},
                    'cache': cache,
                    'automatic_policy': module.spec.automatic_optimizations,
                    'dense_window': [[int(axis.lower), int(axis.stop)] for axis in cell.dense_axes]
                    if cache == 'dense' else [],
                    'dense_logical_domain': [[int(axis.lower), int(axis.stop)] for axis in cell.dense_logical_axes]
                    if cache == 'dense' else [],
                    'numeric_primary_capacity': primary_capacity(cell)
                    if cache == 'native-numeric-keys' else 0,
                    'power_memo': bool(getattr(cell, 'power_memo', None)),
                    'dense_offsets': [int(offset) for offset in cell.cache_offsets]
                    if cache == 'dense' else [],
                    'exact_spill': cache == 'dense' and bool(cell.has_dense_spill),
                    'nogil_qualified': False,
                    'native_helpers': getattr(cell, 'native_helpers', []),
                    'native_numeric_helpers': getattr(cell, 'native_numeric_helpers', []),
                    'native_reductions': getattr(cell, 'native_reductions', 0),
                    'native_enum_comparisons': getattr(cell, 'native_enum_comparisons', 0),
                    'native_enum_constants': getattr(cell, 'native_enum_constants', 0),
                    'reasons': reasons or ['original Cython representation'],
                }
    return result
