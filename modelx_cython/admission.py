"""Describe actual Cython representations without claiming nogil execution."""




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
                    'associative_logical_domain': {
                        axis.key: [int(axis.lower), int(axis.stop)] for axis in cell.associative_logical_axes
                    } if cache in ('native-numeric-keys', 'dictionary') else {},
                    'declared_sizes': _declared_sizes(cls, cell) if cache != 'none' else {},
                    **({'enum_domains': {name: f'{e[0]}.{e[1]} ({e[2]} members)' for name, e in enum_args.items()}}
                       if (enum_args := getattr(getattr(cell, '_rt', None), 'enum_args', None)) else {}),
                    'numeric_primary_capacity': primary_capacity(cell)
                    if cache == 'native-numeric-keys' else 0,
                    'power_memo': bool(getattr(cell, 'power_memo', None)),
                    **(_enum_report(cell) if hasattr(cell, 'native_enum_constants') else {}),
                    'dense_offsets': [int(offset) for offset in cell.cache_offsets]
                    if cache == 'dense' else [],
                    'exact_spill': cache == 'dense' and bool(cell.has_dense_spill),
                    'powers_in_loops': list(getattr(cell, 'powers_in_loops', ())),
                    'guarded_integer_ops': getattr(cell, 'guarded_integer_ops', 0),
                    'object_locals': [{'local': name, 'reason': 'compared by identity (is / is not)'}
                                      for name in getattr(cell, 'python_object_locals', ())],
                    'cache_detail': cache_detail(cell, cache),
                    'slow_paths': slow_paths(cell, cache),
                    'reasons': reasons or ['original Cython representation'],
                }
    return result


def _declared_sizes(cls, cell):
    """{parameter: where its cells_param_size is declared} for a Cell's cache axes."""
    if not cell.params or not hasattr(cls, 'declared_axes'):
        return {}
    cls.cells_arg_sizes, cls.associative_arg_sizes
    return dict(cls.declared_axes.get(tuple(cell.params), {}))


def _window(axes):
    sizes = [int(axis.stop) - int(axis.lower) for axis in axes]
    count = 1
    for size in sizes:
        count *= size
    return [[int(axis.lower), int(axis.stop)] for axis in axes], sizes, count


OVER_LIMIT_WARNING = ('{cell}: window {window} (traced {ranges}) = {count:,} entries exceeds max_dense_entries '
                      '{limit:,}; its values are kept in the hash store. If the cell is '
                      'read only on a regular grid, give it one parameter for the grid position.')


def over_limit_warning(cell):
    """The build warning for a hash-cached Cell whose integer window exceeds the dense limit, else None."""
    if not (cell.has_args() and cell.has_typeinfo() and cell.uses_native_numeric_cache and cell.is_int_args):
        return None
    from .automatic import max_dense_entries
    try:
        bounds, sizes, count = _window(cell.dense_logical_axes)
    except (KeyError, AttributeError, TypeError):
        return None
    limit = max_dense_entries(cell)
    if count <= limit:
        return None
    ranges = ', '.join(f'{name} in [{lo}, {hi})' for name, (lo, hi) in zip(cell.params, bounds))
    return OVER_LIMIT_WARNING.format(cell=getattr(cell, 'model_path', cell.fqname), window=' x '.join(map(str, sizes)), ranges=ranges,
                                     count=count, limit=limit)


USE_WARNING = ('{cell} ({params}): hash store: rectangle {ranges} = {count:,} entries, beyond the dense '
               'ceiling of {limit:,} entries. Remedies, in order: re-parametrise the cell if its parameters are tied '
               '(one parameter for a grid position); mark it not cached if each value is read once; tighten or '
               'correct the declaration.')


def _store_axes(cell):
    """(ranges text, count) of the rectangle the store decision uses: declared axes as declared, sampled ones
    as sampled (observed_axes)."""
    from .automatic import declared_axes, observed_axes
    declared = declared_axes(cell)
    axes = observed_axes(cell, cell.dense_logical_axes)
    count = 1
    for axis in axes:
        count *= int(axis.stop) - int(axis.lower)
    ranges = ' x '.join(f'{axis.key} [{int(axis.lower)}, {int(axis.stop)}) '
                        f'{"declared" if axis.key in declared else "sampled"}' for axis in axes)
    return ranges, count


def use_warning(cell):
    """The build warning for a multi-argument Cell put in the hash store because its rectangle exceeds the
    dense ceiling, else None (an explicit max_dense_entries has its own)."""
    from .automatic import max_dense_entries, max_dense_entries_set
    if not (len(cell.params) >= 2 and cell.has_typeinfo() and cell.uses_native_numeric_cache
            and cell.is_int_args) or max_dense_entries_set(cell):
        return None
    try:
        ranges, count = _store_axes(cell)
    except (KeyError, AttributeError, TypeError):
        return None
    limit = max_dense_entries(cell)
    if count <= limit:
        return None
    return USE_WARNING.format(cell=getattr(cell, 'model_path', cell.fqname), params=', '.join(cell.params),
                              ranges=ranges, count=count, limit=limit)


def store_reason(cell, cache):
    """Per multi-argument integer-keyed Cell, the store chosen and why (build report)."""
    from .automatic import max_dense_entries, max_dense_entries_set
    ranges, count = _store_axes(cell)
    limit = max_dense_entries(cell)
    source = 'explicit max_dense_entries' if max_dense_entries_set(cell) else 'max_dense_entries'
    if cache == 'dense':
        return f'dense: {ranges} = {count:,} entries, within {source} {limit:,}'
    if count <= limit:
        return f'{cache}: {ranges} = {count:,} entries, within {source} {limit:,} but not dense (see hash_reason)'
    return f'{cache}: {ranges} = {count:,} entries exceeds the {source} {limit:,}'


def cache_detail(cell, cache):
    """The cache of a Cell with parameters: kind, window, entries, bytes, and why it is not dense
    (one storage for the Cell, whatever calls it)."""
    if not cell.params or cache in ('none', 'scalar'):
        return {'kind': cache}
    detail = {'kind': cache}
    try:
        if len(cell.params) >= 2 and cell.has_typeinfo() and cell.is_int_args and cache in (
                'dense', 'native-numeric-keys'):
            detail.update(store_reason=store_reason(cell, cache))
        if cache == 'dense':
            window, sizes, count = _window(cell.dense_axes)
            detail.update(window=window, entries=count, bytes=count * 12)
            return detail
        if cache not in ('native-numeric-keys', 'dictionary'):
            return detail
        from .automatic import max_dense_entries
        limit = max_dense_entries(cell)
        if cache == 'dictionary':
            detail['hash_reason'] = ('return or parameter types are not native numbers'
                                     if cell.has_typeinfo() else 'formula not observed by the sample')
        elif not cell.is_int_args:
            detail['hash_reason'] = 'key type: the parameters are not all C integers'
        else:
            window, sizes, count = _window(cell.dense_logical_axes)
            detail.update(window=window, entries=count)
            detail['hash_reason'] = (f'window {" x ".join(map(str, sizes))} = {count:,} entries exceeds '
                                     f'max_dense_entries {limit:,}' if count > limit
                                     else 'negative arguments without a shifted window')
    except Exception as error:      # an axis without bounds: the kind alone
        detail.setdefault('hash_reason', f'no bounded window ({type(error).__name__})')
    return detail


def slow_paths(cell, cache):
    """Work of a compiled formula that still runs on Python objects or a slow cache,
    which the other report keys do not flag as such: a cell with parameters on a hash or
    dictionary cache (and why it is not dense), lowered reductions that iterate or add Python
    objects, table reads whose key parts are Python objects, powers left on Python semantics in
    a loop, and a cell with parameters that returns a Python object. Each entry: kind, detail,
    per_step (the cell has parameters)."""
    per_step = bool(cell.params)
    entries = []
    if per_step and cache in ('native-numeric-keys', 'dictionary'):
        detail = f'{cache} cache'
        try:
            from .automatic import max_dense_entries
            limit = max_dense_entries(cell)
            axes = list(getattr(cell, 'associative_logical_axes', ()) or ())
            if axes and len(axes) == len(cell.params):
                sizes = [int(axis.stop) - int(axis.lower) for axis in axes]
                count = 1
                for size in sizes:
                    count *= size
                detail += (f': dense window {" x ".join(map(str, sizes))} = {count:,} entries exceeds '
                           f'max_dense_entries {limit:,}' if count > limit
                           else ': the arguments are not all sampled C integers in a bounded range')
        except Exception:       # an axis without bounds: the cache kind alone
            pass
        entries.append({'kind': 'hash-cache', 'detail': detail, 'per_step': per_step})
    for power in getattr(cell, 'powers_in_loops', ()) or ():
        entries.append({'kind': 'power-on-objects-in-loop', 'detail': power, 'per_step': per_step})
    if per_step and cache != 'none' and cell.get_rettype_expr(c_style=True) == 'object':
        entries.append({'kind': 'object-return', 'detail': 'returns a Python object on every call '
                        '(boxed at the cell boundary)', 'per_step': per_step})
    return entries


def _enum_report(cell):
    """The enum constants of a Cell of a module with enum definitions (``enums.Constants``); other reports
    are unchanged."""
    return {'native_enum_comparisons': getattr(cell, 'native_enum_comparisons', 0),
            'native_enum_constants': getattr(cell, 'native_enum_constants', 0),
            'native_expressions': getattr(cell, 'enum_memberships', 0),
            'expressions_untyped': expressions_untyped(cell),
            'expressions_object': expressions_object(cell)}


def expressions_untyped(cell):
    """Expression forms left to the interpreter (enum membership), each with its line and reason."""
    entries = [entry for key in ('enum_untyped',) for entry in getattr(cell, key, ())]
    return sorted(entries, key=lambda entry: (entry['line'], entry['form'], entry['expr']))


def expressions_object(cell):
    """Built-in forms compiled with comparisons on Python objects, because an operand's class is not
    known at build time; each with its line and the operand classes."""
    entries = [entry for key in ('enum_object',) for entry in getattr(cell, key, ())]
    return sorted(entries, key=lambda entry: (entry['line'], entry['form'], entry['expr']))
