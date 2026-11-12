"""Frozen v1 collection entry points for compatible pre-versioned exports."""

_mx_collect_adapter_version = 1


def _mx_collect_native_coords(coords):
    return all(type(coord) is int and -(1 << 63) <= coord < (1 << 63) for coord in coords)


def collect_from_space(base, domain, loop_name, names, args, cache=None, evaluate=None,
                       evaluate_serial=None, reuse_in_parallel=False, private_ordered=False):
    """Evaluate an exported collection without retaining policy ItemSpaces."""

    if len(names) != len(args) or loop_name in names:
        raise ValueError('invalid collect arguments')
    if isinstance(domain, (str, bytes, set, frozenset)) or iter(domain) is domain:
        raise TypeError('collect domain must be finite, ordered and re-iterable')
    coords = tuple(domain)
    if not coords:
        raise ValueError('collect domain must not be empty')
    if len(set(coords)) != len(coords):
        raise ValueError('collect domain contains duplicate coordinates')

    key = (coords, args)
    if cache is not None:
        try:
            hash(key)
        except TypeError:
            cache = None
        else:
            found = cache.get(key)
            if found is not None:
                return found

    if (evaluate_serial is not None and _mx_parallel_workers == 1
            and not getattr(_mx_collect_context, 'active', False)
            and sys.gettrace() is None and sys.getprofile() is None):
        _mx_collect_context.active = True
        try:
            result = evaluate_serial(coords)
        finally:
            _mx_collect_context.active = False
        if result is not None:
            return result if cache is None else cache.setdefault(key, result)
    if evaluate is None:
        can_parallel = getattr(base._model, '_mx_collect_parallel_allowed', None)
        # A shared Cell may call collect while holding the model lock. Keep
        # that nested calculation on the owning thread to avoid a lock wait cycle.
        if (_mx_parallel_workers > 1 and len(coords) > 1
                and not getattr(_mx_collect_context, 'active', False)
                and (can_parallel is None or can_parallel())):
            private = None
            if (reuse_in_parallel and evaluate_serial is not None
                    and sys.gettrace() is None and sys.getprofile() is None
                    and _mx_collect_native_coords(coords)):
                private = evaluate_serial
            result = _mx_collect_parallel(base, coords, loop_name, names, args, private, private_ordered)
        else:
            result = _mx_collect_evaluate(base, coords, loop_name, names, args)
    else:
        result = evaluate()
    return result if cache is None else cache.setdefault(key, result)


def _mx_collect_parallel(base, coords, loop_name, names, args, private=None, private_ordered=False):
    from concurrent.futures import ThreadPoolExecutor

    workers = min(_mx_parallel_workers, len(coords))
    chunks = [coords[i * len(coords) // workers:(i + 1) * len(coords) // workers]
              for i in range(workers)]
    result = None
    individual_shape = None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_mx_collect_chunk, base, chunk, loop_name, names, args, private)
                   for chunk in chunks]
        # Only private scalar chunks may merge in completion order.
        ordered_futures = futures
        if private is not None and not private_ordered:
            from concurrent.futures import as_completed
            futures = as_completed(futures)
        for future in futures:
            if private is not None and future.exception() is not None:
                # Keep successful scalar merging in completion order, but
                # propagate the first failure in original domain order.
                for ordered_future in ordered_futures:
                    ordered_future.result()
            fields = future.result()._fields
            if result is None:
                modes = tuple(fields)
                result = {mode: OrderedDict() if mode == 'individual' else None for mode in modes}
            elif tuple(fields) != modes:
                raise ValueError('collect_outputs modes changed between policies')
            for mode, values in fields.items():
                if mode == 'individual':
                    for coordinate, item in values.items():
                        if individual_shape is None:
                            individual_shape = _mx_collect_shape(item._fields)
                        else:
                            _mx_collect_check_shape(individual_shape, item._fields, (mode,))
                        result[mode][coordinate] = item
                else:
                    result[mode] = _mx_collect_sum(result[mode], values._fields, (mode,))
    return CollectResult({mode: value if mode == 'individual' else CollectResult(value)
                          for mode, value in result.items()})


def _mx_collect_chunk(base, coords, loop_name, names, args, private=None):
    _mx_collect_context.active = True
    try:
        if private is not None:
            result = private(coords)
            if result is not None:
                return result
        return _mx_collect_evaluate(base, coords, loop_name, names, args)
    finally:
        _mx_collect_context.active = False
