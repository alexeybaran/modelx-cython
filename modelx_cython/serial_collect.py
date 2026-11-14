"""Emit an opt-in scalar reduction through ordinary compiled leaf Cells."""

from dataclasses import dataclass

import libcst as cst

from modelx_cython.consts import CY_MOD
from .syntax import arguments, dotted_name, literal_value, statements, walk


@dataclass(frozen=True)
class SerialCollect:
    label: str
    cell: str
    parameter: str
    reuse: bool
    private_lookups: bool
    parallel_reuse: bool
    ordered: bool = False


def _self_attribute(node):
    if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name):
        if node.value.value == 'self' and not node.attr.value.startswith('_'):
            return node.attr.value
    return None


def _method(node, name):
    return next((item for item in node.body.body
                 if isinstance(item, cst.FunctionDef) and item.asynchronous is None
                 and item.name.value == name), None)


def _return_expr(method):
    body = statements(method.body) if method else ()
    if len(body) == 1 and isinstance(body[0], cst.Return):
        return body[0].value
    return None


def _single_dict(expr):
    if (isinstance(expr, cst.Dict) and len(expr.elements) == 1
            and isinstance(expr.elements[0], cst.DictElement)):
        key = expr.elements[0].key
        try:
            value = literal_value(key)
        except (ValueError, TypeError):
            return None, None
        if type(value) is str:
            return value, expr.elements[0].value
    return None, None


def collect_contract(node):
    value = _return_expr(_method(node, '_mx_collect_contract_v2'))
    try:
        contract = literal_value(value)
    except (ValueError, TypeError):
        return None
    if (isinstance(contract, tuple) and len(contract) == 4
            and type(contract[0]) is str and isinstance(contract[1], tuple)
            and all(type(name) is str for name in contract[1])
            and type(contract[2]) is str and type(contract[3]) is bool):
        return contract
    return None


def analyze(module):
    """Recognize the numeric sum shape. Policy isolation is a Space contract."""
    options = {name: module.spec.get_spec(cls.fqname)
               for name, cls in module.classes.items()}
    automatic = module.spec.automatic_optimizations
    if not any('serial_collect' in spec or 'parallel_collect_reuse' in spec
               for spec in options.values()) and not automatic:
        return {}
    result = {}
    for node in module.visitor.wrapper.module.body:
        if not isinstance(node, cst.ClassDef) or node.name.value not in module.classes:
            continue
        cls = module.classes[node.name.value]
        spec = options[node.name.value]
        option = spec.get('serial_collect', 'reduce' if automatic else None)
        parallel = spec.get('parallel_collect_reuse', automatic and option == 'reuse')
        if type(parallel) is not bool or (parallel and option != 'reuse'):
            raise ValueError('parallel_collect_reuse requires serial_collect reuse')
        # D18: parallel reuse merges chunk totals in domain order unless the spec asks for completion.
        order = spec.get('parallel_collect_order', 'domain' if parallel is True else 'completion')
        if order not in ('completion', 'domain') or (order == 'domain' and not parallel):
            raise ValueError('parallel_collect_order requires parallel reuse and completion or domain')
        if option is None:
            continue
        if option not in ('reduce', 'reuse'):
            raise ValueError('serial_collect must be reduce or reuse')
        private = spec.get('serial_private_lookups', automatic and option == 'reuse')
        if type(private) is not bool or (private and option != 'reuse'):
            raise ValueError('serial_private_lookups requires serial_collect reuse')
        shape = _match(node, cls, option == 'reuse', private, parallel, order == 'domain')
        if shape is not None:
            result[node.name.value] = shape
            if shape.reuse:
                # D74: budgeted dense windows for hash-cached Cells on the reused root
                from .automatic import reuse_dense_windows
                cls.reuse_dense_windows, cls.reuse_dense_refused = reuse_dense_windows(cls)
    return result


def _match(node, cls, reuse, private_lookups, parallel_reuse, ordered=False):
    if (len(node.bases) != 1 or dotted_name(node.bases[0].value) != '_mx_sys.BaseSpace'
            or cls.spaces or cls.is_locked or node.name.value in cls.visitor.dynamic_ref_classes
            or any(_method(node, hook) is not None or hook in cls.refs
                   for hook in ('_mx_apply_formula', '_mx_collect_attach', '_mx_collect_release'))):
        return None
    contract = collect_contract(node)
    if contract is None or contract[1]:
        return None
    parameter = contract[0]
    if (not isinstance(parameter, str) or parameter not in cls.refs
            or cls.refs[parameter].get_type_expr(c_style=True) != 'long long'):
        return None
    output = _return_expr(_method(node, '_f_collect_outputs'))
    mode, fields = _single_dict(output)
    label, entry = _single_dict(fields)
    if mode != 'sum' or not label or arguments(entry, count=0) is None:
        return None
    cell = _self_attribute(entry.func)
    if (cell not in cls.cells or cls.cells[cell].has_args()
            or cls.cells[cell].get_rettype_expr(c_style=True) != 'double'
            or cls.cells[cell].body_has_closure or cls.cells[cell].called_with_kwargs):
        return None

    return SerialCollect(label, cell, parameter, reuse, private_lookups, parallel_reuse, ordered)


def _check_closure(cls, names):
    """Cells a check Cell reaches through ``self.<cell>`` (transitively), or None when a formula uses
    ``self`` other than as ``self.<attribute>`` (the pre-pass then resets every cache)."""
    defs = cls.visitor.formula_defs.get(cls.name, {})
    seen, todo = set(), list(names)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        node = defs.get(name)
        if node is None:
            return None
        parts = list(walk(node.body))
        bare = sum(1 for part in parts if isinstance(part, cst.Name) and part.value == 'self')
        owned = [part.attr.value for part in parts if isinstance(part, cst.Attribute)
                 and isinstance(part.value, cst.Name) and part.value.value == 'self']
        if bare != len(owned):
            return None
        todo.extend(attr for attr in owned if attr in cls.cells)
    return seen


def _reset_lines(cls, only=None):
    for held, _ in getattr(cls, 'held_table_handles', {}).values():
        # diag-4w (D71): the per-instance table handles go with the per-policy caches
        yield f'root.{held} = None'
    for field in getattr(cls, 'held_table_columns', {}).values():
        # diag-jp (D75): and the column indexes held with them
        yield f'root.{field} = 0'
    for field in getattr(cls, 'held_table_rows', {}).values():
        # D79: and the rows of constant keys held with them
        yield f'root.{field} = 0'
    for name, cell in cls.cells.items():
        if only is not None and name not in only:
            continue
        if memo := getattr(cell, 'power_memo', None):
            yield f'root.{memo}.ready = False'
        if cell.is_special() or not cell.has_formula_def:
            continue
        if not cell.has_args():
            yield f'root._has_{name} = False'
            if cell.get_rettype_expr(c_style=True) not in ('bint', 'double', 'long long'):
                yield f'root._v_{name} = None'
        elif cell.has_typeinfo() and cell.is_arrayable():
            flags = f'root._has_{name}'
            yield f'_mx_collect_memset({CY_MOD}.address({flags}), 0, {CY_MOD}.sizeof({flags}))'
            if cell.has_dense_spill:
                yield f'root._spill_{name} = None'
        elif cell.uses_native_numeric_cache:
            yield f'root._v_{name}.reset()'
        else:
            yield f'root._v_{name} = None'


@dataclass(frozen=True)
class BoundCheck:
    parameter: str
    cell: str
    limit: int          # largest value of the check Cell a policy may have
    bound: int          # size of the parameter on the Cell that limits the check
    margin: int         # traced stop of the parameter minus the traced maximum of the check Cell
    declared: str       # where the bound comes from


def bound_checks(cls):
    """param_bound_checks of a Space: {parameter: Cell}. The check Cell is a zero-argument cached
    Cell whose value is the policy's need for the parameter (a horizon, a maximum age). The sample
    fixes the offset between that value and the arguments the model then passes (``t`` up to
    ``proj_len()``, say): margin = traced stop of the parameter - traced maximum of the Cell. A
    policy is accepted when value + margin <= the parameter's bound, on every cached Cell with
    that parameter."""
    from .config import TransSpec
    spec = cls.module.spec.get_spec(cls.fqname)
    requested = spec.get(TransSpec.PARAM_BOUND_CHECKS, {})
    if not isinstance(requested, dict):
        raise ValueError(f'{cls.fqname}: param_bound_checks must be a dict')
    checks = []
    for parameter, name in requested.items():
        cell = cls.cells.get(name)
        if (cell is None or cell.is_special() or not cell.has_formula_def or cell.has_args()
                or not cell.has_typeinfo()
                or cell.get_rettype_expr(c_style=True) not in ('long long', 'double')):
            raise ValueError(f'{cls.fqname}: param_bound_checks[{parameter!r}] needs a zero-argument '
                             f'cached Cell with a numeric result, got {name!r}')
        check_max = getattr(cell._rt, 'max_ret_value', None)
        if check_max is None:
            raise ValueError(f'{cls.fqname}: the sample gave no numeric value of {name}')
        limit = None
        for signature, sizes in cls.cells_arg_sizes.items():
            if parameter not in signature:
                continue
            stop = sizes[signature.index(parameter)]
            traced = cls._cells_max_args.get(signature, {}).get(parameter)
            if traced is None:
                continue
            margin = traced[0] + 1 - check_max
            allowed = stop - margin
            if limit is None or allowed < limit[0]:
                where = cls.declared_axes.get(signature, {}).get(parameter)
                limit = allowed, stop, margin, (f'cells_param_size at {where}' if where
                                                else 'sampled, not declared')
        if limit is None:
            raise ValueError(f'{cls.fqname}: param_bound_checks[{parameter!r}]: no cached Cell has '
                             f'the integer parameter {parameter!r}')
        checks.append(BoundCheck(parameter, name, int(limit[0]), int(limit[1]), int(limit[2]), limit[3]))
    return checks


def _check_lines(cls, shape, checks):
    """Pre-pass before any projection: every policy's need against the bounds."""
    lines = ['        for coordinate in coords:']
    lines.extend(('            if not first:',))
    # Between policies only the caches the check Cells can reach are reset (all, if unknown).
    closure = _check_closure(cls, [check.cell for check in checks]) if shape.reuse else None
    lines.extend('                ' + line for line in (_restart_lines(cls, shape) if closure is None
                                                         else _reset_lines(cls, closure)))
    if closure is not None:
        lines.append('                pass')
    lines.append(f'            root.{shape.parameter} = coordinate')
    lines.append('            first = False')
    space = cls.name.removeprefix("_c_")
    for check in checks:
        message = (f'{space}: policy {{}} needs {check.parameter} for {check.cell}() = {{}} '
                   f'(sampled margin {check.margin}); the size of {check.parameter} is {check.bound} '
                   f'({check.declared}), which accepts {check.cell}() <= {check.limit}. '
                   f"Raise cells_param_size {{{{{check.parameter!r}: ...}}}} in the spec.")
        failed = (f'{space}: param_bound_checks: {check.cell}() raised for policy {{}} '
                  f'before any projection: {{!r}}')
        lines.extend((
            '            try:',
            f'                need = root.{check.cell}()',
            '            except Exception as error:',
            f'                raise RuntimeError({failed!r}.format(coordinate, error)) from error',
            f'            if need > {check.limit}:',
            f'                raise IndexError({message!r}.format(coordinate, need))',
        ))
    lines.append('        if not first:')
    lines.extend('            ' + line for line in _restart_lines(cls, shape))
    lines.append('        first = True')
    return lines


def _restart_lines(cls, shape):
    if shape.reuse:
        return list(_reset_lines(cls))
    return ['root._mx_cells.clear()', 'root._mx_roots.clear()', 'root._space = None',
            f'root = {cls.name}(self)', 'root._mx_copy_refs(self, self)', 'root._mx_roots.append(root)']


def add_methods(cls, node, shape, *, workers):
    """Add a private evaluator to the existing generated collect entry."""
    if collect_contract(node) is None:
        return node
    can_reuse = f'type(self) is {cls.name} and not self._mx_roots and not self._mx_spaces'
    callback = 'self._mx_evaluate_serial_collect' if shape is not None else 'None'
    parallel = can_reuse if shape is not None and shape.parallel_reuse else 'False'
    ordered = bool(shape and shape.ordered)
    options = cst.parse_statement(
        f'def _mx_collect_options_v2(self):\n    return {workers}, {callback}, {parallel}, {ordered}\n')
    if shape is None:
        return node.with_changes(body=node.body.with_changes(body=(*node.body.body, options)))
    lines = [
        'def _mx_evaluate_serial_collect(self, coords):',
        f'    root: {cls.name}',
        f'    coordinate: {CY_MOD}.longlong',
        f'    total: {CY_MOD}.double = 0.0',
        f'    first: {CY_MOD}.bint = True',
        f'    if type(self) is not {cls.name} or self._mx_roots or self._mx_spaces:',
        '        return None',
        '    if not _mx_sys._mx_collect_native_coords(coords):',
        '        return None',
        f'    root = {cls.name}(self)',
        *(f'    root._v_{name}.window({", ".join(f"{int(a.lower)}, {int(a.stop) - int(a.lower)}" for a in axes)})'
          for name, (axes, _, _) in sorted(getattr(cls, 'reuse_dense_windows', {}).items())
          if shape.reuse),
        '    try:',
        '        root._mx_copy_refs(self, self)',
        '        root._mx_roots.append(root)',
    ]
    if shape.private_lookups:
        for context in getattr(cls, 'native_contexts', {}):
            lines.extend((
                f"        enable = getattr(root.{context}, '_mx_enable_private_lookup', None)",
                '        if enable is not None:',
                '            enable()',
            ))
    checks = bound_checks(cls)
    cls.collect_bound_checks = checks
    if checks:
        lines.extend(_check_lines(cls, shape, checks))
    lines.extend((
        '        for coordinate in coords:',
        '            if not first:',
    ))
    lines.extend('                ' + line for line in _restart_lines(cls, shape))
    lines.extend((
        f'            root.{shape.parameter} = coordinate',
        '            if first:',
        f'                total = root.{shape.cell}()',
        '                first = False',
        '            else:',
        f'                total += root.{shape.cell}()',
        '    finally:',
        '        root._mx_cells.clear()',
        '        root._mx_roots.clear()',
        '        root._space = None',
        f'    return _mx_sys.CollectResult({{"sum": _mx_sys.CollectResult({{{shape.label!r}: total}})}})',
    ))
    helper = cst.parse_module('\n'.join(lines) + '\n').body[0]
    return node.with_changes(body=node.body.with_changes(body=(*node.body.body, helper, options)))
