"""Emit an opt-in scalar reduction through ordinary compiled leaf Cells."""

from dataclasses import dataclass, replace

import libcst as cst
from libcst.metadata import ExpressionContext

from modelx_cython.consts import CY_MOD
from modelx_cython.holds import reset_lines
from .scopes import CstFacts
from .syntax import arguments, code, dotted_name, literal_value, statements, walk


@dataclass(frozen=True)
class Leaf:
    """One output of ``collect_outputs``: ``kind`` is 'scalar' (``self.cell()``, a double), 'dict'
    (``{k: self.cell(k) for k in self.<domain>}``, a double Cell of one C-integer parameter over a
    Reference) or 'python' (anything else, and the whole ``individual`` section), evaluated per item
    and merged as the export merges it."""
    mode: str
    label: str
    kind: str
    cell: str = ''
    domain: str = ''
    expression: str = ''
    reason: str = ''
    domain_expression: str = ''


@dataclass(frozen=True)
class MultiCollect:
    """collect_outputs with several outputs, ``{k: cell(k) for k in D}`` outputs or an ``individual``
    section: the native leaves are summed in C arrays, the others merged by the export's Accumulator."""
    parameter: str
    reuse: bool
    parallel_reuse: bool
    ordered: bool
    leaves: tuple
    extra: tuple = ()       # the collect's other Space parameters: (name, C type or 'object')
    factory: bool = False
    ordered_rows: bool = False
    packed_sums: bool = False
    post_factory: bool = False

    @property
    def native(self):
        return tuple(leaf for leaf in self.leaves if leaf.kind != 'python')

    @property
    def python(self):
        return tuple(leaf for leaf in self.leaves if leaf.kind == 'python')


@dataclass(frozen=True)
class SerialCollect:
    label: str
    cell: str
    parameter: str
    reuse: bool
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


def analyze(module, evaluator_args=False, ordered_rows=False, packed_sums=False):
    """Recognize the numeric sum shape. Policy isolation is a Space contract."""
    options = {name: module.spec.get_spec(cls.fqname)
               for name, cls in module.classes.items()}
    for name, spec in options.items():
        if 'serial_private_lookups' in spec:
            # audit-routes P6 (D126): every compiled root takes its lookup contexts over privately
            raise ValueError(f'{name.removeprefix("_c_")}: the spec key serial_private_lookups was removed; every compiled '
                             'root (collect and direct items) uses private lookups')
        if 'reuse_dense_windows' in spec:
            # D135: the D74 dense window of collect's reused root is removed; both routes keep an over-limit
            # Cell in the hash store
            raise ValueError(f'{name.removeprefix("_c_")}: the spec key reuse_dense_windows was removed (D135): '
                             'collect and the direct route hold a Cell above the dense limit in the same hash '
                             'store; if the Cell is read only on a regular grid, key it on the grid index '
                             '(conversion rule R9)')
    automatic = module.spec.automatic_optimizations
    if not any('serial_collect' in spec or 'parallel_collect_reuse' in spec or 'factory_collect_reuse' in spec
               for spec in options.values()) and not automatic:
        return {}
    result = {}
    for node in module.visitor.wrapper.module.body:
        if not isinstance(node, cst.ClassDef) or node.name.value not in module.classes:
            continue
        cls = module.classes[node.name.value]
        spec = options[node.name.value]
        option = spec.get('serial_collect', 'reduce' if automatic else None)
        factory_reuse = spec.get('factory_collect_reuse', False)
        if type(factory_reuse) is not bool or factory_reuse and option != 'reduce':
            raise ValueError('factory_collect_reuse requires a boolean and serial_collect reduce')
        private_refs = spec.get('factory_collect_private_refs', False)
        if type(private_refs) is not bool or private_refs and not factory_reuse:
            raise ValueError('factory_collect_private_refs requires factory_collect_reuse')
        subtotals = spec.get('factory_collect_subtotals', False)
        if type(subtotals) is not bool or subtotals and not private_refs:
            raise ValueError('factory_collect_subtotals requires factory_collect_private_refs')
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
        shape = _match(node, cls, option == 'reuse', parallel, order == 'domain', evaluator_args)
        if shape is None and private_refs and evaluator_args and packed_sums:
            shape = _match_detached_factory(node, cls, module.visitor.wrapper.module, subtotals)
            if shape is not None:
                result[node.name.value] = shape
                continue
        if shape is not None:
            if packed_sums and isinstance(shape, MultiCollect) and any(leaf.kind == 'dict' for leaf in shape.leaves):
                shape = replace(shape, packed_sums=True)
            if (ordered_rows and isinstance(shape, MultiCollect) and shape.factory and not shape.python
                    and all(leaf.mode == 'sum' for leaf in shape.leaves)):
                shape = replace(shape, ordered_rows=True)
                if factory_reuse:
                    from .factory_reuse import scalar_factory_graph, standard_lifecycle, unshadowed_builtins
                    if (scalar_factory_graph(node, cls, shape) and standard_lifecycle(node, _factory_aliases(node))
                            and unshadowed_builtins(module.visitor.wrapper.module)):
                        shape = replace(shape, reuse=True)
            result[node.name.value] = shape
    return result


def _match_detached_factory(node, cls, module, subtotals):
    from .factory_reuse import declared_aliases, detached_factory_graph, standard_lifecycle, unshadowed_builtins

    contract = collect_contract(node)
    if (contract is None or _method(node, '_mx_apply_formula') is None
            or not _isolated_leaf(node, cls, fresh=True) or bound_checks(cls)):
        return None
    parameter, extra = contract[:2]
    parameters = (parameter, *extra)
    if any(name not in cls.refs or cls.refs[name].get_type_expr(c_style=True) != 'long long'
           for name in parameters):
        return None
    leaves = _leaves(cls, _return_expr(_method(node, '_f_collect_outputs')), parameters, item_domains=True)
    if (not leaves or any(leaf.mode != 'sum' or leaf.kind == 'python' for leaf in leaves)
            or not any(leaf.kind == 'dict' for leaf in leaves)):
        return None
    domains = [_self_attribute(cst.parse_expression(leaf.domain_expression).func)
               for leaf in leaves if leaf.domain_expression]
    aliases = declared_aliases(node)
    if not (standard_lifecycle(node, aliases, remap_references=True) and unshadowed_builtins(module)
            and detached_factory_graph(node, cls, [*(leaf.cell for leaf in leaves), *domains])):
        return None
    varying = {parameter, *aliases}
    fixed_domains = (all(leaf.domain not in varying for leaf in leaves)
                     and detached_factory_graph(node, cls, domains, forbidden_refs=varying, include_factory=False))
    _report(cls, leaves)
    return MultiCollect(parameter, True, subtotals and fixed_domains, True, leaves,
                        tuple((name, 'long long') for name in extra),
                        factory=True, packed_sums=True, post_factory=True)


def _isolated_leaf(node, cls, *, fresh=False):
    """A slotted leaf ItemSpace without child Spaces, model lock, dynamic References or collect hooks:
    its policy root is exactly ``_mx_new_root_C(base)`` (D116)."""
    hooks = ('_mx_collect_attach', '_mx_collect_release')
    if (not fresh or getattr(cls, 'native_contexts', None)
            or getattr(cls, 'collect_shared_item', False)):
        hooks += ('_mx_apply_formula',)
    return not (len(node.bases) != 1 or dotted_name(node.bases[0].value) != '_mx_sys.BaseSpace'
                or cls.spaces or cls.is_locked or node.name.value in cls.visitor.dynamic_ref_classes
                or any(_method(node, hook) is not None or hook in cls.refs
                       for hook in hooks) or '_mx_apply_formula' in cls.refs)


def _factory_aliases(node):
    """A factory which validates arguments and returns References bound to those arguments."""
    formula = _method(node, '_mx_space_formula')
    if formula is None:
        return None
    params = {p.name.value for p in formula.params.params if p.name.value != 'self'}
    body = list(statements(formula.body))
    if not body or not isinstance(body[-1], cst.Return):
        return None
    for statement in body[:-1]:
        if not isinstance(statement, cst.If) or statement.orelse is not None:
            return None
        branch = list(statements(statement.body))
        if len(branch) != 1 or not isinstance(branch[0], cst.Raise):
            return None
    label, refs = _single_dict(body[-1].value)
    if label != 'refs' or not isinstance(refs, cst.Dict):
        return None
    names = []
    for element in refs.elements:
        if (not isinstance(element, cst.DictElement) or not isinstance(element.key, cst.SimpleString)
                or type(element.key.evaluated_value) is not str or not isinstance(element.value, cst.Name)
                or element.value.value not in params):
            return None
        names.append(element.key.evaluated_value)
    return names


def _match(node, cls, reuse, parallel_reuse, ordered=False, evaluator_args=False):
    factory = _method(node, '_mx_apply_formula') is not None
    aliases = _factory_aliases(node) if factory else set()
    if (aliases is None or factory and (reuse or parallel_reuse or bound_checks(cls))
            or not _isolated_leaf(node, cls, fresh=factory)):
        return None
    contract = collect_contract(node)
    if contract is None:
        return None
    parameter = contract[0]
    if (not isinstance(parameter, str) or parameter not in cls.refs
            or cls.refs[parameter].get_type_expr(c_style=True) != 'long long'):
        return None
    extra = []
    for name in contract[1]:            # the collect's other Space parameters, typed from the trace
        if name not in cls.refs:
            return None
        kind = cls.refs[name].get_type_expr(c_style=True)
        extra.append((name, kind if kind in ('long long', 'double') else 'object'))
    output = _return_expr(_method(node, '_f_collect_outputs'))
    mode, fields = _single_dict(output)
    label, entry = _single_dict(fields)
    if not factory and not extra and mode == 'sum' and label and _scalar_cell(cls, entry):
        _report(cls, (Leaf('sum', label, 'scalar', cell=_self_attribute(entry.func)),))
        return SerialCollect(label, _self_attribute(entry.func), parameter, reuse, parallel_reuse, ordered)
    leaves = _leaves(cls, output, (parameter, *contract[1]))
    if leaves is None:
        return None
    if factory and any(leaf.kind == 'python' or leaf.domain in aliases for leaf in leaves):
        return None
    _report(cls, leaves)
    # The export's ordered worker path folds Python sums one policy at a time.
    # Private chunk totals would change the grouping of arbitrary Python addition.
    if any(leaf.mode == 'sum' and leaf.kind == 'python' for leaf in leaves):
        parallel_reuse = False
    shape = MultiCollect(parameter, reuse, parallel_reuse, ordered, leaves, tuple(extra), factory)
    if not shape.native:
        return None
    if extra and not evaluator_args:
        from .collect_adapter import PROTOCOL, _VERSION
        raise ValueError(
            f'{cls.name.removeprefix("_c_")}: its collect takes other Space parameters '
            f'({", ".join(name for name, _ in extra)}), which needs the export protocol {PROTOCOL} extension '
            f'(_mx_collect_evaluator_args in _mx_sys.py, beside {_VERSION} = {PROTOCOL}). '
            'Re-export the model with a modelx that writes it.')
    return shape


def _plain_cell(cls, name):
    cell = cls.cells.get(name)
    return (cell is not None and not cell.body_has_closure and not cell.called_with_kwargs
            and cell.get_rettype_expr(c_style=True) == 'double')


def _scalar_cell(cls, entry):
    """``self.cell()``: a zero-argument Cell with a double result."""
    if arguments(entry, count=0) is None:
        return False
    name = _self_attribute(entry.func)
    return name is not None and _plain_cell(cls, name) and not cls.cells[name].has_args()


def _dict_leaf(cls, entry, parameters, *, item_domains=False):
    """``{k: self.cell(k) for k in self.<domain>}``: (cell, domain) or the reason it is not native."""
    if not isinstance(entry, cst.DictComp):
        return 'not a call of a zero-argument double Cell or {k: cell(k) for k in D}'
    part = entry.for_in
    if (part.inner_for_in is not None or part.ifs or part.asynchronous is not None
            or not isinstance(part.target, cst.Name) or not isinstance(entry.key, cst.Name)
            or entry.key.value != part.target.value):
        return 'the comprehension is not {k: cell(k) for k in D} (one for clause, no condition, the key is k)'
    variable = part.target.value
    value = entry.value
    args = arguments(value, count=1)
    if args is None or not isinstance(args[0], cst.Name) or args[0].value != variable:
        return 'the value is not cell(k) with k its only argument'
    name = _self_attribute(value.func)
    if name is None or not _plain_cell(cls, name):
        return 'the value is not a call of a double Cell'
    cell = cls.cells[name]
    if len(cell.params) != 1 or cell.get_argtype_expr(cell.params[0], c_style=True) != 'long long':
        return f'{name} does not take one C-integer argument'
    domain = _self_attribute(part.iter)
    if domain is None and item_domains:
        # A detached evaluator runs after the factory on the actual item.
        # Evaluate the original domain there in formula order.
        call = part.iter
        if isinstance(call, cst.Call):
            target = cls.cells.get(_self_attribute(call.func))
            args = arguments(call, count=len(target.params)) if target is not None else None
            if (args is not None and not target.called_with_kwargs and all(
                    (attr := _self_attribute(arg)) in parameters and attr in cls.refs
                    and cls.refs[attr].get_type_expr(c_style=True) == 'long long'
                    and target.get_argtype_expr(param, c_style=True) == 'long long'
                    for arg, param in zip(args, target.params, strict=True))):
                return name, '', code(call)
    if domain is None or domain in cls.cells:
        return ('the domain is not a Reference of the Space (a Cell may give a different domain to each '
                'item)')
    if domain not in cls.refs:
        return 'the domain is not a Reference of the Space'
    if domain in parameters:
        return 'a Space parameter domain must be evaluated on the policy item'
    return name, domain, ''


class _Root(cst.CSTTransformer):
    def __init__(self, receiver_reads):
        self.receiver_reads = receiver_reads

    def leave_Name(self, original, updated):
        return updated.with_changes(value='root') if original in self.receiver_reads else updated


def _python_leaf(mode, label, entry, reason):
    facts = CstFacts(entry)
    # Inline only expressions whose lexical names are independent of evaluator
    # locals and which do not rebind or inspect the formula's frame.
    generated = {'root', 'coords', 'args', 'coordinate', 'first', 'need', 'error'}
    introspection = {'dir', 'eval', 'exec', 'globals', 'locals', 'super', 'vars'}
    if (facts.reads & (generated | introspection) or {'root', 'self'} & facts.stores.keys()
            or any(name.startswith('_mx_') for name in facts.reads)
            or any(isinstance(node, cst.NamedExpr) for node in walk(entry))):
        return None
    keywords = {node.keyword for node in walk(entry) if isinstance(node, cst.Arg)}
    receiver_reads = {node for node, context in facts.contexts.items()
                      if isinstance(node, cst.Name) and node.value == 'self'
                      and context is ExpressionContext.LOAD and node not in keywords}
    return Leaf(mode, label, 'python', expression=code(entry.visit(_Root(receiver_reads))), reason=reason)


def _leaves(cls, output, parameters, *, item_domains=False):
    """The outputs of ``collect_outputs`` in source order, or None when it is not a literal
    ``{'individual': ..., 'sum': {name: ...}}``."""
    if not isinstance(output, cst.Dict) or not output.elements:
        return None
    leaves, modes = [], set()
    for element in output.elements:
        if not isinstance(element, cst.DictElement):
            return None
        try:
            mode = literal_value(element.key)
        except (ValueError, TypeError):
            return None
        if mode not in ('individual', 'sum') or mode in modes:
            return None
        modes.add(mode)
        if mode == 'individual':
            leaf = _python_leaf(mode, '', element.value, 'individual outputs are kept per item')
            if leaf is None:
                return None
            leaves.append(leaf)
            continue
        fields = element.value
        if not isinstance(fields, cst.Dict) or not fields.elements:
            return None
        labels = set()
        for item in fields.elements:
            if not isinstance(item, cst.DictElement):
                return None
            try:
                label = literal_value(item.key)
            except (ValueError, TypeError):
                return None
            if type(label) is not str or label in labels:
                return None
            labels.add(label)
            entry = item.value
            if _scalar_cell(cls, entry):
                leaves.append(Leaf(mode, label, 'scalar', cell=_self_attribute(entry.func)))
                continue
            found = _dict_leaf(cls, entry, parameters, item_domains=item_domains)
            if isinstance(found, tuple):
                cell, domain, expression = found
                leaves.append(Leaf(mode, label, 'dict', cell=cell, domain=domain, domain_expression=expression))
            else:
                leaf = _python_leaf(mode, label, entry, found)
                if leaf is None:
                    return None
                leaves.append(leaf)
    return tuple(leaves)


def _report(cls, leaves):
    """Build report: each output of collect_outputs, native or not and why (on the collect_outputs Cell)."""
    cell = cls.cells.get('collect_outputs')
    if cell is not None:
        cell.collect_leaves = [{'mode': leaf.mode, 'output': leaf.label, 'native': leaf.kind != 'python',
                                **({'reason': leaf.reason} if leaf.kind == 'python' else {})}
                               for leaf in leaves]


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
    # R-4: what the item holds from shared inputs goes with the per-policy caches
    yield from reset_lines(cls)
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
        elif getattr(cls, 'collect_shared_item', False):
            if cell.uses_vector_cache:
                from .vectors import cache_constructor
                yield f'root._v_{name} = {cache_constructor(cell)}'
            else:
                yield f'root._v_{name}.clear()'
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
    if shape.reuse:     # what the check and the collected Cell reach (everything when unknown)
        both = _closure(cls, shape, [check.cell for check in checks])
        lines.extend('            ' + line for line in _reset_lines(cls, both))
    else:
        lines.extend('            ' + line for line in _restart_lines(cls, shape))
    lines.append('            pass')
    lines.append('        first = True')
    return lines


def _closure(cls, shape, extra=()):
    """What the collected Cells (and ``extra``) reach; None (everything) when an output is evaluated
    as Python code."""
    if isinstance(shape, MultiCollect):
        if shape.python:
            return None
        return _check_closure(cls, [*dict.fromkeys(leaf.cell for leaf in shape.native), *extra])
    return _check_closure(cls, [shape.cell, *extra])


def _restart_lines(cls, shape):
    if shape.reuse and getattr(shape, 'factory', False):
        # Dependencies and the non-policy coordinates are fixed during collect.
        # The loop rebinds the coordinate and reruns the validated alias factory.
        fresh = replace(shape, reuse=False)
        return ['if _mx_reuse_private and sys.gettrace() is None and sys.getprofile() is None:',
                *('    ' + line for line in _reset_lines(cls)),
                '    pass',
                'else:', *('    ' + line for line in _restart_lines(cls, fresh))]
    if shape.reuse:
        # D107: collect computes the collected Cell only; between policies it resets what that Cell can
        # reach (static closure; everything when unknown). The reused root is private to this entry.
        return list(_reset_lines(cls, _closure(cls, shape)))
    from .item_roots import new_root, release_root
    return [f'{release_root(cls.name)}(root)', f'root = {new_root(cls.name)}(self)', *_bind_extra_lines(shape)]


def _bind_extra_lines(shape):
    if not isinstance(shape, MultiCollect):
        return []
    return [f'root.{name} = args[{position}]' for position, (name, _) in enumerate(shape.extra)]


def _detached_numeric_methods(cls, source):
    """Gather fixed-width financial Cells on the actual policy root after its factory."""
    contract = collect_contract(source)
    if contract is None or cls.is_locked or not _isolated_leaf(source, cls, fresh=True):
        return ()
    output = _return_expr(_method(source, '_f_collect_outputs'))
    leaves = _leaves(cls, output, (contract[0], *contract[1]), item_domains=True)
    if (not leaves or any(leaf.mode != 'sum' or leaf.kind == 'python' for leaf in leaves)
            or not any(leaf.kind == 'dict' for leaf in leaves)):
        return ()
    c = CY_MOD
    lines = [
        'def _mx_collect_numeric_output_v1(self, layouts):',
        f'    if type(self) is not {cls.name} or self._has_collect_outputs:',
        '        return None',
        '    from . import _mx_collect_semantics as semantics',
        f'    buffer: {c}.double[::1]',
        f'    i: {c}.Py_ssize_t',
        f'    key: {c}.longlong',
        '    outputs = {}',
    ]
    for leaf in leaves:
        if leaf.kind == 'scalar':
            lines.append(f'    outputs[{leaf.label!r}] = self.{leaf.cell}()')
        else:
            lines += [
                f'    domain = {leaf.domain_expression or "self." + leaf.domain}',
                '    keys = semantics.numeric_keys(domain, layouts)',
                '    if keys is None:',
                f'        outputs[{leaf.label!r}] = {{k: self.{leaf.cell}(k) for k in domain}}',
                '    else:',
                f'        packed = semantics.PackedFloatSeries(keys, {cls.name}._mx_collect_add_values_v1)',
                '        buffer = packed.values',
                '        for i in range(len(keys)):',
                '            key = keys[i]',
                f'            buffer[i] = self.{leaf.cell}(key)',
                f'        outputs[{leaf.label!r}] = packed',
            ]
    lines.append("    return semantics.detached_numeric_sum(outputs, (_mx_sys.BaseMxObject,))")
    return (*cst.parse_module('\n'.join(lines) + '\n').body, *_numeric_add_method())


def _numeric_add_method():
    c = CY_MOD
    lines = [
        '@staticmethod',
        f'def _mx_collect_add_values_v1(total: {c}.double[::1], values: {c}.double[::1]):',
        f'    i: {c}.Py_ssize_t',
        '    if len(total) != len(values):',
        "        raise ValueError('incompatible numeric collect widths')",
        '    for i in range(len(values)):',
        '        total[i] += values[i]',
    ]
    return cst.parse_module('\n'.join(lines) + '\n').body


def _detached_factory_evaluator(cls, shape):
    from .item_roots import new_root, release_root

    c = CY_MOD
    assignments = '\n'.join('            ' + line for line in _bind_extra_lines(shape))
    resets = '\n'.join('                ' + line for line in _reset_lines(cls))
    bindings = ', '.join([*(f'{name}=args[{index}]' for index, (name, _) in enumerate(shape.extra)),
                          f'{shape.parameter}=coordinate'])
    arg_check = f"""    if (len(args) != {len(shape.extra)}
            or not _mx_sys._mx_collect_native_coords(args)):
        return None
""" if shape.extra else ''
    return f"""\
def _mx_evaluate_serial_collect(self, coords{', args' if shape.extra else ''}):
    from . import _mx_collect_semantics as semantics
    from modelx_cython.runtime.factory_reuse import canonical_factory_builtins
    import sys
    if (type(self) is not {cls.name} or self._mx_roots or self._mx_spaces or not coords
            or not _mx_sys._mx_collect_native_coords(coords)
            or sys.gettrace() is not None or sys.getprofile() is not None or not canonical_factory_builtins()):
        return None
{arg_check}    root: {cls.name} = {new_root(cls.name)}(self)
    coordinate: {c}.longlong
    first: {c}.bint = True
    layouts = {{}}
    accumulator = semantics.Accumulator(_mx_sys._PendingCollectSum)
    try:
        for coordinate in coords:
            if not first:
{resets}
                root._mx_copy_refs(self, self)
{assignments}
            root.{shape.parameter} = coordinate
            root._mx_apply_formula({bindings})
            outputs = root._mx_collect_numeric_output_v1(layouts)
            accumulator.add(coordinate, outputs)
            first = False
        return _mx_sys.CollectResult(accumulator.result()._fields)
    finally:
        {release_root(cls.name)}(root)
"""


def add_methods(cls, node, shape, *, workers, source=None):
    """Add a private evaluator to the existing generated collect entry."""
    if collect_contract(node) is None:
        return node
    can_reuse = f'type(self) is {cls.name} and not self._mx_roots and not self._mx_spaces'
    callback = 'self._mx_evaluate_serial_collect' if shape is not None else 'None'
    parallel = can_reuse if shape is not None and shape.parallel_reuse else 'False'
    # an individual section is returned in domain order whatever order the parallel chunks finish in
    ordered = bool(shape and (shape.ordered or isinstance(shape, MultiCollect)
                              and any(leaf.mode == 'individual' for leaf in shape.leaves)))
    takes_args = ', True' if isinstance(shape, MultiCollect) and shape.extra else ''
    options = cst.parse_statement(
        f'def _mx_collect_options_v2(self):\n    return {workers}, {callback}, {parallel}, {ordered}{takes_args}\n')
    shared = _shared_root_methods(cls, node)
    numeric = _detached_numeric_methods(cls, source) if source is not None else ()
    if getattr(shape, 'post_factory', False) and not numeric:
        shape = None
        options = cst.parse_statement(f'def _mx_collect_options_v2(self):\n    return {workers}, None, False, False\n')
    if not numeric and isinstance(shape, MultiCollect) and shape.packed_sums:
        numeric = _numeric_add_method()
    shared = (*shared, *numeric)
    if shape is None:
        return node.with_changes(body=node.body.with_changes(body=(*node.body.body, *shared, options)))
    from .item_roots import new_root, release_root     # D116: shared with the direct route
    if isinstance(shape, MultiCollect):
        if shape.post_factory:
            helpers = cst.parse_module(_detached_factory_evaluator(cls, shape)).body
            return node.with_changes(body=node.body.with_changes(body=(*node.body.body, *helpers, *shared, options)))
        helpers = list(cst.parse_module('\n'.join(_multi_lines(cls, shape)) + '\n').body)
        if shape.ordered_rows:
            helpers.extend(cst.parse_module('\n'.join(_multi_lines(cls, shape, row_mode=True)) + '\n').body)
            helpers.extend(cst.parse_module(_ordered_row_hooks()).body)
        return node.with_changes(body=node.body.with_changes(body=(*node.body.body, *helpers, *shared, options)))
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
        f'    root = {new_root(cls.name)}(self)',
        '    try:',
    ]
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
        f'        {release_root(cls.name)}(root)',
        f'    return _mx_sys.CollectResult({{"sum": _mx_sys.CollectResult({{{shape.label!r}: total}})}})',
    ))
    helper = cst.parse_module('\n'.join(lines) + '\n').body[0]
    return node.with_changes(body=node.body.with_changes(body=(*node.body.body, helper, *shared, options)))


def _shared_root_methods(cls, node):
    """audit-routes P3: the export's generic collect loop (no evaluator, tracing, nested collect,
    non-native coordinates, workers without reuse) builds and releases each policy root through the
    same generated functions as the evaluator and the direct route (D116), when the class offers them:
    ``_mx_collect_new_root()`` returns ``_mx_new_root_C(self)`` under the evaluator's own condition, else
    None (the export then keeps its generic walk); ``_mx_collect_release_root(root)`` releases it."""
    # The generic dispatcher assigns parameters and executes the Space factory on
    # every new root. Only the allocation and the release of that root are shared.
    if node.name.value not in cls.visitor.itemspace_classes or not _isolated_leaf(node, cls, fresh=True):
        return ()
    from .item_roots import new_root, release_item
    return tuple(cst.parse_module(
        'def _mx_collect_new_root(self):\n'
        f'    if type(self) is {cls.name} and not self._mx_roots and not self._mx_spaces:\n'
        f'        return {new_root(cls.name)}(self)\n'
        '    return None\n'
        '\n'
        'def _mx_collect_release_root(self, root):\n'
        f'    {release_item(cls.name)}(root)\n').body)


def _ordered_row_hooks():
    return f'''def _mx_prepare_ordered_rows_v1(self, coords, args, size, workers):
    # Numeric buffers: one key vector and total, one batch per worker and a retiring batch.
    limit = 1048576 // (size * (workers + 1) + 2)
    fields = self._mx_fill_ordered_rows(coords, args, None, None, limit)
    if fields is None:
        return None
    width = sum(1 if keys is None else len(keys) for _, keys in fields)
    if not 0 < width <= limit:
        return None
    return _mx_sys._OrderedNumericRows(fields, self._mx_fill_ordered_rows, self._mx_merge_ordered_rows, args, width)

def _mx_merge_ordered_rows(self, total: {CY_MOD}.double[::1], rows: {CY_MOD}.double[::1],
                           count: {CY_MOD}.Py_ssize_t, started: {CY_MOD}.bint):
    i: {CY_MOD}.Py_ssize_t
    j: {CY_MOD}.Py_ssize_t
    width: {CY_MOD}.Py_ssize_t = len(total)
    if not width or count < 0 or count > len(rows) // width:
        raise ValueError("invalid numeric collect block")
    for i in range(count):
        if started:
            for j in range(width):
                total[j] += rows[i * width + j]
        else:
            for j in range(width):
                total[j] = rows[i * width + j]
            started = True
'''


def _multi_lines(cls, shape, *, row_mode=False):
    """The evaluator of a MultiCollect. Outputs are evaluated item by item in source order (the order
    ``collect_outputs`` evaluates them). A native output adds in C: the first item assigns and each
    later one adds, as the export's merge does, so a double sums bit for bit as there. The others are
    collected per item and merged by the export's Accumulator. The domain of a dict output is read once
    from the Reference (every item shares it); a domain that is not a tuple, list or range of ints
    within 64 bits, or no items, leaves the whole collect to the export."""
    from .item_roots import new_root, release_root
    c = CY_MOD
    leaves = shape.leaves
    packed = shape.packed_sums and not row_mode
    lines = [
        ("def _mx_fill_ordered_rows(self, coords, args, fields, buffer, limit):" if row_mode else
         f"def _mx_evaluate_serial_collect(self, coords{', args' if shape.extra else ''}):"),
        f'    root: {cls.name}',
        f'    coordinate: {c}.longlong',
        f'    first: {c}.bint = True',
        f'    _mx_i: {c}.Py_ssize_t',
    ]
    if shape.reuse and shape.factory:
        lines.extend(['    import sys',
                      '    from modelx_cython.runtime.factory_reuse import canonical_factory_builtins',
                      '    _mx_reuse_private = sys.gettrace() is None and sys.getprofile() is None and canonical_factory_builtins()'])
    if row_mode:
        lines += [f'    _mx_out: {c}.double[::1]', f'    _mx_pos: {c}.Py_ssize_t = 0',
                  f'    _mx_done: {c}.Py_ssize_t = 0', '    _mx_width = 0']
    elif packed:
        lines += ['    from array import array', '    from . import _mx_collect_semantics as _mx_cs']
    for index, leaf in enumerate(leaves):
        if leaf.kind == 'scalar':
            lines.append(f'    _mx_t{index}: {c}.double = 0.0')
        elif leaf.kind == 'dict':
            lines.append(f'    _mx_n{index}: {c}.Py_ssize_t = 0')
            if row_mode:
                lines.append(f'    _mx_k{index}: {c}.longlong[::1]')
            else:
                lines.append(f'    _mx_k{index}: {c}.p_longlong = {c}.NULL')
                lines.append(f'    _mx_a{index}: {c}.double[::1]' if packed else
                             f'    _mx_a{index}: {c}.p_double = {c}.NULL')
    lines += [
        f'    if type(self) is not {cls.name} or self._mx_roots or self._mx_spaces:',
        '        return None',
        '    if not coords or not _mx_sys._mx_collect_native_coords(coords):',
        '        return None',
    ]
    if shape.extra:     # values of the other parameters the traced type holds, else the export's path
        lines.append(f'    if len(args) != {len(shape.extra)}:')
        lines.append('        return None')
        for position, (name, kind) in enumerate(shape.extra):
            if kind == 'long long':
                lines += [f'    if type(args[{position}]) is not int or not '
                          f'(-9223372036854775808 <= args[{position}] <= 9223372036854775807):',
                          '        return None']
            elif kind == 'double':
                lines += [f'    if type(args[{position}]) is not float:', '        return None']
    for index, leaf in enumerate(leaves):
        if leaf.kind == 'dict':
            containers = '(tuple,)' if shape.factory else '(tuple, list)'
            domain_lines = [f'    _mx_d{index} = self.{leaf.domain}']
            if row_mode:
                domain_lines += [f'    if type(_mx_d{index}) not in (tuple, range):',
                          '        return None',
                          '    try:', f'        _mx_width += len(_mx_d{index})',
                          '    except OverflowError:', '        return None',
                          '    if _mx_width > limit:', '        return None']
            domain_lines += [
                f'    if not (type(_mx_d{index}) is range or type(_mx_d{index}) in {containers}',
                f'            and all(type(_mx_x) is int for _mx_x in _mx_d{index})):',
                '        return None',
                f'    _mx_d{index} = tuple(_mx_d{index})',
                f'    if any(_mx_x < -9223372036854775808 or _mx_x > 9223372036854775807 for _mx_x in _mx_d{index}):',
                '        return None',
                f'    _mx_n{index} = len(_mx_d{index})',
            ]
            if row_mode:
                lines.append('    if fields is None:')
                lines.extend('    ' + line for line in domain_lines)
                lines += ['    else:', f'        _mx_k{index} = fields[{index}]',
                          f'        _mx_n{index} = len(_mx_k{index})', f'        _mx_width += _mx_n{index}']
            else:
                lines.extend(domain_lines)
        elif row_mode:
            lines.append('    _mx_width += 1')
    if row_mode:
        columns = [f'({leaf.label!r}, ' + (f'_mx_d{i})' if leaf.kind == 'dict' else 'None)')
                   for i, leaf in enumerate(leaves)]
        lines += ['    if fields is None:', f'        return ({", ".join(columns)},)',
                  '    if _mx_width != limit or len(buffer) != len(coords) * _mx_width:',
                  '        raise ValueError("invalid numeric collect storage")',
                  '    _mx_out = buffer']
    if shape.python:
        if not packed:
            lines.append('    from . import _mx_collect_semantics as _mx_cs')
        lines.append('    _mx_accumulated = _mx_cs.Accumulator(_mx_sys.CollectResult)')
    lines += [f'    root = {new_root(cls.name)}(self)', '    try:']
    lines.extend('        ' + line for line in _bind_extra_lines(shape))
    for index, leaf in enumerate(leaves):
        if leaf.kind == 'dict' and not row_mode:
            lines.append(f'        _mx_k{index} = {c}.cast({c}.p_longlong, _mx_collect_malloc(({c}.sizeof({c}.longlong)) * (_mx_n{index} + 1)))')
            if packed:
                lines += [
                    f'        if _mx_k{index} == {c}.NULL:',
                    '            raise MemoryError()',
                    f"        _mx_buffer{index} = array('d', [0.0]) * _mx_n{index}",
                    f'        _mx_a{index} = _mx_buffer{index}',
                ]
            else:
                lines += [
                    f'        _mx_a{index} = {c}.cast({c}.p_double, _mx_collect_malloc(({c}.sizeof({c}.double)) * (_mx_n{index} + 1)))',
                    f'        if _mx_k{index} == {c}.NULL or _mx_a{index} == {c}.NULL:',
                    '            raise MemoryError()',
                ]
            lines += [
                f'        for _mx_i in range(_mx_n{index}):',
                f'            _mx_k{index}[_mx_i] = _mx_d{index}[_mx_i]',
            ]
    checks = bound_checks(cls)
    cls.collect_bound_checks = checks
    if checks:
        lines.extend(_check_lines(cls, shape, checks))
    lines += ['        for coordinate in coords:', '            if not first:']
    lines.extend('                ' + line for line in _restart_lines(cls, shape))
    lines.append(f'            root.{shape.parameter} = coordinate')
    if shape.factory:
        bindings = [f'{name}=args[{i}]' for i, (name, _) in enumerate(shape.extra)]
        bindings.append(f'{shape.parameter}=coordinate')
        lines.append(f'            root._mx_apply_formula({", ".join(bindings)})')
    if shape.python:
        lines.append('            _mx_sum = {}')
    for index, leaf in enumerate(leaves):
        if row_mode and leaf.kind == 'scalar':
            lines += [f'            _mx_out[_mx_pos] = root.{leaf.cell}()', '            _mx_pos += 1']
        elif row_mode and leaf.kind == 'dict':
            lines += [f'            for _mx_i in range(_mx_n{index}):',
                      f'                _mx_out[_mx_pos] = root.{leaf.cell}(_mx_k{index}[_mx_i])',
                      '                _mx_pos += 1']
        elif leaf.kind == 'scalar':
            lines += ['            if first:',
                      f'                _mx_t{index} = root.{leaf.cell}()',
                      '            else:',
                      f'                _mx_t{index} += root.{leaf.cell}()']
        elif leaf.kind == 'dict':
            lines += ['            if first:',
                      f'                for _mx_i in range(_mx_n{index}):',
                      f'                    _mx_a{index}[_mx_i] = root.{leaf.cell}(_mx_k{index}[_mx_i])',
                      '            else:',
                      f'                for _mx_i in range(_mx_n{index}):',
                      f'                    _mx_a{index}[_mx_i] += root.{leaf.cell}(_mx_k{index}[_mx_i])']
        elif leaf.mode == 'individual':
            lines.append(f'            _mx_individual = ({leaf.expression})')
        else:
            lines.append(f'            _mx_sum[{leaf.label!r}] = ({leaf.expression})')
    if shape.python:
        modes = list(dict.fromkeys(leaf.mode for leaf in shape.python))
        parts = ', '.join(f"'individual': _mx_individual" if mode == 'individual' else "'sum': _mx_sum"
                          for mode in modes)
        lines.append(f'            _mx_accumulated.add(coordinate, _mx_cs.prepare_output({{{parts}}}, '
                     '(_mx_sys.BaseMxObject,)))')
    lines += ['            first = False']
    if row_mode:
        lines.append('            _mx_done += 1')
    # the result, in the order collect_outputs writes its sections and outputs
    if shape.python:
        lines.append('        _mx_python = _mx_accumulated.result()._fields')
    sections = []
    for mode in dict.fromkeys(leaf.mode for leaf in leaves):
        if mode == 'individual':
            sections.append("'individual': _mx_python['individual']")
            continue
        fields = []
        for index, leaf in enumerate(leaves):
            if leaf.mode != 'sum':
                continue
            if leaf.kind == 'scalar':
                fields.append(f'{leaf.label!r}: _mx_t{index}')
            elif leaf.kind == 'dict':
                if packed:
                    fields.append(f'{leaf.label!r}: _mx_cs.PackedFloatSeries.from_buffer('
                                  f'_mx_d{index}, _mx_buffer{index}, {cls.name}._mx_collect_add_values_v1)')
                else:
                    fields.append(f'{leaf.label!r}: {{_mx_d{index}[_mx_i]: _mx_a{index}[_mx_i] '
                                  f'for _mx_i in range(_mx_n{index})}}')
            else:
                fields.append(f"{leaf.label!r}: _mx_python['sum']._fields[{leaf.label!r}]")
        result_type = '_PendingCollectSum' if packed else 'CollectResult'
        sections.append(f"'sum': _mx_sys.{result_type}({{{', '.join(fields)}}})")
    if row_mode:
        lines += ['        return _mx_done, None', '    except BaseException as _mx_error:',
                  '        return _mx_done, _mx_error']
    else:
        lines.append(f"        return _mx_sys.CollectResult({{{', '.join(sections)}}})")
    lines.append('    finally:')
    for index, leaf in enumerate(leaves):
        if leaf.kind == 'dict' and not row_mode:
            lines.append(f'        _mx_collect_free(_mx_k{index})')
            if not packed:
                lines.append(f'        _mx_collect_free(_mx_a{index})')
    lines.append(f'        {release_root(cls.name)}(root)')
    return lines
