"""Emit an opt-in scalar reduction through ordinary compiled leaf Cells."""

from dataclasses import dataclass

import libcst as cst

from modelx_cython.consts import CY_MOD
from .syntax import arguments, dotted_name, literal_value, statements


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
        order = spec.get('parallel_collect_order', 'completion')
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
    return result


def _match(node, cls, reuse, private_lookups, parallel_reuse, ordered=False):
    if (len(node.bases) != 1 or dotted_name(node.bases[0].value) != '_mx_sys.BaseSpace'
            or cls.spaces or cls.is_locked or node.name.value in cls.visitor.dynamic_ref_classes
            or any(_method(node, hook) is not None or hook in cls.refs
                   for hook in ('_mx_apply_formula', '_mx_collect_attach', '_mx_collect_release'))):
        return None
    call = _return_expr(_method(node, 'collect'))
    if (not isinstance(call, cst.Call) or dotted_name(call.func) != '_mx_sys.collect_from_space'
            or len(call.args) != 6 or any(arg.star for arg in call.args)
            or any(arg.keyword for arg in call.args[:5])
            or dotted_name(call.args[0].value) != 'self'
            or call.args[-1].keyword is None or call.args[-1].keyword.value != 'cache'
            or dotted_name(call.args[-1].value) != 'self._mx_collect_cache'
            or any(not isinstance(arg.value, cst.Tuple) or arg.value.elements for arg in call.args[3:5])):
        return None
    try:
        parameter = literal_value(call.args[2].value)
    except (ValueError, TypeError):
        return None
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


def _reset_lines(cls):
    for name, cell in cls.cells.items():
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


def add_methods(cls, node, shape):
    """Add a private evaluator to the existing generated collect entry."""
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
    lines.extend((
        '        for coordinate in coords:',
        '            if not first:',
    ))
    if shape.reuse:
        lines.extend('                ' + line for line in _reset_lines(cls))
    else:
        lines.extend((
            '                root._mx_cells.clear()',
            '                root._mx_roots.clear()',
            '                root._space = None',
            f'                root = {cls.name}(self)',
            '                root._mx_copy_refs(self, self)',
            '                root._mx_roots.append(root)',
        ))
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
    methods = []
    for item in node.body.body:
        if isinstance(item, cst.FunctionDef) and item.name.value == 'collect':
            stmt = item.body.body[0]
            ret = stmt.body[0]
            arg = cst.Arg(keyword=cst.Name('evaluate_serial'),
                          value=cst.parse_expression('self._mx_evaluate_serial_collect'))
            args = (*ret.value.args, arg)
            if shape.parallel_reuse:
                can_reuse = cst.parse_expression(
                    f'type(self) is {cls.name} and not self._mx_roots and not self._mx_spaces')
                args += (cst.Arg(keyword=cst.Name('reuse_in_parallel'), value=can_reuse),)
                if shape.ordered:
                    args += (cst.Arg(keyword=cst.Name('private_ordered'), value=cst.Name('True')),)
            ret = ret.with_changes(value=ret.value.with_changes(args=args))
            item = item.with_changes(body=item.body.with_changes(
                body=(stmt.with_changes(body=(ret,)),)))
        methods.append(item)
    return node.with_changes(body=node.body.with_changes(body=(*methods, helper)))
