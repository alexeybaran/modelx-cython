"""Emit an opt-in scalar reduction through ordinary compiled leaf Cells."""

import ast
from dataclasses import dataclass

import libcst as cst

from modelx_cython.consts import CY_MOD


@dataclass(frozen=True)
class SerialCollect:
    label: str
    cell: str
    parameter: str
    reuse: bool


def _self_attribute(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == 'self' and not node.attr.startswith('_'):
            return node.attr
    return None


def _method(node, name):
    return next((item for item in node.body
                 if isinstance(item, ast.FunctionDef) and item.name == name), None)


def _return_expr(method):
    if method and len(method.body) == 1 and isinstance(method.body[0], ast.Return):
        return method.body[0].value
    return None


def _single_dict(expr):
    if isinstance(expr, ast.Dict) and len(expr.keys) == 1:
        key = expr.keys[0]
        if isinstance(key, ast.Constant) and type(key.value) is str:
            return key.value, expr.values[0]
    return None, None


def analyze(source, module):
    """Recognize the numeric sum shape. Policy isolation is a Space contract."""
    options = {name: module.spec.get_spec(cls.fqname).get('serial_collect')
               for name, cls in module.classes.items()}
    if all(option is None for option in options.values()):
        return {}
    result = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef) or node.name not in module.classes:
            continue
        cls = module.classes[node.name]
        option = options[node.name]
        if option is None:
            continue
        if option not in ('reduce', 'reuse'):
            raise ValueError('serial_collect must be reduce or reuse')
        shape = _match(node, cls, module, option == 'reuse')
        if shape is not None:
            result[node.name] = shape
    return result


def _match(node, cls, module, reuse):
    if (len(node.bases) != 1 or ast.unparse(node.bases[0]) != '_mx_sys.BaseSpace'
            or cls.spaces or cls.is_locked or node.name in module.visitor.dynamic_ref_classes
            or _method(node, '_mx_apply_formula') is not None):
        return None
    call = _return_expr(_method(node, 'collect'))
    if (not isinstance(call, ast.Call) or ast.unparse(call.func) != '_mx_sys.collect_from_space'
            or len(call.args) != 5 or len(call.keywords) != 1
            or ast.unparse(call.args[0]) != 'self'
            or call.keywords[0].arg != 'cache'
            or ast.unparse(call.keywords[0].value) != 'self._mx_collect_cache'
            or not isinstance(call.args[2], ast.Constant)
            or any(not isinstance(item, ast.Tuple) or item.elts for item in call.args[3:])):
        return None
    parameter = call.args[2].value
    if (not isinstance(parameter, str) or parameter not in cls.refs
            or cls.refs[parameter].get_type_expr(c_style=True) != 'long long'):
        return None
    output = _return_expr(_method(node, '_f_collect_outputs'))
    mode, fields = _single_dict(output)
    label, entry = _single_dict(fields)
    if (mode != 'sum' or not label or not isinstance(entry, ast.Call)
            or entry.args or entry.keywords):
        return None
    cell = _self_attribute(entry.func)
    if (cell not in cls.cells or cls.cells[cell].has_args()
            or cls.cells[cell].get_rettype_expr(c_style=True) != 'double'
            or cls.cells[cell].body_has_closure or cls.cells[cell].called_with_kwargs):
        return None

    return SerialCollect(label, cell, parameter, reuse)


def _reset_lines(cls):
    for name, cell in cls.cells.items():
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
        '        for coordinate in coords:',
        '            if not first:',
    ]
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
            ret = ret.with_changes(value=ret.value.with_changes(args=(*ret.value.args, arg)))
            item = item.with_changes(body=item.body.with_changes(
                body=(stmt.with_changes(body=(ret,)),)))
        methods.append(item)
    return node.with_changes(body=node.body.with_changes(body=(*methods, helper)))


def enable_runtime(path):
    """Wire the opt-in evaluator into the copied export's dispatch and cache."""
    source = path.read_text(encoding='utf-8')
    signature = 'def collect_from_space(base, domain, loop_name, names, args, cache=None, evaluate=None):'
    dispatch = '    if evaluate is None:\n'
    if source.count(signature) != 1 or source.count(dispatch) != 1:
        raise ValueError('serial_collect requires the supported collect runtime')
    source = source.replace(signature, '''
def _mx_collect_native_coords(coords):
    return all(type(coord) is int and -(1 << 63) <= coord < (1 << 63) for coord in coords)


def collect_from_space(base, domain, loop_name, names, args, cache=None, evaluate=None,
                       evaluate_serial=None):'''.lstrip())
    source = source.replace(dispatch, '''    if (evaluate_serial is not None and _mx_parallel_workers == 1
            and not getattr(_mx_collect_context, 'active', False)
            and sys.gettrace() is None and sys.getprofile() is None):
        _mx_collect_context.active = True
        try:
            result = evaluate_serial(coords)
        finally:
            _mx_collect_context.active = False
        if result is not None:
            return result if cache is None else cache.setdefault(key, result)
''' + dispatch)
    path.write_text(source, encoding='utf-8')
