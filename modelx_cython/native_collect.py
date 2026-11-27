"""Lower statically numeric Collect outputs and their typed consumers."""

from dataclasses import dataclass

import libcst as cst

from .consts import FORMULA_PREF
from .native_sources import module_name
from .syntax import arguments, code, dotted_name, literal_value, single_statement, walk


PROVIDER = 'modelx_cython._mx_native_collect'
ALIAS = module_name(PROVIDER)
TYPE = ALIAS + '.NativeCollect'
FLOAT, INTEGER, BOOL, PYTHON = range(4)
NUMERIC_TYPES = {'double': FLOAT, 'long long': INTEGER, 'bint': BOOL}


@dataclass(frozen=True)
class Field:
    name: str
    cell: str
    domain: str = None
    kind: int = FLOAT
    position: int = 0


def _literal_items(value):
    if not isinstance(value, cst.Dict) or not value.elements:
        return None
    items = {}
    for entry in value.elements:
        if not isinstance(entry, cst.DictElement):
            return None
        try:
            key = literal_value(entry.key)
        except ValueError:
            return None
        if not isinstance(key, str) or key in items:
            return None
        items[key] = entry.value
    return items


def _body_return(method):
    if method is None or not isinstance(method.body, cst.IndentedBlock):
        return None
    body = list(method.body.body)
    if body and isinstance(stmt := single_statement(body[0]), cst.Expr) and isinstance(stmt.value, cst.SimpleString):
        body.pop(0)
    stmt = single_statement(body[0]) if len(body) == 1 else None
    return stmt.value if isinstance(stmt, cst.Return) else None


def _uniform(value, methods, refs, params, varying, classes, checked=frozenset()):
    """Prove that a key expression uses the same inputs for all collected items."""
    if isinstance(value, (cst.Integer, cst.Float, cst.SimpleString)):
        return True
    if isinstance(value, cst.Name):
        return value.value in params or value.value in ('True', 'False', 'None')
    if isinstance(value, cst.Attribute):
        return (dotted_name(value.value) == 'self' and value.attr.value in refs
                and value.attr.value not in varying)
    if isinstance(value, cst.UnaryOperation):
        return _uniform(value.expression, methods, refs, params, varying, classes, checked)
    if isinstance(value, cst.BinaryOperation):
        return all(_uniform(item, methods, refs, params, varying, classes, checked)
                   for item in (value.left, value.right))
    if isinstance(value, cst.Comparison):
        return all(_uniform(item, methods, refs, params, varying, classes, checked)
                   for item in (value.left, *(target.comparator for target in value.comparisons)))
    if not isinstance(value, cst.Call) or (args := arguments(value)) is None:
        return False
    if not all(_uniform(arg, methods, refs, params, varying, classes, checked) for arg in args):
        return False
    parts = (dotted_name(value.func) or '').split('.')
    if len(parts) == 1:
        return parts[0] in ('range', 'max', 'min')
    if parts[0] != 'self':
        return False
    if len(parts) == 3:
        ref = refs.get(parts[1])
        if ref is None or parts[1] in varying or (owner := classes.get(ref.decl_type_expr)) is None:
            return False
        cell = owner.cells.get(parts[2])
        return cell is not None and not cell.is_special() and cell.has_formula_def and len(args) == len(cell.params)
    if len(parts) != 2:
        return False
    name = parts[1]
    if name in checked or (method := methods.get(FORMULA_PREF + name)) is None:
        return False
    local_params = {param.name.value for param in method.params.params[1:]}
    return _uniform(_body_return(method), methods, refs, local_params, varying, classes, checked | {name})


class _UniformArguments(cst.CSTTransformer):
    def __init__(self, params):
        self.params = params

    def leave_Attribute(self, original, updated):
        if dotted_name(original.value) == 'self' and original.attr.value in self.params:
            return cst.Name(original.attr.value)
        return updated


def shape(class_node, info, uniform_args, varying_refs):
    methods = {stmt.name.value: stmt for stmt in class_node.body.body if isinstance(stmt, cst.FunctionDef)}
    sections = _literal_items(_body_return(methods.get(FORMULA_PREF + 'collect_outputs')))
    if sections is None or sections.keys() - {'individual', 'sum'}:
        return None
    item = methods.get('_mx_collect_item')
    if item is None:
        return None
    varying = ({param.name.value for param in item.params.params[1:]} - set(uniform_args)) | set(varying_refs)
    rewrite = _UniformArguments(uniform_args)
    layouts = [(), ()]
    position = 0
    for section, expression in sections.items():
        fields = _literal_items(expression)
        if fields is None:
            return None
        layout = []
        for name, node in fields.items():
            when = None
            if isinstance(node, cst.IfExp) and isinstance(node.orelse, cst.Dict) and not node.orelse.elements:
                when, node = node.test, node.body
            domain = None
            if isinstance(node, cst.DictComp):
                loop = node.for_in
                if (not isinstance(loop.target, cst.Name) or not isinstance(node.key, cst.Name)
                        or node.key.value != loop.target.value or loop.ifs or loop.inner_for_in is not None
                        or loop.asynchronous is not None):
                    return None
                if (not _uniform(loop.iter, methods, info.refs, set(), varying, info.module.classes)
                        or when is not None and not _uniform(when, methods, info.refs, set(), varying, info.module.classes)):
                    return None
                domain = code(loop.iter.visit(rewrite))
                if when is not None:
                    domain = f'({domain} if {code(when.visit(rewrite))} else ())'
                args = arguments(node.value, count=1)
                if args is None or dotted_name(args[0]) != loop.target.value:
                    return None
                expr = node.value
            elif when is None and arguments(node, count=0) == ():
                expr = node
            else:
                return None
            parts = (dotted_name(expr.func) or '').split('.')
            if len(parts) != 2 or parts[0] != 'self':
                return None
            cell = info.cells.get(parts[1])
            if cell is None or cell.is_special() or len(cell.params) != (domain is not None):
                return None
            kind = NUMERIC_TYPES.get(cell.get_rettype_expr(c_style=True))
            if kind is None:
                if section != 'individual' or domain is not None:
                    return None
                kind = PYTHON
            if domain is not None and cell.get_argtype_expr(cell.params[0], c_style=True) != 'long long':
                return None
            layout.append(Field(name, parts[1], domain, kind, position))
            position += 1
        layouts[0 if section == 'individual' else 1] = tuple(layout)
    return tuple(layouts) if any(field.kind != PYTHON for fields in layouts for field in fields) else None


def plan(layouts):
    """Build a layout once, falling back on axes outside the native representation."""
    lines = ['    if hasattr(_mx_collect_support, "_native_result"):\n']
    named_layouts = []
    for section, fields in enumerate(layouts):
        descriptors = []
        for index, field in enumerate(fields):
            keys = 'None'
            if field.domain is not None:
                keys = f'_mx_native_keys_{section}_{index}'
                lines.append(f'        {keys} = {ALIAS}.integer_keys({field.domain})\n')
            descriptors.append(f'({field.name!r}, {keys}, {field.kind})')
        named_layouts.append(f'({", ".join(descriptors)},)' if descriptors else '()')
    keys = [f'_mx_native_keys_{section}_{index} is not None' for section, fields in enumerate(layouts)
            for index, field in enumerate(fields) if field.domain is not None]
    indent = '        '
    if keys:
        lines.append(f'{indent}if {" and ".join(keys)}:\n')
        indent += '    '
    lines.append(f'{indent}_mx_native_plan = lambda workers: {ALIAS}.NativeCollect(\n'
                 f'{indent}    _mx_coords, {", ".join(named_layouts)}, workers,\n'
                 f'{indent}    _mx_collect_support._native_result)\n')
    return ''.join(lines)


def gathering(layouts, cls_name):
    lines = ['\n', f'@_mx_cy.cfunc\ndef _mx_gather_{cls_name}(\n',
             f'        root: {cls_name}, target: {TYPE}, _mx_pos: _mx_cy.Py_ssize_t,\n',
             '        _mx_worker: _mx_cy.Py_ssize_t, prepare, live_types) -> object:\n',
             '    _mx_key: _mx_cy.longlong\n', '    _mx_index: _mx_cy.Py_ssize_t\n']
    ordered = sorted(((section == 0, index, field) for section, fields in enumerate(layouts)
                      for index, field in enumerate(fields)), key=lambda entry: entry[2].position)
    opaque = []
    for individual, index, field in ordered:
        write = 'write' if field.kind == FLOAT else 'write_int'
        if field.kind == PYTHON:
            name = f'_mx_opaque_{index}'
            opaque.append((index, field.name, name))
            lines.append(f'    {name} = root.{field.cell}()\n')
        elif field.domain is None:
            lines.append(f'    target.{write}({individual}, _mx_pos, _mx_worker, {index}, 0, root.{field.cell}())\n')
        else:
            lines.append(f'    for _mx_index, _mx_key in enumerate(target.keys({individual}, {index})):\n'
                         f'        target.{write}({individual}, _mx_pos, _mx_worker, {index}, _mx_index,\n'
                         f'                     root.{field.cell}(_mx_key))\n')
    if opaque:
        values = ', '.join(f'{key!r}: {name}' for _, key, name in opaque)
        lines.append(f'    _mx_detached = prepare({{"individual": {{{values}}}}}, live_types)["individual"]\n')
        for index, name, _ in opaque:
            lines.append(f'    target.write_object(_mx_pos, {index}, _mx_detached[{name!r}])\n')
    lines.append('    target.finish_row(_mx_worker)\n')
    lines.append('    return True\n')
    return ''.join(lines)


def _producer(call, info, shapes):
    if not isinstance(call, cst.Call):
        return None
    parts = (dotted_name(call.func) or '').split('.')
    if len(parts) != 3 or parts[0] != 'self' or parts[2] != 'collect':
        return None
    ref = info.refs.get(parts[1])
    return None if ref is None else shapes.get(ref.decl_type_expr)


def _subscript(node):
    if isinstance(node, cst.Subscript) and len(node.slice) == 1 and isinstance(node.slice[0].slice, cst.Index):
        return node.value, node.slice[0].slice.value
    return None, None


def _individual(node, info, shapes):
    base, key = _subscript(node)
    if isinstance(base, cst.Attribute) and base.attr.value == 'individual':
        layouts = _producer(base.value, info, shapes)
        if layouts is not None and layouts[0]:
            return base.value, key, layouts[0]
    return None


def bindings(class_node, info, shapes):
    """Fields of temporary Collect records returned by a Space's factory."""
    method = next((fn for fn in class_node.body.body if isinstance(fn, cst.FunctionDef)
                   and fn.name.value == '_mx_space_formula'), None)
    result = {}
    if method is None:
        return result
    for node in walk(method):
        if isinstance(node, cst.Return) and (outer := _literal_items(node.value)):
            refs = _literal_items(outer.get('refs'))
            if refs is not None:
                for name, value in refs.items():
                    if source := _individual(value, info, shapes):
                        result[name] = source[2]
    return result


class _FactoryReads(cst.CSTTransformer):
    def __init__(self, info, shapes):
        self.info = info
        self.shapes = shapes

    def leave_Return(self, original, updated):
        outer = _literal_items(updated.value)
        if outer is None or _literal_items(outer.get('refs')) is None:
            return updated
        elements = []
        for entry in updated.value.elements:
            if literal_value(entry.key) == 'refs':
                refs = []
                for ref in entry.value.elements:
                    if source := _individual(ref.value, self.info, self.shapes):
                        call, key, _ = source
                        ref = ref.with_changes(value=cst.parse_expression(
                            f'{ALIAS}.individual_at({code(call)}, {code(key)})'))
                    refs.append(ref)
                entry = entry.with_changes(value=entry.value.with_changes(elements=tuple(refs)))
            elements.append(entry)
        return updated.with_changes(value=updated.value.with_changes(elements=tuple(elements)))


def factory_reads(node, info, shapes):
    return node.with_changes(body=node.body.visit(_FactoryReads(info, shapes)))


class _FieldReads(cst.CSTTransformer):
    def __init__(self, info, shapes, refs, integer_params, result_type):
        self.info = info
        self.shapes = shapes
        self.refs = refs
        self.integer_params = integer_params
        self.result_type = result_type
        self.used = False
        self.returns = set()

    def visit_FunctionDef(self, node):
        return False

    visit_Lambda = visit_FunctionDef

    def visit_Return(self, node):
        pending = [node.value]
        while pending:
            value = pending.pop()
            if isinstance(value, cst.IfExp):
                pending.extend((value.body, value.orelse))
            else:
                self.returns.add(value)

    def integer(self, expr):
        if isinstance(expr, cst.Integer):
            return expr.evaluated_value < 1 << 63
        if isinstance(expr, cst.Name):
            return expr.value in self.integer_params
        if isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, (cst.Plus, cst.Minus)):
            return self.integer(expr.expression)
        parts = (dotted_name(expr) or '').split('.')
        if len(parts) == 2 and parts[0] == 'self':
            ref = self.info.refs.get(parts[1])
            return ref is not None and ref.get_type_expr(c_style=True) == 'long long'
        if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, (cst.Add, cst.Subtract)):
            return self.integer(expr.left) and self.integer(expr.right)
        return False

    def read(self, expr, key, index_read=False, whole_return=False):
        if (not whole_return or self.result_type not in ('double', 'long long')
                or not isinstance(expr, cst.Attribute) or key is not None and not self.integer(key)):
            return None
        receiver = expr.value
        name = expr.attr.value
        read = 'read_individual'
        parts = (dotted_name(receiver) or '').split('.')
        fields = self.refs.get(parts[1]) if len(parts) == 2 and parts[0] == 'self' else None
        if fields is not None and key is None:
            return None
        if isinstance(receiver, cst.Attribute) and receiver.attr.value == 'sum':
            layouts = _producer(receiver.value, self.info, self.shapes)
            if layouts is not None:
                read, receiver, fields = 'read_sum', receiver.value, layouts[1]
        elif fields is None and (source := _individual(receiver, self.info, self.shapes)):
            call, coordinate, fields = source
            receiver = cst.parse_expression(f'{ALIAS}.individual_at({code(call)}, {code(coordinate)})')
        if not fields:
            return None
        for index, field in enumerate(fields):
            if field.kind == PYTHON or field.name != name or (field.domain is not None) != (key is not None):
                continue
            if field.kind == FLOAT:
                if self.result_type != 'double':
                    return None
            else:
                if self.result_type != 'long long':
                    return None
                read += '_int'
            self.used = True
            return cst.parse_expression(
                f'{ALIAS}.{read}({code(receiver)}, {index}, {name!r}, '
                f'{code(key) if key is not None else "0"}, {key is not None}, {index_read})')
        return None

    def leave_Attribute(self, original, updated):
        return self.read(updated, None, whole_return=original in self.returns) or updated

    def leave_Call(self, original, updated):
        args = arguments(updated, count=1)
        return (self.read(updated.func, args[0], whole_return=original in self.returns) or updated) if args else updated

    def leave_Subscript(self, original, updated):
        receiver, key = _subscript(updated)
        return (self.read(receiver, key, True, original in self.returns) or updated) if key is not None else updated


def field_reads(node, info, shapes, refs, cells):
    params = {name for name in cells.params if cells.get_argtype_expr(name, c_style=True) == 'long long'}
    transform = _FieldReads(info, shapes, refs, params, cells.get_rettype_expr(c_style=True))
    node = node.with_changes(body=node.body.visit(transform))
    return node, transform.used
