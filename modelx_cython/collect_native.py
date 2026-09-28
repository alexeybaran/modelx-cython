"""Generate a native traversal for an exact, numeric collect sum.

This pass changes only exported classes with one numeric sum leaf. The leaf's
expression is moved to one typed function, used by both collect_outputs and
the native traversal. Other collect contracts keep the exported Python path.
"""

import ast
import json
import re


_NUMERIC = {'_mx_cy.double': 'double', '_mx_cy.longlong': 'long long'}


def _method(cls, name):
    return next((node for node in cls.body
                 if isinstance(node, ast.FunctionDef) and node.name == name), None)


def _string(node):
    return node.value if isinstance(node, ast.Constant) and type(node.value) is str else None


def _collect_contract(cls):
    collect = _method(cls, 'collect')
    formula = _method(cls, '_f_collect_outputs')
    if collect is None or formula is None or len(formula.body) != 1:
        return None
    ret = formula.body[0]
    if not isinstance(ret, ast.Return) or not isinstance(ret.value, ast.Dict):
        return None
    outer = ret.value
    if len(outer.keys) != 1 or _string(outer.keys[0]) != 'sum':
        return None
    leaves = outer.values[0]
    if not isinstance(leaves, ast.Dict) or len(leaves.keys) != 1:
        return None
    output_name = _string(leaves.keys[0])
    expression = leaves.values[0]
    if output_name is None:
        return None
    # A typed Cell call is the first supported contract. Arguments may call
    # other typed numeric Cells, but arbitrary callbacks remain on the general
    # collect route.
    calls = [node for node in ast.walk(expression) if isinstance(node, ast.Call)]
    if not calls or any(not isinstance(c.func, ast.Attribute)
                        or not isinstance(c.func.value, ast.Name)
                        or c.func.value.id != 'self' or c.keywords for c in calls):
        return None
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    for call in calls:
        method = methods.get(call.func.attr)
        if method is None or method.returns is None or ast.unparse(method.returns) not in _NUMERIC:
            return None
    if any(isinstance(node, (ast.NamedExpr, ast.Await, ast.Yield, ast.Lambda,
                             ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
           for node in ast.walk(expression)):
        return None
    leaf_type = ast.unparse(methods[calls[0].func.attr].returns)
    # The first accumulator is a double. Integer results need an overflow-
    # preserving accumulator and retain the general path for now.
    if leaf_type != '_mx_cy.double':
        return None
    body = collect.body
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    call = body[0].value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and ast.unparse(call.func) == '_mx_sys.collect_from_space'
            and len(call.args) == 5 and ast.unparse(call.args[0]) == 'self'
            and isinstance(call.args[2], ast.Constant)
            and type(call.args[2].value) is str
            and isinstance(call.args[3], ast.Tuple) and not call.args[3].elts
            and isinstance(call.args[4], ast.Tuple) and not call.args[4].elts):
        return None
    # The first native route handles one generated dynamic Space without
    # nested Spaces. Nested/outer dynamic roots retain full exported traversal.
    initializer = _method(cls, '__init__')
    if (initializer is None or _method(cls, '_mx_copy_refs') is None
            or _method(cls, '_mx_assign_params') is None
            or not any(isinstance(node, ast.Assign)
                       and any(ast.unparse(target) == 'self._mx_roots'
                               for target in node.targets)
                       and isinstance(node.value, ast.List) and not node.value.elts
                       for node in initializer.body)
            or not any(
            isinstance(node, ast.Assign)
            and any(ast.unparse(target) == 'self._mx_spaces' for target in node.targets)
            and isinstance(node.value, ast.Dict) and not node.value.keys
            for node in initializer.body)):
        return None
    return output_name, expression, leaf_type, call.args[1], call.args[2].value


def transform(source, declarations, receipt_path):
    tree = ast.parse(source)
    selected = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        contract = _collect_contract(cls)
        if contract is None:
            continue
        output_name, expression, leaf_type, domain, parameter = contract
        if not re.search(r'^cdef class ' + re.escape(cls.name) + r'\(', declarations, re.M):
            continue
        if not _method(cls, '_mx_collect_attach') or not _method(cls, '_mx_collect_release'):
            continue
        output_formula = _method(cls, '_f_collect_outputs')
        output_formula.body = ast.parse(
            'return {"sum": {' + repr(output_name) + ': self._mx_collect_value()}}').body
        value_method = ast.parse('''
@_mx_cy.cfunc
@_mx_cy.exceptval(-1, check=True)
def _mx_collect_value(self) -> _mx_cy.double:
    return 0.0
''').body[0]
        value_method.returns = ast.parse(leaf_type, mode='eval').body
        value_method.body = [ast.Return(expression)]
        cls.body.append(value_method)
        old_collect = _method(cls, 'collect')
        old_collect.decorator_list.append(ast.parse(
            '@_mx_cy.locals(_mx_root=' + cls.name
            + ', _mx_coordinate=_mx_cy.longlong, _mx_total=_mx_cy.double, '
            + '_mx_value=_mx_cy.double)\ndef _mx_dummy(): pass').body[0].decorator_list[0])
        fallback = ast.unparse(old_collect.body[0].value)
        domain_text = ast.unparse(domain)
        native = '''
_mx_domain = %s
if type(self) is not %s or self._mx_roots or type(_mx_domain) not in (list, tuple):
    return %s
for _mx_x in _mx_domain:
    if type(_mx_x) is not int or not (-(1 << 63) <= _mx_x < (1 << 63)):
        return %s
_mx_coords = tuple(_mx_domain)
if not _mx_coords or len(set(_mx_coords)) != len(_mx_coords):
    return %s
_mx_scratch = {}
_mx_first = True
_mx_total = 0.0
for _mx_coordinate in _mx_coords:
    _mx_root = %s(self)
    _mx_root._mx_copy_refs(self, self)
    self._mx_assign_params(_mx_root, _mx_coordinate)
    _mx_root._mx_roots.append(_mx_root)
    _mx_root._mx_collect_attach(_mx_scratch, self)
    _mx_value = _mx_root._mx_collect_value()
    _mx_root._v_collect_outputs = {'sum': {%r: _mx_value}}
    _mx_root._has_collect_outputs = True
    if _mx_first:
        _mx_total = _mx_value
        _mx_first = False
    else:
        _mx_total += _mx_value
    _mx_root._mx_collect_release(_mx_scratch, self)
return _mx_sys.CollectResult({'sum': _mx_sys.CollectResult({%r: _mx_total})})
''' % (domain_text, cls.name, fallback, fallback, fallback, cls.name,
       output_name, output_name)
        old_collect.body = ast.parse(native).body
        cls.body.remove(value_method)
        cls.body.insert(cls.body.index(old_collect), value_method)
        marker = next(line for line in declarations.splitlines()
                      if line.startswith('cdef class ' + cls.name + '('))
        declarations = declarations.replace(
            marker, marker + '\n    cdef ' + _NUMERIC[leaf_type]
            + ' _mx_collect_value(' + cls.name + ' self) except? -1', 1)
        selected.append({'class': cls.name, 'output': output_name,
                         'type': leaf_type, 'parameter': parameter,
                         'expression': ast.unparse(expression)})
    if not selected:
        return source, declarations
    ast.fix_missing_locations(tree)
    result = ast.unparse(tree) + '\n'
    ast.parse(result)
    receipt_path.write_text(json.dumps({'schema': 1, 'selected': selected}, indent=2),
                            encoding='utf-8')
    return result, declarations
