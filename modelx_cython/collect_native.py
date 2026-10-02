"""Generate collect traversals beneath the exported Space/Cell interface.

The numeric scalar route keeps one typed formula expression. Other declared
output shapes use the same compiled Cell body and a native policy traversal,
with the exported collector's detachment and accumulation contracts.
"""

import ast
import copy
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
    if collect is None or formula is None:
        return None
    formula_body = formula.body
    if (formula_body and isinstance(formula_body[0], ast.Expr)
            and _string(formula_body[0].value) is not None):
        formula_body = formula_body[1:]
    if len(formula_body) != 1:
        return None
    ret = formula_body[0]
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
            and isinstance(call.args[3], ast.Tuple)
            and isinstance(call.args[4], ast.Tuple)
            and len(call.args[3].elts) == len(call.args[4].elts)
            and all(_string(name) is not None and _string(name) != call.args[2].value
                    for name in call.args[3].elts)):
        return None
    # The exported traversal binds these names as keywords. The native
    # traversal calls the generated binder positionally, so the two routes
    # agree only when the declared parameter order is identical.
    assign = _method(cls, '_mx_assign_params')
    expected = [call.args[2].value] + [_string(name) for name in call.args[3].elts]
    if (assign is None or [arg.arg for arg in assign.args.args[1:]] != expected
            or len(expected) != len(set(expected))):
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
    return (output_name, expression, leaf_type, call.args[1],
            call.args[2].value, call.args[4])


def _traversal_contract(cls):
    """Describe the generated collect call without constraining output types."""
    collect = _method(cls, 'collect')
    if collect is None or _method(cls, '_f_collect_outputs') is None:
        return None
    calls = [node for statement in collect.body for node in ast.walk(statement)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and ast.unparse(node.func) == '_mx_sys.collect_from_space']
    if len(calls) != 1:
        return None
    call = calls[0]
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
            or ast.unparse(call.func) != '_mx_sys.collect_from_space'
            or len(call.args) != 5 or ast.unparse(call.args[0]) != 'self'
            or _string(call.args[2]) is None
            or not isinstance(call.args[3], ast.Tuple)
            or not isinstance(call.args[4], ast.Tuple)
            or len(call.args[3].elts) != len(call.args[4].elts)):
        return None
    names = [_string(value) for value in call.args[3].elts]
    if (any(name is None for name in names) or len(names) != len(set(names))
            or call.args[2].value in names):
        return None
    initializer = _method(cls, '__init__')
    if (initializer is None or not any(
            isinstance(node, ast.Assign)
            and any(ast.unparse(target) == 'self._mx_spaces' for target in node.targets)
            and isinstance(node.value, ast.Dict)
            for node in initializer.body)):
        return None
    return collect, call


def _general_traversal(cls, call):
    """Use the exported fold rules while keeping policy selection in Cython."""
    fallback = ast.unparse(call)
    return '''
_mx_domain = %s
if type(self) is not %s or self._mx_roots:
    return %s
if isinstance(_mx_domain, (str, bytes, set, frozenset)) or iter(_mx_domain) is _mx_domain:
    return %s
_mx_coords = tuple(_mx_domain)
if not _mx_coords or len(set(_mx_coords)) != len(_mx_coords):
    return %s
_mx_names = %s
_mx_args = %s
_mx_scratch = {}
_mx_result = None
_mx_modes = None
_mx_shape = None
for _mx_coordinate in _mx_coords:
    _mx_values = dict(zip(_mx_names, _mx_args))
    _mx_values[%r] = _mx_coordinate
    _mx_root = %s(self)
    try:
        for _mx_space, _mx_template in zip(_mx_root._mx_walk(), self._mx_walk()):
            _mx_space._mx_copy_refs(_mx_template, self)
            self._mx_assign_params(_mx_space, **_mx_values)
            _mx_space._mx_roots.append(_mx_root)
            _mx_attach = getattr(_mx_space, '_mx_collect_attach', None)
            if _mx_attach is not None:
                _mx_attach(_mx_scratch, _mx_template)
        _mx_apply = getattr(_mx_root, '_mx_apply_formula', None)
        if _mx_apply is not None:
            _mx_apply(**_mx_values)
        _mx_outputs = _mx_root.collect_outputs()
        if type(_mx_outputs) is not dict or not _mx_outputs or set(_mx_outputs) - {'individual', 'sum'}:
            raise ValueError('collect_outputs must declare individual and/or sum')
        if _mx_modes is None:
            _mx_modes = tuple(_mx_outputs)
            _mx_result = {mode: None for mode in _mx_modes}
        elif tuple(_mx_outputs) != _mx_modes:
            raise ValueError('collect_outputs modes changed between policies')
        for _mx_mode, _mx_items in _mx_outputs.items():
            if type(_mx_items) is not dict or not _mx_items:
                raise TypeError('collect_outputs sections need named values')
            for _mx_name in _mx_items:
                if type(_mx_name) is not str:
                    raise TypeError('collect_outputs sections need named values')
            _mx_items = _mx_sys._mx_collect_detach(_mx_items)
            if _mx_mode == 'individual':
                _mx_new_shape = {}
                for _mx_name, _mx_item in _mx_items.items():
                    _mx_new_shape[_mx_name] = _mx_sys._mx_collect_shape(_mx_item)
                if _mx_shape is None:
                    _mx_shape = _mx_new_shape
                    _mx_result[_mx_mode] = _mx_sys.OrderedDict()
                else:
                    _mx_sys._mx_collect_check_shape(_mx_shape, _mx_items, (_mx_mode,))
                _mx_result[_mx_mode][_mx_coordinate] = _mx_sys.CollectResult(_mx_items)
            else:
                _mx_result[_mx_mode] = _mx_sys._mx_collect_sum(_mx_result[_mx_mode], _mx_items, (_mx_mode,))
    finally:
        for _mx_space, _mx_template in zip(_mx_root._mx_walk(), self._mx_walk()):
            _mx_release = getattr(_mx_space, '_mx_collect_release', None)
            if _mx_release is not None:
                _mx_release(_mx_scratch, _mx_template)
_mx_final = {}
for _mx_mode, _mx_value in _mx_result.items():
    if _mx_mode == 'individual':
        _mx_final[_mx_mode] = _mx_value
    else:
        _mx_final[_mx_mode] = _mx_sys.CollectResult(_mx_value)
return _mx_sys.CollectResult(_mx_final)
''' % (ast.unparse(call.args[1]), cls.name, fallback, fallback, fallback,
       ast.unparse(call.args[3]), ast.unparse(call.args[4]),
       call.args[2].value, cls.name)


def _preserve_collect_cache(cls, collect, original_body, original_decorators, declarations):
    """Keep the export's result-cache wrapper around its compiled miss traversal."""
    if len(original_body) == 1 and isinstance(original_body[0], ast.Return):
        return declarations
    entry = copy.deepcopy(collect)
    entry.name = '_mx_collect_execute'
    entry.args.args.insert(1, ast.arg('_mx_collect_domain', ast.Name('object', ast.Load())))
    if not (isinstance(entry.body[0], ast.Assign)
            and ast.unparse(entry.body[0].targets[0]) == '_mx_domain'):
        raise ValueError('Generated collection traversal has no domain binding')
    entry.body[0].value = ast.Name('_mx_collect_domain', ast.Load())
    entry.decorator_list = [ast.parse('@_mx_cy.cfunc\ndef entry(): pass').body[0].decorator_list[0]] + [
        node for node in collect.decorator_list
        if isinstance(node, ast.Call) and ast.unparse(node.func) == '_mx_cy.locals']
    entry.returns = ast.Name('object', ast.Load())
    parameters = collect.args.args[1:]
    defaults_from = len(parameters) - len(collect.args.defaults)
    types = {'_mx_cy.double': 'double', '_mx_cy.longlong': 'long long',
             '_mx_cy.bint': 'bint', 'object': 'object', 'str': 'str'}
    arguments = [cls.name + ' self', 'object _mx_collect_domain']
    for index, argument in enumerate(parameters):
        annotation = ast.unparse(argument.annotation) if argument.annotation else 'object'
        arguments.append(types[annotation] + ' ' + argument.arg +
                         ('=*' if index >= defaults_from else ''))
    marker = next(line for line in declarations.splitlines()
                  if line.startswith('cdef class ' + cls.name + '('))
    declarations = declarations.replace(marker, marker + '\n    cdef object _mx_collect_execute('
                                        + ', '.join(arguments) + ')', 1)
    collect.body = original_body
    collect.decorator_list = original_decorators
    calls = [node for statement in collect.body for node in ast.walk(statement)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and ast.unparse(node.func) == '_mx_sys.collect_from_space']
    if len(calls) != 1:
        raise ValueError('Generated collection wrapper has ambiguous traversal')
    domain = copy.deepcopy(calls[0].args[1])
    calls[0].func = ast.Attribute(ast.Name('self', ast.Load()), entry.name, ast.Load())
    calls[0].args = [domain, *(ast.Name(argument.arg, ast.Load()) for argument in parameters)]
    calls[0].keywords = []
    cls.body.append(entry)
    return declarations


def transform(source, declarations, receipt_path):
    tree = ast.parse(source)
    selected = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        traversal = _traversal_contract(cls)
        if traversal is None:
            continue
        original_collect, original_call = traversal
        original_body = copy.deepcopy(original_collect.body)
        original_decorators = copy.deepcopy(original_collect.decorator_list)
        if (original_collect.args.vararg or original_collect.args.kwarg
                or original_collect.args.kwonlyargs):
            continue
        original_collect.body = [ast.Return(copy.deepcopy(original_call))]
        contract = _collect_contract(cls)
        # Frame-backed collect_outputs owns its result in the frame. Keep that
        # authoritative entry instead of publishing a competing class-cache slot.
        output_entry = _method(cls, 'collect_outputs')
        frame_output = output_entry is not None and any(
                isinstance(node, ast.Attribute) and node.attr == '_mx_frame_owner'
                for node in ast.walk(output_entry))
        if contract is None:
            collect, call = traversal
            if not re.search(r'^cdef class ' + re.escape(cls.name) + r'\(', declarations, re.M):
                original_collect.body = original_body
                continue
            collect.decorator_list.append(ast.parse(
                '@_mx_cy.locals(_mx_root=' + cls.name + ')\ndef _mx_dummy(): pass'
            ).body[0].decorator_list[0])
            collect.body = ast.parse(_general_traversal(cls, call)).body
            declarations = _preserve_collect_cache(cls, collect, original_body,
                                                   original_decorators, declarations)
            selected.append({'class': cls.name, 'route': 'mixed-native-traversal',
                             'result_cache_preserved': True})
            continue
        output_name, expression, leaf_type, domain, parameter, extra_args = contract
        if not re.search(r'^cdef class ' + re.escape(cls.name) + r'\(', declarations, re.M):
            original_collect.body = original_body
            continue
        if not _method(cls, '_mx_collect_attach') or not _method(cls, '_mx_collect_release'):
            original_collect.body = original_body
            continue
        output_formula = _method(cls, '_f_collect_outputs')
        docstring = (output_formula.body[:1]
                     if isinstance(output_formula.body[0], ast.Expr)
                     and _string(output_formula.body[0].value) is not None else [])
        if not frame_output:
            output_formula.body = docstring + ast.parse(
                'return {"sum": {' + repr(output_name) + ': self._mx_collect_value()}}').body
        value_method = ast.parse('''
@_mx_cy.cfunc
@_mx_cy.exceptval(-1, check=True)
def _mx_collect_value(self) -> _mx_cy.double:
    return 0.0
''').body[0]
        value_method.returns = ast.parse(leaf_type, mode='eval').body
        value_method.body = ([ast.Return(expression)] if not frame_output else ast.parse(
            'return self.collect_outputs()["sum"][' + repr(output_name) + ']').body)
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
_mx_extra_args = %s
if type(self) is not %s or self._mx_roots or type(_mx_domain) not in (list, tuple, range):
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
    try:
        _mx_root._mx_copy_refs(self, self)
        self._mx_assign_params(_mx_root, _mx_coordinate, *_mx_extra_args)
        _mx_root._mx_roots.append(_mx_root)
        _mx_root._mx_collect_attach(_mx_scratch, self)
        _mx_apply = getattr(_mx_root, '_mx_apply_formula', None)
        if _mx_apply is not None:
            _mx_apply(_mx_coordinate, *_mx_extra_args)
        _mx_value = _mx_root._mx_collect_value()
        _mx_root._v_collect_outputs = {'sum': {%r: _mx_value}}
        _mx_root._has_collect_outputs = True
        if _mx_first:
            _mx_total = _mx_value
            _mx_first = False
        else:
            _mx_total += _mx_value
    finally:
        _mx_root._mx_collect_release(_mx_scratch, self)
return _mx_sys.CollectResult({'sum': _mx_sys.CollectResult({%r: _mx_total})})
''' % (domain_text, ast.unparse(extra_args), cls.name, fallback, fallback, fallback, cls.name,
       output_name, output_name)
        if frame_output:
            # The declaration's compiled entry owns its one object result.
            # Only accumulation is specialized: never publish another class
            # cache, duplicate the declaration body or bypass its exceptions.
            native = native.replace(
                '        _mx_root._v_collect_outputs = {\'sum\': {' + repr(output_name) + ': _mx_value}}\n'
                '        _mx_root._has_collect_outputs = True\n', '')
        old_collect.body = ast.parse(native).body
        cls.body.remove(value_method)
        cls.body.insert(cls.body.index(old_collect), value_method)
        marker = next(line for line in declarations.splitlines()
                      if line.startswith('cdef class ' + cls.name + '('))
        declarations = declarations.replace(
            marker, marker + '\n    cdef ' + _NUMERIC[leaf_type]
            + ' _mx_collect_value(' + cls.name + ' self) except? -1', 1)
        declarations = _preserve_collect_cache(cls, old_collect, original_body,
                                               original_decorators, declarations)
        selected.append({'class': cls.name, 'route': 'numeric-sum', 'output': output_name,
                         'authoritative_output_cache': 'frame' if frame_output else 'class',
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
