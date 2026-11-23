"""Recognize factory computations whose private receiver cannot be published."""
import ast
import symtable

from .syntax import code


_BUILTINS = {'abs', 'bool', 'float', 'int', 'max', 'min', 'round', 'str', 'ValueError'}

_APPLY = """def _mx_apply_formula(self, *args, **kwargs):
    result = self._parent._mx_space_formula(*args, **kwargs)
    if result is None:
        return
    if not isinstance(result, dict) or result.keys() - {'refs'}:
        raise ValueError('Exported Space formula must return only dynamic References')
    refs = result.get('refs')
    if refs is None:
        return
    if type(refs) is not dict or refs.keys() - ALIASES:
        raise ValueError('Unexpected dynamic Reference')
    for name, value in refs.items():
        setattr(self, name, value)
"""


def standard_lifecycle(node, aliases, *, remap_references=False):
    """Recognize exporter-owned construction, reference copy and alias assignment."""
    body = ast.parse(code(node)).body[0].body
    methods = {m.name: m for m in body if isinstance(m, ast.FunctionDef)}
    if any(name in methods for name in ('__new__', '__del__', '__getattribute__', '__setattr__', '__getattr__')):
        return False
    apply = methods.get('_mx_apply_formula')
    if apply is None or apply.decorator_list:
        return False
    # The exporter lists dynamic aliases in declaration order.
    template = ast.parse(_APPLY.replace('ALIASES', repr(tuple(aliases)))).body[0]
    if ast.dump(apply) != ast.dump(template):
        return False
    for name in ('__init__', '_mx_copy_refs'):
        method = methods.get(name)
        if method is None or method.decorator_list:
            return False
        for stmt in method.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                return False
            dest = stmt.targets[0]
            if not (isinstance(dest, ast.Attribute) and isinstance(dest.value, ast.Name)
                    and dest.value.id == 'self'):
                return False
            value = stmt.value
            if name == '_mx_copy_refs':
                direct = _source_reference(value, dest.attr)
                remapped = remap_references and _remapped_reference(value, dest.attr)
                if not (direct or remapped):
                    return False
            elif not (isinstance(value, ast.Constant)
                      or isinstance(value, ast.Dict) and not value.keys
                      or isinstance(value, ast.List) and not value.elts
                      or isinstance(value, ast.Name) and (dest.attr, value.id) in
                      (('_space', 'self'), ('_parent', 'parent'))
                      or isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                      and (dest.attr, value.value.id, value.attr) == ('_model', 'parent', '_model')):
                return False
    return True


def _source_reference(value, name):
    return (isinstance(value, ast.Attribute) and value.attr == name
            and isinstance(value.value, ast.Name) and value.value.id == 'base')


def _remapped_reference(value, name):
    if not (isinstance(value, ast.IfExp) and _source_reference(value.orelse, name)
            and isinstance(value.test, ast.Call) and isinstance(value.test.func, ast.Attribute)
            and value.test.func.attr == '_mx_is_in' and _source_reference(value.test.func.value, name)
            and not value.test.keywords and len(value.test.args) == 1
            and isinstance(value.test.args[0], ast.Name) and value.test.args[0].id == 'base_root'):
        return False
    target = value.body
    while isinstance(target, ast.Attribute) and target.attr != '_parent':
        target = target.value
    return (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
            and target.value.id == 'self')


def unshadowed_builtins(module):
    source = module.code
    symbols = symtable.symtable(source, '<export>', 'exec')
    if any(symbol.get_name() in _BUILTINS | {'setattr', 'type', 'isinstance', 'dict'} and (symbol.is_assigned() or symbol.is_imported())
           for symbol in symbols.get_symbols()):
        return False
    return not any(isinstance(item, ast.ImportFrom) and any(alias.name == '*' for alias in item.names)
                   for item in ast.parse(source).body)


def detached_factory_graph(node, cls, cells, *, forbidden_refs=(), forbidden_cells=(), include_factory=True):
    """Check that the policy receiver cannot become an input or an output.

    This opt-in permits calls on fixed external dependencies. The model author
    must keep those dependencies fixed for the call and use ordinary Cells
    that do not inspect caller frames. A bound policy method may only appear
    as the target of a direct call.
    """
    methods = {m.name: m for m in ast.parse(code(node)).body[0].body if isinstance(m, ast.FunctionDef)}
    pending = ['_f_' + name for name in cells]
    if include_factory:
        pending.append('_mx_space_formula')
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        method = methods.get(name)
        if method is None or name in forbidden_cells:
            return False
        body = ast.Module(body=method.body, type_ignores=[])
        parents = {child: parent for parent in ast.walk(body) for child in ast.iter_child_nodes(parent)}
        reflective = {'locals', 'globals', 'vars', 'eval', 'exec', 'super', 'getattr', 'setattr',
                      '_getframe', 'currentframe', 'f_back', 'f_locals'}
        for expr in ast.walk(body):
            if isinstance(expr, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await,
                                 ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return False
            if isinstance(expr, (ast.Attribute, ast.Subscript)) and not isinstance(expr.ctx, ast.Load):
                return False
            if isinstance(expr, ast.Attribute) and (expr.attr in reflective or expr.attr.startswith('__')):
                return False
            if isinstance(expr, ast.Name):
                if expr.id in reflective:
                    return False
                if expr.id == 'self' and not isinstance(expr.ctx, ast.Load):
                    return False
                if expr.id == 'self' and not isinstance(parents[expr], ast.Attribute):
                    return False
            if (isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name)
                    and expr.value.id == 'self'):
                if expr.attr in cls.cells:
                    parent = parents[expr]
                    if not isinstance(parent, ast.Call) or parent.func is not expr:
                        return False
                    pending.append('_f_' + expr.attr)
                elif expr.attr not in cls.refs or expr.attr.startswith('_') or expr.attr in forbidden_refs:
                    return False
    return True


def declared_aliases(node):
    """Read the aliases from the exporter-owned apply method."""
    for method in ast.parse(code(node)).body[0].body:
        if not isinstance(method, ast.FunctionDef) or method.name != '_mx_apply_formula':
            continue
        for expr in ast.walk(method):
            if (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Sub)
                    and isinstance(expr.right, ast.Tuple)
                    and all(isinstance(key, ast.Constant) and type(key.value) is str
                            for key in expr.right.elts)):
                return tuple(key.value for key in expr.right.elts)
    return ()


def fixed_factory_refs(node):
    """Return the insertion order of a factory's identical literal result schemas."""
    cls = ast.parse(code(node)).body[0]
    factory = next((method for method in cls.body if isinstance(method, ast.FunctionDef)
                    and method.name == '_mx_space_formula'), None)
    if factory is None or factory.decorator_list or not isinstance(factory.body[-1], ast.Return):
        return ()
    aliases = set(declared_aliases(node))
    schema = None
    for result in (item.value for item in ast.walk(factory) if isinstance(item, ast.Return)):
        if (not isinstance(result, ast.Dict) or len(result.keys) != 1
                or not isinstance(result.keys[0], ast.Constant) or result.keys[0].value != 'refs'
                or not isinstance(refs := result.values[0], ast.Dict)
                or any(not isinstance(key, ast.Constant) or type(key.value) is not str for key in refs.keys)):
            return ()
        names = tuple(key.value for key in refs.keys)
        if len(set(names)) != len(names) or not set(names) <= aliases or schema is not None and names != schema:
            return ()
        schema = names
    return schema or ()


def scalar_factory_graph(node, cls, shape):
    """Prove the reachable graph uses numeric inputs and direct calls to local Cells.

    The generated collect evaluator supplies exact primitive arguments. Object
    References, indirect calls, mutation, reflection and callbacks need a separate
    proof. An unrecognized graph keeps fresh policy instances.
    """
    if any(kind not in ('double', 'long long') for _, kind in shape.extra):
        return False
    methods = {m.name: m for m in ast.parse(code(node)).body[0].body if isinstance(m, ast.FunctionDef)}
    pending = ['_mx_space_formula', *('_f_' + leaf.cell for leaf in shape.native)]
    checked = set()
    while pending:
        name = pending.pop()
        if name in checked:
            continue
        method = methods.get(name)
        if method is None:
            return False
        checked.add(name)
        sig = method.args
        arguments = {arg.arg for arg in sig.args}
        if sig.vararg or sig.kwarg or sig.posonlyargs or sig.kwonlyargs or arguments & _BUILTINS:
            return False
        for default in sig.defaults:
            try:
                if type(ast.literal_eval(default)) not in (int, float, bool):
                    return False
            except (TypeError, ValueError):
                return False
        body = ast.Module(body=method.body, type_ignores=[])
        parents = {child: parent for parent in ast.walk(body) for child in ast.iter_child_nodes(parent)}
        for expr in ast.walk(body):
            if isinstance(expr, ast.Call):
                if isinstance(expr.func, ast.Attribute):
                    target = expr.func
                    if not (isinstance(target.value, ast.Name) and target.value.id == 'self'
                            and target.attr in cls.cells):
                        return False
                    pending.append('_f_' + target.attr)
                elif not isinstance(expr.func, ast.Name) or expr.func.id not in _BUILTINS:
                    return False
            elif isinstance(expr, ast.Attribute):
                if not isinstance(expr.value, ast.Name) or expr.value.id != 'self':
                    return False
                if expr.attr in cls.cells:
                    # A bound method itself would carry the private receiver.
                    parent = parents[expr]
                    if not isinstance(parent, ast.Call) or parent.func is not expr:
                        return False
                elif (expr.attr not in cls.refs
                      or cls.refs[expr.attr].get_type_expr(c_style=True) not in ('long long', 'double', 'bint')):
                    return False
            elif isinstance(expr, ast.Name):
                if expr.id == 'self':
                    parent = parents[expr]
                    if not isinstance(parent, ast.Attribute) or parent.value is not expr:
                        return False
                elif expr.id not in arguments and expr.id not in _BUILTINS:
                    return False
            elif not isinstance(expr, (ast.Module, ast.Return, ast.Expr, ast.Constant, ast.If, ast.IfExp,
                                       ast.Raise, ast.Dict, ast.Tuple, ast.Load, ast.BinOp, ast.UnaryOp,
                                       ast.BoolOp, ast.Compare, ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
                return False
    return True
