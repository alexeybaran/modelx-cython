"""Recognize factory computations whose private receiver cannot be published."""
import ast
import symtable

from .syntax import code


_BUILTINS = {'abs', 'bool', 'float', 'int', 'max', 'min', 'round', 'str', 'ValueError'}


def unshadowed_builtins(module):
    source = module.code
    symbols = symtable.symtable(source, '<export>', 'exec')
    if any(symbol.get_name() in _BUILTINS and (symbol.is_assigned() or symbol.is_imported())
           for symbol in symbols.get_symbols()):
        return False
    return not any(isinstance(item, ast.ImportFrom) and any(alias.name == '*' for alias in item.names)
                   for item in ast.parse(source).body)


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
