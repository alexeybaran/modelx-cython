"""Recognize factory computations whose private receiver cannot be published."""
import ast

from .syntax import code


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
        arguments = {arg.arg for arg in method.args.args}
        scalar_builtins = {'abs', 'bool', 'float', 'int', 'max', 'min', 'round', 'str', 'ValueError'}
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
                elif not isinstance(expr.func, ast.Name) or expr.func.id not in scalar_builtins:
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
                elif expr.id not in arguments and expr.id not in scalar_builtins:
                    return False
            elif not isinstance(expr, (ast.Module, ast.Return, ast.Expr, ast.Constant, ast.If, ast.IfExp,
                                       ast.Raise, ast.Dict, ast.Tuple, ast.Load, ast.BinOp, ast.UnaryOp,
                                       ast.BoolOp, ast.Compare, ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
                return False
    return True
