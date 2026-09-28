"""Read explicit category and table-key contracts in exported Cell formulas.

This describes only source forms that state their domain: a category helper
with an Enum reference, an Enum member, or a key helper with a Data table.
It does not infer a type from arithmetic or sampled values. Unrecognized
forms remain object boundaries in the collect graph.
"""

import ast


def _path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ()
    parts.append(node.id)
    return tuple(reversed(parts))


def _explicit_value(node):
    if not isinstance(node, ast.Call) or node.keywords:
        return None
    call = _path(node.func)
    if call == ("self", "lookup", "category") and len(node.args) == 4:
        enum = _path(node.args[3])
        if len(enum) == 2 and enum[0] == "self":
            return {"kind": "enum_ref", "name": enum[1]}
    if call == ("self", "lookup", "key") and len(node.args) == 2:
        table = node.args[0]
        if (isinstance(table, ast.Call) and not table.args and
                not table.keywords):
            table_path = _path(table.func)
            if len(table_path) == 3 and table_path[:2] == ("self", "data"):
                return {"kind": "table_key", "table": table_path[2]}
    return None


def _value(node):
    direct = _explicit_value(node)
    if direct is not None:
        return direct
    path = _path(node)
    if len(path) == 3 and path[0] == "self":
        return {"kind": "enum_ref", "name": path[1]}
    return None


class _Body(ast.NodeVisitor):
    def __init__(self):
        self.returns = []
        self.calls = []

    def visit_FunctionDef(self, node):
        # Nested functions are a different execution scope.
        return

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_Lambda(self, node):
        return

    def visit_Return(self, node):
        self.returns.append(_value(node.value))
        self.generic_visit(node)

    def visit_Call(self, node):
        self.calls.append(node)
        self.generic_visit(node)


def _body(formula):
    visitor = _Body()
    for statement in formula.body:
        visitor.visit(statement)
    return visitor


def derive(formulas, selected):
    """Return deterministic semantic contracts for selected Cell names.

    ``formulas`` maps Cell name to exported ``_f_`` FunctionDef. The result is
    source metadata, not permission to replace public Python object results.
    Argument propagation requires every observed call site to supply the same
    explicit provider type in that position.
    """
    names = set(formulas)
    result_types = {}
    bodies = {name: _body(formula) for name, formula in formulas.items()}
    for name, body in bodies.items():
        if (body.returns and all(value is not None for value in body.returns)
                and all(value == body.returns[0] for value in body.returns)):
            result_types[name] = body.returns[0]

    arguments = {name: {} for name in names}
    seen_calls = {name: {} for name in names}
    for body in bodies.values():
        for call in body.calls:
            path = _path(call.func)
            if len(path) != 2 or path[0] != "self" or path[1] not in names:
                continue
            target = path[1]
            params = [arg.arg for arg in formulas[target].args.posonlyargs +
                      formulas[target].args.args if arg.arg != "self"]
            for position, parameter in enumerate(params):
                if call.keywords or position >= len(call.args):
                    found = None
                else:
                    value = call.args[position]
                    provider = _path(value.func) if isinstance(value, ast.Call) \
                        and not value.args and not value.keywords else ()
                    found = (result_types.get(provider[1]) if len(provider) == 2
                             and provider[0] == "self" else None)
                seen_calls[target].setdefault(parameter, []).append(found)
    for name, params in seen_calls.items():
        for parameter, candidates in params.items():
            if (candidates and all(candidate is not None for candidate in candidates)
                    and all(candidate == candidates[0] for candidate in candidates)):
                arguments[name][parameter] = candidates[0]

    return {name: {"result": result_types.get(name),
                   "arguments": arguments[name]}
            for name in sorted(selected)
            if result_types.get(name) is not None or arguments[name]}
