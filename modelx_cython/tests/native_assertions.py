"""Generated-shape assertions across direct and ordinary Cell dispatch.

Only dispatch spelling and expression formatting are normalized. Operator trees,
operands, constants and behavioral test expectations are unchanged.
"""
import ast


def has_expression(source, expression):
    class OrdinaryDispatch(ast.NodeTransformer):
        def visit_IfExp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.test, ast.Name) and node.test.id == '_mx_exact':
                return node.orelse
            return node
    tree = OrdinaryDispatch().visit(ast.parse(source))
    expected = ast.dump(ast.parse(expression, mode='eval').body)
    return any(isinstance(n, ast.expr) and ast.dump(n) == expected for n in ast.walk(tree))


def has_numeric_array(declarations, ctype, name, size):
    old = f'cdef {ctype}[{size}] {name}'
    owned = f'{ctype} {name}[{size}]'
    return old in declarations or owned in declarations
