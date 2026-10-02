"""Receiver context must not change lexical bindings or formula operations."""
import ast

from modelx_cython.formula_emitter import _separate_receiver_locals


def test_qualified_receiver_does_not_capture_local_container():
    node = ast.parse('''def f(self):
    cell = []
    cell.append(self.cell(2))
    return cell
''').body[0]
    transformed = _separate_receiver_locals(node)
    source = ast.unparse(transformed)
    assert 'self.cell(2)' in source
    assert '_mx_local_cell.append' in source


def test_nested_scope_shadowing_and_outer_closure_stay_distinct():
    node = ast.parse('''def f(self):
    cell = []
    def capture():
        return cell
    def shadow(cell):
        def child():
            return cell
        return child()
    cell.append(self.cell(2))
    return capture(), shadow(9), [cell for cell in range(3)]
''').body[0]
    transformed = _separate_receiver_locals(node)
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[transformed], type_ignores=[])),
                 '<test receiver binding>', 'exec'), scope)
    class Receiver:
        def cell(self, n):
            return n * 2
    assert scope['f'](Receiver()) == ([4], 9, [0, 1, 2])


def test_no_collision_leaves_formula_tree_unchanged():
    node = ast.parse('def f(self):\n result = self.cell(2)\n return result\n').body[0]
    before = ast.dump(node)
    assert ast.dump(_separate_receiver_locals(node)) == before
