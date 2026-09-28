"""The native collect route must retain the exported fallback contract."""

import ast

from modelx_cython.collect_native import transform


SOURCE = '''
import cython as _mx_cy
@_mx_cy.cclass
class _c_Projection:
    def __init__(self, parent):
        self._mx_spaces = {}
        self._mx_roots = []
    def _mx_copy_refs(self, base, base_root):
        pass
    @staticmethod
    def _mx_assign_params(space, idx):
        space.idx = idx
    def _mx_collect_attach(self, context, template):
        pass
    def _mx_collect_release(self, context, template):
        pass
    def _f_collect_outputs(self) -> object:
        return {"sum": {"fib": self.fib(self.depth())}}
    def depth(self) -> _mx_cy.longlong:
        return 2
    def fib(self, n: _mx_cy.longlong) -> _mx_cy.double:
        return 1.0
    @_mx_cy.ccall
    def collect(self) -> object:
        return _mx_sys.collect_from_space(self, self.policy_ids, 'idx', (), ())
'''
DECLARATIONS = 'cdef class _c_Projection(_mx_sys.BaseSpace):\n    cpdef object collect(_c_Projection self)\n'


def test_numeric_sum_uses_one_typed_expression_and_keeps_fallback(tmp_path):
    output, declarations = transform(SOURCE, DECLARATIONS, tmp_path / 'receipt.json')
    tree = ast.parse(output)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    assert ast.unparse(methods['_mx_collect_value'].body[0].value) == 'self.fib(self.depth())'
    assert 'self.fib(self.depth())' not in ast.unparse(methods['_f_collect_outputs'])
    assert '_mx_sys.collect_from_space' in ast.unparse(methods['collect'])
    assert 'cdef double _mx_collect_value' in declarations
    assert (tmp_path / 'receipt.json').exists()


def test_range_domain_uses_native_numeric_traversal(tmp_path):
    source = SOURCE.replace('self.policy_ids', 'range(3)').replace(
        'def _f_collect_outputs(self) -> object:\n',
        'def _f_collect_outputs(self) -> object:\n        """Result contract."""\n')
    output, _ = transform(source, DECLARATIONS, tmp_path / 'range.json')
    assert 'type(_mx_domain) not in (list, tuple, range)' in output
    assert 'Result contract.' in output
    assert (tmp_path / 'range.json').exists()


def test_unrelated_object_cell_does_not_exclude_numeric_collection(tmp_path):
    source = SOURCE.replace(
        '    def depth(self) -> _mx_cy.longlong:\n',
        '    def sex(self) -> object:\n        return "F"\n'
        '    def depth(self) -> _mx_cy.longlong:\n')
    output, declarations = transform(source, DECLARATIONS, tmp_path / 'object.json')
    assert output != source
    assert 'def sex(self) -> object:' in output
    assert 'cdef double _mx_collect_value' in declarations
    assert (tmp_path / 'object.json').exists()


def test_other_output_shapes_and_integer_sum_keep_existing_route(tmp_path):
    variations = (
        SOURCE.replace('"sum"', '"individual"'),
        SOURCE.replace('"fib": self.fib(self.depth())',
                       '"fib": self.fib(self.depth()), "other": self.fib(0)'),
        SOURCE.replace('def fib(self, n: _mx_cy.longlong) -> _mx_cy.double:',
                       'def fib(self, n: _mx_cy.longlong) -> _mx_cy.longlong:'),
        SOURCE.replace('self._mx_spaces = {}', 'self._mx_spaces = {"Child": object()}'),
    )
    for source in variations:
        output, declarations = transform(source, DECLARATIONS, tmp_path / 'receipt.json')
        assert output == source
        assert declarations == DECLARATIONS
        assert not (tmp_path / 'receipt.json').exists()
