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


def test_frame_sum_reads_authoritative_output_entry_without_second_cache(tmp_path):
    source = SOURCE.replace('    def depth(self)',
        '    def collect_outputs(self) -> object:\n'
        '        return self._mx_frame_owner.call("collect_outputs", ())\n'
        '    def depth(self)')
    output, _ = transform(source, DECLARATIONS, tmp_path / 'frame.json')
    tree = ast.parse(output)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    assert ast.unparse(methods['_mx_collect_value'].body[0].value) == "self.collect_outputs()['sum']['fib']"
    assert 'self.fib(self.depth())' in ast.unparse(methods['_f_collect_outputs'])
    assert '_v_collect_outputs =' not in output
    assert '_has_collect_outputs =' not in output


def test_closed_prepared_sum_uses_batch_hook_but_keeps_effectful_space_formula(tmp_path):
    source = SOURCE.replace('    def depth(self)',
        '    def collect_outputs(self) -> object:\n'
        '        return self._mx_frame_owner.call("collect_outputs", ())\n'
        '    def _mx_collect_scalar(self, coordinates, arguments, output):\n'
        '        return 1.0\n'
        '    def depth(self)')
    output,_ = transform(source,DECLARATIONS,tmp_path/'batch.json')
    assert 'self._mx_collect_scalar(_mx_coords, _mx_extra_args,' in output
    assert 'getattr(self, \'_mx_apply_formula\', None) is None' in output
    assert 'for _mx_coordinate in _mx_coords:' in output
    assert '_v_collect_outputs =' not in output


def test_range_domain_uses_native_numeric_traversal(tmp_path):
    source = SOURCE.replace('self.policy_ids', 'range(3)').replace(
        'def _f_collect_outputs(self) -> object:\n',
        'def _f_collect_outputs(self) -> object:\n        """Result contract."""\n')
    output, _ = transform(source, DECLARATIONS, tmp_path / 'range.json')
    assert 'type(_mx_domain) not in (list, tuple, range)' in output
    assert 'Result contract.' in output
    assert (tmp_path / 'range.json').exists()


def test_secondary_projection_parameter_is_bound_for_each_policy(tmp_path):
    source = SOURCE.replace(
        'def _mx_assign_params(space, idx):',
        'def _mx_assign_params(space, idx, scenario):').replace(
        'def collect(self) -> object:',
        'def collect(self, scenario=-1) -> object:').replace(
        "self.policy_ids, 'idx', (), ()",
        "self.policy_ids, 'idx', ('scenario',), (scenario,)")
    output, declarations = transform(source, DECLARATIONS, tmp_path / 'scenario.json')
    assert '_mx_extra_args = (scenario,)' in output
    assert 'self._mx_assign_params(_mx_root, _mx_coordinate, *_mx_extra_args)' in output
    assert '_mx_sys.collect_from_space' in output
    assert 'cdef double _mx_collect_value' in declarations


def test_invalid_secondary_parameter_contract_keeps_fallback(tmp_path):
    source = SOURCE.replace(
        "self.policy_ids, 'idx', (), ()",
        "self.policy_ids, 'idx', ('idx',), (0,)")
    output, declarations = transform(source, DECLARATIONS, tmp_path / 'invalid.json')
    assert output == source and declarations == DECLARATIONS
    assert not (tmp_path / 'invalid.json').exists()


def test_reordered_secondary_parameter_uses_keyword_binding(tmp_path):
    source = SOURCE.replace(
        'def _mx_assign_params(space, idx):',
        'def _mx_assign_params(space, idx, scenario, other):').replace(
        "self.policy_ids, 'idx', (), ()",
        "self.policy_ids, 'idx', ('other', 'scenario'), (1, 2)")
    output, declarations = transform(source, DECLARATIONS, tmp_path / 'reordered.json')
    assert output != source and declarations == DECLARATIONS
    assert 'self._mx_assign_params(_mx_space, **_mx_values)' in output
    assert (tmp_path / 'reordered.json').exists()


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


def test_other_output_shapes_keep_native_traversal(tmp_path):
    variations = (
        SOURCE.replace('"sum"', '"individual"'),
        SOURCE.replace('"fib": self.fib(self.depth())',
                       '"fib": self.fib(self.depth()), "other": self.fib(0)'),
        SOURCE.replace('def fib(self, n: _mx_cy.longlong) -> _mx_cy.double:',
                       'def fib(self, n: _mx_cy.longlong) -> _mx_cy.longlong:'),
    )
    for source in variations:
        output, declarations = transform(source, DECLARATIONS, tmp_path / 'receipt.json')
        assert output != source
        assert declarations == DECLARATIONS
        assert '_mx_sys._mx_collect_detach' in output
        assert (tmp_path / 'receipt.json').exists()


def test_nested_space_uses_general_compiled_traversal(tmp_path):
    source = SOURCE.replace('self._mx_spaces = {}',
                            'self._mx_spaces = {"Child": object()}')
    output, declarations = transform(source, DECLARATIONS, tmp_path / 'nested.json')
    assert output != source and declarations == DECLARATIONS
    assert '_mx_root._mx_walk()' in output
    assert '_mx_sys._mx_collect_detach' in output
    assert (tmp_path / 'nested.json').exists()


def test_cached_collection_preserves_public_cache_and_emits_miss_traversal(tmp_path):
    old = "        return _mx_sys.collect_from_space(self, self.policy_ids, 'idx', (), ())"
    body = """        key = tuple(self.policy_ids)
        if key in self._mx_collect_cache:
            return self._mx_collect_cache[key]
        result = _mx_sys.collect_from_space(self, self.policy_ids, 'idx', (), ())
        self._mx_collect_cache[key] = result
        return result"""
    source = SOURCE.replace(old, body)
    assert source != SOURCE
    output, declarations = transform(source, DECLARATIONS, tmp_path / 'cached.json')
    cls = next(node for node in ast.parse(output).body if isinstance(node, ast.ClassDef))
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    wrapper = ast.unparse(methods['collect'])
    assert 'self._mx_collect_cache[key] = result' in wrapper
    assert 'result = self._mx_collect_execute(self.policy_ids)' in wrapper
    traversal = ast.unparse(methods['_mx_collect_execute'])
    assert 'for _mx_coordinate in _mx_coords:' in traversal
    assert '_mx_apply(_mx_coordinate, *_mx_extra_args)' in traversal
    assert 'finally:' in traversal and '_mx_collect_release' in traversal
    assert 'cdef object _mx_collect_execute(_c_Projection self, object _mx_collect_domain)' in declarations
