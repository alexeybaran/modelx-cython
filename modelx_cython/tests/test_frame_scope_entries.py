"""Only interpreted bodies need eager Cell names in the receiver globals."""
from types import SimpleNamespace

from modelx_cython._frame_runtime.python_calculator import PythonScope


def _scope(*, detached):
    result = {'kind': 'float', 'ctype': 'double'}
    params = [{'name': 'n', 'default': None, 'type': {'kind': 'int', 'ctype': 'long long'}}]
    rows = {
        'child': dict(params=params, result=result, calculator='native', dependencies=[]),
        'compiled': dict(params=params, result=result, calculator='cython_object', dependencies=['child']),
    }
    source = ''
    if detached:
        rows['parent'] = dict(params=params, result=result, calculator='python', dependencies=['child'])
        source = 'def parent(n):\n return child(n) + 1.0\n'
    manifest = dict(package='example', inputs={}, cells=rows, python_references=[],
                    python_cells=['compiled'] + (['parent'] if detached else []),
                    python_enums={}, python_inputs=[], cython_object_cells=['compiled'],
                    cython_numeric_fallback_cells=[], diagnostics=False)
    kernel = SimpleNamespace(call=lambda name, args: float(args[0] * 2))
    return PythonScope(None, manifest, kernel, program=compile(source, '<fixture>', 'exec'))


def test_compiled_only_scope_keeps_external_entries_lazy():
    scope = _scope(detached=False)
    assert scope.entries == {}
    assert 'compiled' not in scope.namespace and 'child' not in scope.namespace
    entry = scope.entry('compiled')
    assert entry(3) == 6.0
    assert scope.entry('compiled') is entry
    assert set(scope.entries) == {'compiled'}


def test_detached_body_keeps_compiled_dependencies_bound():
    scope = _scope(detached=True)
    assert set(scope.entries) == {'parent', 'child'}
    assert 'compiled' not in scope.namespace
    assert scope.calculate('parent', (3,)) == 7.0
    assert scope.entry('compiled')(4) == 8.0
