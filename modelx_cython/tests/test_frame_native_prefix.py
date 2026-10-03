"""Native acceleration retains the original unrestricted dictionary domain."""
import importlib
import json
from types import ModuleType, SimpleNamespace

import pytest

from modelx_cython._frame_runtime.compiler import build
from modelx_cython._frame_runtime.contracts import KernelSpec, cell


@cell
def amount(n: int) -> float:
    return float(n) + offset


@cell
def pair(n: int, m: int) -> float:
    return amount(n) + amount(m)


@cell
def recursive(n: int) -> float:
    return 0.0 if n <= 0 else recursive(n - 1) + 1.0


@cell
def parent(n: int) -> float:
    value = amount(n)
    if fail:
        raise ValueError('retry')
    return value + 1.0


@cell
def report() -> object:
    return {'amount': amount(0)}


@cell
def checked_amount(n: int) -> float:
    # Force a Python-semantic, Cython-compiled numeric body instead of the
    # pure-native arithmetic path. Its callers still use checked typed entries.
    value = float(str(round(amount(n), 2)))
    if fail:
        raise ValueError('checked retry')
    return value


@cell
def checked_report() -> object:
    return {'amount': checked_amount(0)}


@pytest.mark.parametrize('steps', [4, 40])
def test_prefix_overflow_sharing_restore_and_failed_parent(tmp_path, monkeypatch, steps):
    module = ModuleType('native_prefix_' + str(steps))
    module.__file__ = __file__
    for function in (amount, pair, recursive, parent, report):
        setattr(module, function.__name__, function)
    plan = SimpleNamespace(cells={name: SimpleNamespace(
        identity=name, public_result='double',
        public_arguments=tuple((p, 'long long') for p in
            (('n','m') if name == 'pair' else ('n',))))
        for name in ('amount','pair','recursive','parent')})
    plan.cells['report'] = SimpleNamespace(identity='report', public_result='object', public_arguments=())
    spec = KernelSpec(module=module, inputs={'offset': float, 'fail': bool},
        outputs={}, steps=steps, length='0',
        cell_outputs=('amount', 'pair', 'recursive', 'parent', 'report'),
        original_cache_kinds={name: 'dictionary' for name in ('amount','pair','recursive','parent')},
        cython_objects=True, public_emission_plan=plan)
    artifact = tmp_path / 'artifact'
    build(spec, artifact)
    manifest = json.loads((artifact/'manifest.json').read_text())
    assert all(manifest['cells'][name]['native_prefix'] for name in ('amount','pair','recursive','parent'))
    monkeypatch.syspath_prepend(str(artifact))
    native = importlib.import_module(manifest['package'] + '._kernel')
    from modelx_cython._frame_runtime.python_calculator import PythonScope
    def select(owner):
        owner.set_python_scope(PythonScope(artifact, manifest, owner))
        owner.select_scope(0)
    owner = native.Kernel(((0.25, False),))
    select(owner)
    for n in (-1000, -1, 0, steps + 1, steps + 2, 10000):
        assert owner.call('amount', (n,)) == n + 0.25
        assert owner.call('pair', (n, -1)) == n - 0.5
    for n in range(1000, 3000):
        assert owner.call('amount', (n,)) == n + 0.25
    assert owner.call('recursive', (steps + 20,)) == steps + 20
    snapshot = owner.snapshot()
    restored = native.Kernel(((100.0, False),))
    select(restored)
    restored.restore(snapshot)
    for n in (-1000, -1, 0, steps + 1, steps + 2, 2999, 10000):
        assert restored.call('amount', (n,)) == n + 0.25
        assert restored.call('pair', (n, -1)) == n - 0.5
    assert restored.call('recursive', (steps + 20,)) == steps + 20
    bad = native.Kernel(((0.25, True),))
    select(bad)
    with pytest.raises(ValueError, match='retry'):
        bad.call('parent', (0,))
    assert bad.call('amount', (0,)) == 0.25
    assert not any(key == ('parent', 'numeric') and values for key, values in bad.snapshot())


def test_prepared_checked_numeric_body_stays_connected(tmp_path, monkeypatch):
    module = ModuleType('prepared_checked_numeric')
    module.__file__ = __file__
    for function in (amount, checked_amount, checked_report):
        setattr(module, function.__name__, function)
    plan = SimpleNamespace(cells={name: SimpleNamespace(identity=name,
        public_result='double', public_arguments=(('n','long long'),))
        for name in ('amount','checked_amount')})
    plan.cells['checked_report'] = SimpleNamespace(identity='checked_report',
        public_result='object',public_arguments=())
    spec = KernelSpec(module=module, inputs={'offset': float, 'fail': bool},
        outputs={},steps=4,length='0',cell_outputs=('checked_report',),
        cython_objects=True,public_emission_plan=plan)
    artifact = tmp_path/'artifact'
    build(spec,artifact)
    manifest = json.loads((artifact/'manifest.json').read_text())
    assert 'checked_amount' in manifest['original_checked_numeric_bridges']
    kernel_source = next(artifact.rglob('_kernel.pyx')).read_text()
    entry = kernel_source.split(' _native_cell_amount(', 1)[1].split('\n    cdef ', 1)[0]
    assert entry.index('_mx_completed_key] ==') < entry.index('self._invoke(&command)')
    monkeypatch.syspath_prepend(str(artifact))
    native = importlib.import_module(manifest['package']+'._kernel')
    from modelx_cython._frame_runtime.python_calculator import PythonScope
    owner = native.Kernel(((0.255,False),))
    scope = PythonScope(artifact,manifest,owner)
    owner.set_python_scope(scope)
    owner.select_scope(0)
    def forbidden(*args):
        raise AssertionError('Compiled numeric body crossed Python calculate')
    scope.calculate = forbidden
    assert owner.call('checked_report',()) == {'amount': 0.26}
    assert owner.call('checked_amount',(0,)) == 0.26
    bad = native.Kernel(((0.255,True),))
    failed_scope = PythonScope(artifact,manifest,bad)
    bad.set_python_scope(failed_scope)
    bad.select_scope(0)
    failed_scope.calculate = forbidden
    with pytest.raises(ValueError,match='checked retry'):
        bad.call('checked_report',())
    assert bad.call('amount',(0,)) == 0.255
