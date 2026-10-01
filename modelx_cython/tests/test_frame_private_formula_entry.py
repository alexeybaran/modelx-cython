"""The native formula entry remains uncached while using the shared Cell body."""
import importlib
import json
import sys
from types import ModuleType

import pytest

from modelx_cython._frame_runtime.compiler import build
from modelx_cython._frame_runtime.contracts import KernelSpec, cell


@cell
def leaf(x: float) -> float:
    return base + x - 1.0


@cell
def parent(x: float) -> float:
    return leaf(x) + leaf(x)


@cell
def reciprocal(x: float) -> float:
    return 1.0 / x


def test_checked_private_formula_entry_shares_body_but_not_own_cache(tmp_path):
    module = ModuleType('frame_private_formula_fixture')
    module.__file__ = __file__
    module.leaf = leaf
    module.parent = parent
    module.reciprocal = reciprocal
    spec = KernelSpec(module, {'base': float}, {}, 4, '0',
                      cell_outputs=('parent', 'reciprocal'))
    artifact = tmp_path / 'artifact'
    build(spec, artifact, diagnostics=True)
    manifest = json.loads((artifact / 'manifest.json').read_text())
    sys.path.insert(0, str(artifact))
    try:
        kernel_module = importlib.import_module(manifest['package'] + '._kernel')
        kernel = kernel_module.Kernel([(1.0,)])
        with pytest.raises(RuntimeError, match='Select an evaluation scope'):
            kernel.call_formula('leaf', (0.0,))
        assert kernel.evaluate(0, 'leaf', (0.0,)) == 0.0
        initial = kernel.stats()
        assert initial['misses'] == 1
        assert kernel.call_formula('leaf', (0.0,)) == 0.0
        assert kernel.call_formula('leaf', (2.0,)) == 2.0
        assert kernel.call_formula('leaf', (-1.0,)) == -1.0
        assert kernel.stats()['misses'] == 1
        assert kernel.evaluate(0, 'parent', (0.0,)) == 0.0
        assert kernel.stats()['misses'] == 3
        assert kernel.call_formula('parent', (0.0,)) == 0.0
        assert kernel.stats()['misses'] == 3
        with pytest.raises(ZeroDivisionError):
            kernel.call_formula('reciprocal', (0.0,))
        assert kernel.call_formula('reciprocal', (2.0,)) == 0.5
        assert kernel.evaluate(0, 'reciprocal', (2.0,)) == 0.5
        with pytest.raises(TypeError, match='canonical positional arguments'):
            kernel.call_formula('leaf', ())
    finally:
        sys.path.remove(str(artifact))
        for name in tuple(sys.modules):
            if name == manifest['package'] or name.startswith(manifest['package'] + '.'):
                del sys.modules[name]
