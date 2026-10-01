"""Source descriptions feed checked object/numeric Cython without execution."""
import ast
import inspect
import json
from types import ModuleType

import pytest

from modelx_cython.formula_emitter import SourceFormula, with_annotations
from modelx_cython._frame_runtime.compiler import build
from modelx_cython._frame_runtime.contracts import KernelSpec
from modelx_cython._frame_runtime.runtime import CompiledKernel


def test_source_handoff_compiles_mixed_dependency_graph(tmp_path):
    module = ModuleType('source_handoff_fixture')
    module.__file__ = __file__
    descriptions = {
        'amount': ('def amount(i):\n return values[i]\n', {'i': int}, float),
        'report': ('def report():\n return {"amount": amount(0)}\n', {}, object),
    }
    for name, (source, arguments, result) in descriptions.items():
        formula = SourceFormula(ast.parse(source).body[0], arguments, result,
                                filename=__file__, module_name=module.__name__)
        # Creating/rebinding descriptions must never call the formula.
        with pytest.raises(TypeError, match='not an evaluator'):
            formula()
        setattr(module, name, formula)
    rebound = with_annotations(module.amount, {'i': int, 'return': object})
    assert inspect.signature(rebound).return_annotation is object
    assert inspect.signature(module.amount).return_annotation is float
    assert rebound.source == module.amount.source
    spec = KernelSpec(module, {'values': object}, {}, 3, '0',
                      cell_outputs=('amount', 'report'), cython_objects=True,
                      cython_signatures={'amount': {'parameters': {'i': 'int'}, 'return': 'float'}})
    artifact = tmp_path / 'artifact'
    build(spec, artifact)
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert manifest['cells']['amount']['calculator'] == 'cython_scalar'
    assert manifest['cells']['report']['calculator'] == 'cython_object'
    assert manifest['formula_coverage']['counts'].get('detached_python', 0) == 0
    runtime = CompiledKernel(artifact, [{'values': {0: 1.5, 1: -1.0}}])
    with runtime.scope(0) as space:
        assert space.call('amount', 0) == 1.5
        assert space.call('amount', 1) == -1.0
        result = space.call('report')
        assert result == {'amount': 1.5}
        assert space.call('report') is result
