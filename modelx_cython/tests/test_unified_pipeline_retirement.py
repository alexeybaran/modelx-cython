"""The standard product has one frontend/build and no research facade imports."""
import ast
from pathlib import Path

RETIRED = frozenset(('connected_build', 'connected_model', 'connected_package',
    'model', 'package_codegen', 'modelx_frontend_exported', 'source_graph',
    'trace_contracts', 'modelx_frontend_native_mode'))


def test_no_production_caller_or_packaged_module_uses_retired_pipeline():
    root = Path(__file__).parents[1]
    for name in RETIRED:
        assert not (root/'_frame_runtime'/(name+'.py')).exists(), name
    for path in root.rglob('*.py'):
        if 'tests' in path.relative_to(root).parts:
            continue
        source = path.read_text(encoding='utf-8')
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ''
                assert 'reference_frame' not in module, path
                if '_frame_runtime' in module or node.level:
                    assert module.rsplit('.',1)[-1] not in RETIRED, (path, module)
            elif isinstance(node, ast.Import):
                assert all('reference_frame' not in item.name for item in node.names), path
        assert 'modelx_cython.tests.reference_frame' not in source, path
    source = (root/'cli.py').read_text(encoding='utf-8')
    assert '_frame_collect_handler' not in source
    assert 'MX_CY_ORIGINAL_FRAME_SHELL' not in source


def test_resource_adapter_requires_authoritative_shared_space_metadata(tmp_path):
    import pytest
    from modelx_cython._frame_runtime.prepared_resources import from_resolved_resources
    from modelx_cython._frame_runtime.contracts import ContractError
    with pytest.raises(ContractError, match='shared Space emission plan'):
        from_resolved_resources(None, tmp_path)
    assert not any(tmp_path.iterdir())
