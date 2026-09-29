"""Original formula handoff from modelx export to the frame frontend."""
import json

import pytest

from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.modelx_frontend_exported import from_exported


def test_exported_formula_hash_mismatch_fails_before_import(tmp_path):
    package = tmp_path / 'exported'
    package.mkdir()
    (package / '_mx_formula_sources.json').write_text(json.dumps({
        'schema': 'modelx.formula_sources.v1',
        'model': 'Example',
        'spaces': {'Example.Projection': {'cells': {
            'value': {'source': 'lambda: 1', 'sha256': '0' * 64},
        }}},
    }), encoding='utf-8')
    with pytest.raises(ContractError, match='Formula source hash mismatch'):
        from_exported(package, tmp_path / 'source')


def test_unknown_export_source_schema_fails_before_import(tmp_path):
    package = tmp_path / 'exported'
    package.mkdir()
    (package / '_mx_formula_sources.json').write_text(json.dumps({
        'schema': 'modelx.formula_sources.v0', 'model': 'Example', 'spaces': {},
    }), encoding='utf-8')
    with pytest.raises(ContractError, match='Unsupported modelx formula-source'):
        from_exported(package, tmp_path / 'source')
