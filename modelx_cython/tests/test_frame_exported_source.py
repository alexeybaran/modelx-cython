"""Original formula handoff from modelx export to the frame frontend."""
import hashlib
import json

import pytest

from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.modelx_frontend_exported import from_exported
from modelx_cython._frame_runtime.source_graph import describe_source_graph


def _cell(source):
    return {'source': source, 'sha256': hashlib.sha256(source.encode()).hexdigest()}


def test_exported_graph_keeps_connected_spaces_and_external_bindings():
    manifest = {
        'schema': 'modelx.formula_sources.v2', 'model': 'Example',
        'spaces': {
            'Example.Actual': {
                'parameters': ['policy_id'],
                'references': ['ScenarioInputs', 'policy_ids'],
                'cells': {'net_cf': _cell('def net_cf(t):\n    return t\n')},
                'formula': "def _formula(policy_id):\n    return {'refs': {}}\n",
                'collect': {'over': 'policy_id', 'domain': {'kind': 'reference', 'name': 'policy_ids'}},
            },
            'Example.ScenarioInputs': {
                'parameters': [], 'references': ['policy_ids'], 'cells': {},
                'formula': None, 'collect': None,
            },
            'Example.Valuation': {
                'parameters': ['month', 'policy_id'],
                'references': ['ScenarioInputs', 'Actual'],
                'cells': {'liability': _cell('def liability():\n    return 1.0\n')},
                'formula': "def _formula(month, policy_id):\n    return {'refs': {}}\n",
                'collect': None,
            },
            'Example.Outer': {
                'parameters': [], 'references': ['Actual', 'Valuation'],
                'cells': {'result': _cell('def result():\n    return 1.0\n')},
                'formula': None, 'collect': None,
            },
        },
    }
    graph = describe_source_graph(manifest)
    assert {(edge['from'], edge['reference'], edge['to']) for edge in graph['edges']} == {
        ('Example.Actual', 'ScenarioInputs', 'Example.ScenarioInputs'),
        ('Example.Valuation', 'ScenarioInputs', 'Example.ScenarioInputs'),
        ('Example.Valuation', 'Actual', 'Example.Actual'),
        ('Example.Outer', 'Actual', 'Example.Actual'),
        ('Example.Outer', 'Valuation', 'Example.Valuation'),
    }
    assert graph['spaces']['Example.Actual']['external_references'] == ['policy_ids']
    assert graph['spaces']['Example.Valuation']['cells']['liability'] == _cell(
        'def liability():\n    return 1.0\n')['sha256']


def test_exported_graph_rejects_changed_formula_in_any_space():
    manifest = {'schema': 'modelx.formula_sources.v2', 'model': 'Example',
                'spaces': {'Example.Actual': {'parameters': [], 'references': [],
                                             'cells': {'value': _cell('lambda: 1')}}}}
    manifest['spaces']['Example.Actual']['cells']['value']['source'] = 'lambda: 2'
    with pytest.raises(ContractError, match='Formula source hash mismatch'):
        describe_source_graph(manifest)


def test_exported_formula_hash_mismatch_fails_before_import(tmp_path):
    package = tmp_path / 'exported'
    package.mkdir()
    (package / '_mx_formula_sources.json').write_text(json.dumps({
        'schema': 'modelx.formula_sources.v2',
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
