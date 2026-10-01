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


def test_exported_graph_preserves_public_cell_signatures_for_adapter_generation():
    first = _cell('def value(x: float, /, y=2, *, label="a") -> float:\n    return x + y\n')
    first.update(cached=False, allow_none=True)
    second = _cell('lambda policy, period=0: policy + period')
    manifest = {'schema': 'modelx.formula_sources.v2', 'model': 'Example',
                'spaces': {'Example.S': {'parameters': [], 'references': [],
                                         'cells': {'value': first, 'other': second}}}}
    interfaces = describe_source_graph(manifest)['spaces']['Example.S']['cell_interfaces']
    assert interfaces['value'] == {
        'formula_kind': 'function', 'parameters': [
            {'name': 'x', 'kind': 'POSITIONAL_ONLY', 'annotation': 'float'},
            {'name': 'y', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': '2'},
            {'name': 'label', 'kind': 'KEYWORD_ONLY', 'default': "'a'"}],
        'cached': False, 'allow_none': True, 'source_sha256': first['sha256'],
        'return_annotation': 'float'}
    assert interfaces['other']['parameters'] == [
        {'name': 'policy', 'kind': 'POSITIONAL_OR_KEYWORD'},
        {'name': 'period', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': '0'}]


def test_exported_graph_uses_explicit_alias_and_nested_target_identity():
    manifest = {'schema': 'modelx.formula_sources.v2', 'model': 'Example',
                'spaces': {
                    'Example.Root': {'parameters': [], 'references': ['Alias'],
                                     'reference_targets': {'Alias': 'Example.Root.Child'},
                                     'cells': {}},
                    'Example.Root.Child': {'parameters': [], 'references': [],
                                           'cells': {}},
                }}
    graph = describe_source_graph(manifest)
    assert graph['spaces']['Example.Root']['space_references'] == {
        'Alias': 'Example.Root.Child'}
    assert graph['edges'] == [{'from': 'Example.Root', 'reference': 'Alias',
                               'to': 'Example.Root.Child'}]
    manifest['spaces']['Example.Root']['reference_targets']['Alias'] = 'Example.Missing'
    with pytest.raises(ContractError, match='absent from export'):
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


@pytest.mark.parametrize('preserve_metadata', [False, True])
@pytest.mark.parametrize('use_slots', [False, True])
def test_original_export_compiles_static_space_without_policy_table(
        tmp_path, preserve_metadata, use_slots):
    import modelx as mx
    from modelx_cython._frame_runtime.compiler import build, compile_generated
    from modelx_cython._frame_runtime.model import CompiledModel

    model = mx.new_model('OldStyle')
    try:
        space = model.new_space('Projection')
        space.factor = 2.0
        space.new_cells('amount', 'lambda: factor + 0.5')
        exported = tmp_path / 'old_export'
        model.export(exported, use_slots=use_slots)
    finally:
        model.close()
    if not preserve_metadata:
        (exported / '_mx_formula_sources.json').unlink()
    spec = from_exported(exported, tmp_path / 'source')
    assert spec.model['point_table'] is None
    assert 'mxcf_static_scope' not in spec.initial_tables
    if not preserve_metadata:
        assert (tmp_path / 'source' / 'recovered_formula_sources.json').exists()
        assert (tmp_path / 'source' / 'frontend.json').read_text().find(
            'reconstructed_export_methods') >= 0
    artifact = tmp_path / 'artifact'
    build(spec, artifact, compile_native=False)
    assert not list(artifact.rglob('*.pyd'))
    compile_generated(artifact)
    compiled = CompiledModel(artifact, spec.initial_tables)
    try:
        assert compiled.Projection.amount() == 2.5
        assert compiled.Projection.amount() == 2.5
        compiled.Projection.factor = 4.0
        assert compiled.Projection.amount() == 4.5
    finally:
        compiled.close()
    source = artifact / json.loads((artifact / 'manifest.json').read_text())['package'] / '_kernel.pyx'
    source_before = source.read_text(encoding='utf-8')
    source.write_text(source_before + '\n# stale translation\n',
                      encoding='utf-8')
    with pytest.raises(ContractError, match='differs from translation receipt'):
        compile_generated(artifact)
    source.write_text(source_before, encoding='utf-8')
    setup = artifact / 'setup.py'
    setup.write_text(setup.read_text(encoding='utf-8') + '\n# stale setup\n',
                     encoding='utf-8')
    with pytest.raises(ContractError, match='setup.py differs'):
        compile_generated(artifact)


def test_metadata_free_original_export_preserves_dynamic_parameters(tmp_path):
    import modelx as mx
    import pandas as pd
    from modelx_cython._frame_runtime.compiler import build
    from modelx_cython._frame_runtime.model import CompiledModel

    model = mx.new_model('OldDynamic')
    try:
        space = model.new_space('Projection', formula='lambda policy_id: None')
        space.new_cells('amount', 'lambda: policy_id * 2.0')
        exported = tmp_path / 'old_dynamic'
        model.export(exported)
    finally:
        model.close()
    (exported / '_mx_formula_sources.json').unlink()
    from modelx_cython._frame_runtime.modelx_frontend_exported import open_exported_source
    with open_exported_source(exported) as derive:
        recovered = derive.source_manifest
        assert recovered['source_provenance'] == 'reconstructed_export_methods'
        assert recovered['spaces']['OldDynamic.Projection']['cells']['amount']['source']
        assert derive.source_graph['spaces']['OldDynamic.Projection']['cells']['amount']
    records = pd.DataFrame(index=pd.Index([1, 3], name='policy_id'))
    spec = from_exported(exported, tmp_path / 'source_dynamic',
                         policy_records=records, point_table='points')
    artifact = tmp_path / 'artifact_dynamic'
    build(spec, artifact)
    compiled = CompiledModel(artifact, spec.initial_tables)
    try:
        assert compiled.Projection[1].amount() == 2.0
        assert compiled.Projection[3].amount() == 6.0
        assert compiled.Projection[1].amount() == 2.0
    finally:
        compiled.close()


@pytest.mark.parametrize('preserve_metadata', [False, True])
def test_original_dynamic_export_needs_no_policy_table(tmp_path, preserve_metadata):
    import modelx as mx
    from modelx_cython._frame_runtime.compiler import build
    from modelx_cython._frame_runtime.model import CompiledModel

    model = mx.new_model('UnboundedDynamic')
    try:
        space = model.new_space('Projection', formula='lambda policy_id: None')
        space.new_cells('amount', 'lambda: policy_id * 2.0')
        space.new_cells('label', 'lambda: ("policy", policy_id)')
        exported = tmp_path / 'exported'
        model.export(exported)
    finally:
        model.close()
    if not preserve_metadata:
        (exported / '_mx_formula_sources.json').unlink()
    spec = from_exported(exported, tmp_path / 'source')
    assert spec.model['point_table'] is None
    assert spec.initial_tables == {}
    artifact = tmp_path / 'artifact'
    build(spec, artifact)
    compiled = CompiledModel(artifact, {})
    try:
        first = compiled.Projection[1]
        second = compiled.Projection[3]
        assert first.amount() == 2.0
        assert second.amount() == 6.0
        assert first.amount() == 2.0
        assert first.label() == ('policy', 1)
        assert second.label() == ('policy', 3)
        assert compiled.Projection['S'].label() == ('policy', 'S')
        assert first is compiled.Projection[1]
        compiled.reset()
        assert first.amount() == 2.0
        assert second.label() == ('policy', 3)
    finally:
        compiled.close()


def test_metadata_free_export_preserves_cell_defaults_and_keyword_calls(tmp_path):
    import modelx as mx
    import pandas as pd
    from modelx_cython._frame_runtime.compiler import build
    from modelx_cython._frame_runtime.model import CompiledModel

    model = mx.new_model('OldSignatures')
    try:
        space = model.new_space('Projection', formula='lambda policy_id: None')
        space.new_cells('amount', 'lambda t=2: policy_id + t')
        space.new_cells('label', 'lambda t=2: ("policy", policy_id, t)')
        exported = tmp_path / 'old_signatures'
        model.export(exported)
    finally:
        model.close()
    (exported / '_mx_formula_sources.json').unlink()
    records = pd.DataFrame(index=pd.Index([1, 3], name='policy_id'))
    spec = from_exported(exported, tmp_path / 'source_signatures',
                         policy_records=records, point_table='points')
    artifact = tmp_path / 'artifact_signatures'
    build(spec, artifact)
    compiled = CompiledModel(artifact, spec.initial_tables)
    try:
        item = compiled.Projection[1]
        assert item.amount() == 3
        assert item.amount(t=3) == 4
        assert item.amount(4) == 5
        assert item.amount() == 3
        assert item.label(t=4) == ('policy', 1, 4)
        assert item.label() == ('policy', 1, 2)
        with pytest.raises(TypeError):
            item.amount(2, t=3)
    finally:
        compiled.close()
