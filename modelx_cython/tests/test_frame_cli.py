"""The opt-in frame build must use the standard trace and source handoff."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from modelx_cython import cli


def _args():
    return SimpleNamespace(
        translate_only=False, compile_only=False, no_spec=True, setup='',
        sample='sample.py', frame_space='Projection',
        frame_point_table='points', frame_steps=480)


def _source_session(manifest, derive):
    @contextmanager
    def open_source(_package):
        derive.source_manifest = manifest
        yield derive
    return open_source


def test_frame_cli_routes_standard_trace_to_source_preserved_build(tmp_path, monkeypatch):
    exported = tmp_path / 'exported'
    exported.mkdir()
    manifest = {'model': 'M', 'spaces': {'M.Projection': {'cells': {'value': {}}}}}
    (exported / '_mx_formula_sources.json').write_text(json.dumps(manifest))
    calls = []
    logger = object()
    contracts = {'value': {'return': float}}
    spec = SimpleNamespace(model={'collect': {'over': 'point', 'sum': ['value']}})

    monkeypatch.setattr(cli, 'run_sample', lambda path, sample: (
        calls.append(('trace', path, sample)) or logger))
    from modelx_cython._frame_runtime import trace_contracts, modelx_frontend_exported, compiler
    monkeypatch.setattr(trace_contracts, 'from_mx2cy_trace', lambda log, source, space: (
        calls.append(('contracts', log, source, space)) or contracts))
    monkeypatch.setattr(modelx_frontend_exported, 'open_exported_source',
                        _source_session(manifest,
                            lambda *a, **kw: (calls.append(('source', a, kw)) or spec)))
    monkeypatch.setattr(compiler, 'build',
                        lambda *a: calls.append(('build', a)))

    assert cli._frame_collect_handler(_args(), exported) == 0
    assert [row[0] for row in calls] == ['trace', 'contracts', 'source', 'build']
    assert calls[2][2]['contracts'] is contracts
    assert calls[2][2]['steps'] == 480
    assert calls[3][1][1] == tmp_path / 'exported_cy'


def test_frame_cli_refuses_overwrite_before_tracing(tmp_path, monkeypatch):
    exported = tmp_path / 'exported'
    exported.mkdir()
    (exported / '_mx_formula_sources.json').write_text('{}')
    (tmp_path / 'exported_cy').mkdir()
    monkeypatch.setattr(cli, 'run_sample', lambda *a: pytest.fail('trace ran'))
    with pytest.raises(FileExistsError, match='already exists'):
        cli._frame_collect_handler(_args(), exported)


def test_frame_cli_builds_a_model_without_collect(tmp_path, monkeypatch):
    exported = tmp_path / 'exported'
    exported.mkdir()
    (exported / '_mx_formula_sources.json').write_text(json.dumps({
        'model': 'M', 'spaces': {'M.Projection': {'cells': {'value': {}}}},
    }))
    monkeypatch.setattr(cli, 'run_sample', lambda *_: object())
    from modelx_cython._frame_runtime import trace_contracts, modelx_frontend_exported, compiler
    monkeypatch.setattr(trace_contracts, 'from_mx2cy_trace', lambda *_a, **_k: {})
    monkeypatch.setattr(modelx_frontend_exported, 'open_exported_source',
                        _source_session({'model': 'M', 'spaces': {'M.Projection': {'cells': {}}}},
                            lambda *_a, **_k: SimpleNamespace(model={'collect': None})))
    built = []
    monkeypatch.setattr(compiler, 'build', lambda spec, destination: built.append((spec, destination)))

    assert cli._frame_collect_handler(_args(), exported) == 0
    assert built[0][1] == tmp_path / 'exported_cy'


def test_frame_cli_accepts_metadata_free_original_export(tmp_path, monkeypatch):
    exported = tmp_path / 'exported'
    exported.mkdir()
    monkeypatch.setattr(cli, 'run_sample', lambda *_: object())
    from modelx_cython._frame_runtime import trace_contracts, modelx_frontend_exported, compiler
    recovered = {'model': 'M', 'spaces': {'M.Projection': {'cells': {'value': {}}}}}
    monkeypatch.setattr(trace_contracts, 'from_mx2cy_trace',
                        lambda _logger, source, **_kw: (
                            {'value': {'return': float}} if source is recovered
                            else pytest.fail('Recovered source manifest was not traced')))
    seen = []
    monkeypatch.setattr(modelx_frontend_exported, 'open_exported_source',
                        _source_session(recovered,
                            lambda *_a, **kwargs: seen.append(kwargs) or object()))
    monkeypatch.setattr(compiler, 'build', lambda *_a: None)
    assert cli._frame_collect_handler(_args(), exported) == 0
    assert seen[0]['contracts'] == {'value': {'return': float}}


def test_frame_cli_translate_then_compile_without_retracing(tmp_path, monkeypatch):
    exported = tmp_path / 'exported'
    exported.mkdir()
    (exported / '_mx_formula_sources.json').write_text(json.dumps({
        'model': 'M', 'spaces': {'M.Projection': {'cells': {'value': {}}}},
    }))
    traced = []
    monkeypatch.setattr(cli, 'run_sample', lambda *_: traced.append(1) or object())
    from modelx_cython._frame_runtime import trace_contracts, modelx_frontend_exported, compiler
    monkeypatch.setattr(trace_contracts, 'from_mx2cy_trace', lambda *_a, **_k: {})
    monkeypatch.setattr(modelx_frontend_exported, 'open_exported_source',
                        _source_session({'model': 'M', 'spaces': {'M.Projection': {'cells': {}}}},
                            lambda *_a, **_k: object()))
    built = []
    monkeypatch.setattr(compiler, 'build',
                        lambda *_a, **kwargs: built.append(kwargs))
    compiled = []
    monkeypatch.setattr(compiler, 'compile_generated',
                        lambda destination: compiled.append(destination))
    args = _args()
    args.translate_only = True
    assert cli._frame_collect_handler(args, exported) == 0
    assert built == [{'compile_native': False}]
    args.translate_only = False
    args.compile_only = True
    assert cli._frame_collect_handler(args, exported) == 0
    assert traced == [1]
    assert compiled == [tmp_path / 'exported_cy']


def test_frame_cli_traces_a_metadata_free_export_before_compiling(tmp_path):
    import modelx as mx
    from modelx_cython._frame_runtime.model import CompiledModel

    model = mx.new_model('RecoveredCLI')
    try:
        model.new_space('Projection').new_cells('value', 'lambda: 4.25')
        exported = tmp_path / 'exported'
        model.export(exported)
    finally:
        model.close()
    (exported / '_mx_formula_sources.json').unlink()
    sample = tmp_path / 'sample.py'
    sample.write_text('from exported import mx_model\n'
                      'assert mx_model.Projection.value() == 4.25\n')
    args = _args()
    args.sample = str(sample)
    args.frame_point_table = None
    args.frame_steps = 2
    assert cli._frame_collect_handler(args, exported) == 0
    artifact = tmp_path / 'exported_cy'
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert manifest['cells']['value']['result']['kind'] == 'float'
    compiled = CompiledModel(artifact, {})
    try:
        assert compiled.Projection.value() == 4.25
    finally:
        compiled.close()
