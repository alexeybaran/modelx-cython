"""The opt-in frame build must use the standard trace and source handoff."""

import json
from types import SimpleNamespace

import pytest

from modelx_cython import cli


def _args():
    return SimpleNamespace(
        translate_only=False, compile_only=False, no_spec=True, setup='',
        sample='sample.py', frame_space='Projection',
        frame_point_table='points', frame_steps=480)


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
    monkeypatch.setattr(modelx_frontend_exported, 'from_exported',
                        lambda *a, **kw: (calls.append(('source', a, kw)) or spec))
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
    monkeypatch.setattr(modelx_frontend_exported, 'from_exported', lambda *_a, **_k:
                        SimpleNamespace(model={'collect': None}))
    built = []
    monkeypatch.setattr(compiler, 'build', lambda spec, destination: built.append((spec, destination)))

    assert cli._frame_collect_handler(_args(), exported) == 0
    assert built[0][1] == tmp_path / 'exported_cy'


def test_frame_cli_accepts_metadata_free_original_export(tmp_path, monkeypatch):
    exported = tmp_path / 'exported'
    exported.mkdir()
    monkeypatch.setattr(cli, 'run_sample', lambda *_: object())
    from modelx_cython._frame_runtime import trace_contracts, modelx_frontend_exported, compiler
    monkeypatch.setattr(trace_contracts, 'from_mx2cy_trace',
                        lambda *_a, **_k: pytest.fail('no source manifest to trace'))
    seen = []
    monkeypatch.setattr(modelx_frontend_exported, 'from_exported',
                        lambda *_a, **kwargs: seen.append(kwargs) or object())
    monkeypatch.setattr(compiler, 'build', lambda *_a: None)
    assert cli._frame_collect_handler(_args(), exported) == 0
    assert seen[0]['contracts'] == {}


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
    monkeypatch.setattr(modelx_frontend_exported, 'from_exported',
                        lambda *_a, **_k: object())
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
