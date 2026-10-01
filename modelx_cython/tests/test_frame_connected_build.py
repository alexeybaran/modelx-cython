"""Collect-connected compilation is selected from source metadata, not names."""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelx_cython._frame_runtime.connected_build import (
    build_collect_package, plan_collect_regions, plan_package, build_package)
from modelx_cython._frame_runtime.contracts import ContractError


def _graph():
    def node(refs=(), *, collect=False):
        return {'cells': {'value': 'source'}, 'space_references': {
            name: 'Example.' + name for name in refs},
            'collect': {'over': 'policy_id'} if collect else None}
    return {'schema': 'mxcf.exported_space_graph.v1', 'model': 'Example',
            'spaces': {'Example.Inputs': node(),
                       'Example.Policies': node(('Inputs',), collect=True),
                       'Example.Scenarios': node(('Inputs',), collect=True),
                       'Example.Report': node(('Policies', 'Scenarios')),
                       'Example.Unrelated': node()}}


def test_collect_regions_include_all_roots_and_consumers_in_dependency_order():
    plan = plan_collect_regions(_graph(), external_names=('Inputs',))
    assert plan['collect_roots'] == ['Policies', 'Scenarios']
    assert set(plan['compiled_spaces']) == {'Policies', 'Scenarios', 'Report'}
    assert plan['compiled_spaces'][-1] == 'Report'
    assert plan['external_spaces'] == ['Inputs']
    assert 'Unrelated' not in plan['compiled_spaces']


def test_collect_regions_reject_missing_input_and_plan_reference_cycles():
    with pytest.raises(ContractError, match='Unbound external Space'):
        plan_collect_regions(_graph())
    graph = _graph()
    graph['spaces']['Example.Policies']['space_references']['Report'] = 'Example.Report'
    plan = plan_collect_regions(graph, external_names=('Inputs',))
    assert set(plan['compiled_spaces']) == {'Policies', 'Scenarios', 'Report'}


def test_complete_package_uses_same_graph_without_collect_declarations():
    graph = _graph()
    for node in graph['spaces'].values():
        node['collect'] = None
    plan = plan_package(graph)
    assert plan['collect_roots'] == []
    assert set(plan['compiled_spaces']) == {
        'Inputs', 'Policies', 'Scenarios', 'Report', 'Unrelated'}
    assert plan['compiled_spaces'].index('Inputs') < plan['compiled_spaces'].index('Policies')
    assert plan['compiled_spaces'].index('Policies') < plan['compiled_spaces'].index('Report')
    assert plan['external_spaces'] == []


@pytest.mark.parametrize('fixture, model_name, spaces', [
    ('ref_space', 'RefSpace', {'Bar', 'Bar.Qux', 'Foo'}),
    ('nested_params', 'NestedParams',
     {'TopLevel', 'TopLevel.SecondLevel', 'TopLevel.SecondLevel.ThirdLevel'}),
    ('duplicated_params', 'DuplicatedParams',
     {'Space1', 'Space1.Space2', 'Space1.Space2.Space3'}),
])
def test_original_nested_exports_reach_one_package_plan(
        tmp_path, fixture, model_name, spaces):
    import modelx as mx
    from modelx_cython._frame_runtime.modelx_frontend_exported import open_exported_source

    source = Path(__file__).parent / 'samples' / fixture / model_name
    exported = tmp_path / 'exported'
    model = mx.read_model(source)
    try:
        model.export(exported)
    finally:
        model.close()
    with open_exported_source(exported) as derive:
        plan = plan_package(derive.source_graph)
        if fixture == 'ref_space':
            assert derive.source_graph['spaces']['RefSpace.Foo'][
                'base_parameter_values'] == {'i': 1}
            parent = derive(tmp_path / 'parent_sources', projection='Bar')
            assert parent.model['projection'] == 'Bar'
            assert set(parent.cells) == {'baz'}
    assert set(plan['compiled_spaces']) == spaces
    assert plan['collect_roots'] == []
    assert plan['external_spaces'] == []


def test_one_build_invocation_uses_discovered_spaces_not_option_keys(tmp_path, monkeypatch):
    from modelx_cython._frame_runtime import compiler, connected_package
    from modelx_cython._frame_runtime import modelx_frontend_exported

    source = 'def value():\n    return 1\n'
    def space(references, collect=None):
        return {'parameters': [], 'references': list(references),
                'cells': {'value': {'source': source,
                                    'sha256': hashlib.sha256(source.encode()).hexdigest()}},
                'formula': None, 'collect': collect}
    exported = tmp_path / 'exported'
    exported.mkdir()
    (exported / '_mx_formula_sources.json').write_text(json.dumps({
        'schema': 'modelx.formula_sources.v2', 'model': 'OtherModel',
        'spaces': {
            'OtherModel.Inputs': space(()),
            'OtherModel.Policies': space(('Inputs',),
                {'over': 'policy_id', 'domain': {'kind': 'reference', 'name': 'ids'}}),
            'OtherModel.Total': space(('Policies',)),
            'OtherModel.Unrelated': space(()),
        }}))
    generated = []
    sessions = []
    def fake_frontend(package, destination, **options):
        generated.append(('frontend', options['projection'], options.get('steps')))
        return SimpleNamespace(projection=options['projection'])
    def fake_build(spec, artifact):
        generated.append(('build', spec.projection))
        artifact.mkdir(parents=True)
    def fake_assemble(package, graph, artifacts, **kwargs):
        generated.append(('assemble', sorted(artifacts)))
        return {'native': {}}
    @contextmanager
    def fake_source(package):
        sessions.append(package)
        def derive(destination, **options):
            return fake_frontend(package, destination, **options)
        from modelx_cython._frame_runtime.source_graph import describe_source_graph
        derive.source_graph = describe_source_graph(json.loads(
            (package / '_mx_formula_sources.json').read_text()))
        yield derive
    monkeypatch.setattr(modelx_frontend_exported, 'open_exported_source', fake_source)
    monkeypatch.setattr(compiler, 'build', fake_build)
    monkeypatch.setattr(connected_package, 'assemble_connected_package', fake_assemble)
    output = tmp_path / 'output'
    result = build_collect_package(exported, output,
        external_imports={'Inputs': 'provider:inputs'},
        space_options={'Policies': {'steps': 3}}, package_name='CompiledOther')
    assert result['plan']['compiled_spaces'] == ['Policies', 'Total']
    assert generated == [('frontend', 'Policies', 3), ('build', 'Policies'),
                         ('frontend', 'Total', None), ('build', 'Total'),
                         ('assemble', ['Policies', 'Total'])]
    assert result['package'] == output / 'CompiledOther'
    assert sessions == [exported]
    assert json.loads((output / 'build_plan.json').read_text())['collect_roots'] == ['Policies']


def test_complete_package_builds_single_space_without_collect(tmp_path, monkeypatch):
    from modelx_cython._frame_runtime import compiler, connected_package
    from modelx_cython._frame_runtime import modelx_frontend_exported

    source = 'def value():\n    return 1\n'
    exported = tmp_path / 'exported'
    exported.mkdir()
    (exported / '_mx_formula_sources.json').write_text(json.dumps({
        'schema': 'modelx.formula_sources.v2', 'model': 'Single',
        'spaces': {'Single.Space': {
            'parameters': [], 'references': [], 'formula': None,
            'cells': {'value': {'source': source,
                                'sha256': hashlib.sha256(source.encode()).hexdigest()}},
        }}}))
    called = []
    @contextmanager
    def fake_source(_package):
        def derive(destination, **options):
            return SimpleNamespace(projection=options['projection'])
        from modelx_cython._frame_runtime.source_graph import describe_source_graph
        derive.source_graph = describe_source_graph(json.loads(
            (_package / '_mx_formula_sources.json').read_text()))
        yield derive
    def fake_build(spec, artifact):
        called.append(spec.projection)
        artifact.mkdir(parents=True)
    monkeypatch.setattr(modelx_frontend_exported, 'open_exported_source', fake_source)
    monkeypatch.setattr(compiler, 'build', fake_build)
    monkeypatch.setattr(connected_package, 'assemble_connected_package',
                        lambda *args, **kwargs: {})
    result = build_package(exported, tmp_path / 'out', external_imports={})
    assert called == ['Space']
    assert result['plan']['compiled_spaces'] == ['Space']
    assert result['plan']['collect_roots'] == []


@pytest.mark.parametrize('preserve_metadata', [True, False])
def test_complete_package_compiles_ordinary_single_space(tmp_path, preserve_metadata):
    import modelx as mx
    from modelx_cython._frame_runtime.connected_build import build_package
    from modelx_cython._frame_runtime.connected_model import load_connected

    model = mx.new_model('OrdinaryPackage')
    try:
        model.new_space('Projection').new_cells('value', 'lambda: 4.25')
        exported = tmp_path / 'exported'
        model.export(exported)
        if not preserve_metadata:
            (exported / '_mx_formula_sources.json').unlink()
    finally:
        model.close()
    result = build_package(exported, tmp_path / 'compiled', external_imports={})
    assert result['plan']['collect_roots'] == []
    connected = load_connected(
        tmp_path / 'compiled' / '_sources' / 'Projection' / 'source_graph.json',
        result['artifacts'], external_bindings={})
    try:
        assert connected.Projection.value() == 4.25
    finally:
        connected.close()


def test_complete_connected_package_imports_an_eager_model_without_providers(
        tmp_path, monkeypatch):
    import importlib
    import sys
    import modelx as mx

    model = mx.new_model('EagerNative')
    try:
        model.new_space('Projection').new_cells('value', 'lambda: 4.25')
        exported = tmp_path / 'exported'
        model.export(exported)
    finally:
        model.close()
    result = build_package(exported, tmp_path / 'compiled', external_imports={},
                           package_name='EagerNativeCompiled', eager=True)
    assert result['manifest']['external_imports'] == {}
    monkeypatch.syspath_prepend(str(tmp_path / 'compiled'))
    package = importlib.import_module('EagerNativeCompiled')
    try:
        assert package.mx_model is package.EagerNative
        assert package.mx_model.Projection.value() == 4.25
    finally:
        package.mx_model.close()
        sys.modules.pop(package.__name__, None)


def test_computational_data_space_is_not_executed_as_build_input(tmp_path):
    import modelx as mx
    from modelx_cython._frame_runtime.connected_package import load_connected_package

    model = mx.new_model('ComputationalData')
    try:
        data = model.new_space('Data')
        data.calls = {'scale': 0}
        data.log = []
        data.new_cells('scale', 'def scale():\n'
                     '    calls["scale"] += 1\n'
                     '    log.append("scale")\n'
                     '    return 2.0')
        data.new_cells('rate', 'lambda t: scale() * t')
        projection = model.new_space('Projection')
        projection.Data = data
        projection.data = data
        projection.log = data.log
        projection.new_cells('value', 'lambda t: Data.rate(t)')
        projection.new_cells('value_lower', 'def value_lower(t):\n'
                             '    log.append(t)\n'
                             '    return data.rate(t)')
        exported = tmp_path / 'exported'
        model.export(exported)
    finally:
        model.close()

    result = build_package(exported, tmp_path / 'compiled', external_imports={})
    connected = load_connected_package(result['package'] / '__init__.py')
    try:
        assert connected.Data.calls == {'scale': 0}
        assert connected.Projection.value(2) == 4.0
        assert connected.Projection.value(2) == 4.0
        assert connected.Projection.value_lower(2) == 4.0
        assert connected.Data.log is connected.Projection.log
        assert connected.Data.log == ['scale', 2]
        assert connected.Data.calls == {'scale': 1}
    finally:
        connected.close()


def test_original_locked_export_preserves_nested_parent_and_shared_references(
        tmp_path, monkeypatch):
    import math
    import shutil
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from modelx_cython._frame_runtime.connected_package import load_connected_package

    fixture = Path(__file__).parent / 'samples' / 'locked_spaces'
    source = tmp_path / 'source'
    shutil.copytree(fixture, source)
    subprocess.run([sys.executable, str(source / 'build_model.py')],
                   cwd=source, check=True)
    monkeypatch.syspath_prepend(str(source))
    from LockedSpaces_nomx import LockedSpaces as exported

    result = build_package(source / 'LockedSpaces_nomx', tmp_path / 'built',
                           external_imports={})
    compiled = load_connected_package(result['package'] / '__init__.py')
    try:
        assert compiled.Data.log is compiled.Table.log is compiled.Table.Child.log
        assert compiled.Table['B'].Child.tbl is compiled.Table['B']
        def values(policy):
            return [compiled.Projection[policy].pv(t) for t in range(8)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            actual = list(pool.map(values, range(16)))
        for policy, row in enumerate(actual):
            for period, value in enumerate(row):
                assert math.isclose(value, exported.Projection[policy].pv(period),
                                    rel_tol=1e-9, abs_tol=1e-7)
        log = compiled.Data.log
        assert len(log) == len(set(log))
    finally:
        compiled.close()


def test_complete_package_wires_real_cyclic_space_references(tmp_path):
    import modelx as mx
    from modelx_cython._frame_runtime.connected_build import build_package
    from modelx_cython._frame_runtime.connected_model import load_connected

    model = mx.new_model('CyclicSpaces')
    try:
        first = model.new_space('First')
        second = model.new_space('Second')
        first.Other = second
        second.Other = first
        first.new_cells('value', 'lambda: Other.value()')
        second.new_cells('value', 'lambda: 7')
        second.new_cells('via_first', 'lambda: Other.value()')
        exported = tmp_path / 'cyclic_export'
        model.export(exported)
    finally:
        model.close()
    result = build_package(
        exported, tmp_path / 'cyclic_compiled', external_imports={})
    graph = tmp_path / 'cyclic_compiled' / '_sources' / 'First' / 'source_graph.json'
    connected = load_connected(graph, result['artifacts'], external_bindings={})
    try:
        assert connected.First.value() == 7
        assert connected.Second.via_first() == 7
        assert connected.First.Other is connected.Second
        assert connected.Second.Other is connected.First
    finally:
        connected.close()


@pytest.mark.parametrize('use_slots', [True, False])
def test_complete_package_preserves_nested_references_and_base_parameters(
        tmp_path, use_slots):
    import modelx as mx
    from modelx_cython._frame_runtime.connected_package import load_connected_package

    source = Path(__file__).parent / 'samples' / 'ref_space' / 'RefSpace'
    live = mx.read_model(source)
    try:
        exported = tmp_path / 'exported'
        live.export(exported, use_slots=use_slots)
        expected = [(live.Foo[i].foo(), live.Foo[i].quux()) for i in range(10)]
        base = live.Bar.Qux.corge()
    finally:
        live.close()

    result = build_package(exported, tmp_path / 'compiled', external_imports={})
    assert result['plan']['generated_spaces']['Bar.Qux'] != 'Bar.Qux'
    connected = load_connected_package(result['package'] / '__init__.py')
    try:
        assert connected.Bar.Qux.corge() == connected.Foo.foo() == base
        assert [(connected.Foo[i].foo(), connected.Foo[i].quux())
                for i in range(10)] == expected
    finally:
        connected.close()


def test_complete_package_keeps_static_cell_free_parent_as_space_shell(tmp_path):
    import modelx as mx
    from modelx_cython._frame_runtime.connected_package import load_connected_package

    live = mx.new_model('StaticNested')
    try:
        parent = live.new_space('Parent')
        child = parent.new_space('Child')
        child.new_cells('value', 'lambda: 7')
        exported = tmp_path / 'exported'
        live.export(exported)
    finally:
        live.close()

    result = build_package(exported, tmp_path / 'compiled', external_imports={})
    assert result['plan']['compiled_spaces'] == ['Parent', 'Parent.Child']
    assert set(result['artifacts']) == {'Parent.Child'}
    assert result['manifest']['shells'] == ['Parent']
    connected = load_connected_package(result['package'] / '__init__.py')
    try:
        assert connected.Parent.Child.value() == 7
        assert connected.Parent.Child is connected._spaces['Parent.Child']
    finally:
        connected.close()


@pytest.mark.parametrize('fixture, model_name, preserve_metadata', [
    ('nested_params', 'NestedParams', True),
    ('nested_params', 'NestedParams', False),
    ('duplicated_params', 'DuplicatedParams', True),
    ('duplicated_params', 'DuplicatedParams', False),
])
def test_parameterized_cell_free_parents_bind_inherited_values(
        tmp_path, fixture, model_name, preserve_metadata):
    import modelx as mx
    from modelx_cython._frame_runtime.connected_package import load_connected_package

    source = Path(__file__).parent / 'samples' / fixture / model_name
    live = mx.read_model(source)
    try:
        exported = tmp_path / 'exported'
        live.export(exported)
        if not preserve_metadata:
            (exported / '_mx_formula_sources.json').unlink()
        if fixture == 'nested_params':
            expected = [live.TopLevel[i].SecondLevel[j].ThirdLevel[k].Cells1()
                        for i, j, k in ((1, 2, 3), (2, 3, 4), (0, 0, 0))]
        else:
            expected = [live.Space1[i, j].Space2[k, l].Space3.Cells1()
                        for i, j, k, l in ((1, 2, 3, 4), (2, 9, 5, 6), (0, 0, 0, 0))]
    finally:
        live.close()

    result = build_package(exported, tmp_path / 'compiled', external_imports={})
    connected = load_connected_package(result['package'] / '__init__.py')
    try:
        if fixture == 'nested_params':
            actual = [connected.TopLevel[i].SecondLevel[j].ThirdLevel[k].Cells1()
                      for i, j, k in ((1, 2, 3), (2, 3, 4), (0, 0, 0))]
            assert connected.TopLevel[1] is connected.TopLevel[1]
            assert connected.TopLevel[1].SecondLevel[2] is connected.TopLevel[1].SecondLevel[2]
            assert connected.TopLevel[1].SecondLevel[2].ThirdLevel[3].Cells1() == expected[0]
            assert connected.TopLevel(i=1).SecondLevel(j=2).ThirdLevel(k=3).Cells1() == expected[0]
        else:
            actual = [connected.Space1[i, j].Space2[k, l].Space3.Cells1()
                      for i, j, k, l in ((1, 2, 3, 4), (2, 9, 5, 6), (0, 0, 0, 0))]
            assert connected.Space1[1, 2] is connected.Space1[1, 2]
            assert connected.Space1[1, 2].Space2[3, 4].Space3.Cells1() == expected[0]
            assert connected.Space1(1, 2).Space2(3, 4).Space3.Cells1() == expected[0]
        assert actual == expected
    finally:
        connected.close()


def test_complete_package_preserves_uncached_and_fractional_power_cells(tmp_path):
    import modelx as mx
    from modelx_cython._frame_runtime.connected_package import load_connected_package

    source = Path(__file__).parent / 'samples' / 'fractional_power' / 'FractionalPower'
    live = mx.read_model(source)
    try:
        exported = tmp_path / 'exported'
        live.export(exported)
        expected = [(live.Space1.uncached_rate(t), live.Space1.mth_rate(t),
                     live.Space1.int_pow(t)) for t in range(11)]
    finally:
        live.close()
    result = build_package(exported, tmp_path / 'compiled', external_imports={})
    assert result['plan']['collect_roots'] == []
    connected = load_connected_package(result['package'] / '__init__.py')
    try:
        for t, values in enumerate(expected):
            assert connected.Space1.uncached_rate(t) == pytest.approx(values[0])
            assert connected.Space1.mth_rate(t) == pytest.approx(values[1])
            assert connected.Space1.int_pow(t) == values[2]
        assert connected.Space1.uncached_rate(2) == pytest.approx(expected[2][0])
    finally:
        connected.close()


def test_complete_package_uncached_cell_repeats_side_effects(tmp_path):
    import modelx as mx
    from modelx_cython._frame_runtime.connected_package import load_connected_package

    live = mx.new_model('UncachedConnected')
    try:
        space = live.new_space('Space1')
        space.counter = {'n': 0}
        space.new_cells('tick', 'def tick():\n    counter["n"] += 1\n    return counter["n"]')
        space.tick.is_cached = False
        space.new_cells('twice', 'lambda: tick() + tick()')
        exported = tmp_path / 'exported'
        live.export(exported)
    finally:
        live.close()

    result = build_package(exported, tmp_path / 'compiled', external_imports={})
    connected = load_connected_package(result['package'] / '__init__.py')
    try:
        assert connected.Space1.tick() == 1
        assert connected.Space1.tick() == 2
        assert connected.Space1.twice() == 7
        assert connected.Space1.twice() == 7
        assert connected.Space1.tick() == 5
        assert connected.Space1.counter == {'n': 5}
    finally:
        connected.close()


@pytest.mark.parametrize('preserve_metadata', [True, False])
def test_complete_package_preserves_original_various_types(tmp_path, preserve_metadata):
    import modelx as mx
    import numpy as np
    from modelx_cython._frame_runtime.connected_package import load_connected_package

    source = Path(__file__).parent / 'samples' / 'various_types' / 'VariousTypes'
    live = mx.read_model(source)
    try:
        exported = tmp_path / 'exported'
        live.export(exported)
        if not preserve_metadata:
            (exported / '_mx_formula_sources.json').unlink()
        expected = {
            'int': [live.IntArg.foo(10), live.IntArg.bar(10),
                    live.IntArg.baz(11), live.IntArg.qux(10),
                    live.IntArg.quux(10), live.IntArg.grault(10),
                    live.IntArg.garply(11)],
            'static': [live.NoArg.foo(), live.NoArg.qux(),
                       live.NoArg.grault()],
            'multi': [live.IntArgs.foo(1, 2), live.IntArgs.qux(3, 4),
                      live.IntArgs.garply(3, 4)],
        }
    finally:
        live.close()

    result = build_package(exported, tmp_path / 'compiled', external_imports={})
    connected = load_connected_package(result['package'] / '__init__.py')
    try:
        actual = {
            'int': [connected.IntArg.foo(10), connected.IntArg.bar(10),
                    connected.IntArg.baz(11), connected.IntArg.qux(10),
                    connected.IntArg.quux(10), connected.IntArg.grault(10),
                    connected.IntArg.garply(11)],
            'static': [connected.NoArg.foo(), connected.NoArg.qux(),
                       connected.NoArg.grault()],
            'multi': [connected.IntArgs.foo(1, 2), connected.IntArgs.qux(3, 4),
                      connected.IntArgs.garply(3, 4)],
        }
        for group in expected:
            for value, reference in zip(actual[group], expected[group]):
                if isinstance(reference, np.ndarray):
                    np.testing.assert_array_equal(value, reference)
                else:
                    assert value == reference
        # The original CLI's out-of-range assertions depend on the traced
        # capacity specification. This connected build has no such spec; its
        # object/array result parity is a separate coverage gate.
    finally:
        connected.close()
