"""The optional exporter ABI tolerates formatting, but not unknown semantics."""
import ast
from pathlib import Path

import libcst as cst
import pytest

from modelx_cython.collect_adapter import CollectRuntime, _native
from modelx_cython.helpers import read_resource


def current():
    return read_resource('modelx.export', '_mx_sys.py').decode('utf-8')


def legacy():
    return (Path(__file__).parent/'samples/collect_v0.txt').read_text(encoding='utf-8')


@pytest.mark.parametrize('source', [legacy])
def test_known_legacy_upgrade_is_structural_and_preserves_other_code(source):
    code = source() + '\nclass Authored:\n    value = "unchanged"\n'
    # The old exact-text adapter rejected both of these harmless edits.
    code = code.replace('def collect_from_space(base,', 'def collect_from_space(\n    base,')
    code = code.replace('_mx_parallel_workers = 1', '_mx_parallel_workers  =  1 # exported')
    adapter = CollectRuntime(code)
    assert adapter.supported
    output = adapter.render(4, native=True)
    assert _native(cst.parse_module(output))
    assert 'class Authored:\n    value = "unchanged"' in output
    assert '_mx_parallel_workers  =  4 # exported' in output
    assert CollectRuntime(output).render(4, native=True) == output
    assert not _native(cst.parse_module(adapter.render(4)))


@pytest.mark.parametrize('change', [
    lambda code: code.replace('key = (coords, args)', 'key = (args, coords)'),
    lambda code: code + '\ndef _mx_collect_chunk(*args): pass\n',
    lambda code: code + '\n_mx_collect_adapter_version = 17\n',
    lambda code: code + '\n_mx_collect_native_coords = object()\n',
    lambda code: code + '\ndef _mx_collect_adapter_version(): pass\n',
    lambda code: code.replace('scratch = {}', 'scratch = {"authored": True}'),
    lambda code: code.replace('scratch = {}', 'scratch = {"authored": f"{base}" "text"}'),
])
def test_unknown_adapter_keeps_the_general_runtime(change):
    source = change(legacy())
    adapter = CollectRuntime(source)
    assert not adapter.supported
    assert adapter.render() == source
    with pytest.raises(ValueError, match='unsupported native'):
        adapter.render(native=True)


def test_native_export_requires_no_runtime_transplant():
    source = current()
    adapter = CollectRuntime(source)
    assert adapter.supported and adapter.template is None
    assert adapter.render(native=True) == source
    output = adapter.render(7, native=True)
    tree = cst.parse_module(output)
    assert tree.code == output
    assert _native(tree, 2 if adapter.hook else 1)
    if adapter.hook:
        assert output == source


def test_legacy_export_with_older_modelx_keeps_ordinary_collection(monkeypatch):
    from modelx_cython import collect_adapter
    monkeypatch.setattr(collect_adapter, 'read_resource', lambda *args: legacy().encode())
    adapter = CollectRuntime(legacy())
    assert not adapter.supported
    assert adapter.render() == legacy()


def test_parallel_worker_setting_requires_an_unambiguous_export():
    for source in ('pass\n', '_mx_parallel_workers = 1\n_mx_parallel_workers = 2\n'):
        with pytest.raises(ValueError, match='collection-enabled'):
            CollectRuntime(source).render(4)


def test_legacy_protocol_uses_cst_and_ignores_parenthesized_layout():
    from modelx_cython.collect_adapter import _legacy
    source = legacy().replace('key = (coords, args)', 'key = ((coords, args))')
    assert _legacy(cst.parse_module(source))
