"""Original discovery/specification results remain authoritative for emission."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _translate(folder, model, sample, spec=None):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(folder) + os.pathsep + env.get('PYTHONPATH', '')
    args = [sys.executable, '-m', 'modelx_cython', str(folder / model),
            '--sample', str(folder / sample), '--translate-only']
    args += ['--spec', str(folder / spec)] if spec else ['--no-spec']
    run = subprocess.run(args, env=env, cwd=folder, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert run.returncode == 0, run.stdout
    return json.loads((folder / (model + '_cy') / '_mx_compilation_plan.json').read_text())


@pytest.mark.parametrize('sample,model', [
    ('nested_params', 'NestedParams'),
    ('duplicated_params', 'DuplicatedParams'),
    ('ref_space', 'RefSpace'),
])
def test_plan_preserves_original_scopes_and_reference_targets(tmp_path, sample, model):
    import modelx as mx
    src = Path(__file__).parent / 'samples' / sample
    shutil.copytree(src, tmp_path, dirs_exist_ok=True)
    live = mx.read_model(tmp_path / model)
    try:
        live.export(tmp_path / (model + '_nomx'))
    finally:
        live.close()
    # Rich author-source metadata is optional, including for nested models.
    (tmp_path / (model + '_nomx') / '_mx_formula_sources.json').unlink(missing_ok=True)
    manifest = _translate(tmp_path, model + '_nomx', 'sample.py', 'spec.py')
    spaces = manifest['spaces']
    for identity, space in spaces.items():
        assert space['identity'] == identity
        for child in space['children']:
            assert child in spaces
            assert spaces[child]['parent'] == identity
        for target in space['reference_targets'].values():
            assert target in spaces
    if sample == 'nested_params':
        prefix = model + '_nomx_cy.TopLevel'
        assert prefix + '.SecondLevel.ThirdLevel' in spaces
        assert spaces[prefix]['parameters'] == ['i']
        assert spaces[prefix + '.SecondLevel']['parameters'] == ['j']
        assert spaces[prefix + '.SecondLevel.ThirdLevel']['parameters'] == ['k']
    elif sample == 'duplicated_params':
        prefix = model + '_nomx_cy.Space1'
        assert spaces[prefix]['parameters'] == ['i', 'j']
        assert spaces[prefix + '.Space2']['parameters'] == ['j', 'k']
    else:
        prefix = model + '_nomx_cy'
        assert spaces[prefix + '.Foo']['reference_targets']['bar'] == prefix + '.Bar'


def test_plan_preserves_final_types_uncached_defaults_and_capacity(tmp_path):
    import modelx as mx
    live = mx.new_model('PlanTypes')
    s = live.new_space('S')
    s.calls = {}
    s.new_cells('dense', 'def dense(i, j):\n return float(i + j)\n')
    s.new_cells('observed', 'def observed(i):\n calls[i] = calls.get(i, 0) + 1\n return i\n')
    s.observed.is_cached = False
    s.new_cells('defaulted', 'def defaulted(i=2):\n return dense(i, 1)\n')
    s.new_cells('unseen', 'def unseen():\n raise RuntimeError("not traced")\n')
    s.new_cells('receiver_value', 'def receiver_value():\n return _space\n')
    live.export(tmp_path / 'model')
    live.close()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\n'
        'import numpy as np\n'
        's=mx_model.S\ns.dense(np.int64(14),3)\ns.observed(1)\ns.defaulted()\n', encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'spaces': {'S': {
        'cells_param_size': {('i', 'j'): (9, 13)},
        'cells': {'dense': {'return_type': 'object'}}}}}), encoding='utf-8')
    manifest = _translate(tmp_path, 'model', 'sample.py', 'spec.py')
    cells = manifest['spaces']['model_cy.S']['cells']
    assert cells['dense']['public_result'] == 'object'
    assert cells['dense']['cache_kind'] == 'dictionary'  # explicit object override
    assert cells['observed']['cache_kind'] == 'uncached'
    assert not cells['unseen']['traced'] and cells['unseen']['public_result'] == 'object'
    assert cells['receiver_value']['receiver_requirements'] == ['original receiver member: _space']
    assert 'i=2' in cells['defaulted']['signature']
    assert cells['defaulted']['dependencies'] == ['dense']
    # Without the return override the *same* original spec determines both axes.
    (tmp_path / 'spec2.py').write_text(repr({'spaces': {'S': {
        'cells_param_size': {('i', 'j'): (9, 13)}}}}), encoding='utf-8')
    second = _translate(tmp_path, 'model', 'sample.py', 'spec2.py')
    dense = second['spaces']['model_cy.S']['cells']['dense']
    assert dense['cache_domains'] == [
        {'argument': 'i', 'lower': 0, 'upper': 15},
        {'argument': 'j', 'lower': 0, 'upper': 13}]
