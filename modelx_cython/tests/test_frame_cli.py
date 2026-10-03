"""The retired frame flag uses the standard public-package compilation pipeline."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from modelx_cython import cli


def test_compatibility_options_share_specification_and_preserve_precedence():
    args = SimpleNamespace(frame_collect=True, frame_space='Top.P',
                           frame_point_table='points', frame_steps=480)
    specification = {'spaces': {'Top': {'spaces': {'P': {
        'cells_param_size': {'n': 9},
        'native_context': {'steps': 32, 'point_table': 'author_points'}}}}}}
    result = cli._frame_option_spec(specification, args)
    p = result['spaces']['Top']['spaces']['P']
    assert p['native_context'] == {'steps':32, 'point_table':'author_points'}
    assert p['cells_param_size'] == {'n':9}
    assert not hasattr(cli, '_frame_collect_handler')


def test_compatibility_options_do_not_invent_a_policy_domain():
    args = SimpleNamespace(frame_collect=True, frame_space='S',
                           frame_point_table=None, frame_steps=480)
    assert cli._frame_option_spec({}, args) == {}
    args.frame_space = '../bad'
    with pytest.raises(cli.HandlerError):
        cli._frame_option_spec({}, args)


@pytest.mark.parametrize('alias', [False, True])
def test_standard_translate_compile_and_original_interfaces_without_metadata(tmp_path, alias):
    import modelx as mx
    m = mx.new_model('UnifiedCLI')
    try:
        m.new_space('Projection').new_cells('value', 'lambda: 4.25')
        m.export(tmp_path/'model')
    finally:
        m.close()
    (tmp_path/'model/_mx_formula_sources.json').unlink()
    sample = tmp_path/'sample.py'
    sample.write_text('from model import mx_model\nmx_model.Projection.value()\n')
    environment = os.environ.copy()
    environment.pop('MX_CY_ORIGINAL_FRAME_SHELL', None)
    environment['PYTHONPATH'] = str(tmp_path)+os.pathsep+environment.get('PYTHONPATH','')
    def run(arguments):
        result = subprocess.run([sys.executable,'-m','modelx_cython',str(tmp_path/'model'),*arguments],
            cwd=tmp_path, env=environment, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout[-2000:]+result.stderr[-2000:]
    options = ['--frame-collect'] if alias else []
    setup = tmp_path/'custom_setup.py'
    run(['--sample',str(sample),'--no-spec','--translate-only','--setup',str(setup),*options])
    package = tmp_path/'model_cy'
    assert (package/'_mx_classes.pxd').exists()
    routes = json.loads((package/'_mx_classes.frame-shell.json').read_text())
    assert any(r['cell']=='value' and r['execution']=='frame_exact' for r in routes['cells'])
    assert not (package/'model.py').exists()  # No replacement CompiledModel facade.
    sample.write_text('raise AssertionError("compile-only must not trace")\n')
    run(['--compile-only','--setup',str(setup),*options])
    result = subprocess.run([sys.executable,'-c',
        'from model_cy import mx_model\nassert mx_model.Projection.value()==4.25\n'
        'assert "modelx_cython" not in __import__("sys").modules\n'],
        cwd=tmp_path,env=environment,text=True,capture_output=True)
    assert result.returncode == 0, result.stderr[-2000:]
