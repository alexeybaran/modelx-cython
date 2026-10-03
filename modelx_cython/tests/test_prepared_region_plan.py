"""Rich resource declarations use the standard compiler and original classes."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def isolate_export_imports():
    def clear():
        for name in tuple(sys.modules):
            if name in ('model', 'model_cy') or name.startswith(('model.', 'model_cy.')):
                sys.modules.pop(name, None)
    clear()
    yield
    clear()


def test_standard_command_resolves_prepared_context_without_frontend_hook(tmp_path, monkeypatch):
    import modelx as mx
    from modelx_cython import cli
    from lifelib.helpers import lookup
    from modelx_cython.tests.reference_frame import modelx_frontend_exported
    from modelx_cython._frame_runtime import space_shell
    m = mx.new_model('ResourcePlan')
    data = m.new_space('Data')
    data.points_data = pd.DataFrame({'value': [1.5, -1.0]},
                                   index=pd.Index([0, 1], name='policy_id'))
    data.new_cells('points', 'def points():\n return points_data\n')
    policies = m.new_space('Policies', formula='def Policies(policy_id):\n return {"refs": {}}\n')
    policies.data = data
    policies.lookup = lookup
    policies.new_cells('amount', 'def amount():\n return lookup.field(data.points(), policy_id, "value")\n')
    policies.new_cells('report', 'def report():\n return {"amount": amount()}\n')
    policies.new_cells('uncached', 'def uncached():\n return report()["amount"]\n')
    policies.uncached.is_cached = False
    m.export(tmp_path/'model')
    m.close()
    (tmp_path/'model/_mx_formula_sources.json').unlink()
    (tmp_path/'sample.py').write_text('from model import mx_model\n'
        'for i in (0,1):\n p=mx_model.Policies[i]\n p.amount(); p.report(); p.uncached()\n')
    (tmp_path/'spec.py').write_text(repr({'spaces': {'Policies': {
        'native_context': {'point_table': 'points', 'steps': 8}}}}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Standard planning must not reopen the alternate frontend')
    monkeypatch.setattr(modelx_frontend_exported, 'open_exported_source', forbidden)
    monkeypatch.setenv('MX_CY_ORIGINAL_FRAME_SHELL', '0')
    planner = space_shell.plan
    captured = []
    def observe(unit, **kwargs):
        captured.append(kwargs['prepared_contexts'])
        return planner(unit, **kwargs)
    monkeypatch.setattr(space_shell, 'plan', observe)
    status = cli.main_handler(argparse.Namespace(model_path=str(tmp_path/'model'),
        sample=str(tmp_path/'sample.py'),spec=str(tmp_path/'spec.py'),no_spec=False,
        setup=None,frame_collect=False,compile_only=False,translate_only=False),
        io.StringIO(),io.StringIO())
    assert status == 0
    assert captured and 'model_cy._mx_classes._c_Policies' in captured[0]
    package = tmp_path/'model_cy'
    inventory = json.loads((package/'_mx_compilation_plan.json').read_text())
    assert inventory['spaces']['model_cy.Policies']['native_context']['point_table'] == 'points'
    assert (package/'_mx_classes.pxd').exists()
    # Execute relocated package interfaces in a fresh process, with no compiler
    # import needed and no replacement Space/Model facade.
    environment = os.environ.copy()
    # The model's declared helper dependency remains available. The compiler
    # checkout is deliberately absent from this relocated package's path.
    environment['PYTHONPATH'] = os.pathsep.join((str(tmp_path), str(Path(lookup.__file__).parents[2])))
    result = subprocess.run([sys.executable, '-c',
        'from model_cy import mx_model\n'
        'p=mx_model.Policies[0]; q=mx_model.Policies[1]\n'
        'assert p.amount()==1.5 and q.amount()==-1.0\n'
        'assert p.report()=={"amount":1.5}\n'
        'assert p.report() is p.report()\n'
        'assert p.uncached()==1.5 and q.uncached()==-1.0\n'
        'assert p.amount()==1.5\n'
        'assert "modelx_cython" not in __import__("sys").modules\n'],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-3000:]


def test_unknown_prepared_options_fail_before_any_artifact(tmp_path):
    from types import SimpleNamespace
    from modelx_cython._frame_runtime.prepared_region_plan import resolve
    space = SimpleNamespace(native_context={'typo': True})
    package = SimpleNamespace(spaces={'Example.P':space})
    with pytest.raises(ValueError, match='Unknown native resource'):
        resolve(package, None, tmp_path/'generated')
    assert not (tmp_path/'generated').exists()
