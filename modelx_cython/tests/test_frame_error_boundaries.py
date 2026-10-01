"""Native errors stop at checked object frames and preserve Cython cleanup."""
import gc
import json
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest


def test_numeric_object_failures_cleanup_retry_and_nested_guard(tmp_path, monkeypatch):
    from modelx_cython._frame_runtime.compiler import build
    from modelx_cython._frame_runtime.model import CompiledModel
    from modelx_cython._frame_runtime.modelx_frontend_native_mode import from_modelx

    tracker = ModuleType('mxcf_error_boundary_tracker')
    tracker.created = tracker.destroyed = tracker.child_evals = 0
    tracker.fail = True
    class Token:
        def __init__(self):
            tracker.created += 1
        def __del__(self):
            tracker.destroyed += 1
    tracker.token = Token
    def child():
        tracker.child_evals += 1
    def fail():
        if tracker.fail:
            raise KeyError('object failure')
    tracker.child = child
    tracker.maybe_fail = fail
    monkeypatch.setitem(sys.modules, tracker.__name__, tracker)

    def formula(policy_id):
        return None
    def source(text):
        return SimpleNamespace(formula=SimpleNamespace(source=text),
                               is_cached=True, allow_none=False, parent=None)
    space = SimpleNamespace(parameters=('policy_id',), spaces={},
        formula=SimpleNamespace(func=formula, source='lambda policy_id: None'),
        refs={'tracker': tracker}, _impl=SimpleNamespace(collect_over_spec=None), cells={
            'leaf': source('def leaf(n: int) -> float:\n if n == 2: raise ValueError("numeric failure")\n return float(n-2)\n'),
            'pure_parent': source('def pure_parent(n: int) -> float:\n return leaf(n) + 1.0\n'),
            'caught': source('def caught(n: int) -> object:\n token = tracker.token()\n try:\n  return {"value": pure_parent(n)}\n except ValueError as error:\n  return {"caught": str(error)}\n'),
            'child': source('def child(n: int) -> object:\n tracker.child()\n return {"value": float(n)}\n'),
            'failing_object': source('def failing_object(n: int) -> object:\n token = tracker.token()\n completed = child(n)\n tracker.maybe_fail()\n return completed\n'),
            'numeric_parent': source('def numeric_parent(n: int) -> float:\n return float(failing_object(n)["value"])\n'),
            'nested': source('def nested(n: int) -> object:\n token = tracker.token()\n tracker.callback()\n return {"value": pure_parent(n)}\n'),
        })
    points = pd.DataFrame(index=pd.Index([0], name='policy_id'))
    model_source = SimpleNamespace(name='ErrorBoundary', Projection=space,
        Data=SimpleNamespace(parameters=(), spaces={}, refs={'points': points}, cells={}))
    spec = from_modelx(model_source, tmp_path/'source', point_table='points',
                      cython_objects=True, steps=4)
    artifact = tmp_path/'artifact'
    build(spec, artifact)
    manifest = json.loads((artifact/'manifest.json').read_text())
    assert manifest['cells']['pure_parent']['coverage']['body_backend'] == 'cython_nogil'
    assert manifest['cells']['caught']['coverage']['body_backend'] == 'cython_gil'
    assert manifest['cells']['numeric_parent']['coverage']['body_backend'] == 'cython_gil'
    model = CompiledModel(artifact, spec.initial_tables)
    try:
        item = model.Projection[0]
        assert item.caught(2) == {'caught': 'numeric failure'}
        # Legitimate sentinel values cross the checked boundary unchanged.
        assert item.leaf(1) == -1.0
        for _ in range(2):
            with pytest.raises(KeyError, match='object failure'):
                item.numeric_parent(1)
        assert tracker.child_evals == 1  # completed child was not rolled back
        gc.collect()
        assert tracker.created == tracker.destroyed == 3
        tracker.fail = False
        assert item.numeric_parent(1) == 1.0
        assert tracker.child_evals == 1
        tracker.callback = lambda: item.pure_parent(2)
        with pytest.raises(RuntimeError, match='active Cell evaluation'):
            item.nested(1)
        assert model._active == 0
        def nested_failure():
            with pytest.raises(ValueError, match='numeric failure'):
                # Internal checked entry on the same owner, not a new public
                # calculation inside the facade's active-call restriction.
                item._context.call('pure_parent', 2)
        tracker.callback = nested_failure
        assert item.nested(1) == {'value': 0.0}
        assert item.pure_parent(1) == 0.0
        with pytest.raises(ValueError, match='numeric failure'):
            item.pure_parent(2)
        gc.collect()
        assert tracker.created == tracker.destroyed == 6
    finally:
        model.close()
