"""Resource wiring uses original resolved Data interfaces, not a new model."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from modelx_cython._frame_runtime import space_owner
from modelx_cython._frame_runtime.data_binding import DataView
from modelx_cython._frame_runtime.prepared_space_inputs import PreparedSpaceInputs


@pytest.mark.parametrize('synthetic_domain', [False, True])
def test_reference_tables_memoryviews_and_checked_scope_bindings(monkeypatch, synthetic_domain):
    points = pd.DataFrame({'age': [40, 41]}, index=pd.Index([10, 11], name='point_id'))
    rates = pd.Series([.01, .02])
    array = np.array([40, 41], dtype='int64')
    manifest = dict(public_space_binding={'data_reference': 'data_alias'},
        python_references=['data', 'point_id'], model=dict(
            tables={'points': {'object_kind': 'DataFrame'},
                    'rates': {'object_kind': 'Series'}, 'ages': {'object_kind': 'ndarray'}},
            reference_defaults={}, data_reference_names=['rates'], point_table='points',
            point_parameter='point_id', parameter_defaults={}))
    if synthetic_domain:
        manifest['public_space_binding'].update(synthetic_policy_table='domain', domain_entry='_f_domain')
        manifest['model']['tables']['domain'] = dict(object_kind='DataFrame', index_names=['point_id'])
        manifest['model']['point_table'] = 'domain'
    monkeypatch.setattr(space_owner, '_metadata', lambda module: (None, manifest))
    monkeypatch.setattr(space_owner, 'import_module', lambda module: None)
    captured = []
    class Prepared:
        def __init__(self, manifest, lookups, tables, *, references):
            captured.append(tables)
        def record(self, position, *, parameters):
            return (position,)
    monkeypatch.setattr(PreparedSpaceInputs, '__init__', Prepared.__init__)
    monkeypatch.setattr(PreparedSpaceInputs, 'record', Prepared.record)
    class Owner:
        def __init__(self, module, record):
            self.module, self.record = module, record
            self._scope = None
            self.selected = None
            self.kernel = SimpleNamespace(replace_records=self.replace_records,
                                          select_native_scope=self.select_scope,
                                          attach_python_scope_current=lambda: None)
        def prepare_checked(self):
            self._scope = SimpleNamespace(bindings=self._checked_bindings)
        def replace_records(self, records):
            self.record = records[0]
        def select_scope(self, index):
            self.selected = index
    monkeypatch.setattr(space_owner, 'SpaceFrameOwner', Owner)
    data = SimpleNamespace(points=lambda:points, rates=rates, ages=lambda:memoryview(array))
    receiver = SimpleNamespace(data_alias=data, point_id=10, _f_domain=lambda:(10,11))
    resources = {}
    owner = space_owner.prepared_owner(receiver, 'kernel.module', resources)
    assert captured[0]['rates'] is rates
    assert captured[0]['points'] is points
    assert np.shares_memory(captured[0]['ages'], array)
    if synthetic_domain:
        assert captured[0]['domain'].index.tolist() == [10,11]
    assert owner._scope is None  # Numeric work does not prepare object formulas.
    binding = owner._checked_bindings[0]
    assert binding['point_id'] == 10
    assert binding['data'].points() is points
    assert binding['data'].rates is rates
    owner.prepare_checked()
    receiver.point_id = 11
    assert space_owner.prepared_owner(receiver, 'kernel.module', resources, owner) is owner
    assert len(captured) == 1
    assert owner.record == (1,)
    assert owner._scope.bindings[0]['point_id'] == 11
    assert owner.selected == 0


def test_table_lookup_adapter_is_shared_with_existing_facade():
    from modelx_cython._frame_runtime.model import DataView as ExistingDataView
    assert ExistingDataView is DataView
    source = pd.DataFrame({'value': [1., 2.]})
    view = DataView({'points': source})
    calls = []
    def factory(frame):
        calls.append(frame)
        return object()
    first = view._prepare_lookup('points', factory)
    assert view._prepare_lookup('points', factory) is first
    assert calls == [source]
    assert view.points() is source
    import pickle
    restored = pickle.loads(pickle.dumps(view))
    assert restored.points().equals(source)
    assert not restored._prepared
