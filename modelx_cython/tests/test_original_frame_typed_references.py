"""Resolved public Space types remain typed inside checked frame formulas."""
import json

import numpy as np
import pytest

from modelx_cython.tests.test_original_frame_shell import _run


def test_compiled_array_reference_preserves_rebinding_cache_and_none(tmp_path):
    import modelx as mx
    model = mx.new_model('TypedReferences')
    data = model.new_space('Data')
    data.inputs = np.array([1.5, -1.0, 3.0, 4.0])
    data.new_cells('values', 'def values():\n return inputs\n')
    space = model.new_space('S')
    space.data = data
    space.other = data
    space.hook = None
    for name, source in {
        'amount': 'def amount(n):\n return data.values()[n]\n',
        'alias': 'def alias(n):\n return other.values()[n]\n',
        'uncached': 'def uncached(n):\n return data.values()[n]\n',
        'after_callback': 'def after_callback(n):\n hook()\n return data.values()[n] + amount(n)\n',
    }.items():
        space.new_cells(name, source)
    space.uncached.is_cached = False
    model.export(tmp_path / 'model')
    model.close()
    (tmp_path / 'model/_mx_formula_sources.json').unlink(missing_ok=True)
    (tmp_path / 'sample.py').write_text('''from model import mx_model
mx_model.S.hook = lambda: None
for n in range(4):
    mx_model.S.amount(n); mx_model.S.alias(n)
    mx_model.S.uncached(n); mx_model.S.after_callback(n)
''')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--no-spec',
                    '--sample', str(tmp_path / 'sample.py')], 'build')
    folder = tmp_path / 'model_cy/_mx_frame_S'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['original_reference_types'] == {
        'data': 'model_cy._mx_classes._c_Data',
        'other': 'model_cy._mx_classes._c_Data',
    }
    source = next(folder.rglob('_kernel.pyx')).read_text()
    assert 'cdef inline _mx_reference_type_0 _mx_live_data(self):' in source
    assert 'from model_cy._mx_classes cimport _c_Data as _mx_reference_type_0' in source
    assert '@cython.nonecheck(True)' in source
    assert all(not row['coverage']['detached_python'] for row in manifest['cells'].values())
    (tmp_path / 'overrides.pyx').write_text('''from model_cy._mx_classes cimport _c_S, _c_Data
cdef class PrivateData(_c_Data):
    cdef const double[:] _f_values(self):
        return self.inputs
cdef class PublicData(_c_Data):
    cpdef const double[:] values(self):
        return self.inputs
def owner(_c_S space):
    return space._mx_frame_owner
''')
    (tmp_path / 'overrides_setup.py').write_text('''from setuptools import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize('overrides.pyx', language_level=3))
''')
    _run(tmp_path, ['overrides_setup.py', 'build_ext', '--inplace'], 'overrides-build')
    (tmp_path / 'verify.py').write_text('''import pickle
import numpy as np
from model_cy import mx_model
from overrides import owner, PrivateData, PublicData
m = type(mx_model)(); s = m.S
other = type(mx_model)()
other.Data.inputs = np.array([10.0, 20.0, 30.0, 40.0])
assert s.amount(0) == s.alias(0) == s.uncached(0) == 1.5
assert owner(s) is not None
s.data = other.Data
assert s.amount(0) == 1.5
assert s.amount(1) == s.uncached(1) == 20.0
assert s.alias(1) == -1.0
other.Data.inputs[1] = 22.0
assert s.amount(1) == 20.0 and s.uncached(1) == 22.0
def callback():
    s.data = m.Data
s.hook = callback
assert s.after_callback(2) == 6.0 and s.data is m.Data
s.data = None
assert s.amount(0) == 1.5
try:
    s.uncached(3)
except AttributeError:
    pass
else:
    raise AssertionError('A None Space receiver must raise, not dereference a native pointer')
s.data = other.Data
assert s.uncached(3) == 40.0
# The accepted ordinary compiler also cannot pickle a populated native
# memoryview field. Keep this source-specific limitation explicit; normal
# populated model serialization remains covered by the existing shell suite.
s.hook = None
try:
    pickle.dumps(m)
except TypeError as error:
    assert '__reduce__' in str(error)
else:
    raise AssertionError('Unexpected serialization contract for native array field')
completed = owner(s).kernel.snapshot()
assert pickle.loads(pickle.dumps(completed)) == completed
for cls in (PrivateData, PublicData):
    replacement = cls(m)
    replacement.inputs = np.array([5.0, 6.0, 7.0, 8.0])
    s.data = replacement
    assert s.uncached(3) == 8.0  # typed reference retains virtual dispatch
    replacement.inputs[3] = 9.0
    assert s.uncached(3) == 9.0
''')
    _run(tmp_path, ['verify.py'], 'verify')


def test_typed_reference_contract_requires_resolved_live_object_field():
    from modelx_cython._frame_runtime.contracts import KernelSpec, ContractError
    from types import ModuleType
    for extra in (
        {'original_space_shell': False},
        {'original_live_fields': ()},
        {'inputs': {'data': float}},
        {'original_reference_types': {'data': 'not a compiled identity'}},
    ):
        args = dict(module=ModuleType('fixture'), inputs={'data': object},
                    outputs={}, cell_outputs=('amount',), steps=4, length='0',
                    original_space_shell=True, original_live_fields=('data',),
                    original_reference_types={'data': 'model_cy._mx_classes._c_Data'})
        args.update(extra)
        with pytest.raises(ContractError, match='resolved original live object fields'):
            KernelSpec(**args)
