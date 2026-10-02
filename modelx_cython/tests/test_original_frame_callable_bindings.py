"""Reference fields and checked bound Cells retain their distinct public ABIs."""
import json

from test_original_frame_shell import _run


def test_checked_memoryview_cell_uses_bound_dispatch(tmp_path):
    import modelx as mx
    import numpy as np

    model = mx.new_model('CallableBinding')
    space = model.new_space('S')
    space.np = np
    space.factor = 2.0
    space.new_cells('array', 'def array():\n return np.array([1.0, 3.0, 5.0])\n')
    space.new_cells('value', 'def value(i):\n return array()[i] * factor\n')
    model.export(tmp_path / 'model')
    model.close()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\nfor i in range(3): mx_model.S.value(i)\n',
        encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'),
                    '--no-spec', '--sample', str(tmp_path / 'sample.py')], 'build')
    manifest = json.loads((tmp_path / 'model_cy/_mx_frame_S/manifest.json').read_text())
    assert 'array' in manifest['inputs']
    assert 'array' not in manifest['original_live_fields']
    assert 'factor' in manifest['original_live_fields']
    assert manifest['cells']['value']['coverage']['body_backend'] == 'cython_gil'
    source = next((tmp_path / 'model_cy/_mx_frame_S').rglob('_kernel.pyx')).read_text()
    assert 'from model_cy._mx_classes cimport _c_S' in source
    assert 'ctypedef _c_S _mx_original_space' in source
    assert "getattr(receiver, 'array')" in source
    assert '_mx_live_array' not in source
    (tmp_path / 'override.pyx').write_text('''
from model_cy._mx_classes cimport _c_S
import numpy as np
cdef class Override(_c_S):
    cdef const double[:] _f_array(self):
        return np.array([10.0, 30.0, 50.0])
''', encoding='utf-8')
    (tmp_path / 'override_setup.py').write_text(
        "from setuptools import setup\nfrom Cython.Build import cythonize\n"
        "setup(ext_modules=cythonize('override.pyx'))\n", encoding='utf-8')
    _run(tmp_path, ['override_setup.py', 'build_ext', '--inplace'], 'override')
    (tmp_path / 'verify.py').write_text('''
from model_cy import mx_model
from override import Override
model = type(mx_model)()
item = model.S
assert item.value(0) == 2.0
item.factor = 3.0
assert item.value(0) == 2.0  # completed result is still cached
assert item.value(1) == 9.0  # uncached work sees the live field
sub = Override(model)
sub.factor = 4.0
assert sub.value(0) == 40.0
assert sub.value(1) == 120.0
''', encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')
