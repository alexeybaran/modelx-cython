"""External object inputs keep the original compiled receiver and public ABI."""
import json
from test_original_frame_shell import _run


def test_object_input_closure_uses_authoritative_original_receiver(tmp_path):
    import modelx as mx
    model = mx.new_model('ObjectInputBoundary')
    data = model.new_space('Data')
    data.inputs = {'value': -1.0}
    data.new_cells('row', 'def row():\n return inputs\n')
    s = model.new_space('S')
    s.data = data
    s.new_cells('value', 'def value(n):\n return data.row()["value"] + float(n)\n')
    s.new_cells('parent', 'def parent(n):\n return value(n) + value(n+1)\n')
    model.export(tmp_path/'model')
    model.close()
    (tmp_path/'sample.py').write_text('from model import mx_model\nmx_model.S.parent(0)\n')
    (tmp_path/'spec.py').write_text(repr({'spaces':{'S':{'cells_param_size':{'n':8}}}}))
    _run(tmp_path,['-m','modelx_cython',str(tmp_path/'model'),'--sample',str(tmp_path/'sample.py'),
                   '--spec',str(tmp_path/'spec.py')],'build')
    ledger = json.loads((tmp_path/'model_cy/_mx_classes.frame-shell.json').read_text())
    rows = [r for r in ledger['cells'] if r['space'].endswith('._c_S')]
    assert len(rows) == 2
    assert all(r['execution'] == 'original_cython' for r in rows)
    assert all('external object Cell ABI' in r['reason'] for r in rows)
    (tmp_path/'private.pyx').write_text('from model_cy._mx_classes cimport _c_S\ndef private_value(_c_S s):\n return s._f_value(0)\n')
    (tmp_path/'private_setup.py').write_text('from setuptools import setup\nfrom Cython.Build import cythonize\nsetup(ext_modules=cythonize("private.pyx", language_level=3))\n')
    _run(tmp_path,['private_setup.py','build_ext','--inplace'],'private-build')
    (tmp_path/'verify.py').write_text('''from model_cy import mx_model
from private import private_value
m=type(mx_model)();s=m.S
assert s.parent(0)==-1.0
m.Data.inputs['value']=10.0
assert s.value(0)==-1.0
assert private_value(s)==10.0
assert s.parent(0)==-1.0
assert s.parent(2)==25.0
''')
    _run(tmp_path,['verify.py'],'verify')
