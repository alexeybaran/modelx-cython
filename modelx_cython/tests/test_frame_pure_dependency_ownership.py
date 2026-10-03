"""An object output must not add reentry ownership to a pure numeric closure."""
import json
from test_original_frame_shell import _run


def test_pure_recursive_child_rollback_inside_object_formula(tmp_path):
    import modelx as mx
    model = mx.new_model('PureDependencyOwnership')
    s = model.new_space('S')
    s.fail = True
    s.hook = None
    s.new_cells('value', '''def value(n):
 if n == 0: return -1.0
 child = value(n-1)
 if fail and n == 3: raise ValueError("numeric failure")
 return child + 1.0
''')
    s.new_cells('output', '''def output(n):
 try: return {"value": value(n)}
 except ValueError: return {"child": value(2)}
''')
    s.new_cells('callback', '''def callback(n):
 hook()
 return {"value": value(n)}
''')
    model.export(tmp_path/'model')
    model.close()
    (tmp_path/'sample.py').write_text('from model import mx_model\ns=mx_model.S\ns.fail=False\ns.hook=lambda:None\ns.output(4)\ns.callback(4)\n')
    (tmp_path/'spec.py').write_text(repr({'spaces':{'S':{'cells_param_size':{'n':16}}}}))
    _run(tmp_path, ['-m','modelx_cython',str(tmp_path/'model'),'--sample',str(tmp_path/'sample.py'),
                    '--spec',str(tmp_path/'spec.py')], 'build')
    manifest = json.loads((tmp_path/'model_cy/_mx_frame_S/manifest.json').read_text())
    assert manifest['cells']['value']['invocation_owner_required'] is False
    source = next((tmp_path/'model_cy/_mx_frame_S').rglob('_kernel.pyx')).read_text()
    entry = source[source.index('cdef inline double _cell_value('):]
    entry = entry[:entry.index('\ncdef inline', 1)]
    assert '_busy_owner' not in entry
    (tmp_path/'verify.py').write_text('''from model_cy import mx_model
s=type(mx_model)().S
assert s.output(4)=={"child":1.0}
assert s.value(2)==1.0
s.fail=False
assert s.value(4)==3.0
q=type(mx_model)().S
def nested():
 assert q.output(4)=={"child":1.0}
 q.fail=False
 assert q.value(4)==3.0
q.hook=nested
assert q.callback(4)=={"value":3.0}
assert q.value(0)==-1.0
''')
    _run(tmp_path,['verify.py'],'verify')
