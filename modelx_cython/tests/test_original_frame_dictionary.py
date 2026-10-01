"""Original growing dictionary behavior through shared frame cache descriptors."""
import json
from test_original_frame_shell import _run


def test_original_shell_integer_keyed_object_dictionary_domain(tmp_path):
    import modelx as mx
    m = mx.new_model('ObjectDictionary')
    s = m.new_space('S')
    s.offset = 0.0; s.counts = {}; s.flags = {'fail': False}
    s.new_cells('child', 'def child():\n return offset + 5.0\n')
    s.new_cells('label', 'def label(i):\n return str(i)\n')
    s.new_cells('record', 'def record(i, j):\n value = child()\n counts[(i,j)] = counts.get((i,j),0) + 1\n if flags["fail"]: raise ValueError("record retry")\n return {"key": (i,j), "value": value, "label": label(i)}\n')
    m.export(tmp_path/'model'); m.close()
    (tmp_path/'sample.py').write_text('from model import mx_model\ns=mx_model.S\ns.record(1,2)\n',encoding='utf-8')
    _run(tmp_path, ['-m','modelx_cython',str(tmp_path/'model'),'--no-spec',
                    '--sample',str(tmp_path/'sample.py')], 'build')
    routes = json.loads((tmp_path/'model_cy/_mx_classes.frame-shell.json').read_text())
    assert all(r['execution']=='frame_exact' for r in routes['cells'])
    manifest=json.loads((tmp_path/'model_cy/_mx_frame_S/manifest.json').read_text())
    assert manifest['original_cache_kinds']['record']=='dictionary'
    assert manifest['cells']['record']['storage']=='python_dict'
    assert manifest['cells']['label']['storage']=='python_dict'
    (tmp_path/'verify.py').write_text("""
import pickle
from model_cy import mx_model
s = type(mx_model)().S
for i,j in ((-100,-3),(0,0),(1,2),(100,999)):
 r=s.record(i,j)
 assert r=={'key':(i,j),'value':5.0,'label':str(i)}
 assert s.record(i,j) is r and s.counts[(i,j)]==1
s.flags['fail']=True
for _ in range(2):
 try:s.record(-2,200)
 except ValueError as e:assert str(e)=='record retry'
 else:raise AssertionError('failed dictionary parent cached')
s.offset=100.0; s.flags['fail']=False
assert s.record(-2,200)['value']==5.0
assert s.counts[(-2,200)]==3
q=pickle.loads(pickle.dumps(s))
q.offset=-100.0
for i,j in ((-100,-3),(0,0),(1,2),(100,999),(-2,200)):
 r=q.record(i,j)
 assert r['value']==5.0 and r['label']==str(i)
 assert q.record(i,j) is r
assert q.counts==s.counts
assert q.label(9999)=='9999'
print('unbounded original integer dictionary keys, retries, completed children and populated serialization passed')
""",encoding='utf-8')
    _run(tmp_path,['verify.py'],'verify')
