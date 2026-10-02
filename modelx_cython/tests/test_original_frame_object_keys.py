"""Object-keyed numeric Cells use checked Cython within the shared frame."""
import json
from test_original_frame_shell import _run


def test_original_shell_object_keys_share_frame_cache_and_checked_cleanup(tmp_path):
    import modelx as mx
    m=mx.new_model('ObjectKeys'); s=m.new_space('S')
    s.offset=0.0; s.counts={}
    s.new_cells('value', 'def value(key):\n counts["value"] = counts.get("value",0)+1\n return float(len(key)) + offset\n')
    s.new_cells('axes', 'def axes(i, key):\n return float(i) + value(key)\n')
    s.new_cells('parent', 'def parent(key):\n return value(key) + value(key)\n')
    s.new_cells('recursive', 'def recursive(key):\n return 0.0 if key[0] <= 0 else recursive((key[0]-1,)) + 1.0\n')
    m.export(tmp_path/'model'); m.close()
    (tmp_path/'hooks.py').write_text('''class Key:
    def __init__(self, length, fail=False): self.length=length; self.fail=fail
    def __len__(self): return self.length
    def __hash__(self):
        if self.fail: raise ValueError('hash retry')
        return 0
    def __eq__(self, other): return type(other) is Key and self.length==other.length
''',encoding='utf-8')
    (tmp_path/'sample.py').write_text('from model import mx_model\nfrom hooks import Key\ns=mx_model.S\ns.parent("aa")\ns.parent(Key(3))\ns.recursive((3,))\ns.axes(1, "aa")\ns.axes(2, Key(3))\n',encoding='utf-8')
    _run(tmp_path,['-m','modelx_cython',str(tmp_path/'model'),'--sample',str(tmp_path/'sample.py'),'--no-spec'],'build')
    routes=json.loads((tmp_path/'model_cy/_mx_classes.frame-shell.json').read_text())['cells']
    assert all(r['execution']=='frame_exact' and not r['checked_cell_dependencies'] for r in routes)
    manifest=json.loads((tmp_path/'model_cy/_mx_frame_S/manifest.json').read_text())
    assert all(manifest['cells'][n]['storage']=='typed_python_dict' for n in ('value','parent','recursive'))
    assert all(manifest['cells'][n]['calculator']=='cython_scalar' for n in ('value','parent','recursive','axes'))
    assert manifest['cells']['axes']['storage']=='typed_epoch_slabs'
    (tmp_path/'verify.py').write_text('''import pickle
from model_cy import mx_model
from hooks import Key
s=type(mx_model)().S
assert s.parent(Key(3))==6.0 and s.parent(Key(3))==6.0 and s.counts=={'value':1}
for i in range(30): assert s.parent(Key(i))==2*i
assert s.counts=={'value':30}
k=Key(99, True)
for _ in range(2):
    try: s.parent(k)
    except ValueError as e: assert str(e)=='hash retry'
    else: raise AssertionError('key error swallowed')
assert s.counts=={'value':30}
k.fail=False; assert s.parent(k)==198.0 and s.counts=={'value':31}
assert s.recursive((30,))==30.0
for i in (-100, -1, 0, 1, 2, 10000):
    assert s.axes(i,'aa')==i+2.0 and s.axes(i,Key(3))==i+3.0
q=pickle.loads(pickle.dumps(s)); q.offset=1000.0
assert q.parent(Key(99))==198.0 and q.value(Key(3))==3.0 and q.counts=={'value':32}
for i in range(31): assert q.recursive((i,))==float(i)
for i in (-100, -1, 0, 1, 2, 10000):
    assert q.axes(i,'aa')==i+2.0 and q.axes(i,Key(3))==i+3.0
assert q.counts=={'value':32}
print('object-key numeric Cells share frame caches: collisions, hash failures/retries, recursion and serialization passed')
''',encoding='utf-8')
    _run(tmp_path,['verify.py'],'verify')
