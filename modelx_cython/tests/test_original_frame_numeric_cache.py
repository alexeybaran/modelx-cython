"""Accepted growing-key caches under original public classes and frame ownership."""
import json
from test_original_frame_shell import _run


def test_original_shell_native_numeric_growth_failure_and_restore(tmp_path):
    # Force prefix-to-growing migration at a small capacity. It remains an
    # acceleration hint, not a key-domain or workload bound.
    import modelx as mx
    m = mx.new_model('NativeKeys')
    s = m.new_space('S'); s.offset = 0.25
    for name, body in {
        'affine': 'def affine(x):\n return x + offset\n',
        'pair': 'def pair(i, x):\n return i + x + offset\n',
        'anchor': 'def anchor():\n return offset + 10.0\n',
        'parent': 'def parent(x):\n child = anchor()\n return child + affine(x)\n',
        'recursive': 'def recursive(x):\n return 0.0 if x <= 0 else recursive(x-1.0) + 1.0\n',
        'fallback_child': "def fallback_child(x):\n if x == 2.0: fallback_child(float('inf'))\n return 0.0 if x == float('inf') else x\n",
    }.items(): s.new_cells(name, body)
    m.export(tmp_path/'model'); m.close()
    (tmp_path/'sample.py').write_text('from model import mx_model\ns=mx_model.S\ns.parent(0.0)\ns.pair(0,0.25)\ns.recursive(3.0)\ns.fallback_child(0.25)\n', encoding='utf-8')
    _run(tmp_path, ['-m','modelx_cython',str(tmp_path/'model'),'--no-spec',
                    '--sample',str(tmp_path/'sample.py'),'--translate-only'], 'translate')
    manifest=json.loads((tmp_path/'model_cy/_mx_frame_S/manifest.json').read_text())
    for name in ('affine','pair','parent','recursive','fallback_child'):
        assert manifest['cells'][name]['storage']=='growing_numeric'
        assert manifest['cells'][name]['native_prefix']
    assert manifest['cells']['affine']['slots'] == 16
    routes=json.loads((tmp_path/'model_cy/_mx_classes.frame-shell.json').read_text())
    assert all(r['execution']=='frame_exact' and not r['checked_cell_dependencies'] for r in routes['cells'])
    source=next((tmp_path/'model_cy/_mx_frame_S').rglob('_kernel.pyx'))
    text=source.read_text()
    # Fault injection belongs only to this generated test artifact.
    needle='from libc.stdlib cimport calloc, free'
    assert text.count(needle)==1
    text=text.replace(needle, '''from libc.stdlib cimport calloc as _mx_real_calloc, free
from libc.stddef cimport size_t
cdef int _mx_test_fail = 0
def _test_fail_next():
    global _mx_test_fail
    _mx_test_fail = 1
cdef void* calloc(size_t count, size_t size) noexcept nogil:
    global _mx_test_fail
    if _mx_test_fail:
        _mx_test_fail = 0
        return NULL
    return _mx_real_calloc(count, size)
''')
    source.write_text(text, encoding='utf-8')
    _run(tmp_path,['-m','modelx_cython',str(tmp_path/'model'),'--compile-only'],'build')
    (tmp_path/'verify.py').write_text(f'''
import gc, importlib, math, pickle
from model_cy import mx_model
module=importlib.import_module({manifest['package']+'._kernel'!r})
factory=type(mx_model); m=factory(); s=m.S
for i in range(16): assert s.affine(float(i))==i+0.25
module._test_fail_next()
try: s.parent(16.0)
except MemoryError: pass
else: raise AssertionError('growth failure swallowed')
s.offset=100.25
assert s.affine(15.0)==15.25
assert s.anchor()==10.25
assert s.parent(16.0)==126.5
assert s.parent(16.0)==126.5
for i in range(3000): assert s.pair(i, i+0.25)==2*i+100.5
s.offset=-200.0
for i in reversed(range(3000)): assert s.pair(i, i+0.25)==2*i+100.5
assert s.recursive(300.0)==300.0
q=pickle.loads(pickle.dumps(s)); q.offset=-1000.0
assert q.parent(16.0)==126.5 and q.pair(2999,2999.25)==6098.5
for i in range(301): assert q.recursive(float(i))==float(i)
assert s.fallback_child(0.25)==0.25 and s.fallback_child(2.0)==2.0
assert s.fallback_child(0.25)==0.25 and s.fallback_child(float('inf'))==0.0
q=pickle.loads(pickle.dumps(s))
assert q.fallback_child(2.0)==2.0 and q.fallback_child(float('inf'))==0.0
p=factory().S; p.offset=-1.0
assert p.affine(-0.0)==-1.0
p.offset=100.0; assert p.affine(0.0)==-1.0
assert math.isinf(p.affine(float('inf'))) and p.affine(0.0)==-1.0
assert math.isnan(p.affine(float('nan')))
for i in range(50):
    m=factory(); m.S.pair(i,0.25); m.S.affine(float('inf'))
    del m
gc.collect()
print('native growing caches: growth failure/retry, completed children, collisions, signed zero, fallback, recursion, serialization and destruction passed')
''',encoding='utf-8')
    _run(tmp_path,['verify.py'],'verify')


def test_original_shell_numeric_collect_reset_and_direct_ownership(tmp_path):
    import modelx as mx
    m=mx.new_model('NumericCollect')
    ids=(1,2,3)
    p=m.new_space('Policies', formula='def Policies(policy):\n return {"refs": {"offset": float(policy)}}\n',
                  refs={'offset':0.0,'policy_ids':ids}, collect_over={'policy':ids},
                  collect_outputs='lambda: {"sum": {"total": amount(0.25)}, "individual": {"values": [amount(0.25), amount(-0.5)]}}')
    p.new_cells('amount','def amount(x):\n return offset + x\n')
    m.export(tmp_path/'model'); m.close()
    (tmp_path/'sample.py').write_text('from model import mx_model\nfor i in (1,2,3): mx_model.Policies[i].amount(0.25)\nmx_model.Policies.collect()\n',encoding='utf-8')
    _run(tmp_path,['-m','modelx_cython',str(tmp_path/'model'),'--sample',str(tmp_path/'sample.py'),'--no-spec'],'build')
    manifest=json.loads((tmp_path/'model_cy/_mx_frame_Policies/manifest.json').read_text())
    assert manifest['cells']['amount']['storage']=='growing_numeric'
    (tmp_path/'verify.py').write_text('''
import pickle
from model_cy import mx_model
m=type(mx_model)(); p=m.Policies
earlier=p[1]; assert earlier.amount(0.25)==1.25
r=p.collect()
assert r.sum.total==6.75
for i in (1,2,3): assert r.individual[i].values==[i+0.25,i-0.5]
assert p.collect() is r
assert earlier.amount(0.25)==1.25 and earlier.amount(-0.5)==0.5
q=pickle.loads(pickle.dumps(m)); assert q.Policies.collect().sum.total==6.75
assert q.Policies[1].amount(0.25)==1.25
print('growing cache reset across collect policies, detached outputs and independent direct owners passed')
''',encoding='utf-8')
    _run(tmp_path,['verify.py'],'verify')
