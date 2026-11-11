"""Representative traces size storage; they do not bound valid Cell arguments."""
import os
import subprocess
import sys
import modelx as mx
import pytest


@pytest.mark.parametrize('explicit', [False, True])
def test_inferred_dense_spill_preserves_cache_errors_and_overrides(tmp_path, explicit):
    model=mx.new_model('InferredDomain')
    try:
        p=model.new_space('Projection', refs={'factor': 2., 'fail': True})
        p.new_cells('value', '''def value(t):
    if t == 9 and fail:
        raise ValueError('retry')
    return factor*t
''')
        p.new_cells('forward', 'lambda t: value(t+1)')
        p.new_cells('tupled', 'lambda t, k: value(t) + k')
        model.export(tmp_path/'exported')
    finally:model.close()
    (tmp_path/'sample.py').write_text('''from exported import mx_model as m
for t in range(3):
 m.Projection.value(t);m.Projection.forward(t);m.Projection.tupled(t, 0)
''')
    args=['--no-spec']
    if explicit:
        (tmp_path/'spec.py').write_text(repr({'spaces':{'Projection':{'cells_param_size':{'t':4,('t','k'):(4,1)}}}}))
        args=['--spec','spec.py']
    build=subprocess.run([sys.executable,'-B','-m','modelx_cython','exported','--sample','sample.py',*args],
                         cwd=tmp_path,env=dict(os.environ),capture_output=True,text=True)
    (tmp_path/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0,build.stdout+build.stderr
    probe='''import pickle,sys
from exported_cy import mx_model as m
p=m.Projection
if sys.argv[1]=='True':
 for key in (-1, 4, 12):
  try:p.value(key)
  except IndexError:pass
  else:raise AssertionError('explicit range was not enforced')
else:
 for key in (-4, 7, 12, 1<<60):
  assert p.value(key)==2.*key
  assert p.forward(key)==2.*(key+1)
  assert p.tupled(key, -2)==2.*key-2
 assert p.value(8)==16.
 p.factor=3.
 assert p.value(8)==16.
 try:p.value(9)
 except ValueError as error:assert error.args==('retry',)
 else:raise AssertionError('error not raised')
 p.fail=False
 assert p.value(9)==27.
 q=pickle.loads(pickle.dumps(p))
 assert q.value(8)==16. and q.value(9)==27.
 class Derived(type(p)):
  def value(self,t):return 7.*t
 d=Derived(m);d._mx_copy_refs(p,m)
 assert d.value(12)==84.
 assert d.forward(12)==91.
'''
    result=subprocess.run([sys.executable,'-B','-c',probe,str(explicit)],cwd=tmp_path,
                          env=dict(os.environ),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
