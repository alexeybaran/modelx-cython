"""Native identity traversal keeps natural generator syntax and no mapper body."""
import json
import os
import subprocess
import sys

import modelx as mx
from lifelib import expressions


def test_identity_generator_expressions_through_normal_cli(tmp_path):
    model = mx.new_model('IdentityStreams')
    try:
        p = model.new_space('Projection', refs={'e': expressions})
        p.new_cells('cf', 'lambda t: t + .5')
        p.new_cells('ranges', """def ranges(start, stop, step):
    return tuple(e.over(value for value in e.periods(start, stop, step)))
""")
        p.new_cells('values', '''def values(n):
    return e.over(cf(t) for t in e.periods(n))
''')
        p.new_cells('total', '''def total(n):
    return sum(e.over(cf(t) for t in e.periods(n)))
''')
        p.new_cells('filtered', '''def filtered(n):
    return tuple(e.over((cf(t) for t in e.periods(n)), where=lambda value: value > 1.0))
''')
        p.new_cells('nested', '''def nested():
    return tuple(e.over((first, second) for first in e.periods(3) for second in e.periods(first)))
''')
        p.new_cells('retained', '''def retained(kind=None):
    return tuple(e.over(cf(t) if kind is None else 2 * cf(t) for t in e.periods(3)))
''')
        assert list(p.values(3)) == [.5, 1.5, 2.5]
        assert p.total(3) == 4.5 and type(p.total(0)) is int
        model.export(tmp_path/'exported')
    finally:
        model.close()
    (tmp_path/'sample.py').write_text('''from exported import mx_model as m
p=m.Projection
limit=(1<<63)-1
for bounds in [(limit,-limit-1,-limit),(-limit-1,limit,limit),(limit-1,limit,2),(-limit,-limit-1,-2),(0,-limit-1,-limit-1),(1<<80,(1<<80)+5,2),(0,2,1<<80),(3,0,-1),(0,0,1)]:p.ranges(*bounds)
list(p.values(3));p.total(-1);p.total(0);p.total(3);p.filtered(3);p.nested();p.retained();p.retained("double")
''')
    run = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
        '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, env=dict(os.environ),
        capture_output=True, text=True)
    (tmp_path/'build.log').write_text(run.stdout+run.stderr)
    assert run.returncode == 0, run.stdout+run.stderr
    source = (tmp_path/'exported_cy/_mx_classes.py').read_text()
    assert '__pyx_gb_' in (tmp_path/'exported_cy/_mx_classes.c').read_text()
    admission=json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    for cell in ['values','total','filtered','nested','retained','ranges']:
        entries=[v for k,v in admission.items() if k.endswith('._f_'+cell)]
        assert len(entries)==1 and 'lifelib._native_streams.over' in entries[0]['native_helpers']
        assert entries[0]['native_reductions']==0
        assert 'lifelib._native_streams.periods' in entries[0]['native_helpers']
    probe='''import sys
from exported_cy import mx_model as prototype
from lifelib import expression_streams as h
for repetition in range(2):
 p=type(prototype)().Projection
 forbidden={h.over.__code__,h._stream_generator.__code__,h._identity_generator.__code__,h._InputCursor.__next__.__code__}
 calls=[]
 def guard(frame,event,arg):
  if event=='call' and frame.f_code in forbidden:calls.append(frame.f_code.co_name)
 sys.setprofile(guard)
 try:
  limit=(1<<63)-1
  for bounds in [(limit,-limit-1,-limit),(-limit-1,limit,limit),(limit-1,limit,2),(-limit,-limit-1,-2),(0,-limit-1,-limit-1),(1<<80,(1<<80)+5,2),(0,2,1<<80),(3,0,-1),(0,0,1)]:
   assert p.ranges(*bounds)==tuple(range(*bounds))
  try:p.ranges(0,1,0)
  except ValueError:pass
  else:raise AssertionError('zero step accepted')
  stream=p.values(3)
  try:stream.send(7)
  except TypeError:pass
  else:raise AssertionError('fresh identity stream accepted non-None send')
  assert next(stream)==.5 and stream.send(99)==1.5 and next(stream)==2.5
  assert list(stream)==[]
  assert p.total(3)==4.5 and p.total(-1)==0 and type(p.total(0))is int
  assert p.filtered(3)==(1.5,2.5)
  assert p.nested()==((1,0),(2,0),(2,1))
  assert p.retained()==(.5,1.5,2.5) and p.retained("double")== (1.,3.,5.)
 finally:sys.setprofile(None)
 assert calls==[],calls
from types import SimpleNamespace
from lifelib import expressions as e
calls=[]
def current(source,**keywords):
 calls.append(keywords)
 return h.over(source,**keywords)
fresh=type(prototype)().Projection
fresh.e=SimpleNamespace(over=current,periods=e.periods)
assert fresh.filtered(3)==(1.5,2.5)
assert len(calls)==1 and set(calls[0])=={"where"},calls
'''
    run=subprocess.run([sys.executable,'-c',probe],cwd=tmp_path,env=dict(os.environ),capture_output=True,text=True)
    (tmp_path/'native-probe.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
