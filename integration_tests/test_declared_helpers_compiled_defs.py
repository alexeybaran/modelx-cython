"""Compiled-def closure/default handling must retain declared native bindings."""
import json
import os
import subprocess
import sys

import modelx as mx
from modelx_helpers import expressions


def test_declared_helpers_in_captured_default_and_yield_formulas(tmp_path):
    model=mx.new_model('CompiledDefs')
    try:
        p=model.new_space('Projection',refs={'expressions':expressions})
        p.new_cells('cf','lambda t: t + 1.0')
        p.new_cells('total','''def total(kind=None):
    return sum(expressions.over(range(3), lambda t: cf(t) if kind is None else 2.0*cf(t)))
''')
        p.new_cells('values','''def values(n):
    yield from expressions.over(range(n), lambda t: cf(t))
''')
        p.new_cells('uncached','lambda: sum(expressions.over(range(2), lambda t: cf(t)))')
        p.uncached.is_cached=False
        assert p.total()==6.0 and p.total('double')==12.0
        model.export(tmp_path/'exported')
    finally:model.close()
    (tmp_path/'sample.py').write_text('from exported import mx_model as m\np=m.Projection\np.total();p.total("double");list(p.values(3));p.uncached()\n')
    run=subprocess.run([sys.executable,'-m','modelx_cython','exported','--sample','sample.py','--no-spec'],
        cwd=tmp_path,env=dict(os.environ),capture_output=True,text=True)
    (tmp_path/'build.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
    source=(tmp_path/'exported_cy/_mx_classes.py').read_text()
    assert 'def _f_total(self, kind=None):' in source
    assert 'sum(_mx_helper_' in source
    report=json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    for suffix in ['._f_total','._f_values']:
        matches=[v for k,v in report.items() if k.endswith(suffix)]
        assert len(matches)==1 and 'modelx_cython.runtime._native_streams.over' in matches[0]['native_helpers']
    probe='''import sys
from exported_cy import mx_model as model
from modelx_helpers import expression_streams as h
for fresh in range(2):
 p=type(model)().Projection
 calls=[]
 def guard(frame,event,arg):
  if event=='call' and frame.f_code is h.over.__code__:calls.append('Python over')
 sys.setprofile(guard)
 try:
  assert p.total()==6.0 and p.total('double')==12.0
  assert list(p.values(3))==[1.0,2.0,3.0]
  assert p.uncached()==3.0 and p.uncached()==3.0
 finally:sys.setprofile(None)
 assert calls==[],calls
'''
    run=subprocess.run([sys.executable,'-c',probe],cwd=tmp_path,env=dict(os.environ),capture_output=True,text=True)
    (tmp_path/'native-probe.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
