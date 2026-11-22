"""Differential execution of defects found by the pre-coverage audit."""
import functools
from http import HTTPStatus
import json
import math
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import modelx as mx
import numpy as np
from modelx_cython.cli import create_setup


def test_pickle_restores_distinct_multidimensional_native_array_shapes(tmp_path):
    (tmp_path/'array_state.py').write_text('''import cython
@cython.cclass
class Arrays:
    small: cython.double[2][2]
    large: cython.double[5][2]
    def __init__(self):
        for i in range(2):
            for j in range(2): self.small[i][j] = 10*i+j
            for j in range(5): self.large[i][j] = 100*i+j
    def values(self):
        return ([self.small[i][j] for i in range(2) for j in range(2)],
                [self.large[i][j] for i in range(2) for j in range(5)])
''')
    create_setup('array_state',[Path('array_state.py')],tmp_path/'setup.py')
    built=subprocess.run([sys.executable,'-B','setup.py','build_ext','--inplace'],
                         cwd=tmp_path,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert built.returncode==0,built.stdout
    assert 'warning C4048' not in built.stdout,built.stdout
    subprocess.run([sys.executable,'-B','-c','''import pickle
from array_state import Arrays
original=Arrays()
assert pickle.loads(pickle.dumps(original)).values()==original.values()
'''],cwd=tmp_path,check=True)


def compile_model(tmp_path, sample, integer_checks=None):
    (tmp_path/'sample.py').write_text(sample,encoding='utf-8')
    spec = ['--no-spec']
    if integer_checks is not None:
        (tmp_path/'spec.py').write_text(repr({'integer_checks': integer_checks}),encoding='utf-8')
        spec = ['--spec','spec.py']
    subprocess.run([sys.executable,'-B','-m','modelx_cython','exported',*spec],
                   cwd=tmp_path,check=True)


def test_observed_wide_integers_and_buffer_formats_fall_back_to_objects(tmp_path):
    model=mx.new_model('ObservedRepresentations')
    try:
        s=model.new_space('Values',refs={'wide':2**80,
            'a32':np.array([1.25,2.5],dtype=np.float32),
            'i32':np.array([11,22],dtype=np.int32),
            'swapped':np.array([1.25,2.5],dtype='>f8'),
            'zero':np.array(1.25),
            'native':np.array([1.25,2.5],dtype=np.float64)})
        for name in ('a32','i32','swapped','native'):
            s.new_cells(name+'_array','lambda: '+name)
            s.new_cells(name+'_at','lambda k: '+name+'_array()[k]')
        s.new_cells('echo','lambda x: x')
        s.new_cells('zero_array','lambda: zero')
        s.new_cells('zero_at','lambda: zero_array()[()]')
        s.new_cells('reference','lambda: wide')
        model.export(tmp_path/'exported')
    finally:model.close()
    compile_model(tmp_path,'from exported import mx_model\ns=mx_model.Values\n'
        's.echo(2**80);s.echo(-2**80);s.reference();s.zero_at()\n'
        'for k in (0,1):s.a32_at(k);s.i32_at(k);s.swapped_at(k);s.native_at(k)\n'
        'import json\n'
        'with open("formats.json","w") as f:json.dump({"swapped_native":s.swapped.dtype.isnative},f)\n')
    admission=json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    for name in ('a32_array','i32_array','zero_array','echo','reference'):
        assert admission[f'exported_cy._mx_classes._c_Values._f_{name}']['return_type']=='object'
    # Protocol 4 normalizes endianness during export; protocol 5 preserves it.
    formats=json.loads((tmp_path/'formats.json').read_text())
    swapped_type=admission['exported_cy._mx_classes._c_Values._f_swapped_array']['return_type']
    assert swapped_type==('const double[:]' if formats['swapped_native'] else 'object')
    for package in ('exported','exported_cy'):
        subprocess.run([sys.executable,'-B','-c',f'''from {package} import mx_model
s=mx_model.Values
for value in (2**80,-2**80,2**100):assert s.echo(value)==value
assert s.reference()==2**80
assert s.zero_at()==1.25
assert s.a32_at(1)==s.swapped_at(1)==s.native_at(1)==2.5
assert s.i32_at(1)==22
'''],cwd=tmp_path,check=True)
