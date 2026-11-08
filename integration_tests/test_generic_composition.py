"""Compiler features compose without narrowing legal Python values/providers."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from http import HTTPStatus

import modelx as mx
from lifelib.typed_inputs import EnumDefinitions, sum_f64


def compile_model(path, sample):
    (path/'sample.py').write_text(sample, encoding='utf-8')
    subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported', '--no-spec'],
                   cwd=path, check=True)


def test_generator_literals_preserve_integer_width_and_sign(tmp_path):
    values = {'minimum': -(1 << 63), 'maximum': (1 << 63)-1,
              'above': 1 << 63, 'below': -(1 << 63)-1, 'large': 1 << 80}
    low, high = -(1 << 63), (1 << 63)-1
    ranges = {'positive_overshoot': (high-1, high, 2),
              'negative_overshoot': (low+1, low, -2),
              'positive_safe': (high-2, high, 2), 'negative_safe': (low+2, low, -2),
              'empty': (high-1, high-2, 2)}
    model = mx.new_model('LiteralDomains')
    try:
        space = model.new_space('Values', refs={'sum_f64': sum_f64, 'Status':HTTPStatus,
                                'definitions':EnumDefinitions(('Status',)), 'stride': 2})
        for name, value in values.items():
            space.new_cells(name, f'lambda: sum_f64(k * 1.0 for k in ({value},))')
        space.new_cells('step', f'lambda: sum_f64(k * 1.0 for k in range(0, 2, {1 << 80}))')
        space.new_cells('filtered', f'lambda: sum_f64(k * 1.0 for k in ({1 << 63},) if k > 0)')
        for name, args in ranges.items():
            space.new_cells(name, f'lambda: sum_f64(k * 1.0 for k in range{args!r})')
        space.new_cells('dynamic_stride', f'lambda epoch=0: sum_f64(k * 1.0 for k in range({high-1}, {high}, stride))')
        space.new_cells('authored_enum_prefix', '''def authored_enum_prefix():
    _mx_enum_const_0 = -123
    if _mx_enum_const_0 < 0:
        return Status.CONTINUE
    return 0
''')
        space.new_cells('assigned', '''def assigned(epoch=0):
    total = sum_f64(k * 1.0 for k in range(3))
    return str(total)
''')
        space.new_cells('weighted', '''def weighted(epoch=0):
    factor = 1.5
    total = sum_f64(k * factor for k in range(3))
    return total
''')
        model.export(tmp_path/'exported')
    finally:
        model.close()
    names = (*values, *ranges, 'dynamic_stride', 'step', 'filtered', 'authored_enum_prefix', 'assigned', 'weighted')
    compile_model(tmp_path, 'from exported import mx_model\n'
                  + '\n'.join(f'mx_model.Values.{name}()' for name in names)
                  + '\nfor epoch in (1,2): mx_model.Values.assigned(epoch); mx_model.Values.weighted(epoch)\n')
    expected = {name: float(value) for name, value in values.items()}
    expected.update({name: sum(k * 1.0 for k in range(*args)) for name, args in ranges.items()})
    expected['dynamic_stride'] = float(high-1)
    expected.update(step=0.0, filtered=float(1 << 63), authored_enum_prefix=100,
                    assigned='3.0', weighted=4.5)
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-B', '-c',
            f'from {package} import mx_model\n'
            f'expected={expected!r}\n'
            'for name,value in expected.items():\n'
            '    actual=getattr(mx_model.Values,name)()\n'
            '    assert actual==value,(name,actual,value)\n'], cwd=tmp_path, check=True, timeout=30)
    admission = json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    for name in ('above', 'below', 'large', 'step', 'filtered',
                 'positive_overshoot', 'negative_overshoot', 'dynamic_stride'):
        assert admission[f'exported_cy._mx_classes._c_Values._f_{name}']['native_reductions'] == 0
    for name in ('assigned', 'weighted', 'positive_safe', 'negative_safe', 'empty'):
        assert admission[f'exported_cy._mx_classes._c_Values._f_{name}']['native_reductions'] == 1
    for package in ('exported', 'exported_cy'):
        script=f'''from {package} import mx_model
s=mx_model.Values
s.stride=0
try: s.dynamic_stride(1)
except ValueError as error: assert str(error)=='range() arg 3 must not be zero'
else: raise AssertionError('zero range step did not raise')
class Value:
    def __str__(self): return 'custom result'
def replacement(values):
    assert list(values)==[0.0,1.0,2.0]
    return Value()
s.sum_f64=replacement
assert s.assigned(1)=='custom result'
failure=RuntimeError('assigned reduction fallback')
def broken(values):
    assert next(values)==0.0
    raise failure
s.sum_f64=broken
try: s.assigned(2)
except RuntimeError as actual: assert actual is failure
else: raise AssertionError('assigned reduction exception lost')
'''
        subprocess.run([sys.executable,'-B','-c',script],cwd=tmp_path,check=True)


def test_two_providers_in_one_formula_keep_distinct_native_contexts(tmp_path):
    package = tmp_path/'test_providers'
    package.mkdir()
    (package/'__init__.py').write_text('''def left(table,key,column): return table[key][column]
def right(table,key,column): return table[key][column] + 1.0
for function,provider in ((left,'first'),(right,'second')):
    function.__modelx_cython_helper__ = ('test_providers.'+provider,'lookup')
    function.__modelx_cython_numeric__ = 'lookup_f64'
''', encoding='utf-8')
    declarations = '''cdef class LookupContext:
    pass
cdef object lookup(object function, object table, object key, object column)
cdef double lookup_f64(object function, LookupContext context, object table,
                       long long key, object column) except? -1
'''
    implementation = '''# cython: freethreading_compatible=True
cdef class LookupContext:
    def __reduce__(self): return type(self), ()
cdef object lookup(object function, object table, object key, object column):
    return function(table,key,column)
cdef double lookup_f64(object function, LookupContext context, object table,
                       long long key, object column) except? -1:
    return function(table,key,column)
'''
    for name in ('first', 'second'):
        (package/(name+'.pxd')).write_text(declarations, encoding='utf-8')
        (package/(name+'.pyx')).write_text(implementation, encoding='utf-8')
    # Prepare in a subprocess so authored helper References have their real
    # importable package identity, as they do in ordinary model exports.
    formula = "def value(flag, epoch=0):\n    if flag: return left(table,1,'v')\n    return right(table,1,'v')\n"
    prepare = f'''import modelx as mx
from test_providers import left,right
from modelx_cython.helpers import module_name
model=mx.new_model('ProviderComposition')
s=model.new_space('Values',refs={{'left':left,'right':right,'table':{{1:{{'v':2.0}}}}}})
s.new_cells('value',{formula!r})
s.new_cells('value'+module_name('test_providers.second'),"lambda flag, epoch=0: left(table,1,'v')")
model.export('exported')
model.close()
'''
    subprocess.run([sys.executable, '-B', '-c', prepare], cwd=tmp_path, check=True)
    compile_model(tmp_path, 'from exported import mx_model\n'
                  'from modelx_cython.helpers import module_name\n'
                  'for epoch in range(3):\n'
                  '    mx_model.Values.value(True,epoch); mx_model.Values.value(False,epoch)\n'
                  '    getattr(mx_model.Values,"value"+module_name("test_providers.second"))(True,epoch)\n')
    for name in ('exported', 'exported_cy'):
        script = f'''import pickle
from {name} import mx_model
from modelx_cython.helpers import module_name
s=mx_model.Values
assert s.value(True)==2.0 and s.value(False)==3.0
assert getattr(s,'value'+module_name('test_providers.second'))(True)==2.0
restored=pickle.loads(pickle.dumps(mx_model)).Values
assert restored.value(True,1)==2.0 and restored.value(False,1)==3.0
s.left=lambda *args:7.0
assert s.value(True,1)==7.0
failure=LookupError('provider replacement')
def broken(*args): raise failure
s.right=broken
try: s.value(False,2)
except LookupError as actual: assert actual is failure
else: raise AssertionError('helper exception lost')
'''
        subprocess.run([sys.executable, '-B', '-c', script], cwd=tmp_path, check=True)
    tree = ast.parse((tmp_path/'exported_cy/_mx_classes.py').read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    context_types = [ast.dump(node.annotation) for node in owner.body if isinstance(node, ast.AnnAssign)
                     and isinstance(node.annotation, ast.Attribute) and node.annotation.attr=='LookupContext']
    assert len(context_types)==3 and len(set(context_types))==2
