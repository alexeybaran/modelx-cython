"""Normal compiler route preserves extrema effects and IEEE edge selection."""
import os
import subprocess
import sys
import textwrap

import modelx as mx


def test_ordered_builtin_extrema_native_contract(tmp_path):
    model = mx.new_model('OrderedArguments')
    try:
        space = model.new_space('Values', refs={'values': (1.0, 2.0, 3.0),
            'events': [], 'failures': (None, None, None)})
        space.new_cells('term', '''def term(index):
    events.append(index)
    if failures[index] is not None:
        raise failures[index]
    return values[index]
''')
        space.new_cells('maximum', 'lambda: max(term(0), term(1), term(2))')
        space.new_cells('minimum', 'lambda: min(term(0), term(1), term(2))')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model\nm=mx_model.Values\nm.maximum();m.minimum()\n')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
        '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, capture_output=True,
        text=True, env=dict(os.environ))
    (tmp_path / 'build.log').write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text()
    assert 'max(*(' in source and 'min(*(' in source
    probe = textwrap.dedent('''
        import importlib,itertools,math,struct,sys
        model=importlib.import_module(sys.argv[1]).mx_model
        def fresh(values,failures=(None,None,None)):
            result=type(model)().Values
            result.values=values;result.events=[];result.failures=failures
            return result
        samples=[float('nan'),float('inf'),float('-inf'),0.0,-0.0,1.0,-1.0]
        for name,operation in [('maximum',max),('minimum',min)]:
            for values in itertools.product(samples,repeat=3):
                space=fresh(values)
                actual=getattr(space,name)();expected=operation(values)
                assert space.events==[0,1,2],(name,space.events)
                if math.isnan(expected):assert math.isnan(actual)
                else:assert struct.pack('d',actual)==struct.pack('d',expected)
            first,second=ValueError('first'),LookupError('second')
            space=fresh((1.0,2.0,3.0),(first,second,None))
            try:getattr(space,name)()
            except ValueError as error:assert error is first
            else:raise AssertionError('first exception was lost')
            assert space.events==[0]
    ''')
    for package in ['exported', 'exported_cy']:
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                             capture_output=True, text=True, env=dict(os.environ))
        (tmp_path / (package + '-validation.log')).write_text(run.stdout + run.stderr)
        assert run.returncode == 0, run.stdout + run.stderr
