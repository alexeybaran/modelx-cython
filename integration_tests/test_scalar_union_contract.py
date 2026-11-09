"""Empty ordinary sums keep integer identity through native Cell caches."""
import os
import subprocess
import sys

import modelx as mx


def test_scalar_sum_variant_cache_modes(tmp_path):
    model = mx.new_model('ScalarVariants')
    try:
        space = model.new_space('Values')
        space.new_cells('term', 'lambda t: 1.5*t')
        space.new_cells('total', 'lambda n: sum(term(t) for t in range(n))')
        space.new_cells('twice', 'lambda n: 2*total(n)')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model\nfor n in [0,1,3]:\n    mx_model.Values.total(n);mx_model.Values.twice(n)\n')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
        '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, capture_output=True,
        text=True, env=dict(os.environ))
    (tmp_path / 'build.log').write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    probe = '''import importlib,pickle,sys
import pandas as pd
m=importlib.import_module(sys.argv[1]).mx_model
for root in [type(m)(),pickle.loads(pickle.dumps(type(m)()))]:
    s=root.Values
    for _ in range(2):
        assert s.total(0)==0 and type(s.total(0)) is int
        assert s.twice(0)==0 and type(s.twice(0)) is int
        assert s.total(3)==4.5 and type(s.total(3)) is float
        assert s.twice(3)==9.0 and type(s.twice(3)) is float
        assert str(pd.DataFrame({'amount':[s.total(0)]}).dtypes.iloc[0])=='int64'
'''
    for package in ['exported', 'exported_cy']:
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                             capture_output=True, text=True, env=dict(os.environ))
        (tmp_path / (package + '-validation.log')).write_text(run.stdout + run.stderr)
        assert run.returncode == 0, run.stdout + run.stderr
