"""The generic provider never admits a replaced builtin as primitive all/any."""
from pathlib import Path
import subprocess
import sys


def test_builtin_reduction_identity_provider(tmp_path):
    source = Path(__file__).parents[1] / 'modelx_cython'
    for suffix in ('.pyx', '.pxd'):
        (tmp_path / ('native' + suffix)).write_bytes((source / ('_native_iterables' + suffix)).read_bytes())
    (tmp_path / 'probe.pyx').write_text('''cimport native
def accepts_all(function):
    return native.all_accepts(function)
def accepts_any(function):
    return native.any_accepts(function)
def accepts_min(function):
    return native.min_accepts(function)
def accepts_max(function):
    return native.max_accepts(function)
def absolute(double value):
    return native.f64_abs(value)
''')
    (tmp_path / 'setup.py').write_text('''from setuptools import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize(['native.pyx', 'probe.pyx'],compiler_directives={'freethreading_compatible':True}))
''')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=tmp_path, check=True)
    subprocess.run([sys.executable, '-c', '''import builtins
import probe
original_all, original_any = builtins.all, builtins.any
assert probe.accepts_all(original_all) and probe.accepts_any(original_any)
assert not probe.accepts_all(original_any) and not probe.accepts_any(original_all)
assert probe.accepts_min(min) and probe.accepts_max(max)
assert not probe.accepts_min(max) and not probe.accepts_max(min)
import math
assert math.copysign(1.0, probe.absolute(-0.0)) == 1.0
def replacement(values): return 17
builtins.all = replacement
assert not probe.accepts_all(builtins.all)
assert probe.accepts_all(original_all)
builtins.all = original_all
'''], cwd=tmp_path, check=True)
    subprocess.run([sys.executable, '-c', '''import builtins
def replacement(values): return 17
builtins.all = replacement
import probe
assert not probe.accepts_all(replacement)
assert probe.accepts_any(builtins.any)
'''], cwd=tmp_path, check=True)
