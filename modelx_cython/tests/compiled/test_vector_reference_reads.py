"""Cross-Space vector reads (review r4 vectors): ``projection.vec(n)[i]`` through a Reference to a Space
declared with its class reads the typed vector (vec_get on the compiled entry), where it built a new list
per read (4,000 reads: 0.46 s compiled, 0.001 s live). Python uses still get a new list."""
import subprocess
import sys

from vector_live import build, live_and_compiled

PROJECTION = {
    'vec': 'def vec(n):\n    return [0.5 * i for i in range(n)]\n',
    'ivec': 'def ivec(n):\n    return [i % 7 for i in range(n)]\n',
}
PORTFOLIO = {
    'total': 'def total(n):\n    s = 0.0\n    for i in range(n):\n        s += projection.vec(n)[i]\n    return s\n',
    'local': ('def local(n):\n    v = projection.vec(n)\n    s = 0.0\n    for i in range(n):\n        s += v[i] * ivec(i)\n'
              '    return s + len(projection.ivec(n))\n'),
    'ivec': 'def ivec(i):\n    return projection.ivec(i + 1)[i]\n',
    'python': 'def python(n):\n    return sorted(projection.vec(n), reverse=True)[:3]\n',
}
N = 4000
TIMING = '''\
import sys, time
from importlib import import_module
m = import_module(sys.argv[1]).mx_model
m.Projection.vec({n})
start = time.perf_counter()
m.Portfolio.total({n})
print(time.perf_counter() - start)
'''


def test_vector_reads_through_a_reference(tmp_path):
    build(tmp_path, {'Projection': (PROJECTION, {}), 'Portfolio': (PORTFOLIO, {'projection': 'Projection'})},
          f'm.Portfolio.total({N})\nm.Portfolio.local({N})\nm.Portfolio.python({N})\n'
          + ''.join(f'm.Portfolio.ivec({i})\n' for i in range(N)))
    exprs = [f'm.Portfolio.{name}({N})' for name in ('total', 'local', 'python')] + ['m.Portfolio.ivec(17)']
    live, compiled = live_and_compiled(tmp_path, exprs)
    assert compiled == live
    assert all(kind == 'value' for kind, _ in live)
    source = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    assert '.vec_get(self.projection._mx_cell_vec(n), i)' in source
    assert 'v = self.projection._mx_cell_vec(n)' in source
    (tmp_path / 'timing.py').write_text(TIMING.format(n=N), encoding='utf-8')
    seconds = {}
    for package in ('exported', 'exported_cy'):
        runs = [subprocess.run([sys.executable, 'timing.py', package], cwd=tmp_path, capture_output=True,
                               text=True, check=True) for _ in range(3)]
        seconds[package] = min(float(r.stdout) for r in runs)
    # one list per read took about 400 times live; a typed read is faster than live
    assert seconds['exported_cy'] < 5 * seconds['exported'] + 0.005, seconds
