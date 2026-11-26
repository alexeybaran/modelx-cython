"""A Cell with integer parameters called from Python goes through its Python entry: an integer argument
through the index protocol (an exact int, a bool, a NumPy integer, an int subclass) gives live modelx's
value, and a subclass or an override of the Space is called as written."""
import json
import subprocess
import sys
import textwrap

import modelx as mx

PROBE = textwrap.dedent('''\
    import enum, json
    import numpy as np
    from exported_cy import mx_model as m

    class Small(int):
        pass

    class Level(enum.IntEnum):
        LOW = 2

    def run(call):
        try:
            return ['value', float(call())]
        except Exception as error:
            return [type(error).__name__, str(error)]

    S = m.S

    class Sub(type(S)):           # a Python subclass: Cython dispatches through the attribute
        pass

    class Over(type(S)):
        def f(self, t):
            return -1.0

    sub, over = Sub(m), Over(m)
    out = {
        'sub': run(lambda: sub.f(3)),
        'over': run(lambda: over.f(3)), 'over_g': run(lambda: over.g(3, 4)),
        'exact': run(lambda: S.f(3)), 'two': run(lambda: S.g(3, 4)), 'default': run(lambda: S.g(3)),
        'bool': run(lambda: S.f(True)), 'numpy': run(lambda: S.f(np.int64(5))),
        'subclass': run(lambda: S.f(Small(6))), 'enum': run(lambda: S.g(Level.LOW, Level.LOW)),
        'keyword': run(lambda: S.g(u=4, t=3)), 'keyword_numpy': run(lambda: S.g(3, u=np.int32(7))),
        'mixed': run(lambda: S.h(2, 0.5)), 'dynamic': run(lambda: m.D[2].c(3)),
        'gate': [S.f.__name__, S.f.__wrapped__.__name__, S.f.__doc__ == type(S).f.__doc__, callable(S.f)],
    }
    print(json.dumps(out))
''')

def test_python_calls_check_integer_arguments(tmp_path):
    model = mx.new_model('CallEntry')
    try:
        s = model.new_space('S')
        s.new_cells('f', 'lambda t: t * 1.5')
        s.new_cells('g', 'lambda t, u=2: t * 1.5 + u')
        s.new_cells('h', 'lambda t, x: t * x')
        d = model.new_space('D', formula="lambda k: {'refs': {'scale': k * 10}}")
        d.scale = 1
        d.new_cells('c', 'lambda t: t * scale + 0.5')
        live = {'exact': s.f(3), 'two': s.g(3, 4), 'default': s.g(3), 'bool': s.f(True), 'numpy': s.f(5),
                'subclass': s.f(6), 'enum': s.g(2, 2), 'keyword': s.g(3, 4), 'keyword_numpy': s.g(3, 7),
                'mixed': s.h(2, 0.5), 'dynamic': d[2].c(3)}
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'for t in range(10):\n    m.S.f(t); m.S.g(t, t); m.S.h(t, 0.5); m.D[2].c(t)\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, capture_output=True)
    out = json.loads(subprocess.run([sys.executable, '-c', PROBE], cwd=tmp_path, check=True,
                                    capture_output=True, text=True).stdout)
    for name, value in live.items():
        assert out[name] == ['value', float(value)], name
    assert out['gate'] == ['f', 'f', True, True]
    assert out['sub'] == out['exact']
    assert out['over'] == ['value', -1.0] and out['over_g'] == out['two']   # an override is called as written
