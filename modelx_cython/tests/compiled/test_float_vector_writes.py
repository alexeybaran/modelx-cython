"""R3 (review r4 vectors): a value not proven a float that enters a float vector (a comprehension's
element, ``v[i] = x``, ``v.append(x)``, ``[x] * n``) is checked: a float, or an int a double holds exactly,
is stored; a larger int raises TypeError where a C conversion rounded it silently."""
from vector_live import build, live_and_compiled

FORMULAS = {
    # an object-typed cell: floats on the sampled calls of the vector cells, a large int with scale 2.0
    'obj': 'def obj(z, scale):\n    return None if scale == 0.0 else (z + 0.5 if scale == 1.0 else 2 ** 60 + 1 + z)\n',
    'comp': 'def comp(t, scale):\n    return tuple(obj(z, scale) for z in range(t))\n',
    'written': ('def written(t, scale):\n    v = [0.0] * t\n    for i in range(t):\n        v[i] = obj(i, scale)\n'
                '    return tuple(v)\n'),
    'appended': ('def appended(t, scale):\n    v = []\n    for i in range(t):\n        v.append(obj(i, scale))\n'
                 '    return tuple(v)\n'),
    'filled': 'def filled(t, scale):\n    return (obj(1, scale),) * t\n',
    'local': ('def local(t, scale):\n    v = [0.0] * 2\n    v[0] = obj(t, scale)\n'
              '    return int(v[0]) - int(obj(t, scale)) + vec(t)[0]\n'),
    'vec': 'def vec(t):\n    return tuple([0.5 * t, 1.0])\n',
}
CELLS = ['comp', 'written', 'appended', 'filled', 'local']
SAMPLE = ('m.Space1.obj(0, 0.0)\n'
          + ''.join(f'm.Space1.{name}({t}, 1.0)\n' for name in CELLS for t in range(5)))


def test_non_floats_entering_float_vectors_are_checked(tmp_path):
    build(tmp_path, {'Space1': (FORMULAS, {})}, SAMPLE)
    exprs = [f'm.Space1.{name}(4, {scale})' for name in CELLS for scale in (1.0, 2.0)]
    live, compiled = live_and_compiled(tmp_path, exprs)
    for expr, a, b in zip(exprs, live, compiled):
        assert a[0] == 'value', (expr, a)
        if expr.endswith('1.0)'):
            assert a == b, (expr, a, b)
        else:
            # live holds the int 2 ** 60 + 1 + z: compiled equals live or raises, never a rounded float
            assert b in (a, ('raises', 'TypeError')), (expr, a, b)
