"""R2 (review r4 vectors): a value not proven an integer that enters an integer vector (a comprehension's
element, ``v[i] = x``, ``v.append(x)``, ``[x] * n``, ``v[i] += x``) is checked: an int is stored, a float
raises TypeError where a C conversion truncated it silently. The sample sees only integers."""
from vector_live import build, live_and_compiled

FORMULAS = {
    # an object-typed cell: an int on the sampled calls of the vector cells, a float with scale 1.5
    'raw': 'def raw(i, scale):\n    return i if scale == 1.0 else i * scale\n',
    'comp': 'def comp(t, scale):\n    return tuple(raw(z, scale) for z in range(t))\n',
    'written': ('def written(t, scale):\n    acc = [0] * t\n    for i in range(t):\n        acc[i] = raw(i, scale)\n'
                '    return tuple(acc)\n'),
    'appended': ('def appended(t, scale):\n    acc = []\n    for i in range(t):\n        acc.append(raw(i, scale))\n'
                 '    return tuple(acc)\n'),
    'filled': 'def filled(t, scale):\n    return (raw(3, scale),) * t\n',
    'updated': ('def updated(t, scale):\n    acc = [1] * t\n    for i in range(t):\n        acc[i] += raw(i, scale)\n'
                '    return tuple(acc)\n'),
    'local_sum': ('def local_sum(t, scale):\n    acc = [0] * t\n    for i in range(t):\n        acc[i] = raw(i, scale)\n'
                  '    return sum(acc) * 1.0 + len(acc)\n'),
}
VECTORS = ['comp', 'written', 'appended', 'filled', 'updated', 'local_sum']
SAMPLE = ('m.Space1.raw(1, 2.5)\n'      # raw returns floats and ints: an object result
          + ''.join(f'm.Space1.{name}({t}, 1.0)\n' for name in VECTORS for t in range(5)))


def test_non_integers_entering_integer_vectors_are_checked(tmp_path):
    build(tmp_path, {'Space1': (FORMULAS, {})}, SAMPLE)
    exprs = [f'm.Space1.{name}(4, {scale})' for name in VECTORS for scale in (1.0, 1.5)]
    live, compiled = live_and_compiled(tmp_path, exprs)
    for expr, a, b in zip(exprs, live, compiled):
        assert a[0] == 'value', (expr, a)
        if expr.endswith('1.0)'):
            assert a == b, (expr, a, b)
        else:
            # live holds the floats 1.5, 4.5: compiled equals live or raises, never a truncated integer
            assert b in (a, ('raises', 'TypeError')), (expr, a, b)
    assert compiled[VECTORS.index('comp') * 2 + 1] == ('raises', 'TypeError')
