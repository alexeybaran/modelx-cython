"""R4 (review r4 vectors): a conditional expression whose branches are vectors (vector cell calls or
typed vector locals) is a vector where its value is used natively, and a list where Python code uses it."""
from vector_live import build, live_and_compiled

FORMULAS = {
    'vec': 'def vec(t):\n    return tuple([0.5 * t, 1.5, 2.5 * t])\n',
    'returned': 'def returned(t):\n    return tuple(vec(t)) if t > 2 else tuple(vec(2))\n',
    'chosen': ('def chosen(t):\n    a = [1.0] * 3\n    b = [0.0] * 3\n    b[1] = 0.25 * t\n'
               '    c = a if t > 2 else b\n    return c[1] + sum(c) + len(c)\n'),
    'read': 'def read(t):\n    return (vec(t) if t % 2 else vec(t + 1))[2]\n',
    'listed': 'def listed(t):\n    return sorted(vec(t) if t else vec(1), reverse=True)\n',
}
NAMES = ['returned', 'chosen', 'read', 'listed']


def test_conditional_vectors(tmp_path):
    build(tmp_path, {'Space1': (FORMULAS, {})},
          ''.join(f'm.Space1.{name}({t})\n' for name in NAMES for t in range(5)))
    exprs = [f'm.Space1.{name}({t})' for name in NAMES for t in range(5)]
    live, compiled = live_and_compiled(tmp_path, exprs)
    assert compiled == live
    assert all(kind == 'value' for kind, _ in live)
