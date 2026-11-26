"""R6 of reviewer b (review r4 vectors): an integer vector local whose element update gives a float
(``v[i] += 0.5``, ``v[i] /= 2``, ``v[i] *= x`` with a float x) stays a Python list; the build stopped
(Cython cannot assign a double to a long long)."""
from vector_live import build, live_and_compiled

FORMULAS = {
    'rate': 'def rate(t):\n    return 0.25 * t\n',
    'halves': 'def halves(t):\n    v = [0] * 3\n    v[0] += 0.5\n    v[1] = t\n    return v[0] + v[1] + sum(v)\n',
    'divided': 'def divided(t):\n    v = [t] * 3\n    v[2] /= 2\n    return v\n',
    'scaled': ('def scaled(t):\n    v = []\n    for i in range(3):\n        v.append(i + t)\n    v[1] *= rate(t)\n'
               '    return sum(v) * 1.0\n'),
    'kept': 'def kept(t):\n    v = [1] * 3\n    v[0] += t\n    return v\n',
}


def test_integer_vectors_updated_with_floats(tmp_path):
    build(tmp_path, {'Space1': (FORMULAS, {})}, ''.join(f'm.Space1.{n}({t})\n' for n in FORMULAS for t in range(4)))
    exprs = [f'm.Space1.{n}({t})' for n in FORMULAS for t in range(4)]
    live, compiled = live_and_compiled(tmp_path, exprs)
    assert compiled == live
    assert all(kind == 'value' for kind, _ in live)
