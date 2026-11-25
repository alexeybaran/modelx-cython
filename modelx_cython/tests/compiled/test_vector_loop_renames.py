"""R5 of reviewer b (review r4 vectors): in a comprehension or generator over vectors expanded to a C loop,
the loop target is renamed; a keyword argument or an attribute with the target's name is not a variable
and keeps its name (``w(t, s=s)`` raised TypeError on every call)."""
import math

from vector_live import build, live_and_compiled

FORMULAS = {
    'iv': 'def iv(t):\n    return [t, t + 1, t + 2]\n',
    'fv': 'def fv(t):\n    return [0.5 * t, 1.5]\n',
    'w': 'def w(t, s=0):\n    return 0.25 * t + s\n',
    'summed': 'def summed(t):\n    return sum(w(t, s=s) for s in iv(t)) + 0.0\n',
    'listed': 'def listed(t):\n    return [w(t, s=s) for s in iv(t)]\n',
    'attribute': 'def attribute(t):\n    return [math.e * e for e in fv(t)]\n',
}
NAMES = ['summed', 'listed', 'attribute']


def test_keywords_and_attributes_keep_their_names(tmp_path):
    build(tmp_path, {'Space1': (FORMULAS, {'math': math})},
          ''.join(f'm.Space1.{name}({t})\n' for name in NAMES for t in range(4)))
    exprs = [f'm.Space1.{name}({t})' for name in NAMES for t in range(4)]
    live, compiled = live_and_compiled(tmp_path, exprs)
    assert compiled == live
    assert all(kind == 'value' for kind, _ in live)
