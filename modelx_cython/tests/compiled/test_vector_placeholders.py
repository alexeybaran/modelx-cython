"""R5 (review r4 vectors): ``[None] * n`` (or another value no vector holds) filled in a loop and returned
by a vector cell stays a Python list and is converted where it is returned; it stopped the build."""
from vector_live import build, live_and_compiled

FORMULAS = {
    'floats': ('def floats(t):\n    out = [None] * t\n    for i in range(t):\n        out[i] = 0.5 * i\n'
               '    return out\n'),
    'ints': ('def ints(t):\n    out = [None] * t\n    for i in range(t):\n        out[i] = 2 * i\n'
             '    return out\n'),
    'texts': ('def texts(t):\n    out = [""] * t\n    for i in range(t):\n        out[i] = 1.5 * i\n'
              '    return out\n'),
    'reader': 'def reader(t):\n    return sum(floats(t)) + sum(ints(t)) + len(texts(t))\n',
}


def test_placeholder_lists_returned_as_vectors(tmp_path):
    build(tmp_path, {'Space1': (FORMULAS, {})}, ''.join(f'm.Space1.reader({t})\n' for t in range(5)))
    # (from t = 1 for reader: sum of an empty float vector is 0.0 where live's sum([]) is 0, equal values)
    exprs = [f'm.Space1.{name}({t})' for name in FORMULAS for t in range(name == 'reader', 5)]
    live, compiled = live_and_compiled(tmp_path, exprs)
    assert compiled == live
    assert all(kind == 'value' for kind, _ in live)
