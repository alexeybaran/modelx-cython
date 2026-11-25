"""R7 of reviewer b (review r4 vectors): a float vector cell that returns an integer vector cell's value on a
path the sample did not reach converts the integers (``vec_of`` of an I64Vec); it raised TypeError."""
from vector_live import build, live_and_compiled

FORMULAS = {
    'iv': 'def iv(t):\n    return tuple([t + 7, t + 8])\n',
    'fv': 'def fv(t, x):\n    if x > 1.0:\n        return tuple(iv(t))\n    return tuple([0.5 * t, 1.5])\n',
}


def test_float_vector_of_an_integer_vector(tmp_path):
    build(tmp_path, {'Space1': (FORMULAS, {})}, ''.join(f'm.Space1.fv({t}, 0.0)\nm.Space1.iv({t})\n' for t in range(4)))
    # live returns the ints, compiled the equal floats
    exprs = [f'[float(x) for x in m.Space1.fv({t}, {x})]' for t in range(4) for x in (0.0, 2.0)]
    live, compiled = live_and_compiled(tmp_path, exprs)
    assert compiled == live
    assert all(kind == 'value' for kind, _ in live)
