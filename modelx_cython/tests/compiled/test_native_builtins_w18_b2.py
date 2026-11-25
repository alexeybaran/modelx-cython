"""W1.8: built-in calls left in Python by W1.4, compiled natively; differential tests: nested reductions (B2)."""
import modelx as mx

from test_native_builtins_w18 import _build, _cell, _differential, _pool, _PROBE_HEAD


def test_nested_reductions_lifted(tmp_path):
    floats, _ = _pool(81, 12)
    model = mx.new_model('NestedW18')
    try:
        s = model.new_space('Values', refs={'vals': tuple(floats), 'top': 12, 'tol': 1e-9})
        s.new_cells('v', 'lambda i: vals[i]')
        s.new_cells('n', 'lambda i: i * 3 - 4')
        s.new_cells('a', 'def a(k) -> object:\n    return bool(all(abs(v(i)) <= 10.0 for i in range(k)))\n')
        s.new_cells('b', 'def b(k) -> object:\n    if not all(v(i) < 5.0 for i in range(k)):\n'
                    '        return False\n    return sum(v(i) for i in range(k)) / 2.0\n')
        s.new_cells('c', 'def c(k) -> object:\n    return n(k) * sum(v(i) for i in range(k, top))\n')
        s.new_cells('d', 'def d(k) -> object:\n    ages = [n(k)]\n    if k > 2:\n'
                    '        ages.append(n(k - 2))\n    return 12 * (top - min(ages))\n')
        s.new_cells('e', 'def e(k) -> object:\n'
                    '    return (sum(v(i) for i in range(k)) + v(k % 12) + sum(v(i) for i in range(k, top)))\n')
        s.new_cells('f', 'def f(k) -> object:\n    return k > 3 and all(v(i) >= 0 for i in range(k))\n')
        s.new_cells('g', 'def g(k) -> object:\n    y = min(n(k), int(max(n(i) for i in range(k + 1))))\n    return y\n')
        s.new_cells('h', 'def h(k) -> object:\n    return max(0.0, max(v(i) for i in range(k)) - v(0))\n')
        s.new_cells('m', 'def m(k) -> object:\n    xs = [v(i) for i in range(k)]\n'
                    '    return 1.0 - tol / min(x for x in xs if x > 0.0)\n')
        s.new_cells('a2', 'def a2(k) -> object:\n    return k % 3 or any(v(i) > 1.0 for i in range(k))\n')
        # branches of different classes: the lifted result keeps each branch's class
        s.new_cells('ty', 'def ty(k) -> object:\n    return (max(n(i) for i in range(k + 1)) if k % 2 else 0.5) * 1\n')
        s.new_cells('ty2', 'def ty2(k) -> object:\n    return k > 4 and max(n(i) for i in range(k + 1))\n')
        # an augmented assignment is not lifted (its target is read first): kept, listed
        s.new_cells('o', 'def o(k) -> object:\n    x = 1.0\n    x += 2 * sum(v(i) for i in range(k))\n    return x\n')
        # a conditional expression with reductions in its branches becomes an if statement
        s.new_cells('r', 'def r(k) -> object:\n    free = [i for i in range(k) if v(i) <= 0.0]\n'
                    '    return (min(free) if free else max(n(i) for i in range(k + 1)) + 1) + 1\n')
        report = _build(tmp_path, model, '''\
            from exported import mx_model as m
            w = m.Values
            for k in range(12):
                for f in (w.a, w.b, w.c, w.d, w.e, w.f, w.g, w.m, w.r, w.a2, w.o, w.ty, w.ty2):
                    try:
                        f(k)
                    except ValueError:
                        pass
                if k:
                    w.h(k)
            ''')
    finally:
        model.close()
    for name in ('a', 'b', 'c', 'd', 'e', 'g', 'h', 'm', 'r', 'a2', 'ty', 'ty2'):
        assert _cell(report, name)['expressions_untyped'] == [], name
    # and/or: the left operand once, the reduction only when the operator evaluates it
    assert _cell(report, 'f')['expressions_untyped'] == []
    o = _cell(report, 'o')['expressions_untyped']
    assert [e['form'] for e in o] == ['sum'] and 'augmented assignment' in o[0]['reason']
    _differential(tmp_path, _PROBE_HEAD + '''\
    for k in range(12):
        for f in (w.a, w.b, w.c, w.d, w.e, w.f, w.g, w.h, w.m, w.r, w.a2, w.o, w.ty, w.ty2):
            call(f, k)
    json.dump(out, open(sys.argv[2], 'w'))
    ''')
