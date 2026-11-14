"""W1.8: built-in calls left in Python by W1.4, compiled natively; differential tests.

Each test builds one model, compiles it with a sample that reaches every formula, and
compares the exported (interpreted) model with the compiled one on the same calls:
result type and repr, or exception type and message. Property cases come from a seeded
pool with ties, signed zeros, infinities, nan and integers near 2**62; edge cases are
listed explicitly.
"""
import json
import math
import random
import subprocess
import sys
import textwrap

import modelx as mx


def _pool(seed, size):
    rng = random.Random(seed)
    edges = [0.0, -0.0, 1.0, -1.0, 1.5, 0.5, math.inf, -math.inf, math.nan,
             2.0 ** 62, -(2.0 ** 62), 1e-300, 5e-324]
    floats = edges + [rng.choice((rng.uniform(-3, 3), float(rng.randint(-3, 3)),
                                  rng.uniform(-1e6, 1e6))) for _ in range(size - len(edges))]
    ints = [0, 1, -1, 3, -7, 2 ** 62, -(2 ** 62), 2 ** 62 + 1] + [
        rng.randint(-5, 5) for _ in range(size - 8)]
    return floats, ints


def _build(tmp_path, model, sample):
    model.export(tmp_path / 'exported')
    (tmp_path / 'sample.py').write_text(textwrap.dedent(sample), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    return source, report


def _differential(tmp_path, probe):
    """Run ``probe`` on both packages; the outcome lists must be identical."""
    results = []
    for package in ('exported', 'exported_cy'):
        out = tmp_path / f'{package}.json'
        subprocess.run([sys.executable, '-c', textwrap.dedent(probe), package, str(out)],
                       cwd=tmp_path, check=True)
        results.append(json.loads(out.read_text()))
    live, compiled = results
    assert len(live) == len(compiled) and live, len(live)
    for index, (a, b) in enumerate(zip(live, compiled)):
        assert a == b, (index, a, b)
    return live


_PROBE_HEAD = '''\
    import json, sys
    from importlib import import_module
    w = import_module(sys.argv[1]).mx_model.Values
    out = []
    def call(f, *args):
        try:
            r = f(*args)
            out.append([type(r).__name__, repr(r)])
        except Exception as e:
            out.append(['raise', type(e).__name__, str(e)])
'''


def _cell(report, name):
    return report[f'exported_cy._mx_classes._c_Values._f_{name}']


def test_minmax_operand_kinds_and_objects(tmp_path):
    floats, ints = _pool(18, 24)
    # ints and floats with one str: an object cell (the tracer widens int|float to float)
    mixed = [floats[i] if i % 3 else ints[i] for i in range(23)] + ['x']
    model = mx.new_model('MinMaxW18')
    try:
        s = model.new_space('Values', refs={'floats': tuple(floats), 'ints': tuple(ints),
                                            'mixed': tuple(mixed), 'words': ('a', 'b', 'a')})
        s.new_cells('v', 'lambda i: floats[i]')
        s.new_cells('n', 'lambda i: ints[i]')
        s.new_cells('o', 'lambda i: mixed[i]')
        s.new_cells('wd', 'lambda i: words[i]')
        # locals of one kind (fixpoint), nested min/max, the cached-cell form
        s.new_cells('loc', 'def loc(i, j) -> object:\n    worst = 0.0\n    f = v(i)\n'
                    '    worst = max(worst, -min(f, 0.0), max(f - 1.0, 0.0))\n'
                    '    return max(worst, f - v(j))\n')
        # built-ins as operands: float, abs, len, round(x, n)
        s.new_cells('blt', 'def blt(i, j) -> object:\n'
                    '    return min(abs(v(i)), float(n(j)), round(v(j), 2))\n')
        s.new_cells('ln', 'def ln(i) -> object:\n    xs = [1, 2]\n    return max(len(xs), n(i))\n')
        # mixed classes: objects, int and float literals, three or more mixed operands
        s.new_cells('obj', 'def obj(i, j) -> object:\n    return max(0, o(i), v(j))\n')
        s.new_cells('obj2', 'def obj2(i, j) -> object:\n    return min(o(i), o(j))\n')
        s.new_cells('mix3', 'def mix3(i, j) -> object:\n    return max(n(i), v(j), n(j))\n')
        s.new_cells('mix4', 'def mix4(i, j) -> object:\n    return min(1, v(i), n(j), 0.5)\n')
        s.new_cells('ints2', 'def ints2(i, j) -> object:\n    k = n(i) + 1\n    return min(k, n(j))\n')
        # a comparison error is raised from the same pair as the built-in
        s.new_cells('err', 'def err(i, j) -> object:\n    return max(o(i), wd(j % 3))\n')
        # a comprehension variable of a parameter's name is not the parameter
        s.new_cells('shadow', 'def shadow(i) -> object:\n    return [max(i, 0) for i in (0.5, -1, v(i))]\n')
        source, report = _build(tmp_path, model, f'''\
            from exported import mx_model as m
            w = m.Values
            def t(f, *args):
                try:
                    f(*args)
                except TypeError:
                    pass
            for i in range(24):
                for j in range(24):
                    for f in (w.loc, w.blt, w.obj, w.obj2, w.mix3, w.mix4, w.ints2):
                        t(f, i, j)
                w.ln(i); w.shadow(i)
            ''')
    finally:
        model.close()
    assert 'max(*(' not in source and 'min(*(' not in source
    for name in ('loc', 'blt', 'ln', 'ints2'):
        assert _cell(report, name)['expressions_object'] == [], name
        assert _cell(report, name)['expressions_untyped'] == [], name
    for name in ('obj', 'obj2', 'mix3', 'mix4', 'err', 'shadow'):
        assert [e['form'] for e in _cell(report, name)['expressions_object']], name
        assert _cell(report, name)['expressions_untyped'] == [], name
    _differential(tmp_path, _PROBE_HEAD + '''\
    for i in range(24):
        for j in range(24):
            for f in (w.loc, w.blt, w.obj, w.obj2, w.mix3, w.mix4, w.ints2, w.err):
                call(f, i, j)
        call(w.ln, i); call(w.shadow, i)
    json.dump(out, open(sys.argv[2], 'w'))
    ''')


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
        source, report = _build(tmp_path, model, '''\
            from exported import mx_model as m
            w = m.Values
            for k in range(12):
                for f in (w.a, w.b, w.c, w.d, w.e, w.f, w.g, w.m):
                    try:
                        f(k)
                    except ValueError:
                        pass
                if k:
                    w.h(k)
            ''')
    finally:
        model.close()
    for name in ('a', 'b', 'c', 'd', 'e', 'g', 'h', 'm'):
        assert _cell(report, name)['expressions_untyped'] == [], name
    f = _cell(report, 'f')['expressions_untyped']
    assert [e['form'] for e in f] == ['all'] and 'one branch' in f[0]['reason']
    _differential(tmp_path, _PROBE_HEAD + '''\
    for k in range(12):
        for f in (w.a, w.b, w.c, w.d, w.e, w.f, w.g, w.h, w.m):
            call(f, k)
    json.dump(out, open(sys.argv[2], 'w'))
    ''')


def test_object_membership_and_isclose(tmp_path):
    rng = random.Random(1808)
    close = [0.0, -0.0, 1.0, 1.0 + 1e-10, 1.0 + 1e-8, math.inf, -math.inf, math.nan, 1e-9, 2e-9,
             5e300, -5e300, 1e-320]
    close += [rng.choice((1.0, 100.0, 1e-9)) * (1 + rng.choice((1e-12, 1e-9, 1e-7, -1e-10)))
              for _ in range(11)]
    members = (0, 1, 2.0, math.nan, 'a', None, 3, -0.0, True, 2 ** 62)
    model = mx.new_model('OtherW18')
    try:
        s = model.new_space('Values', refs={'math': math, 'close': tuple(close),
                                            'members': members, 'rt': 1e-9, 'k3': 3})
        s.new_cells('v', 'lambda i: close[i]')
        s.new_cells('o', 'lambda i: close[i] if i != 10 else 3')
        s.new_cells('mo', 'lambda i: members[i]')
        s.new_cells('c1', 'def c1(i, j) -> object:\n    return math.isclose(v(i), v(j))\n')
        s.new_cells('c2', 'def c2(i, j) -> object:\n'
                    '    return math.isclose(v(i), v(j), rel_tol=1e-9, abs_tol=1e-8)\n')
        s.new_cells('c3', 'def c3(i, j) -> object:\n    x = o(j)\n'
                    '    return math.isclose(o(i), x, abs_tol=1e-9, rel_tol=rt)\n')
        s.new_cells('c4', 'def c4(i) -> object:\n    return math.isclose(v(i), 1.0, rel_tol=-1.0)\n')
        s.new_cells('c5', 'def c5(i) -> object:\n    return math.isclose(mo(i), 0)\n')
        s.new_cells('mem', 'def mem(i) -> object:\n    x = mo(i)\n'
                    '    return (x in (1, 2, k3), x not in (0.0, "a"), x in (None,), x in (2 ** 62, -0.0))\n')
        source, report = _build(tmp_path, model, '''\
            from exported import mx_model as m
            w = m.Values
            for i in range(24):
                for j in range(24):
                    w.c1(i, j); w.c2(i, j); w.c3(i, j)
            for i in range(10):
                w.mem(i)
            ''')
    finally:
        model.close()
    assert '.isclose(' not in source.replace('namespace.isclose', '')
    for name in ('c1', 'c2', 'c3', 'c4'):
        assert _cell(report, name)['expressions_untyped'] == [], name
    _differential(tmp_path, _PROBE_HEAD + '''\
    for i in range(24):
        for j in range(24):
            for f in (w.c1, w.c2, w.c3):
                call(f, i, j)
        call(w.c4, i)
    for i in range(10):
        call(w.c5, i); call(w.mem, i)
    json.dump(out, open(sys.argv[2], 'w'))
    ''')
