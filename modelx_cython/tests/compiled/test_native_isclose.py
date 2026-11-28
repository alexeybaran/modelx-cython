"""Check native math.isclose against the exported model on values, types and exceptions."""
import json
import math
import random
import subprocess
import sys
import textwrap

import modelx as mx


def _build(tmp_path, model, sample):
    model.export(tmp_path / 'exported')
    (tmp_path / 'sample.py').write_text(textwrap.dedent(sample), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    return json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())


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
        s.new_cells('rebound', 'def rebound(a, b) -> object:\n'
                    '    return math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-8)\n').is_cached = False
        s.new_cells('mem', 'def mem(i) -> object:\n    x = mo(i)\n'
                    '    return (x in (1, 2, k3), x not in (0.0, "a"), x in (None,), x in (2 ** 62, -0.0))\n')
        report = _build(tmp_path, model, '''\
            from exported import mx_model as m
            w = m.Values
            for i in range(24):
                for j in range(24):
                    w.c1(i, j); w.c2(i, j); w.c3(i, j)
            for i in range(10):
                w.mem(i)
            w.rebound(1.0, 2.0)
            ''')
    finally:
        model.close()
    for name in ('c1', 'c2', 'c3', 'c4', 'rebound'):
        cell = _cell(report, name)
        assert cell['expressions_untyped'] == [], name
        assert cell['native_helpers'] == ['modelx_cython.runtime._native_math.m_isclose'], name
    _differential(tmp_path, _PROBE_HEAD + '''\
    for i in range(24):
        for j in range(24):
            for f in (w.c1, w.c2, w.c3):
                call(f, i, j)
        call(w.c4, i)
    for i in range(10):
        call(w.c5, i); call(w.mem, i)
    from types import SimpleNamespace
    calls = []
    def replacement(a, b, **tolerances):
        calls.append((a, b, tuple(tolerances)))
        return True
    w.math = SimpleNamespace(isclose=replacement)
    call(w.rebound, 1.0, 2.0)
    assert calls == [(1.0, 2.0, ('abs_tol', 'rel_tol'))], calls
    json.dump(out, open(sys.argv[2], 'w'))
    ''')
