"""R1 (review r4 vectors): a local list the formula changes in place (a list method other than append,
``del v[i]``, a callee that may change it, a second name for it) stays a Python list, in a module with
vector cells and in one without (D94), so compiled equals live. A cached vector changed in place stops
the build."""
import subprocess
import sys

import modelx as mx
import pytest

from vector_live import build, live_and_compiled

MUTATORS = {
    'vec': 'def vec(t):\n    return [0.5 * (3 - i) + t for i in range(4)]\n',
    'sorted_out': ('def sorted_out(t):\n    out = []\n    for i in range(4):\n        out.append(0.5 * (3 - i) + t)\n'
                   '    out.sort()\n    return out\n'),
    'reversed_out': 'def reversed_out(t):\n    out = [0.0] * 3\n    out[0] = 1.5 * t\n    out.reverse()\n    return out\n',
    'extended': 'def extended(t):\n    out = [0.0] * 2\n    out.extend([2.5, 3.5])\n    return sum(out) + len(out)\n',
    'popped': 'def popped(t):\n    out = [0.0] * 3\n    out[2] = 4.5 + t\n    last = out.pop()\n    return last + len(out)\n',
    'deleted': 'def deleted(t):\n    out = [1.0] * 3\n    out[0] = 2.0 + t\n    del out[0]\n    return len(out) + out[0]\n',
    'called': 'def called(t):\n    out = [0.0] * 3\n    out[1] = 2.0 + t\n    list.reverse(out)\n    return out[0]\n',
    'aliased': ('def aliased(t):\n    out = [0.0] * 3\n    other = out\n    other.insert(0, 9.5)\n'
                '    out[1] = t + 0.5\n    return out[0] + out[1] + len(out)\n'),
    'contained': ('def contained(t):\n    out = [0.0] * 2\n    box = (out, 1)\n    box[0].append(7.5)\n'
                  '    return sum(out) + len(out) + t\n'),
    'bound_method': ('def bound_method(t):\n    out = [3.0, 1.0, 2.0 + t]\n    fix = out.sort\n    fix()\n'
                     '    return out[0] * 100.0 + out[2]\n'),
}
NO_VECTOR_CELL = {
    # D94: a module without vector cells
    'smallest': ('def smallest(t):\n    xs = []\n    for i in range(3):\n        xs.append(0.5 * (3 - i) + t)\n'
                 '    xs.sort()\n    return xs[0] + xs[1] * 10.0\n'),
    'cleared': 'def cleared(t):\n    xs = [0.5] * 3\n    xs.clear()\n    return len(xs) + t * 1.0\n',
}


def _check(tmp_path, formulas):
    calls = [f'm.Space1.{name}({t})' for name in formulas for t in (0, 1, 2)]
    build(tmp_path, {'Space1': (formulas, {})}, ''.join(c + '\n' for c in calls))
    live, compiled = live_and_compiled(tmp_path, calls)
    assert compiled == live
    assert all(kind == 'value' for kind, _ in live)


def test_list_mutators_in_a_module_with_vector_cells(tmp_path):
    _check(tmp_path, MUTATORS)


def test_list_mutators_in_a_module_without_vector_cells(tmp_path):
    _check(tmp_path, NO_VECTOR_CELL)


@pytest.mark.parametrize('formula', [
    'def changes(t):\n    v = vec(t)\n    v.sort()\n    return v[0]\n',
    'def changes(t):\n    vec(t).reverse()\n    return 1.0\n',
])
def test_a_cached_vector_changed_in_place_stops_the_build(tmp_path, formula):
    model = mx.new_model('CachedChanged')
    try:
        space = model.new_space('Space1')
        space.new_cells(formula=MUTATORS['vec'])
        space.new_cells(formula=formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\nm.Space1.changes(1)\n')
    result = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                             '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'cached vectors are immutable' in result.stdout + result.stderr
