"""Integer guards without a run-time mode: C integer ``+ - *`` and unary minus of a formula set an
overflow flag that the formula tests once before it returns, raising OverflowError naming the Cell.
Nothing depends on the sample: the reproductions below wrapped silently (or flipped with an unrelated
traced value) when a model-wide mode followed the largest traced integer. Each must raise or equal
live; small numbers equal live; a float that reaches an integer Reference raises, and a loop target
read after a loop that ran no iteration raises as live."""
import json
import re
import subprocess
import sys
import textwrap

MODEL = '''\
import modelx as mx
def build():
    m = mx.new_model('Ovf')
    s = m.new_space('S')
    s.new_cells('base', 'lambda: 3_000_000')
    s.new_cells('local_typed', 'def local_typed():\\n    k = base()\\n    return (k * k * k) // (k * k)\\n')
    s.new_cells('inferred', 'def inferred(n):\\n    x = 1\\n    for i in range(n):\\n        x = x * 1000\\n    return x % 7\\n')
    s.new_cells('augmented', 'def augmented(n):\\n    x = 1\\n    for i in range(n):\\n        x *= 1000\\n    return x % 7\\n')
    s.new_cells('il', 'lambda n: n * n * n * n')
    s.new_cells('float_argument', 'lambda: il(1e5)')
    s.new_cells('vec', 'lambda k: [k, 2]')
    s.new_cells('element', 'def element(k):\\n    v = vec(k)\\n    return (v[0] * v[0]) % 1000\\n')
    s.new_cells('top', 'lambda: 2_100_000')
    s.new_cells('reduced', 'lambda: sum(float(t * t * t) for t in range(top()))')
    s.new_cells('lowest', 'lambda: -9223372036854775807 - 1')
    s.new_cells('negated', 'lambda: -lowest()')
    s.new_cells('after_loop', 'def after_loop(n):\\n    for i in range(n):\\n        pass\\n    return float(i)\\n')
    s.new_cells('scaled', 'lambda: iref * 3')
    s.new_cells('squared', 'def squared():\\n    t = kref + 1\\n    return t * t - 1\\n')
    s.new_cells('small', 'lambda t: (t * 7 + 3) * (t - 2) - (t // 3)')
    s.iref = 2
    s.kref = 5
    return m
'''

SAMPLE = '''\
from exported import mx_model
S = mx_model.S
S.local_typed(); S.inferred(2); S.augmented(2); S.il(3); S.vec(3); S.element(3); S.reduced(); S.lowest()
S.negated(); S.after_loop(3); S.scaled(); S.squared()
[S.small(t) for t in range(50)]
'''

CHECKS = '''\
import json, sys
sys.path.insert(0, '.')
from model import build
from exported_cy import mx_model
live = build()
comp = type(mx_model)()

def outcome(model, call):
    try:
        return ['value', repr(eval('model.S.' + call))]
    except BaseException as error:      # modelx wraps the formula's error in FormulaError
        text = str(error)
        for name in ('OverflowError', 'TypeError', 'UnboundLocalError', 'IndexError'):
            if type(error).__name__ == name or name + ':' in text:
                return [name, text.splitlines()[-1] if type(error).__name__ != name else text]
        return [type(error).__name__, text]

calls = ['local_typed()', 'inferred(2)', 'inferred(10)', 'augmented(2)', 'augmented(10)', 'float_argument()',
         'element(1)', 'reduced()', 'negated()', 'after_loop(3)', 'after_loop(0)', 'squared()']
calls += [f'small({t})' for t in range(50)]
result = {call: [outcome(comp, call), outcome(live, call)] for call in calls}
live.S.iref = 0.5
try:
    comp.S.iref = 0.5
    result['iref'] = [['value', repr(comp.S.scaled())], ['value', repr(live.S.scaled())]]
except TypeError as error:
    result['iref'] = [['TypeError', str(error)], ['value', repr(live.S.scaled())]]
fresh = type(mx_model)()
fresh.S.kref = live.S.kref = 4_000_000_000
result['kref'] = [outcome(fresh, 'squared()'), outcome(live, 'squared()')]
print(json.dumps(result))
'''

RAISES_OVERFLOW = {'local_typed()': 'S.local_typed', 'inferred(10)': 'S.inferred', 'augmented(10)': 'S.augmented',
                   'reduced()': 'S.reduced', 'negated()': 'S.negated', 'kref': 'S.squared'}


def test_flagged_integer_arithmetic_raises_or_equals_live(tmp_path):
    (tmp_path / 'model.py').write_text(MODEL, encoding='utf-8')
    (tmp_path / 'sample.py').write_text(SAMPLE, encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'cells_param_size': {'n': 30, 'k': 4}}), encoding='utf-8')
    (tmp_path / 'checks.py').write_text(CHECKS, encoding='utf-8')
    run = lambda *args: subprocess.run([sys.executable, '-B', *args], cwd=tmp_path, capture_output=True,
                                       text=True, timeout=900)
    exported = run('-c', 'from model import build; m = build(); m.export("exported"); m.close()')
    assert exported.returncode == 0, exported.stderr
    built = run('-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--spec', 'spec.py')
    assert built.returncode == 0, built.stdout + built.stderr
    source = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    # no model-wide directive; every C integer + - * of these formulas is flagged, except two left to
    # the per-formula fallback in this compiler: an element of a vector Cell's value (a Python object:
    # vector Cells are a bridge layer) and a sum over a generator (a nested scope: reductions are one too)
    assert 'overflowcheck' not in source.splitlines()[0], source.splitlines()[0]
    report = json.loads((tmp_path / 'exported_cy' / '_mx_integer_checks.json').read_text())
    assert report == {'mode': 'checked', 'overflowcheck_fallback': ['S.element', 'S.reduced']}, report
    assert len(re.findall(r'_mx_sys\._mx_flag_(add|sub|mul|neg)\(', source)) >= 10
    checked = run('checks.py')
    assert checked.returncode == 0, checked.stdout + checked.stderr
    result = json.loads(checked.stdout.strip().splitlines()[-1])
    for call, (compiled, live) in result.items():
        if call in RAISES_OVERFLOW:
            assert compiled[0] == 'OverflowError' and compiled[1].startswith(RAISES_OVERFLOW[call]), (call, compiled)
            assert live[0] == 'value', (call, live)
        elif call == 'float_argument()':        # a float argument of an integer Cell raises (not truncated)
            assert compiled[0] == 'TypeError', (call, compiled)
        elif call == 'after_loop(0)':           # unbound after a loop of no iteration, as live
            assert compiled[0] == live[0] == 'UnboundLocalError', (call, compiled, live)
        elif call == 'iref':                    # a float written to an integer Reference raises (not truncated)
            assert compiled[0] == 'TypeError' and 'S.iref' in compiled[1], (call, compiled)
        else:
            assert compiled == live, (call, compiled, live)
