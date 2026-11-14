"""M1 (W1.1): table-read admission keys, the per-step warning, the spec keys and the
NativeTable index forwarding, on the round-2 helper (no typed handles yet)."""

import json
import subprocess
import sys

import modelx as mx
from modelx_helpers.immutable_inputs import Table
import pytest

PREFIX = 'exported_cy._m_Portfolio._mx_classes._c_Projection._f_'
INPUTS = 'exported_cy._m_Portfolio._mx_classes._c_MarketInputs._f_'
# Round-2's helper index has no first(): the forwarding test adds one, in every process.
PATCH = ('from modelx_helpers import immutable_inputs as _ii\n'
         '_ii._TableIndex.first = lambda self: min(self._table._keys)\n')


def _model(tmp_path):
    model = mx.new_model('Market')
    try:
        portfolio = model.new_space('Portfolio')
        source = portfolio.new_space('MarketInputs', refs={'rates': Table([1, 2, 3], [(0.05,), (0.06,), (0.07,)], ['rate'])})
        source.new_cells('curve', '''\
def curve() -> "modelx_helpers.immutable_inputs.Table":
    return rates
''')
        source.new_cells('upper', '''\
def upper(t):
    \"\"\"Round-2 clamp shape (four statements).\"\"\"
    table = curve()
    t = min(t, table.index.max_int())
    return table.lookup_float(t, "rate")
''')
        projection = portfolio.new_space('Projection', refs={'market': source})
        projection.new_cells('nested', '''\
def nested(t):
    check = lambda k: k in market.curve().index
    return check(t)
''')
        projection.new_cells('captured', '''\
def captured(t, k=2):
    check = lambda: k in market.curve().index
    return check() and t > 0
''')
        projection.new_cells('listed', 'lambda t: t in market.curve().index')
        projection.new_cells('count', '''\
def count():
    table = market.curve()
    return len(table.index)
''')
        projection.new_cells('seen', '''\
def seen(t):
    return sum(1 for k in range(t) if k in market.curve().index)
''')
        projection.new_cells('first', '''\
def first(t):
    table = market.curve()
    return table.index.first() + t
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        PATCH + 'from exported import mx_model\n'
        'p = mx_model.Portfolio.Projection\n'
        'assert p.listed(1) and not p.listed(4) and p.count() == 3 and p.seen(4) == 3\n'
        'assert p.first(1) == 2 and p.nested(2) and not p.nested(0) and p.captured(1)\n'
        'u = mx_model.Portfolio.MarketInputs.upper\n'
        'assert [u(t) for t in range(1, 6)] == [0.05, 0.06, 0.07, 0.07, 0.07]\n', encoding='utf-8')


def _translate(tmp_path, spec=None, translate_only=True):
    args = [sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py']
    if spec is None:
        args.append('--no-spec')
    else:
        (tmp_path / 'spec.py').write_text(repr(spec), encoding='utf-8')
        args += ['--spec', 'spec.py']
    if translate_only:
        args.append('--translate-only')
    done = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    return report, done.stdout + done.stderr


def test_untyped_reads_are_reported_and_per_step_ones_warned(tmp_path):
    _model(tmp_path)
    report, output = _translate(tmp_path)
    listed = report[PREFIX + 'listed']    # typed_table_reads is on by default (D45)
    assert listed['table_reads_typed'] == 1 and listed['table_reads_untyped'] == []
    assert listed['table_reads_keys'] == ['key in index, 1 part: int (parameter t is long long)']
    count = report[PREFIX + 'count']      # len(T.index) reads the handle's row count (W1.9)
    assert count['table_reads_typed'] == 1 and count['table_reads_untyped'] == []
    seen = report[PREFIX + 'seen']       # inside a generator, lowered
    assert seen['table_reads_typed'] == 1 and seen['table_reads_untyped'] == []
    first = report[PREFIX + 'first']['table_reads_untyped']
    assert [e['read'] for e in first] == ['table.index.first()']
    nested = report[PREFIX + 'nested']['table_reads_untyped']        # inside a closure
    assert [(e['read'], e['per_step']) for e in nested] == [('self.market.curve().index', True)]
    captured = report[PREFIX + 'captured']['table_reads_untyped']    # formula kept as Python
    assert [(e['read'], e['per_step']) for e in captured] == [('self.market.curve().index', True)]
    assert captured[0]['reason'].startswith('formula kept as a Python method (captured default)')
    upper = report[INPUTS + 'upper']
    assert 'modelx_cython.runtime._native_inputs.read_clamped_table_float' in upper['native_numeric_helpers']
    assert [e['read'] for e in upper['table_reads_untyped']] == ['table.index.max_int()']
    assert upper['table_reads_untyped'][0]['reason'] == (
        'index.max_int(...) is not in the typed vocabulary; runs as a Python call'), upper
    warned = [line for line in output.splitlines() if 'per-step table read is not typed:' in line]
    assert len(warned) == 4 and not any('_f_count' in line for line in warned), output
    assert any(f'{PREFIX}first: per-step table read is not typed: table.index.first() '
               '(formula line 3): index.first(...) is not in the typed vocabulary; runs as a Python call'
               in line for line in warned), output
    # an unrelated cell carries the keys too, empty
    assert report[INPUTS + 'curve']['table_reads_typed'] == 0
    assert report[INPUTS + 'curve']['table_reads_keys'] == []
    classes = (tmp_path / 'exported_cy/_m_Portfolio/_mx_classes.py').read_text(encoding='utf-8')
    assert classes.startswith('# cython: infer_types=True, overflowcheck=True\n')    # default on (W1.5, D57)


def test_spec_keys_infer_types_and_table_recognisers(tmp_path):
    _model(tmp_path)
    report, _ = _translate(tmp_path, {'infer_types': True, 'table_recognisers': False})
    classes = (tmp_path / 'exported_cy/_m_Portfolio/_mx_classes.py').read_text(encoding='utf-8')
    assert classes.startswith('# cython: infer_types=True, overflowcheck=True\n'), classes[:200]
    upper = report[INPUTS + 'upper']     # the round-2 clamp recogniser is off
    assert not any('clamped' in name for name in upper['native_numeric_helpers']), upper
    assert upper['table_reads_untyped'][0]['reason'] == (
        'index.max_int(...) is not in the typed vocabulary; runs as a Python call')
    assert 'read_clamped_table_float' not in classes


def test_table_recognisers_off_builds_and_matches(tmp_path):
    _model(tmp_path)
    _translate(tmp_path, {'table_recognisers': False}, translate_only=False)
    check = subprocess.run([sys.executable, '-c', PATCH + '''
from exported_cy import mx_model
u = mx_model.Portfolio.MarketInputs.upper
assert [u(t) for t in range(1, 6)] == [0.05, 0.06, 0.07, 0.07, 0.07]
'''], cwd=tmp_path, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr


@pytest.mark.parametrize('value', ['yes', 1])
def test_spec_keys_must_be_booleans(value):
    from modelx_cython.config import CompilerPolicy
    for name in ('infer_types', 'table_recognisers'):
        with pytest.raises(ValueError, match=name):
            CompilerPolicy.resolve({name: value})
    policy = CompilerPolicy.resolve({})
    assert policy.table_recognisers is True and policy.infer_types is True


def test_native_table_index_forwards_public_names(tmp_path):
    _model(tmp_path)
    _translate(tmp_path, translate_only=False)
    check = subprocess.run([sys.executable, '-c', PATCH + '''
from exported_cy import mx_model
import exported_cy._mx_classes
p = mx_model.Portfolio.Projection
assert p.first(1) == 2 and p.first(3) == 4
assert p.listed(3) and not p.listed(0) and p.count() == 3 and p.seen(4) == 3
table = mx_model.Portfolio.MarketInputs.curve()
assert type(table).__name__ == "NativeTable", type(table)
assert table.index.first() == 1
try:
    table.index._table
except AttributeError:
    pass
else:
    raise AssertionError("private names are not forwarded")
'''], cwd=tmp_path, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
