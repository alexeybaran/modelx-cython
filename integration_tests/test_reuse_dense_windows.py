"""D74: a hash-cached Cell is dense on the reused collect root; spec key max_dense_entries.

`surv(s, t)` has a 2-D integer window. With ``max_dense_entries`` below its size it is not inline dense
(the hash store, on every instance); on the Space whose collect reuses one root per worker the root holds
the window dense (heap storage, set by ``window()``), and the per-policy reset clears only the entries
that policy wrote. Values differ by policy, so a stale entry would change the total. The report states the
cache kind, window, entries, bytes, and why the cell is on the hash store. Compiled at one and four
workers equals the exported model; an item evaluated outside collect (hash store) equals it too.
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx

PREFIX = 'exported_cy._m_Portfolio._mx_classes._c_Projection._f_'


def _model(tmp_path, spec):
    keys = tuple(range(1, 31))
    model = mx.new_model('ReuseDense')
    try:
        portfolio = model.new_space('Portfolio')
        projection = portfolio.new_space(
            'Projection', formula='lambda key: None', refs={'keys': keys},
            collect_over={'key': keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        projection.new_cells('rate', 'lambda t: 0.01 * ((t + key) % 7)')
        projection.new_cells('surv', '''\
def surv(s, t):
    if t <= s:
        return 1.0
    return surv(s, t - 1) * (1.0 - rate(t - 1))
''')
        projection.new_cells('value', '''\
def value():
    total = 0.0
    for t in range(12):
        for s in range(0, t + 1, 1 + key % 3):
            total += surv(s, t) * (t + 1)
    return total
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'spec.py').write_text(repr(spec), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Portfolio.Projection\n'
        'p.collect()\n'
        'for key in (1, 2, 3):\n'
        '    p[key].value(); [p[key].surv(s, t) for s in range(12) for t in range(12)]\n', encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return ((tmp_path / 'exported_cy/_m_Portfolio/_mx_classes.py').read_text(encoding='utf-8'),
            json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8')))


PROBE = textwrap.dedent('''\
    import sys
    from importlib import import_module
    m = import_module(sys.argv[1]).mx_model
    p = m.Portfolio.Projection
    if sys.argv[1] == 'exported_cy':
        serial = type(m)()
        serial._mx_collect_set_workers_v2(1)
        one = serial.Portfolio.Projection.collect().sum.value
        again = serial.Portfolio.Projection.collect().sum.value
        four = p.collect().sum.value
        print(repr(one), repr(again), repr(four))
    else:
        v = p.collect().sum.value
        print(repr(v), repr(v), repr(v))
    item = p[7]
    print(repr([item.surv(s, t) for s in range(12) for t in range(12)] + [item.value()]))
    ''')


def _run(tmp_path):
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', PROBE, package], cwd=tmp_path, capture_output=True,
                             text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    return out


def test_hash_cell_is_dense_on_the_reused_root_and_values_stay(tmp_path):
    spec = {'parallel_workers': 4, 'spaces': {'Portfolio': {'spaces': {'Projection': {
        'serial_collect': 'reuse', 'parallel_collect_reuse': True, 'parallel_collect_order': 'domain',
        'cells': {'surv': {'max_dense_entries': 16}}}}}}}
    generated, report = _model(tmp_path, spec)
    assert '.NumericCache(' in generated and 'root._v_surv.window(0, ' in generated, generated
    assert 'root._v_surv.reset()' in generated
    assert '.dense_find(_mx_dense_cache, ' in generated and '.dense_put(_mx_dense_cache, ' in generated  # the field bound to a typed local
    detail = report[PREFIX + 'surv']['cache_detail']
    assert detail['kind'] == 'native-numeric-keys', detail
    assert 'exceeds max_dense_entries 16' in detail['hash_reason'], detail
    assert detail['reused_root']['kind'] == 'dense' and detail['reused_root']['bytes'] == (
        detail['reused_root']['entries'] * 13), detail
    assert not any(e['kind'] == 'hash-cache' for e in report[PREFIX + 'surv']['slow_paths'])
    out = _run(tmp_path)
    one, again, four = out['exported_cy'][0].split()
    assert one == again == out['exported'][0].split()[0], out        # reset leaves nothing stale
    # four chunk totals merged in domain order can differ from the serial sum in the last bits (config
    # docs: parallel_collect_order); every policy value is exact (one worker and the item read)
    assert abs(float(four) - float(one)) <= 1e-12 * abs(float(one)), out
    assert out['exported_cy'][1] == out['exported'][1], out


def test_reuse_dense_windows_off_keeps_the_hash_store(tmp_path):
    spec = {'spaces': {'Portfolio': {'spaces': {'Projection': {
        'serial_collect': 'reuse', 'reuse_dense_windows': False, 'max_dense_entries': 16}}}}}
    generated, report = _model(tmp_path, spec)
    assert '.window(' not in generated
    detail = report[PREFIX + 'surv']['cache_detail']
    assert detail['reused_root'] == {'kind': 'native-numeric-keys',
                                     'reason': 'reuse_dense_windows is false for the Space'}, detail
    assert any(e['kind'] == 'hash-cache' for e in report[PREFIX + 'surv']['slow_paths'])
    out = _run(tmp_path)
    assert out['exported_cy'][0].split()[0] == out['exported'][0].split()[0], out
    assert out['exported_cy'][1] == out['exported'][1], out


ITEM_PROBE = textwrap.dedent('''\
    import pickle, sys
    from importlib import import_module
    m = import_module(sys.argv[1]).mx_model
    p = m.Portfolio.Projection
    out = []
    for key in (4, 9, 17):                 # items outside collect, then after a collect
        item = p[key]
        out.append([item.surv(s, t) for s in range(12) for t in range(12)] + [item.value()])
    total = p.collect().sum.value
    fresh = p[23]
    partial = [fresh.surv(0, 5), fresh.surv(2, 7)]
    out.append(partial + [fresh.surv(s, t) for s in range(12) for t in range(12)] + [fresh.value()])
    out.append([total, p[4].value()])
    print(repr(out))
    if sys.argv[1] == 'exported_cy':
        # a model whose collect reuses a root holds a thread-local, so its items do not pickle (as before
        # D123); the item's window store itself round-trips: every entry written into the dense window
        # comes back from the pickle
        def caches(value, depth=0):
            if type(value).__name__ == 'NumericCache':
                yield value
            elif isinstance(value, tuple) and depth < 4:
                for part in value:
                    yield from caches(part, depth + 1)
        state = p[9].__reduce_ex__(2)
        found = list(caches(state))
        assert found, state
        for cache in found:
            entries = sorted(cache.__reduce__()[2])
            assert entries, 'no dense entries written'
            assert sorted(pickle.loads(pickle.dumps(cache)).__reduce__()[2]) == entries
        print('pickled', len(found))
    ''')


def test_items_hold_the_same_dense_window_and_pickle(tmp_path):
    # D123: every instance built by the shared constructor holds the window; values equal the exported model,
    # also for an item restored from a pickle with part of its window computed
    spec = {'spaces': {'Portfolio': {'spaces': {'Projection': {
        'serial_collect': 'reuse', 'cells': {'surv': {'max_dense_entries': 16}}}}}}}
    generated, _ = _model(tmp_path, spec)
    assert 'root._v_surv.window(0, ' in generated
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', ITEM_PROBE, package], cwd=tmp_path, capture_output=True,
                             text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    assert out['exported_cy'][0] == out['exported'][0], out
    assert out['exported_cy'][1].startswith('pickled '), out
