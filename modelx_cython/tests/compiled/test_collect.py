"""A Space declared with modelx's collect compiles with the collect hook (version 1).

One build checks that individual values equal the export and that sums agree within rounding
error at one and four workers. Other Space parameters reach the items, temporary items are
released, retained Direct items survive, and an invalid calculation or shared Space raises.
"""
import json
import subprocess
import sys
import textwrap

import pytest
import modelx as mx

from modelx_cython.config import CompilerPolicy

pytestmark = pytest.mark.skipif(
    not hasattr(mx.core.space.UserSpace, 'set_collect'), reason='modelx without collect')

PROBE = textwrap.dedent('''\
    import gc, json, sys
    from exported import mx_model as export
    from exported_cy import mx_model as compiled

    def flat(result):
        out = {'sum': {k: (dict(v) if isinstance(v, dict) else v) for k, v in result.sum._fields.items()}}
        out['individual'] = {str(c): r._fields for c, r in result.individual.items()}
        return out

    def error(call):
        try:
            call()
        except Exception as exc:
            return [type(exc).__name__, str(exc)]
        return None

    out = {'hook': type(compiled.Projection)._mx_collect_hook_version,
           'export': flat(export.Projection.collect()), 'export2': flat(export.Projection.collect(scen=2))}
    for workers in (1, 4):
        compiled._mx_collect_workers = workers
        compiled.Projection._mx_collect_cache.clear()
        compiled.Projection.fail_keys = ()
        kept = compiled.Projection[3]
        kept_pv = kept.pv()
        gc.disable()
        result = flat(compiled.Projection.collect())
        items = sum(type(o) is type(compiled.Projection) for o in gc.get_objects())
        gc.enable()
        out[workers] = {
            'value': result, 'value2': flat(compiled.Projection.collect(scen=2)),
            'kept': compiled.Projection[3] is kept and kept.pv() == kept_pv,
            'items': items,
            'total': compiled.Book.total()}
        del compiled.Projection[3, 1]
        compiled.Projection._mx_collect_cache.clear()
        compiled.Projection.fail_keys = (7, 4)
        out[workers]['error'] = error(compiled.Projection.collect)
        compiled.Loose._mx_collect_cache.clear()
        out[workers]['loose'] = error(compiled.Loose.collect)
        out[workers]['item'] = error(lambda: compiled.Projection[2].collect)
        out[workers]['item_has'] = hasattr(compiled.Projection[3], 'collect')
    export.Projection.fail_keys = (7, 4)
    export.Projection._mx_collect_cache.clear()
    out['export_error'] = error(export.Projection.collect)
    out['export_item'] = error(lambda: export.Projection[2].collect)
    print(json.dumps(out))
''')


def _model(tmp_path):
    model = mx.new_model('CollectModel')
    try:
        data = model.new_space('Data')
        data.new_cells('factor', 'lambda t: 1.0 + 0.01 * t')
        proj = model.new_space('Projection', formula='lambda key, scen=1: None',
                               refs={'data': data, 'fail_keys': ()})
        proj.new_cells('keys', 'lambda: tuple(range(1, 41))')
        proj.new_cells('cf', '''def cf(t):
    if key in fail_keys:
        raise LookupError(f'no rate for {key}')
    return key * 0.1 * scen * data.factor(t) / 3.0
''')
        proj.new_cells('pv', '''def pv():
    total = 0.0
    for t in range(12):
        total = total + cf(t) / (1.03 ** t)
    return total
''')
        proj.set_collect(over={'key': proj.keys}, outputs='''lambda: {
    'individual': {'pv': pv()},
    'sum': {'pv': pv(), 'cfs': {t: cf(t) for t in range(3)}}}''')
        loose_data = model.new_space('LooseData')
        loose_data.new_cells('factor', 'lambda t: 2.0 + t')
        loose = model.new_space('Loose', formula='lambda key: None', refs={'data': loose_data})
        loose.new_cells('keys', 'lambda: (1, 2, 3)')
        loose.new_cells('value', 'lambda: key * data.factor(1)')
        loose.set_collect(over={'key': loose.keys}, outputs='lambda: {"sum": {"value": value()}}')
        book = model.new_space('Book', refs={'proj': proj})
        book.new_cells('total', 'lambda: proj.collect().sum.pv')
        live = {'pv': proj.collect().sum.pv, 'pv2': proj.collect(scen=2).sum.pv,
                'cfs': dict(proj.collect().sum.cfs), 'ind': proj.collect().individual[5].pv,
                'item': not hasattr(proj[2], 'collect')}
        model.export(tmp_path / 'exported', locked_spaces=['Data'])
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'm.Book.total(); m.Projection.collect(scen=2); m.Loose.collect()\n'
        'for k in range(1, 41):\n    m.Projection[k].collect_outputs()\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, capture_output=True)
    return live


def test_compiled_collect(tmp_path):
    live = _model(tmp_path)
    run = subprocess.run([sys.executable, '-c', PROBE], cwd=tmp_path, check=True,
                         capture_output=True, text=True)
    out = json.loads(run.stdout)
    export = out['export']
    assert out['hook'] == 1
    assert export['sum']['pv'] == live['pv'] and out['export2']['sum']['pv'] == live['pv2']
    assert export['sum']['cfs'] == {str(k): v for k, v in live['cfs'].items()}
    assert export['individual']['5']['pv'] == live['ind']
    assert live['item']             # live modelx: collect is not an attribute of an ItemSpace
    assert out['export_error'] == ['LookupError', 'no rate for 4']
    for workers in ('1', '4'):
        got = out[workers]
        for actual, expected in ((got['value'], export), (got['value2'], out['export2'])):
            assert actual.keys() == expected.keys(), workers
            assert actual['individual'] == expected['individual'], workers
            assert actual['sum'].keys() == expected['sum'].keys(), workers
            assert actual['sum']['pv'] == pytest.approx(expected['sum']['pv'], rel=1e-14, abs=1e-13)
            assert actual['sum']['cfs'] == pytest.approx(expected['sum']['cfs'], rel=1e-14, abs=1e-13)
        assert got['total'] == pytest.approx(live['pv'], rel=1e-14, abs=1e-13)
        assert got['kept']                                       # an existing item is used and kept
        # base, the kept item and at most an item per worker whose free a thread defers: the rest released
        assert got['items'] <= 2 + int(workers), got['items']
        assert got['error'] in (['LookupError', 'no rate for 4'], ['LookupError', 'no rate for 7'])
        # collect belongs to the declared Space: an ItemSpace has no collect, as in live modelx
        assert got['item'] == out['export_item'] and got['item'][0] == 'AttributeError'
        assert got['item_has'] is False
        assert got['loose'] == (None if workers == '1' else [
            'ValueError', 'collect of Loose with 4 workers: the Reference data is a Space without a lock; '
            'export it in locked_spaces or use one worker'])


def test_collect_workers_key():
    assert CompilerPolicy.resolve({}).collect_workers == 1
    assert CompilerPolicy.resolve({'collect_workers': 4}).collect_workers == 4
    for bad in (0, 2.0, True):
        with pytest.raises(ValueError, match='collect_workers'):
            CompilerPolicy.resolve({'collect_workers': bad})


LIVE_PROBE = textwrap.dedent('''\
    import gc, json, sys
    from exported_cy import mx_model as compiled
    compiled.Projection.n = int(sys.argv[1])
    compiled._mx_collect_workers = int(sys.argv[2])
    gc.collect()
    gc.disable()
    before = len(gc.get_objects())
    compiled.Projection.collect()
    compiled.Projection._mx_collect_cache.clear()     # the result itself, held as live modelx holds it
    objects = gc.get_objects()
    items = sum(type(o) is type(compiled.Projection) for o in objects)
    print(json.dumps({'items': items, 'objects': len(objects) - before}))
''')

STALE_PROBE = textwrap.dedent('''\
    import json
    from exported import mx_model as export
    from exported_cy import mx_model as compiled

    def flat(result):
        return {str(c): {k: (dict(v) if isinstance(v, dict) else v) for k, v in r._fields.items()}
                for c, r in result.individual.items()}

    out = {'export': flat(export.Projection.collect())}
    for workers in (1, 4):
        compiled._mx_collect_workers = workers
        compiled.Projection._mx_collect_cache.clear()
        out[workers] = flat(compiled.Projection.collect())
    print(json.dumps(out))
''')


def test_reused_items_hold_nothing_from_the_previous_policy(tmp_path):
    """Each worker reuses one item: values of every kind of cache differ from one policy to the next, one
    Cell is reached by one policy only and one by every other policy, so a value left from an earlier
    policy on the same item would show. The live items and objects do not grow with the number of items."""
    model = mx.new_model('StaleCollect')
    try:
        proj = model.new_space('Projection', formula='lambda key: None', refs={'n': 24})
        proj.new_cells('keys', 'lambda: tuple(range(1, n + 1))')
        proj.new_cells('wide', 'lambda t: key * 0.5 + t')
        proj.new_cells('dense', 'lambda t: key * 10.0 + t')
        proj.new_cells('scalar', 'lambda: key * 1.5 + dense(3)')
        proj.new_cells('count', 'lambda: key * 7')
        proj.new_cells('keyed', 'lambda name: key * len(name)')
        proj.new_cells('only', 'lambda: key * 100.0')
        proj.new_cells('odd', 'lambda t: key * 0.25 + t')
        proj.new_cells('pv', '''def pv():
    total = 0.0
    for t in range(6):
        total = total + dense(t) / (1.03 ** t)
    return total
''')
        proj.set_collect(over={'key': proj.keys}, outputs='''lambda: {'individual': {
    'scalar': scalar(), 'count': count(), 'wide': sum(wide(t) for t in range(4000)), 'keyed': keyed('xyz'), 'pv': pv(),
    'dense': {t: dense(t) for t in range(4)},
    'only': only() if key == 5 else 0.0,
    'odd': odd(2) if key % 2 else 0.0}}''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nm.Projection.collect()\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, capture_output=True)
    assert '_mx_reset_root__c_Projection' in (tmp_path / 'exported_cy' / '_mx_classes.pxd').read_text()
    # the reused item is kept in the slot the evaluator passes to each worker, not in a threading.local
    hook = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text().split('def _mx_collect_hook(')[1]
    assert 'def _mx_outputs(_mx_spare, ' in hook and '_mx_spare[0] = _mx_fast' in hook and 'threading.local' not in hook
    run = subprocess.run([sys.executable, '-c', STALE_PROBE], cwd=tmp_path, check=True,
                         capture_output=True, text=True)
    out = json.loads(run.stdout)
    assert out['1'] == out['export'] and out['4'] == out['export']
    assert out['export']['5']['only'] == 500.0 and out['export']['3']['odd'] == 2.75
    # items are released at one worker as at four, counted directly (not by resident memory, whose unit and
    # allocator differ by platform): the live items and the objects the garbage collector tracks after collect
    # of 20 and of 3,000 coordinates, with the collector off, so a kept item or cached value would count
    def live(n, workers):
        return json.loads(subprocess.run([sys.executable, '-c', LIVE_PROBE, str(n), workers], cwd=tmp_path,
                                         check=True, capture_output=True, text=True).stdout)
    for workers in ('1', '4'):
        small, large = live(20, workers), live(3000, workers)
        # base, the item per worker and at most an item per worker whose free a thread defers
        assert large['items'] <= 2 + int(workers), (workers, small, large)
        # 3,000 coordinates: one object more per kept coordinate would add 3,000
        assert large['objects'] - small['objects'] < 300, (workers, small, large)


def test_a_declaration_without_reset_rule_stops_the_build():
    from modelx_cython import collect
    block = 'cdef class _c_P(_mx_sys.BaseSpace):\n    cdef RateTable _v_rate\n'
    with pytest.raises(ValueError, match=r"P: collect cannot reset .*_v_rate.* \(Cell rate\)"):
        collect.reset_function('_c_P', block, {'_c_P': None}, set())
