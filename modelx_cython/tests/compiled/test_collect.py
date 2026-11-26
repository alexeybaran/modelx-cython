"""A Space declared with modelx's collect compiles with the collect hook (version 1).

One build checks: values equal the export and live modelx at one and four workers, bit for bit
(sums are added in domain order); other Space parameters reach the items; items that collect
creates are released and existing items are used and kept; the first failing coordinate raises
at any worker count; more than one worker over an unlocked shared Space raises naming it.
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
    for workers in ('1', '4'):
        got = out[workers]
        assert got['value'] == export, workers                   # bit for bit, domain order
        assert got['value2'] == out['export2'], workers
        assert got['total'] == live['pv']
        assert got['kept']                                       # an existing item is used and kept
        # base, the kept item and at most an item per worker whose free a thread defers: the rest released
        assert got['items'] <= 2 + int(workers), got['items']
        assert got['error'] == out['export_error'] == ['LookupError', 'no rate for 4']
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


REBIND_PROBE = textwrap.dedent('''\
    import json
    from exported import mx_model as export
    from exported_cy import mx_model as compiled

    def run(m, scale):
        m.Projection.scale = scale
        m.Projection._mx_collect_cache.clear()
        return {str(c): r.v for c, r in m.Projection.collect().individual.items()}

    print(json.dumps({'export': run(export, 1.0), 'compiled': run(compiled, 1.0),
                      'export3': run(export, 3.0), 'compiled3': run(compiled, 3.0)}))
''')


def test_a_reference_rebound_during_collect_reaches_the_later_items(tmp_path):
    """A reused item keeps the References it copied and copies them again when one is no longer the Space's: a
    formula that rebinds a Reference of the Space at coordinate 3 changes the items after it, as in live modelx,
    and the caller's rebinding between two collects reaches the next one."""
    model = mx.new_model('RebindCollect')
    try:
        proj = model.new_space('Projection', formula='lambda key: None', refs={'scale': 1.0})
        proj.new_cells('keys', 'lambda: tuple(range(1, 7))')
        proj.new_cells('poke', '''def poke():
    if key == 3:
        _model.Projection.scale = scale * 2.0
    return 0.0
''')
        proj.new_cells('v', 'lambda: poke() + key * scale')
        proj.set_collect(over={'key': proj.keys}, outputs='lambda: {"individual": {"v": v()}}')
        live = {str(c): r.v for c, r in proj.collect().individual.items()}
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    assert live == {'1': 1.0, '2': 2.0, '3': 3.0, '4': 8.0, '5': 10.0, '6': 12.0}
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nm.Projection.collect()\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, capture_output=True)
    assert '_mx_refs_held__c_Projection' in (tmp_path / 'exported_cy' / '_mx_classes.pxd').read_text()
    out = json.loads(subprocess.run([sys.executable, '-c', REBIND_PROBE], cwd=tmp_path, check=True,
                                    capture_output=True, text=True).stdout)
    assert out['compiled'] == out['export'] == live
    assert out['compiled3'] == out['export3'] and out['compiled3']['4'] == 24.0


REACH_PROBE = textwrap.dedent('''\
    import json
    from exported import mx_model as export
    from exported_cy import mx_model as compiled

    def run(m, name):
        return {str(c): r._fields for c, r in getattr(m, name).collect().individual.items()}

    print(json.dumps({name: [run(export, name), run(compiled, name)] for name in ('Static', 'Dynamic')}))
''')


def test_the_reset_skips_only_cells_collect_outputs_cannot_reach(tmp_path):
    """The reset of a reused item leaves the caches of a Cell that collect_outputs cannot reach; a Cell reached
    through ``_space`` (dynamically) makes the reset clear every Cell, so its value never comes from an earlier
    coordinate."""
    model = mx.new_model('ReachCollect')
    try:
        for name, outputs in (('Static', '{"o": odd(2), "s": stat()}'), ('Dynamic', '{"d": dyn(), "s": stat()}')):
            space = model.new_space(name, formula='lambda key: None')
            space.new_cells('keys', 'lambda: tuple(range(1, 9))')
            space.new_cells('odd', 'lambda t: key * 0.25 + t')
            space.new_cells('unused', 'lambda t: key * 3.0 + t')
            space.new_cells('stat', 'lambda: key * 2.0 + unused(1) * 0.0 if key < 0 else key * 2.0')
            space.new_cells('dyn', 'lambda: _space.odd(2)')
            space.set_collect(over={'key': space.keys}, outputs=f'lambda: {{"individual": {outputs}}}')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nm.Static.collect(); m.Dynamic.collect()\n'
        'for k in (1, 2):\n    m.Static[k].unused(1); m.Dynamic[k].unused(1)\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, capture_output=True)
    pxd = (tmp_path / 'exported_cy' / '_mx_classes.pxd').read_text()
    resets = {name: pxd.split(f'_mx_reset_root__c_{name}(')[1].split('\ncdef ')[0] for name in ('Static', 'Dynamic')}
    assert 'root._has_odd' in resets['Static'] and 'root._has_stat' in resets['Static']
    assert 'root._has_dyn' not in resets['Static']                  # reached by no output of Static
    assert 'root._has_dyn' in resets['Dynamic'] and 'root._has_unused' in resets['Dynamic']   # every Cell
    out = json.loads(subprocess.run([sys.executable, '-c', REACH_PROBE], cwd=tmp_path, check=True,
                                    capture_output=True, text=True).stdout)
    for name in ('Static', 'Dynamic'):
        assert out[name][1] == out[name][0], name
    assert out['Dynamic'][0]['3']['d'] == 2.75


SUM_PROBE = textwrap.dedent('''\
    import json, struct
    from exported import mx_model as export
    from exported_cy import mx_model as compiled

    def bits(result):
        return {k: (struct.pack('<d', v).hex() if type(v) is float else repr(v)) for k, v in result.sum._fields.items()}

    def error(call):
        try:
            call()
        except Exception as exc:
            return [type(exc).__name__, str(exc)]

    out = {'export': bits(export.Projection.collect()), 'export_bad': error(export.Bad.collect)}
    for workers in (1, 4):
        compiled._mx_collect_workers = workers
        compiled.Projection._mx_collect_cache.clear()
        compiled.Bad._mx_collect_cache.clear()
        out[workers] = [bits(compiled.Projection.collect()), error(compiled.Bad.collect)]
    print(json.dumps(out))
''')


def test_float_sums_equal_live_bit_for_bit(tmp_path):
    """Sums of floats added without the Accumulator equal live modelx bit for bit, with NaN, infinities, values
    far apart in size and a sum that mixes int and float outputs; an output of another structure raises live's
    error at its coordinate, at one worker and at four."""
    model = mx.new_model('SumCollect')
    try:
        proj = model.new_space('Projection', formula='lambda key: None')
        proj.new_cells('keys', 'lambda: tuple(range(1, 41))')
        proj.new_cells('a', '''def a():
    if key == 7:
        return float('nan')
    return key * 0.1 + 1e-17 * key
''')
        proj.new_cells('b', '''def b():
    if key == 5:
        return float('inf')
    if key == 9:
        return float('-inf')
    return 1e300 * key
''')
        proj.new_cells('c', 'lambda: (1e-310 if key % 2 else 3.0) / 7.0')
        proj.new_cells('n', 'lambda: key if key % 3 == 0 else key * 0.5')
        proj.set_collect(over={'key': proj.keys},
                         outputs='lambda: {"sum": {"a": a(), "b": b(), "c": c(), "n": n()}}')
        bad = model.new_space('Bad', formula='lambda key: None')
        bad.new_cells('keys', 'lambda: tuple(range(1, 9))')
        bad.new_cells('x', 'lambda: key * 0.5')
        bad.set_collect(over={'key': bad.keys},
                        outputs='lambda: {"sum": {"x": x(), "y": x()}} if key == 5 else {"sum": {"x": x()}}')
        live = {k: v for k, v in proj.collect().sum._fields.items()}
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nm.Projection.collect()\n'
        'for k in range(1, 9):\n    m.Bad[k].collect_outputs()\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, capture_output=True)
    out = json.loads(subprocess.run([sys.executable, '-c', SUM_PROBE], cwd=tmp_path, check=True,
                                    capture_output=True, text=True).stdout)
    import struct
    assert out['export'] == {k: struct.pack('<d', v).hex() for k, v in live.items()}
    assert out['export_bad'][0] == 'ValueError'
    for workers in ('1', '4'):
        assert out[workers][0] == out['export'], workers
        assert out[workers][1] == out['export_bad'], workers
