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
        out[workers]['item'] = error(compiled.Projection[2].collect)
    export.Projection.fail_keys = (7, 4)
    export.Projection._mx_collect_cache.clear()
    out['export_error'] = error(export.Projection.collect)
    out['export_item'] = error(export.Projection[2].collect)
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
        # collect belongs to the declared Space: on an ItemSpace the export and the compiled class raise
        # TypeError (live modelx, which defines collect on the declared Space only, raises AttributeError)
        assert got['item'][0] == out['export_item'][0] == 'TypeError' and got['item'] == out['export_item']
        assert got['loose'] == (None if workers == '1' else [
            'ValueError', 'collect of Loose with 4 workers: the Reference data is a Space without a lock; '
            'export it in locked_spaces or use one worker'])


def test_collect_workers_key():
    assert CompilerPolicy.resolve({}).collect_workers == 1
    assert CompilerPolicy.resolve({'collect_workers': 4}).collect_workers == 4
    for bad in (0, 2.0, True):
        with pytest.raises(ValueError, match='collect_workers'):
            CompilerPolicy.resolve({'collect_workers': bad})
