"""The compiled numeric collector keeps its results native until Python reads them."""

import subprocess
import sys
import textwrap

import modelx as mx
import pytest


collect_support = pytest.importorskip('modelx.export._mx_collect')


@pytest.mark.skipif(not hasattr(collect_support, '_native_result'), reason='export without native result support')
def test_native_records_and_reductions(tmp_path):
    model = mx.new_model('NativeBook')
    try:
        source = model.new_space('Source', formula='lambda idx, close=0, shock=-1: None',
                                 refs={'indices': (7, 2, 8, 1), 'months': (-1, 2, 6)})
        source.new_cells('value', 'lambda t: idx * 1.0 + t')
        source.new_cells('total', 'lambda: idx * 2.0')
        source.new_cells('count', 'lambda: idx')
        source.new_cells('large', 'lambda: 1 << 62')
        source.new_cells('label', 'lambda: {"kind": "even" if idx % 2 == 0 else "odd"}')
        source.new_cells('forward_months', 'lambda t: range(max(0, 4 - t))')
        source.set_collect(over={'idx': 'indices'}, outputs=(
            'lambda: {"individual": {"value": {t: value(t) for t in months}, "count": count(), "label": label()}, '
            '"sum": {"total": total(), "series": {t: value(t) for t in months}, "count": count(), "large": large(), '
            '"forward": {t: value(t) for t in forward_months(close)} if shock == -1 else {}}}'))
        use = model.new_space('Use', refs={'Source': source, 'record': None}, formula=textwrap.dedent('''\
            def factory(idx, use_collect):
                if use_collect:
                    return {'refs': {'record': Source.collect().individual[idx]}}
                return {'refs': {'record': Source[idx]}}
            '''))
        use.new_cells('read', 'lambda t: record.value(t)')
        use.new_cells('python_read', 'lambda t: getattr(record, "value")(t)')
        use.new_cells('same_record', 'lambda: record is Source.collect().individual[idx]')
        use.new_cells('bound_cell', 'lambda: callable(record.count)')
        book = model.new_space('Book', refs={'Source': source})
        book.new_cells('read_total', 'lambda: Source.collect().sum.total')
        book.new_cells('read_count', 'lambda: Source.collect().sum.count')
        book.new_cells('wide_arithmetic', 'lambda: Source.collect().sum.large // 8')
        book.new_cells('read_series', 'lambda t: Source.collect().sum.series[t]')
        book.new_cells('format_series', 'lambda t: repr(Source.collect().sum.series[t])')
        book.new_cells('read_row', 'lambda idx, t: Source.collect().individual[idx].value[t]')
        book.new_cells('python_row', 'lambda idx: Source.collect().individual[idx]')
        datum = model.new_space('Data', refs={'lengths': [2, 1]})
        datum.new_cells('axis', 'lambda: range(lengths.pop())').is_cached = False
        varying = model.new_space('Varying', formula='lambda idx: None', refs={'data': datum, 'indices': (0, 1)})
        varying.new_cells('value', 'lambda t: t + idx * 1.0')
        varying.set_collect(over={'idx': 'indices'},
                            outputs='lambda: {"individual": {"value": {t: value(t) for t in data.axis()}}}')
        opaque = model.new_space('Opaque', formula='lambda idx: None', refs={'indices': (8,), 'values': []})
        opaque.new_cells('payload', 'lambda: values')
        opaque.new_cells('finish', 'lambda: values.append(idx) or 1.0')
        opaque.new_cells('zero', 'lambda: -0.0')
        opaque.set_collect(over={'idx': 'indices'}, outputs=(
            'lambda: {"individual": {"payload": payload(), "value": finish()}, "sum": {"zero": zero()}}'))
        model.export(tmp_path / 'exported', locked_spaces=['Source', 'Data'])
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(textwrap.dedent('''\
        from exported import mx_model as m
        m.Source.collect()
        m.Source.collect(1, 0)
        m.Use[7, True].read(2)
        m.Use[7, True].python_read(2)
        m.Use[7, False].read(6)
        m.Use[7, False].bound_cell()
        m.Book.read_total()
        m.Book.read_count()
        m.Book.wide_arithmetic()
        m.Book.read_series(2)
        m.Book.format_series(2)
        m.Book.format_series(6)
        m.Book.read_row(8, 6)
        m.Book.python_row(7)
        m.Varying.collect()
        m.Opaque.collect()
        '''), encoding='utf-8')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                           cwd=tmp_path, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    generated = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    for expression in ('_mx_gather__c_Source', '.individual_at(', '.read_individual(', '.read_sum('):
        assert expression in generated
    assert '_mx_gather__c_Varying' not in generated

    child = textwrap.dedent('''\
        import pickle
        import math
        from exported_cy import mx_model
        from exported_cy._mx_collect import CollectResult
        Model = type(mx_model)
        collision = CollectResult({'_data': 42, '_mx_native': 7})
        assert collision._data == 42 and collision._mx_native == 7
        def storage(result):
            return result._CollectResult__storage
        for workers in (1, 4):
            model = Model()
            model._mx_collect_workers = workers
            result = model.Source.collect()
            assert storage(result).native is not None
            assert storage(result).views == {}
            assert model.Book.read_total() == 36.0
            assert model.Book.read_count() == 18
            assert model.Book.read_series(2) == 26.0
            assert model.Book.read_row(8, 6) == 14.0
            assert model.Use[7, True].read(2) == model.Use[7, False].read(2) == 9.0
            assert storage(result).views == {}
            assert model.Use[7, True].python_read(6) == 13.0
            assert storage(result).views == {}
            assert model.Source.collect() is result
            assert model.Book.python_row(7).value(6) == 13.0
            assert set(storage(result).views) == {'individual'}
            assert model.Use[7, True].same_record()
            assert model.Use[7, False].bound_cell()
            assert dict(result.individual[2].value) == {-1: 1.0, 2: 4.0, 6: 8.0}
            assert result.individual[2].count == 2 and result.individual[2].label['kind'] == 'even'
            assert result.sum.large == 1 << 64
            result.sum.series[2] = 26
            assert model.Book.format_series(2) == '26'
            result.sum.series[6] = 'unboxed Python value'
            assert model.Book.format_series(6) == "'unboxed Python value'"
            assert model.Book.wide_arithmetic() == 1 << 61
            assert dict(model.Source.collect(1).sum.forward) == {0: 18.0, 1: 22.0, 2: 26.0}
            assert model.Source.collect(1, 0).sum.forward == {}
            assert pickle.loads(pickle.dumps(result)).sum.total == 36.0
            result.individual[7].value[6] = 101.0
            assert model.Use[7, True].read(6) == 101.0
            assert model.Use[7, False].read(6) == 13.0
            opaque = model.Opaque.collect()
            assert storage(opaque).native is not None
            assert opaque.individual[8].payload == [8]
            assert math.copysign(1.0, opaque.sum.zero) == -1.0
        model = Model()
        result = model.Varying.collect()
        assert type(result._fields) is dict
        assert tuple(result.individual[0].value) == (0,)
        assert tuple(result.individual[1].value) == (0, 1)
        ''')
    checked = subprocess.run([sys.executable, '-c', child], cwd=tmp_path, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
