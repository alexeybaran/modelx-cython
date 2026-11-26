"""Mandatory integration contract, separate from upstream's dependency matrix.

On a model that builds on upstream modelx: inherited References, ItemSpaces and their
identity, no live dependency invalidation, an uncached Cell, exceptions passed through, locked
Spaces, a read-only vector value, totals over the items, and a Cell of an ItemSpace reading a
shared Space through a Reference.
"""
import json
import os
import subprocess
import sys
import textwrap

import modelx as mx


def test_exported_contract_on_the_direct_route(tmp_path):
    model = mx.new_model('CompiledContract')
    try:
        base = model.new_space('Base', refs={'offset': 10.0})
        base.new_cells('value', 'lambda t=0.5: offset + t')
        derived = model.new_space('Derived', bases=base)
        derived.offset = 30.0
        rows = {'a': {'value': 7.0}, 'b': {'value': 2.5}}
        keys = tuple(rows)
        policies = model.new_space('Policies', formula='lambda key: None',
                                   refs={'rows': rows, 'keys': keys, 'adjust': float})
        policies.new_cells('row', 'lambda: rows[key]')
        policies.new_cells('total', 'lambda: adjust(row()["value"])')
        policies.new_cells('uncached', 'lambda: adjust(1.0)').is_cached = False
        shared = model.new_space('Shared', refs={'adjust': float})
        shared.new_cells('value', 'lambda: adjust(10.0)')
        # an ItemSpace formula returning References is development modelx's: the Cell computes it
        nested = model.new_space('Nested', refs={'Shared': shared, 'keys': (7, 3)},
                                 formula='lambda date, key: None')
        nested.new_cells('state', 'lambda: [Shared.value() + date + key]')
        caller = model.new_space('Caller', refs={'Nested': nested, 'keys': (7, 3)})
        caller.new_cells('total', 'lambda date: sum(Nested(date, key).state()[0] for key in keys)')
        model.export(tmp_path / 'exported', locked_spaces=['Shared', 'Caller'])
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'm.Base.value(); m.Derived.value()\n'
        'for key in m.Policies.keys: m.Policies[key].total()\n'
        'm.Policies["a"].uncached(); m.Nested(2, 7).state(); m.Nested(2, 3).state(); m.Caller.total(2)\n',
        encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, env=dict(os.environ), check=True)
    admission = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    assert admission['exported_cy._mx_classes._c_Base._f_value']['return_type'] == 'double'
    assert admission['exported_cy._mx_classes._c_Derived._f_value']['return_type'] == 'double'
    assert admission['exported_cy._mx_classes._c_Policies._f_uncached']['cache'] == 'none'
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        gil = getattr(sys, '_is_gil_enabled', lambda: True)
        before = gil()
        m = import_module(sys.argv[1]).mx_model
        assert gil() == before
        for space, initial in ((m.Base, 10.0), (m.Derived, 30.0)):
            assert space.value(t=0.5) == initial + 0.5
            space.offset = 99.0
            assert space.value(0.5) == initial + 0.5
            assert space.value(1.5) == 100.5
        p = m.Policies['a']
        assert p is m.Policies['a']
        assert p.row() is p.row()
        assert p.total() == 7.0
        p.row()['value'] = 25.0
        assert p.row()['value'] == 25.0
        assert p.total() == 7.0  # No live dependency invalidation.
        m.Policies.rows['new'] = {'value': 5.0}
        assert m.Policies['new'].total() == 5.0
        calls = []
        p.adjust = lambda v: calls.append(v) or v + 1.0
        assert p.uncached() == p.uncached() == 2.0
        assert calls == [1.0, 1.0]
        del m.Policies['a']
        assert m.Policies['a'] is not p
        assert m.Policies['a'].total() == 25.0
        b = m.Policies['b']
        failure = LookupError('retry')
        def fail(value):
            raise failure
        b.adjust = fail
        try:
            b.total()
        except LookupError as error:
            assert error is failure
        else:
            raise AssertionError('missing exception')
        b.adjust = lambda v: v + 2.0
        assert b.total() == 4.5
        # the items as they now are (b's own adjust)
        assert sum(m.Policies[key].total() for key in m.Policies.keys) == 25.0 + 4.5
        calls = []
        m.Shared.adjust = lambda v: calls.append(v) or v
        assert [m.Nested(2, key).state() for key in (7, 3)] == [[19.0], [15.0]]
        # A compiled vector cell value is read-only; live modelx lets a caller
        # mutate the cached list. Callers copy first and the cached value stays as it was.
        state = list(m.Nested(2, 7).state())
        state[0] = -1.0
        assert state == [-1.0] and m.Nested(2, 7).state() == [19.0]
        assert m.Caller.total(2) == 34.0
        assert calls == [10.0]
        assert gil() == before
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package],
                       cwd=tmp_path, env=dict(os.environ), check=True, timeout=90)
