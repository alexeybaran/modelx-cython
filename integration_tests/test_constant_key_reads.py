"""D79 (diag-vec): table reads whose key parts are build-time constants.

(1) An IntEnum member admitted at build (enum definition) used as a table key part is its module
constant (``_mx_enum_const_N``), not an attribute read of the enum class per call.
(2) A read through a held handle (D71) whose key parts are all constants holds its row next to the
handle (``held_row_n``, as D75's column): resolved on the first read, a missing key raises KeyError at
the read as before, reset with the handle on the reused collect root.
Results must not change: compiled at one and four workers equals the exported model.
"""
from http import HTTPStatus
import subprocess
import sys
import textwrap

import modelx as mx
from modelx_helpers.immutable_inputs import Table
from modelx_helpers.typed_inputs import EnumDefinitions


def _model(tmp_path):
    keys = tuple(range(1, 41))
    model = mx.new_model('ConstKeys')
    try:
        portfolio = model.new_space('Portfolio')
        data = portfolio.new_space('Inputs', refs={
            'params': Table([200, 201, 202], [(0.5, 2.0), (0.25, 3.0), (0.125, 5.0)], ['value', 'load']),
            'grid': Table([(200, 1), (200, 2), (201, 1), (201, 2)],
                          [(1.5,), (2.5,), (3.5,), (4.5,)], ['f'])})
        data.new_cells('basis', 'def basis() -> "modelx_helpers.immutable_inputs.Table":\n    return params\n')
        data.new_cells('cube', 'def cube() -> "modelx_helpers.immutable_inputs.Table":\n    return grid\n')
        projection = portfolio.new_space(
            'Projection', formula='lambda key: None', refs={'market': data, 'keys': keys, 'P': HTTPStatus},
            collect_over={'key': keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        projection.enum_definitions = EnumDefinitions(('P',))
        projection.new_cells('rate', textwrap.dedent('''\
            def rate(t):
                mix = market.basis().at[P.OK, "value"]
                load = market.basis().loc[P.CREATED, "load"]
                g = market.cube().at[(P.ACCEPTED, 2), "f"] if False else market.cube().at[(P.CREATED, 2), "f"]
                return mix * (t + key) + load * 0.001 * t + g
            '''))
        projection.new_cells('value', textwrap.dedent('''\
            def value():
                total = 0.0
                for t in range(24):
                    total += rate(t)
                return total
            '''))
        projection.new_cells('rare', textwrap.dedent('''\
            def rare(t):
                if t > 20:
                    return market.basis().at[P.NOT_FOUND, "value"]
                return market.basis().at[P.ACCEPTED, "value"] * t
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    spec = {'parallel_workers': 4, 'spaces': {'Portfolio': {'spaces': {'Projection': {
        'serial_collect': 'reuse', 'parallel_collect_reuse': True, 'parallel_collect_order': 'domain'}}}}}
    (tmp_path / 'spec.py').write_text(repr(spec), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Portfolio.Projection\n'
        'p.collect()\n'
        'for key in (1, 2, 3):\n'
        '    [p[key].rate(t) for t in range(24)]; p[key].value(); [p[key].rare(t) for t in range(3)]\n',
        encoding='utf-8')


def test_constant_key_reads_use_enum_constants_and_held_rows(tmp_path):
    _model(tmp_path)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    generated = (tmp_path / 'exported_cy/_m_Portfolio/_mx_classes.py').read_text(encoding='utf-8')
    # (1) members are constants, (2) constant keys are held rows, reset with the handle
    assert '_mx_definition_P.OK' not in generated and '_mx_definition_P.CREATED' not in generated, generated
    assert '.held_row_1(_mx_cy.address(self._mx_tr_' in generated, generated
    assert '.held_row_2(_mx_cy.address(self._mx_tr_' in generated, generated
    assert 'root._mx_tr_0 = 0' in generated and 'root._mx_th_0 = None' in generated, generated
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        m = import_module(sys.argv[1]).mx_model
        p = m.Portfolio.Projection
        if sys.argv[1] == 'exported_cy':
            serial = type(m)()
            serial._mx_collect_set_workers_v2(1)
            print(repr(serial.Portfolio.Projection.collect().sum.value), repr(p.collect().sum.value))
        else:
            print(repr(p.collect().sum.value), repr(p.collect().sum.value))
        item = p[7]
        for attempt in range(2):                # the second read finds the slot still unresolved
            try:
                item.rare(22)
            except KeyError as error:
                # the key as an int: a C-integer key part has raised KeyError(404) since the typed reads
                # (unchanged here); the exported model raises KeyError(<P.NOT_FOUND: 404>), an equal key
                print('KeyError', int(error.args[0]))
            else:
                print('no error')
        print(repr([item.rare(t) for t in range(5)]))
        values = [item.rate(t) for t in range(24)] + [item.value()]
        assert [item.rate(t) for t in range(24)] + [item.value()] == values
        print(repr(values))
        ''')
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    one, four = out['exported_cy'][0].split()
    assert one == out['exported'][0].split()[0], out
    assert float(four) == float(one), out
    assert out['exported_cy'][1:] == out['exported'][1:], out
