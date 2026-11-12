"""A cached clamp can keep its native input and an arbitrary Python fallback key."""

import json
import subprocess
import sys
import textwrap

import modelx as mx

from modelx_helpers import pdn
from modelx_helpers.immutable_inputs import Table


def test_cached_table_clamp_rebinding(tmp_path):
    model = mx.new_model('ClampBindings')
    try:
        space = model.new_space('Rates', refs={
            'source': Table((0, 1), ((2.0,), (5.0,)), ('rate',)),
            'pdn': pdn,
            'convert': float,
        })
        space.new_cells('schedule', '''\
def schedule() -> "modelx_helpers.immutable_inputs.Table":
    return source
''')
        space.new_cells('upper', '''\
def upper(t):
    """Read a rate with an upper clamp."""
    table = schedule()
    t = min(t, table.index.max_int())
    return table.lookup_float(t, "rate")
''')
        space.new_cells('bounded', '''\
def bounded(t):
    t = int(t)
    table = schedule()
    t = min(max(t, int(pdn.first_key(table))), table.index.max_int())
    return table.lookup_float(t, "rate")
''')
        space.new_cells('converted', '''\
def converted(t):
    t = int(t)
    table = schedule()
    key = min(t, table.index.max_int())
    return table.lookup_float(key, "rate")
''')
        space.new_cells('shadowed', '''\
def shadowed(t):
    int = convert
    t = int(t)
    table = schedule()
    key = min(t, table.index.max_int())
    return table.lookup_float(key, "rate")
''')
        space.new_cells('reassigned', '''\
def reassigned(t):
    t = int(t)
    t = t / 2
    table = schedule()
    key = min(t, table.index.max_int())
    return table.lookup_float(key, "rate")
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        's = mx_model.Rates\n'
        's.upper(2); s.bounded(2); s.converted(2); s.shadowed(2); s.reassigned(2)\n',
        encoding='utf-8',
    )
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--no-spec'],
                   cwd=tmp_path, check=True)
    generated = tmp_path / 'exported_cy'
    report = json.loads((generated / '_mx_admission.json').read_text(encoding='utf-8'))
    code = (generated / '_mx_classes.py').read_text(encoding='utf-8')
    provider = 'modelx_cython.runtime._native_inputs.read_clamped_table_float'
    for name in ('upper', 'bounded', 'converted'):
        assert provider in report[f'exported_cy._mx_classes._c_Rates._f_{name}']['native_numeric_helpers']
        assert f'def _f_{name}(self, t: _mx_cy.longlong)' in code
    for name in ('shadowed', 'reassigned'):
        assert provider not in report[f'exported_cy._mx_classes._c_Rates._f_{name}']['native_numeric_helpers']
        assert f'def _f_{name}(self, t: object)' in code
    assert '_mx_clamp_fallback_key: object' in code

    probe = textwrap.dedent('''\
        from importlib import import_module
        import sys
        from modelx_helpers.immutable_inputs import Table

        model = import_module(sys.argv[1]).mx_model
        cells = ('upper', 'bounded', 'converted')
        for name in cells + ('shadowed', 'reassigned'):
            assert getattr(model.Rates, name)(2) == 5.0

        wide_key = -(1 << 80)
        seen = []
        class WideTable(Table):
            def lookup_float(self, key, column):
                seen.append((key, type(key)))
                return super().lookup_float(key, column)

        fresh = type(model)()
        fresh.Rates.source = WideTable((wide_key,), ((7.0,),), ('rate',))
        for name in cells:
            assert getattr(fresh.Rates, name)(2) == 7.0
        assert seen == [(wide_key, int)] * len(cells)
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True)
