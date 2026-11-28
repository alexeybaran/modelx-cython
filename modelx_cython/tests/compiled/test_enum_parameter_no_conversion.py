"""An enum parameter is converted to its member only where its class shows.

A C-integer parameter traced as one IntEnum class once became its member at every object use (a returned
or stored tuple, a list, a dict, a comparison, an argument of sorted), a call per use on every evaluation.
A formula that never shows the class must compile as if the parameter had no enum class: no conversion
(``enum_member``) in its generated function. A ``.name`` use still converts, and every value equals live's (``==``: a parameter left a C integer is the
integer where live has the member; its class shows nowhere in such a formula).
Found when this test was written: such a compiled Cell returns the integer where live returns the enum
member (``(0, 1)`` against ``(<Leg.CALL: 0>, 1)``), equal by ``==`` but not by type or ``repr``.
"""
import json
import os
import re
import subprocess
import sys
import textwrap

import modelx as mx

ENUMS = textwrap.dedent('''\
    import enum

    class Leg(enum.IntEnum):
        CALL = 0
        PUT = 1
        BOTH = 2
    ''')

CELLS = {
    'plain': 'def plain(leg):\n    pair = (leg, 1)\n    held = [leg]\n    keyed = {leg: 1.0}\n'
             '    return pair, held, keyed, leg == Leg.PUT, sorted([leg, Leg.CALL])\n',
    'named': 'lambda leg: leg.name',
}
CALLS = ['p.plain(p.Leg.CALL)', 'p.plain(p.Leg.PUT)', 'p.named(p.Leg.PUT)', 'p.named(p.Leg.BOTH)']
PROBE = textwrap.dedent('''\
    import json
    from exported_cy import mx_model as m
    p = m.P
    {norm}
    print(json.dumps([repr(norm(v)) for v in ({calls})]))
    ''')
NORM = ('norm = lambda v: (type(v)(norm(x) for x in v) if isinstance(v, (tuple, list)) else '
        '{norm(k): norm(x) for k, x in v.items()} if isinstance(v, dict) else '
        'int(v) if isinstance(v, int) and not isinstance(v, bool) else v)')


def _function(source, name):
    """The text of the generated formula function of Cell ``name``."""
    found = re.search(rf'\n( *)def _f_{name}\(.*?(?=\n\1(?:def |@)|\n\S|\Z)', source, re.S)
    assert found, f'no formula function of {name}'
    return found.group(0)


def test_an_enum_parameter_is_converted_only_where_its_class_shows(tmp_path, monkeypatch):
    (tmp_path / 'probe_enum_param.py').write_text(ENUMS, encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    import probe_enum_param as e
    model = mx.new_model('EnumParamUses')
    try:
        s = model.new_space('P', refs={'Leg': e.Leg})
        for name, formula in CELLS.items():
            s.new_cells(name, formula)
        p = s
        scope = {'p': p}
        exec(NORM, scope)
        live = [repr(scope['norm'](eval(c, scope))) for c in CALLS]
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\np = m.P\n' + ''.join(f'{c}\n' for c in CALLS), encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, capture_output=True, env=env)
    source = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    assert 'enum_member(' not in _function(source, 'plain'), _function(source, 'plain')
    assert 'enum_member(' in _function(source, 'named'), 'the .name use is no longer converted: the check is void'
    out = subprocess.run([sys.executable, '-c', PROBE.format(calls=', '.join(CALLS), norm=NORM)], cwd=tmp_path, env=env,
                         check=True, capture_output=True, text=True).stdout
    assert json.loads(out) == live
