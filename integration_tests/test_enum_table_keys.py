"""IntEnum members as keys of prepared table reads: compiled equals live, every read native.

Members become integer literals before the table reads are lowered, so a key part that is a member written in
place, an enum parameter, a Cell returning members or a Reference holding a member is a C integer: a scalar read
(``T.loc[(m, k), "c"]``, one level or two), a row and its fields, a view of a prefix keyed by a member, membership
and the extreme of a prefix. The tables hold the members' integers (as read from a file) or the members
themselves. The collect half is ``test_enum_table_keys_collect``.
"""
import math
import os
import subprocess
import sys
import textwrap

from forms_probe import native_reads

ENUMS = 'from enum import IntEnum\n\n\nclass Dec(IntEnum):\n    DEATH = 1\n    LAPSE = 2\n    MAT = 4\n'

CELLS = {
    'dur': ('t', 'return t // 3'),
    'pick': ('t', 'return Dec.DEATH if t % 2 else Dec.LAPSE'),
    'lit': ('t', 'return data.ints().loc[(Dec.LAPSE, min(dur(t), 4)), "q"]'),
    'lit_m': ('t', 'return data.members().loc[(Dec.MAT, min(dur(t), 4)), "q"]'),
    'by_par': ('t, d', 'return data.members().loc[(d, min(dur(t), 4)), "q"] + data.ints().loc[(d, 0), "q"]'),
    'by_cell': ('t', 'return data.ints().loc[(pick(t), min(dur(t), 4)), "q"]'),
    'by_ref': ('t', 'return data.members().loc[(basis, min(dur(t), 4)), "q"]'),
    'one_level': ('t', 'return data.flat().loc[pick(t), "w"] * t'),
    'row': ('t', 'r = data.flat().loc[Dec.MAT]\nreturn r["w"] + r["v"] * t'),
    'row_par': ('t, d', 'return data.flat().loc[d]["v"] * t'),
    'view': ('t', 'v = data.members().loc[Dec.DEATH]\nreturn v.loc[min(dur(t), 4), "q"]'),
    'view_par': ('t, d', 'v = data.ints().loc[d]\nreturn v.loc[min(dur(t), 4), "q"] + v.index.max()'),
    'has': ('t', 'return 1.0 if (pick(t), t) in data.ints().index else 0.0'),
    'highest': ('t, d', 'return float(data.members().loc[d].index.max() + t)'),
}
NATIVE = {
    'lit': ['f64_c'], 'lit_m': ['f64_c'], 'by_par': ['f64_c', 'f64_c'], 'by_cell': ['f64_c'], 'by_ref': ['f64_c'],
    'one_level': ['f64_1'], 'row': ['f64_row', 'f64_row'], 'row_par': ['f64_row'],
    'view': ['lcode', 'view_start', 'f64_c'], 'view_par': ['kcode', 'view_start', 'f64_c', 'label_ll'],
    'has': ['contains_c'], 'highest': ['label_ll'],
}

MODEL = textwrap.dedent('''\
    import modelx as mx
    import pandas as pd
    from table_enums import Dec

    CELLS = {cells!r}
    INTS = ('pd.DataFrame({{"q": [0.001 * d * (1 + k) for d in (1, 2, 4) for k in range(5)]}}, '
            'index=pd.MultiIndex.from_tuples([(d, k) for d in (1, 2, 4) for k in range(5)], names=["dec", "dur"]))')
    MEMBERS = ('pd.DataFrame({{"q": [0.002 * int(d) * (1 + k) for d in Dec for k in range(5)]}}, '
               'index=pd.MultiIndex.from_tuples([(d, k) for d in Dec for k in range(5)], names=["dec", "dur"]))')
    FLAT = 'pd.DataFrame({{"w": [1.5, 2.5, 3.5], "v": [10.0, 20.0, 40.0]}}, index=pd.Index(list(Dec), name="dec"))'


    def build(collect=False):
        m = mx.new_model('EnumKeys')
        data = m.new_space('Data', refs={{'pd': pd, 'Dec': Dec}})
        for name, src in (('ints', INTS), ('members', MEMBERS), ('flat', FLAT)):
            data.new_cells(name, f'def {{name}}():\\n    return {{src}}\\n')
        p = m.new_space('P', formula='lambda policy_id: None', refs={{'Dec': Dec, 'basis': Dec.MAT, 'data': data}})
        for name, (params, body) in CELLS.items():
            p.new_cells(name, f'def {{name}}({{params}}):\\n' + ''.join('    ' + l + '\\n' for l in body.splitlines()))
        p.new_cells('total', 'def total():\\n    return sum(' + ' + '.join(
            f'{{n}}(t, Dec.MAT)' if ',' in a else f'{{n}}(t)' for n, (a, _) in CELLS.items() if n not in ('dur', 'pick'))
            + ' for t in range(12))\\n')
        if collect:
            p.set_collect(over={{'policy_id': (1, 2, 3)}}, outputs='lambda: {{"sum": {{"total": total()}}}}')
        return m


    def calls(space):
        for name, (params, _) in CELLS.items():
            if ',' in params:
                yield from ((name, (t, d)) for t in range(12) for d in space.Dec)
            else:
                yield from ((name, (t,)) for t in range(12))
''')

CHECK = textwrap.dedent('''\
    import math, sys
    from importlib import import_module
    from enum_keys_model import build, calls
    compiled, live = import_module('exported_cy').mx_model.P, build().P
    bad = []
    for i in (1, 2, 3):
        for name, args in calls(live[i]):
            want = getattr(live[i], name)(*args)
            got = getattr(compiled[i], name)(*args)
            if not (got == want or math.isclose(got, want, rel_tol=1e-10, abs_tol=1e-7)):
                bad.append((i, name, args, got, want))
        assert math.isclose(compiled[i].total(), live[i].total(), rel_tol=1e-10)
    print(bad if bad else 'equal')
''')


def build(tmp_path, collect=False):
    (tmp_path / 'table_enums.py').write_text(ENUMS, encoding='utf-8')
    (tmp_path / 'enum_keys_model.py').write_text(MODEL.format(cells=CELLS), encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    run = lambda *args: subprocess.run([sys.executable, *args], cwd=tmp_path, env=env, capture_output=True, text=True)
    done = run('-c', f'from enum_keys_model import build\nm = build({collect})\nm.export("exported")\nm.close()\n')
    assert done.returncode == 0, done.stderr[-3000:]
    sample = ('from exported import mx_model as m\nm.P.collect()\n' if collect else
              'from exported import mx_model as m\nfrom enum_keys_model import calls\nfor i in (1, 2):\n'
              '    for name, args in calls(m.P[i]):\n        getattr(m.P[i], name)(*args)\n    m.P[i].total()\n')
    (tmp_path / 'sample.py').write_text(sample, encoding='utf-8')
    done = run('-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec')
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return env, run


def test_member_keys_are_native_and_equal_live(tmp_path):
    env, run = build(tmp_path)
    source = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    for cell, reads in NATIVE.items():
        assert native_reads(source, cell) == reads, (cell, native_reads(source, cell))
    assert 'Dec.' not in source.split('class _c_P', 1)[1].split('def _mx_assign_refs', 1)[0]
    (tmp_path / 'check.py').write_text(CHECK, encoding='utf-8')
    done = run('check.py')
    assert done.returncode == 0 and done.stdout.strip() == 'equal', done.stdout[-3000:] + done.stderr[-3000:]
