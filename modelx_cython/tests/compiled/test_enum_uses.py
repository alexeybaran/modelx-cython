"""IntEnum members and values in every kind of use (enums.plan): compiled equals the exported model.

The cases of two reviews of the enum pass and two shapes of real models, in one small model: a member or an
enum-typed value where its class shows (``repr``, ``str``, ``format``, ``isinstance``, a container), read by
``.name``, tested by identity, passed to a helper function, appended to a list, used as the key of an item Space
or of a table by its name, in a comprehension, from an untraced Cell, across Spaces, and of a class with its own
operators. Two probes raise ``TypeError`` where the model gives a value, because a member of another class
reaches a parameter or result that is compiled as an integer of its traced class and converted back.
"""
import os
import subprocess
import sys
import textwrap

import modelx as mx

ENUMS = textwrap.dedent('''\
    import enum

    class Leg(enum.IntEnum):
        CALL = 0
        PUT = 1

    class Kind(enum.IntEnum):
        X = 0
        Y = 1

    class Timing(enum.IntEnum):
        BEF = 0
        AFT = 1
        END = 2

    class Pretty(enum.IntEnum):
        M = 0
        F = 1

        def __str__(self):
            return self.name

        def __format__(self, spec):
            return self.name

    class Ord(enum.IntEnum):
        LOW = 2
        HIGH = 1

        def __lt__(self, other):
            return int(self) > int(other)

    def label(leg):
        return leg.name
    ''')

CELLS = {
    'comp_name': 'lambda: [x.name for x in (Leg.CALL, Leg.PUT)]',
    'comp_repr': 'lambda: [repr(x) for x in (Leg.CALL, Leg.PUT)]',
    'comp_isinst': 'lambda: [isinstance(x, Leg) for x in (Leg.CALL, Leg.PUT)]',
    'list_loop': 'def list_loop():\n    out = []\n    for x in [Leg.CALL, Leg.PUT]:\n        out.append(x.name)\n    return out\n',
    'nm': 'lambda k: k.name',
    'cross': 'lambda: [nm(x) for x in (Kind.X, Kind.Y)]',                   # not sampled
    'fmt_r': 'lambda: "{!r}".format(Leg.CALL)',
    'fmt_pretty': 'lambda s: "sex={}".format(s) + str(s)',
    'str_tuple': 'lambda k: str((k, 1))',
    'shown': 'lambda k: (repr(k), str(k), f"{k}|{k!r}", "%s %r" % (k, k), type(k).__name__, isinstance(k, Leg))',
    'appended': 'def appended():\n    out = []\n    out.append(Leg.CALL)\n    return out\n',
    'app_names': 'lambda: [x.name for x in appended()]',
    'opt_local': 'def opt_local(t):\n    x = Leg.PUT if t > 0 else None\n    return x.name if x is not None else ""\n',
    'ord_lt': 'lambda: Ord.LOW < Ord.HIGH',
    'ord_param': 'lambda k: k < Ord.HIGH',
    'pick': 'lambda t: Leg.PUT if t > 0 else Kind.X',
    'pick_nm': 'lambda t: pick(t).name',
    'un': 'lambda k: k.name',
    'call_un': 'lambda t: un(Leg.PUT) if t > 0 else ""',                      # sampled with t = 0 only
    'lab': 'lambda: label(Leg.PUT)',
    'ret_member': 'lambda: Leg.PUT',
    'ret_name': 'lambda: ret_member().name',
    'ident': 'lambda k: (k is Leg.PUT, k is not Leg.CALL, ret_member() is Leg.PUT)',
    'is_tuple': 'lambda: tuple(x for x in (Leg.CALL,))[0] is (Leg.CALL, Leg.PUT)[0]',
    'legs': 'lambda: (Leg.CALL, Leg.PUT)',
    'first': 'lambda: Leg.CALL',
    'ref_nm': 'lambda: (first_leg.name, first_leg == Leg.PUT, repr(first_leg))',
    'sub_nm': 'lambda: Sub[Leg.PUT].val()',
    'value': 'lambda t, k: 10.0 * t + k',
    'total': 'lambda t: sum(value(t, k) for k in (Leg.CALL, Leg.PUT)) + value(t, Leg.PUT)',
    'chosen': 'lambda t, k: value(t, k) if k in (Leg.PUT,) else (-1.0 if k == Leg.CALL else 0.0)',
    # the Child_KR_S shape: the name of a member is the key of a table read
    'table': 'lambda: pd.DataFrame({"value": [1.5, 2.5]}, index=["CALL", "PUT"])',
    'basis_param': 'lambda name: float(table().loc[name.name, "value"])',
    'basis': 'lambda: basis_param(Leg.CALL) + 10.0 * basis_param(Leg.PUT)',
    # the VA_KR_S shape: a timing that is compared, passed on and shown in an error
    'fund_at': textwrap.dedent('''\
        def fund_at(t, timing):
            if timing == Timing.BEF:
                return 1.0 * t
            if timing == Timing.AFT:
                return fund_at(t, Timing.BEF) + 0.5
            raise ValueError("invalid timing: " + str(timing) + " " + repr(timing))
        '''),
}
SAMPLE = '''
for name in ('comp_name', 'comp_repr', 'comp_isinst', 'list_loop', 'fmt_r', 'app_names', 'ord_lt', 'lab', 'ret_name',
             'is_tuple', 'ref_nm', 'sub_nm', 'basis'):
    getattr(p, name)()
p.nm(p.Leg.CALL); p.fmt_pretty(p.Pretty.M); p.str_tuple(p.Leg.CALL); p.shown(p.Leg.CALL); p.opt_local(0)
p.opt_local(1); p.ord_param(p.Ord.LOW); p.pick_nm(1); p.call_un(0); p.ident(p.Leg.PUT)
for t in range(3):
    p.total(t); p.chosen(t, p.Leg.CALL); p.chosen(t, p.Leg.PUT); p.fund_at(t, p.Timing.AFT)
p.C.ident(); p.C.idents()
'''
PROBES = [f'p.{name}()' for name in (
    'comp_name', 'comp_repr', 'comp_isinst', 'list_loop', 'fmt_r', 'app_names', 'ord_lt', 'lab', 'ret_name',
    'is_tuple', 'ref_nm', 'sub_nm', 'basis')] + [
    'p.nm(p.Leg.PUT)', 'p.fmt_pretty(p.Pretty.F)', 'p.str_tuple(p.Leg.PUT)', 'p.shown(p.Leg.PUT)', 'p.opt_local(1)',
    'p.ord_param(p.Ord.LOW)', 'p.pick_nm(1)', 'p.call_un(1)', 'p.ident(p.Leg.PUT)', 'p.ident(p.Leg.CALL)',
    'p.ret_member() == p.Leg.PUT', 'p.Sub[p.Leg.CALL].val()', 'p.total(2)', 'p.chosen(1, p.Leg.PUT)',
    'p.chosen(1, 0)', 'p.fund_at(2, p.Timing.AFT)', 'p.fund_at(2, p.Timing.END)', 'p.shown(1)[4:]',
    'p.C.ident()', 'p.C.idents()', 'p.basis_param(p.Leg.PUT)']
# a member of another class given to a converted parameter or result; probed in a process of their own: the
# exported model keeps one cached value for members of two classes that are equal as integers
RAISES = ['p.cross()', 'p.pick_nm(0)']


def test_every_use_equals_live(tmp_path, monkeypatch):
    (tmp_path / 'uses_enums.py').write_text(ENUMS, encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    import pandas as pd
    import uses_enums as e
    model = mx.new_model('EnumUses')
    try:
        s = model.new_space('P', refs={'Leg': e.Leg, 'Kind': e.Kind, 'Timing': e.Timing, 'Pretty': e.Pretty,
                                       'Ord': e.Ord, 'label': e.label, 'first_leg': e.Leg.PUT, 'pd': pd})
        for name, formula in CELLS.items():
            s.new_cells(name, formula)
        s.new_space('Sub', formula=lambda leg: None).new_cells('val', 'lambda: (leg.name, leg == 1, [leg])')
        c = s.new_space('C', refs={'par': s})           # another module, without an enum Reference of its own
        c.new_cells('ident', 'lambda: par.legs()[0] is par.first()')
        c.new_cells('idents', 'lambda: [v is w for v in par.legs() for w in (par.first(),)]')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\np = m.P\n' + SAMPLE, encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                          cwd=tmp_path, capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        p = import_module(sys.argv[1]).mx_model.P
        for probe in sys.argv[2:]:
            try:
                print(probe, repr(eval(probe)))
            except Exception as error:
                print(probe, type(error).__name__, str(error).partition(": ")[2] if "P." in str(error) else error)
        ''')
    out = {}
    for package in ('exported', 'exported_cy'):
        for probes in (RAISES, PROBES):         # two processes: see RAISES
            run = subprocess.run([sys.executable, '-c', probe, package, *probes], cwd=tmp_path,
                                 capture_output=True, text=True, env=env)
            assert run.returncode == 0, package + run.stdout + run.stderr
            out.setdefault(package, []).extend(run.stdout.splitlines())
    live, compiled = out['exported'][len(RAISES):], out['exported_cy'][len(RAISES):]
    assert compiled == live, '\n'.join(f'{a}  |  {b}' for a, b in zip(live, compiled) if a != b)
    assert "p.shown(p.Leg.PUT) ('<Leg.PUT: 1>'," in ''.join(compiled)
    assert "p.fund_at(2, p.Timing.END) ValueError invalid timing: 2 <Timing.END: 2>" in compiled
    assert out['exported'][:len(RAISES)] == ["p.cross() ['X', 'Y']", "p.pick_nm(0) 'X'"]
    assert [line.split()[1] for line in out['exported_cy'][:len(RAISES)]] == ['TypeError', 'TypeError'], out['exported_cy']
