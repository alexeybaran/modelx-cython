"""Tuple experiment (r4 w1-vectors): a tuple of floats or integers BUILT AS A SEQUENCE (``tuple(<generator or
list>)``, ``(x,) * n``, or those concatenated with other tuples, ``(x,) + tuple(...)``) returned by a cell is
a vector cell, immutable for readers by construction; Python code gets an equal tuple. A literal display
``(a, b)``, a tuple used as a cell argument or a dictionary key, and a small record stay Python tuples.

The model repeats the shapes of IP_UK_S (lifelib r4/l1-lean-b) ``claim_rate_vectors`` and ``sick_cohorts``.
"""
import ast
import math

from vector_live import build, live_and_compiled, reports

CLAIMS = {
    'max_dur': 'def max_dur():\n    return 6\n',
    'proj_len': 'def proj_len():\n    return 8\n',
    'rec_rate_mth': 'def rec_rate_mth(z):\n    return 0.02 + 0.001 * z\n',
    'mort_rate_sick_mth': 'def mort_rate_sick_mth(z):\n    return 0.001 * z\n',
    'pols_inception': 'def pols_inception(t):\n    return 10.0 + t\n',
    'claim_rate_vectors': '''def claim_rate_vectors():
    zs = range(1, max_dur() + 1)
    rec = tuple(rec_rate_mth(z) for z in zs)
    dth = tuple(mort_rate_sick_mth(z) for z in zs)
    return rec, dth, tuple((1.0 - r) * (1.0 - d) for r, d in zip(rec, dth))
''',
    'sick_cohorts': '''def sick_cohorts(t):
    n = max_dur()
    if t < 0 or t >= proj_len():
        return (0.0,) * n
    if t == 0:
        seed = 5.0
        z0 = 2
        return tuple(seed if z == z0 else 0.0 for z in range(1, n + 1))
    prev = sick_cohorts(t - 1)
    surv = claim_rate_vectors()[2]
    return (pols_inception(t - 1),) + tuple(
        prev[z - 1] * surv[z - 1] for z in range(1, n))
''',
    'pols_sick_dur': '''def pols_sick_dur(t, z):
    v = sick_cohorts(t)
    return v[z - 1] if 1 <= z <= len(v) else 0.0
''',
    'pols_sick': 'def pols_sick(t):\n    return sum(sick_cohorts(t))\n',
    'pols_sick_surv': '''def pols_sick_surv(t):
    v = sick_cohorts(t)
    surv = claim_rate_vectors()[2]
    return sum(a * b for a, b in zip(v, surv) if a != 0.0)
''',
    'pols_death_sick': '''def pols_death_sick(t):
    v = sick_cohorts(t)
    rec, dth, _ = claim_rate_vectors()
    return sum(a * (1.0 - r) * d
               for a, r, d in zip(v, rec, dth) if a != 0.0)
''',
    'counts': 'def counts(t):\n    return tuple(z * t for z in range(4))\n',
    'count_total': 'def count_total(t):\n    c = counts(t)\n    return c[1] + len(c) + sum(c)\n',
    'keyed': 'def keyed(t):\n    return {counts(t): 1.5}[counts(t)] + len(set([sick_cohorts(t)]))\n',
}
# Python tuples: a literal display (a record of ints, of floats), a key, a cell argument
RECORDS = {
    'record': 'def record(t):\n    return (t, 2 * t)\n',
    'small': 'def small(t):\n    return (0.5 * t, 1.5)\n',
    'key_of': 'def key_of(t):\n    return (t % 2, 1)\n',
    'lookup': 'def lookup(t):\n    d = {(0, 1): 1.5, (1, 1): 2.5}\n    return d[key_of(t)]\n',
    'scaled': 'def scaled(pair):\n    return pair[0] * pair[1]\n',
    'with_arg': 'def with_arg(t):\n    return scaled((t, 1.5)) + scaled(small(t))\n',
}
CALLS = (['m.Claims.claim_rate_vectors()']
         + [f'm.Claims.{name}({t})' for name in ('sick_cohorts', 'pols_sick', 'pols_sick_surv', 'pols_death_sick',
                                                 'counts', 'count_total', 'keyed', 'record', 'small', 'key_of',
                                                 'lookup', 'with_arg') for t in range(-1, 9)]
         + [f'm.Claims.pols_sick_dur({t}, {z})' for t in range(9) for z in range(0, 8)])
def _same(a, b):
    """Equal values: numbers within rel 1e-10, abs 1e-7 (a double cell gives 0.0 for live's sum of nothing,
    0), containers of the same type."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


VECTOR_CELLS = {'claim_rate_vectors': ['T', ['V', 'V', 'V']], 'sick_cohorts': 'V', 'counts': 'I'}


def test_tuples_built_as_sequences_are_vectors(tmp_path):
    build(tmp_path, {'Claims': ({**CLAIMS, **RECORDS}, {})}, ''.join(c + '\n' for c in CALLS))
    live, compiled = live_and_compiled(tmp_path, CALLS + [f'type({c}).__name__' for c in CALLS[:12]])
    assert all(kind == 'value' for kind, _ in live)
    for expr, (_, a), (kind, b) in zip(CALLS, live, compiled):
        assert kind == 'value' and _same(ast.literal_eval(a), ast.literal_eval(b)), (expr, a, b)
    assert compiled[len(CALLS):] == live[len(CALLS):]      # the types of the values
    found = reports(tmp_path)
    shapes = {name: (found[name].get('vectors') or {}).get('shape') for name in {**CLAIMS, **RECORDS}}
    for name, shape in shapes.items():
        assert shape == VECTOR_CELLS.get(name), (name, shape)
    source = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    for name in RECORDS:        # records, keys and arguments: no vector code
        start = source.index(f'def _f_{name}(')
        assert 'Vec' not in source[start:source.index('\n    @', start)], name
