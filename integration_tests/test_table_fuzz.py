"""Shape fuzz of the table-read recognisers (T-5c): generated frame expressions on the probe's tables.

A small grammar combines receivers (a table, a prefix of it, a view local), operations (loc/at/iloc, index[i],
asof, searchsorted, get_level_values extremes, membership, walks, iterrows, comprehensions and for loops with
plain, tuple, nested and starred targets) and keys (literals in odd forms, tuples, slices, lists, calls, locals,
starred parts). Every case is a cell of the probe model (``test_table_probe``).

- Translation of every case runs under strict lowering: an internal error of a recogniser fails the build.
- A seeded subset is compiled and every case is compared with live modelx as the probe compares: equal, or one of
  the probe's accepted boundary types. A case either compiles natively or is left to pandas; both must equal live.
"""
import json
import os
import random
import re

import pytest

import test_table_probe as probe

SEED = 20261119
COUNT = int(os.environ.get('MX_CY_FUZZ_CASES', '3000'))
COMPILED = int(os.environ.get('MX_CY_FUZZ_COMPILED', '500'))

KEYS = {   # by table: one-part keys, complete keys, prefixes
    'flat': ['20', '20 + t', '0x14', '20.0', 'True', "'a'", '(20,)', '((20,),)', '[20]', '20:21', 'int(20.0)',
             't + 19.5', '-1', 'None', 'k1', '(t + 20)', '20 + t % 4', 'abs(-21)'],
    'grid': ["('A', 20)", "('A', 20 + t)", "('B', 0x15)", "('A',)", "'A'", "('A', 'B')[t % 2]", 'k2', 'k1',
             "(('A', 20),)", "['A']", "('C', 20)", "('A', 20, 1)", "('A', slice(None))", "(*k2,)", 'tuple(k2)',
             "'A' if t else 'B'", "('A', 20.0)", '20'],
    'tri': ["('A', 'F', 20)", "('A', 'F')", "'A'", "('B', 'M', 21)", 'k3', "('A', 'F', 20 + t % 2)", "('A', 'q')",
            "(('A', 'F'), 20)", 'k2', "('A',)", "('A', 'F', 22)"],
}
COLUMNS = ["'q'", "'n'", "('q', 'n')[t % 2]", "'zz'", '0', "['q']", "'x'"]
LOCALS = "k1 = 21 + t % 3\nk2 = ('A', 20 + t % 4) if t % 3 else 'B'\nk3 = ('A', 'F', 20) if t else ('B', 'M')\n"
OPS = [
    'R.loc[K, C]', 'R.loc[K]', 'R.at[K, C]', 'R.loc[K][C]', 'R[C][K]', 'R.iloc[I]', 'R.index[I]',
    'R.index.asof(X)', 'int(R.index.asof(X))', 'R.index.searchsorted(X)', "R.index.searchsorted(X, side='right')",
    'R.index.get_level_values(L).max()', 'R.index.max()', 'min(R.index)', 'len(R.index)', 'K in R.index',
    'K not in R.index', '[k for k in R.index]', 'sorted(R.index)', '[(k, r[C]) for k, r in R.iterrows()]',
    'sum(v for v in R[C])', 'R.loc[K, C] if K in R.index else -1.0', 'float(R.loc[K, C])', 'R[C].tolist()',
    '{a: 1 for a in R.index}', 'R.loc[R.index[I], C]', 'R.loc[K, C] + R.loc[K, C]',
]
TARGETS = ['k', 'a, b', '(a, b)', 'a, (b, c)', '*a, b', 'a, b, c', '[a, b]', 'a, *b']
INDEXES = ['0', '-1', 't', 'True', '0x1', '1.0', 't - 1', '9']
XS = ['20.5', '21 + t', 't', "'a'", '1e9', 'float("nan")', '-5']
LEVELS = ['0', '1', '-1', "'age'", "'plan'", '2', '0x1']


def _receivers(source):
    yield 'T', ''
    if source != 'flat':
        prefix = "('A', 'B')[t % 2]"
        yield f'T.loc[{prefix}]', ''
        yield 'sub', f'sub = T.loc[{prefix}]\n'


def cases(count, seed=SEED):
    """``count`` (name, source, body) cases, deterministic for ``seed``."""
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        source = rng.choice(sorted(KEYS))
        receiver, setup = rng.choice(list(_receivers(source)))
        fill = {'R': receiver, 'K': rng.choice(KEYS[source]), 'C': rng.choice(COLUMNS),
                'I': rng.choice(INDEXES), 'X': rng.choice(XS), 'L': rng.choice(LEVELS)}
        if rng.random() < 0.2:      # a for loop with a target of any shape
            target = rng.choice(TARGETS)
            walked = rng.choice([f'{receiver}.index', f'{receiver}.iterrows()', f"{receiver}[{fill['C']}].items()"])
            body = (f'{LOCALS}{setup}total = 0\nfor {target} in {walked}:\n    total += 1\n'
                    f'    if total > 3:\n        break\nreturn total')
        else:
            expression = re.sub(r'\b([RKCIXL])\b', lambda m: fill[m.group(1)], rng.choice(OPS))   # one pass
            body = f'{LOCALS}{setup}return {expression}'
        try:                        # only syntactically valid formulas (a slice key outside a subscript is not)
            compile('def f(t):\n' + ''.join('    ' + line + '\n' for line in body.split('\n')), 'case', 'exec')
        except SyntaxError:
            continue
        out.append((f'z{len(out)}', source, body))
    return out


def _model(tmp_path, chosen):
    stmts = {name: (source, body) for name, source, body in chosen}
    (tmp_path / 'probe_model.py').write_text(
        probe.MODEL.format(ops={}, stmts=stmts, data=probe.DATA, t5_items={}), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(probe.SAMPLE, encoding='utf-8')
    (tmp_path / 'checks.py').write_text(probe.CHECKS, encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'parallel_workers': 1, **SPEC}), encoding='utf-8')
    probe._run(tmp_path, 'probe_model.py')


SPEC = {} if os.environ.get('MX_CY_FUZZ_BASELINE') else {'strict_lowering': True}


def _build(tmp_path, *extra):
    import subprocess
    import sys
    env = dict(os.environ, MX_CY_STRICT_LOWERING='1')
    return subprocess.run([sys.executable, '-m', 'modelx_cython', 'probe_nomx', '--sample', 'sample.py',
                           '--spec', 'spec.py', *extra], cwd=tmp_path, env=env, timeout=7200,
                          capture_output=True, text=True)


def test_fuzz_translation_never_raises(tmp_path):
    _model(tmp_path, cases(COUNT))
    built = _build(tmp_path, '--translate-only')
    assert built.returncode == 0, built.stdout[-4000:] + built.stderr[-8000:]
    reads = json.loads((tmp_path / 'probe_nomx_cy' / '_mx_table_reads.json').read_text())
    errors = [e for es in reads.values() if isinstance(es, list) for e in es
              if isinstance(e, dict) and 'internal error' in e.get('reason', '')]
    assert not errors, errors[:10]


def test_fuzz_compiled_equals_live(tmp_path):
    chosen = random.Random(SEED + 1).sample(cases(COUNT), min(COMPILED, COUNT))
    _model(tmp_path, chosen)
    built = _build(tmp_path)
    assert built.returncode == 0, built.stdout[-4000:] + built.stderr[-8000:]
    verdicts = json.loads(probe._run(tmp_path, 'checks.py').stdout)
    differing = {name: v for name, v in verdicts.items()
                 if v != 'same' and not (v.startswith('boundary: ')
                                         and set(v[len('boundary: '):].split('; ')) <= probe.BOUNDARY)}
    assert not differing, '\n'.join(f'{n}: {v}' for n, v in sorted(differing.items())[:40])


# Two-formula mode: pairs of formulas of one item class that reuse the same local names, cells and tables with
# different prefixes and keys. Held fields are per class, so a key shared by two different reads would make one
# formula read the other's codes or rows; each pair is also generated in the other order.
PREFIXES = {   # table -> prefix expressions of an item (literals, item cells, per-step choices)
    'grid': ["'A'", "'B'", 'plan()', "('A', 'B')[t % 2]", "('B', 'A')[t % 2]"],
    'sgrid': ["'A'", "'B'", 'plan()'],
    'rates': ["(plan(), sex())", "(plan(), 'F')", "('A', 'M')", "plan()", "('B', sex())"],
}
READS = {
    'grid': ["sub.loc[20 + t % 4, 'q']", "int(sub.index[-1])", "len(sub.index)", "sum(sub.loc[k, 'q'] for k in sub.index)",
             "float(sub.index.asof(20.5 + t))", "sub.loc[(21, 22)[t % 2], 'n']"],
    'rates': ["sub.loc[2000 + t % 3, 'q']", "len(sub.index)", "[int(k) for k in sub.index]"],
}


def pairs(count, seed=SEED + 2):
    """``count`` pairs of item formulas (as T5_ITEMS entries), each also in the other order."""
    rng = random.Random(seed)
    out = {}
    while len(out) < 4 * count:
        table = rng.choice(sorted(PREFIXES))
        a, b = rng.sample(PREFIXES[table], 2)
        reads = READS['grid' if table != 'rates' else 'rates']
        if table == 'sgrid':        # unordered: pandas' asof raises (and the sample would stop on it)
            reads = [r for r in reads if 'asof' not in r]
        ra, rb = rng.choice(reads), rng.choice(reads)
        first = f"sub = data.{table}().loc[{a}]\nreturn {ra}"
        second = f"sub = data.{table}().loc[{b}]\nreturn {rb}"
        n = len(out) // 4
        out[f'pa{n}'], out[f'pb{n}'] = first, second
        out[f'qb{n}'], out[f'qa{n}'] = second, first          # the other order
    return out


def test_fuzz_two_formulas(tmp_path):
    items = pairs(int(os.environ.get('MX_CY_FUZZ_PAIRS', '60')))
    (tmp_path / 'probe_model.py').write_text(
        probe.MODEL.format(ops={}, stmts={}, data=probe.DATA, t5_items=items), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(probe.SAMPLE, encoding='utf-8')
    (tmp_path / 'checks.py').write_text(probe.CHECKS, encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'parallel_workers': 1, **SPEC}), encoding='utf-8')
    probe._run(tmp_path, 'probe_model.py')
    built = _build(tmp_path)
    assert built.returncode == 0, built.stdout[-4000:] + built.stderr[-8000:]
    verdicts = json.loads(probe._run(tmp_path, 'checks.py').stdout)
    differing = {name: v for name, v in verdicts.items()
                 if v != 'same' and not (v.startswith('boundary: ')
                                         and set(v[len('boundary: '):].split('; ')) <= probe.BOUNDARY)}
    assert not differing, '\n'.join(f'{n}: {v}' for n, v in sorted(differing.items())[:40])
