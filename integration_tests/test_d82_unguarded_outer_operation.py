"""D82 conditions (a) and (c) for diag-us2 0001 (D76 guards read as their native expression).

(a) Where a value is taken as long long and only its outer operation is unproven, the guard is
dropped. Each operator on full-range C integer leaves must then either give the exact value or
raise OverflowError, never wrap: +, -, *, //, %, unary minus (including -(-2**63)), with leaves of
the widths the passes produce (long long parameters, a bint cell, an integer cell result).
(c) A seg_*-shaped cell (local ``m = t - age``, recursive call ``f(t - 1, t - 1 - m)``, argument
``t - m``) keeps its window proof: the cache entry carries ``_mx_proven_axes``, ``m`` is a plain C
difference, and no guard is emitted, so a later change of pass order cannot silently undo it.
"""

import re
import subprocess
import sys
import textwrap

import modelx as mx

OPERATIONS = {
    'add': 'a + b', 'sub': 'a - b', 'mul': 'a * b', 'fdiv': 'a // b', 'mod': 'a % b', 'neg': '-a',
    'flag_add': 'a + flag()', 'cell_sub': 'a - width()',
}


def test_unguarded_outer_operations_are_exact_or_raise(tmp_path):
    model = mx.new_model('D82Outer')
    try:
        values = model.new_space('Values')
        values.new_cells('term', 'lambda index: float(index)', is_cached=False)
        values.new_cells('flag', 'lambda: True')
        values.new_cells('width', 'lambda: 7')
        for name, expression in OPERATIONS.items():
            values.new_cells(name, textwrap.dedent(f'''\
                def {name}(a, b):
                    return term({expression})
                '''), is_cached=False)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    calls = ''.join(f'v.{name}(7, 3)\n' for name in OPERATIONS)
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\nv = mx_model.Values\nv.flag(); v.width()\n' + calls, encoding='utf-8')
    subprocess.run(
        [sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
        cwd=tmp_path, check=True)
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    for name in OPERATIONS:
        body = re.search(rf'    def {name}\(.*?(?=\n    @|\n    def |\nclass |\Z)', source, re.S).group(0)
        assert '>> 63' not in body, body       # no guard: the outer operation is the boundary

    probe = textwrap.dedent('''\
        from exported_cy import mx_model
        v = mx_model.Values
        LO, HI = -(1 << 63), (1 << 63) - 1
        edges = (LO, LO + 1, -(1 << 62), -(1 << 31) - 1, -7, -1, 0, 1, 3, (1 << 31), (1 << 62), HI - 1, HI)
        python = {
            'add': lambda a, b: a + b, 'sub': lambda a, b: a - b, 'mul': lambda a, b: a * b,
            'fdiv': lambda a, b: a // b, 'mod': lambda a, b: a % b, 'neg': lambda a, b: -a,
            'flag_add': lambda a, b: a + 1, 'cell_sub': lambda a, b: a - 7,
        }
        checked = raised = 0
        for name, exact in python.items():
            cell = getattr(v, name)
            for a in edges:
                for b in edges:
                    try:
                        expected = exact(a, b)
                    except ZeroDivisionError:
                        try:
                            cell(a, b)
                        except ZeroDivisionError:
                            continue
                        raise AssertionError((name, a, b, 'no ZeroDivisionError'))
                    if LO <= expected <= HI:
                        got = cell(a, b)
                        assert got == float(expected), (name, a, b, got, expected)
                        checked += 1
                    else:
                        try:
                            got = cell(a, b)
                        except OverflowError:
                            raised += 1
                            continue
                        raise AssertionError((name, a, b, 'wrapped to', got, 'exact', expected))
        assert checked and raised
        try:
            v.neg(LO, 0)               # -(-2**63) is outside the range
        except OverflowError:
            pass
        else:
            raise AssertionError('-(-2**63) did not raise')
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)


def test_seg_shaped_cell_keeps_its_window_proof(tmp_path):
    model = mx.new_model('D82Seg')
    try:
        values = model.new_space('Values')
        values.new_cells('seg_share', textwrap.dedent('''\
            def seg_share(t, segment_age):
                m = t - segment_age
                return 0.5 + 0.01 * m
            '''))
        values.new_cells('seg_bal', textwrap.dedent('''\
            def seg_bal(t, segment_age):
                m = t - segment_age
                if t == 0 or segment_age == 0:
                    return 1.0
                return seg_bal(t - 1, t - 1 - m) * seg_share(t, t - m)
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'v = mx_model.Values\n'
        'for t in range(24):\n'
        '    for s in range(6):\n'
        '        v.seg_bal(t, s)\n'
        '        v.seg_share(t, s)\n', encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({
        'dense_bounds_proofs': True, 'dense_one_axis': True, 'dense_miss_binding': True,
        'propagate_exact_receiver': True}), encoding='utf-8')
    subprocess.run(
        [sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--spec', 'spec.py'],
        cwd=tmp_path, check=True)
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert '_mx_proven_axes' in source, 'the cache entries lost their window proof'
    entries = re.findall(r'def (_mx_cache_entry_\d+)\(self[^)]*_mx_proven_axes', source)
    assert entries, 'no cache entry takes _mx_proven_axes'
    c_source = (tmp_path / 'exported_cy/_mx_classes.c').read_text(encoding='utf-8')

    def c_local_type(function, local):
        """C declaration of a local in the generated C function named after ``function``."""
        start = re.search(rf'\n[^\n;]*\b__pyx_\w*{function}\([^;{{]*\{{\n', c_source)
        assert start, function
        block = c_source[start.end():c_source.index('\n}\n', start.end())]
        found = re.search(rf'^  (PY_LONG_LONG|PyObject \*)\s*__pyx_v_{local}\b', block, re.M)
        return found.group(1) if found else None

    for name in ('seg_share', 'seg_bal'):
        dense = re.search(rf'def _f_{name}\(.*?(_mx_dense_formula_\d+)\(', source, re.S).group(1)
        body = re.search(rf'\ndef {dense}\(.*?(?=\n@|\ndef |\nclass |\Z)', source, re.S).group(0)
        # properties: no overflow guard, no boxed operand, and `m` a C long long in the compiled function
        assert '>> 63' not in body and '_mx_cy.cast(object' not in body, body
        assert c_local_type(dense, 'm') == 'PY_LONG_LONG', (name, dense)
    probe = textwrap.dedent('''\
        from exported_cy import mx_model
        v = mx_model.Values
        def seg_bal(t, s):
            m = t - s
            if t == 0 or s == 0:
                return 1.0
            return seg_bal(t - 1, t - 1 - m) * (0.5 + 0.01 * (t - (t - m)))
        for t in range(24):
            for s in range(6):
                assert v.seg_bal(t, s) == seg_bal(t, s), (t, s)
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
