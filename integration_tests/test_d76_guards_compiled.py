"""D76 guards that the range proofs make redundant are emitted as the plain native expression.

diag-us2: the guards of guarded_integers are written before entry lowering and the integer
boundary pass. A guard always has the exact value of its native expression, so those passes
read through it; where the native expression is proven not to overflow from the Cell's argument
window, or where its value is taken as long long and only its outer operation can overflow, the
guard is dropped. The values stay exact, and a value outside the 64-bit range still raises.
"""

import re
import subprocess
import sys
import textwrap

import modelx as mx


def test_proven_guards_are_native(tmp_path):
    model = mx.new_model('ProvenGuards')
    try:
        values = model.new_space('Values')
        values.new_cells('share', textwrap.dedent('''\
            def share(t, age):
                m = t - age
                return 0.5 + 0.01 * m
            '''))
        values.new_cells('bal', textwrap.dedent('''\
            def bal(t, age):
                m = t - age
                if t == 0 or age == 0:
                    return 1.0
                return bal(t - 1, t - 1 - m) * share(t, t - m)
            '''))
        values.new_cells('term', 'lambda index: float(index)', is_cached=False)
        values.new_cells('gap', textwrap.dedent('''\
            def gap(a, b):
                return term(a - b)
            '''), is_cached=False)
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'v = mx_model.Values\n'
        'for t in range(12):\n'
        '    for age in range(4):\n'
        '        v.bal(t, age)\n'
        'v.gap(5, 3)\n',
        encoding='utf-8')
    subprocess.run(
        [sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
        cwd=tmp_path, check=True)
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')

    c_source = (tmp_path / 'exported_cy/_mx_classes.c').read_text(encoding='utf-8')

    def body(name):
        """The Python text that computes the cell: its formula method, or the module-level
        entry function the method delegates to (test_native_integer_boundaries does the same)."""
        found = (re.search(rf'def _f_{name}\(.*?(?=\n    @|\n    def |\nclass |\ndef |\Z)', source, re.S)
                 or re.search(rf'    def {name}\(.*?(?=\n    @|\n    def |\nclass |\ndef |\Z)', source, re.S))
        assert found, name
        text = found.group(0)
        entry = re.search(r'return (_mx_dense_formula_\d+)\(', text)
        if entry:
            text = re.search(rf'def {entry.group(1)}\(.*?(?=\n@|\nclass |\ndef |\Z)', source, re.S).group(0)
        return text

    def c_local_type(function, local):
        """C declaration of a local in the generated C function named after ``function``."""
        start = re.search(rf'\n[^\n;]*\b__pyx_\w*{function}\([^;{{]*\{{\n', c_source)
        assert start, function
        block = c_source[start.end():c_source.index('\n}\n', start.end())]
        found = re.search(rf'^  (PY_LONG_LONG|PyObject \*)\s*__pyx_v_{local}\b', block, re.M)
        return found.group(1) if found else None

    # proven from the argument window: no overflow guard where `m` is computed, and `m` is a
    # C long long there (whether annotated or inferred)
    for name in ('share', 'bal'):
        text = body(name)
        assert 'm = (t - age)' in text.replace('m: _mx_cy.longlong = (t - age)', 'm = (t - age)'), text
        assert '>> 63' not in text and '_mx_cy.cast(object, t)' not in text, text
        function = re.search(r'def (\w+)\(', text).group(1)
        assert c_local_type(function, 'm') == 'PY_LONG_LONG', (name, function)
    # `a - b` taken as a long long argument: one outer operation, no guard; exact or raising
    assert '>> 63' not in body('gap'), body('gap')

    probe = textwrap.dedent('''\
        from exported_cy import mx_model
        v = mx_model.Values

        def bal(t, age):
            m = t - age
            if t == 0 or age == 0:
                return 1.0
            return bal(t - 1, t - 1 - m) * (0.5 + 0.01 * (t - (t - m)))

        for t in range(12):
            for age in range(4):
                assert v.bal(t, age) == bal(t, age), (t, age)
        bound = (1 << 63) - 1
        for a, b in ((5, 3), (bound, 0), (-bound - 1, 0), (bound, bound), (-bound - 1, -bound - 1),
                     (bound - 5, -5), (-bound, 1)):
            assert v.gap(a, b) == float(a - b), (a, b)
        for a, b in ((bound, -1), (-bound - 1, 1), (bound, -bound - 1)):
            try:
                v.gap(a, b)
            except OverflowError:
                pass
            else:
                raise AssertionError((a, b))
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
