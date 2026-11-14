"""D76: integer arithmetic on C-integer locals in guarded form (exact on every value)."""

import libcst as cst

from modelx_cython.guarded_integers import guard_integers

SOURCE = '''def f(self, start, limit, g):
    total = 0.0
    for k in range(1, limit + 1):
        off = (k - 1) * 3 if g > 0.0 else k * 3
        s = start + off
        cert = 1.0 if k * 3 <= g else 0.0
        total += ((s - start) / 12.0) + float(s * s)
        k2 = k
        k2 += 1
        same = k2 is k
    return total
'''
LOCALS = {'k': 'int', 'off': 'int', 's': 'int', 'k2': 'int', 'total': 'float', 'cert': 'float'}
PARAMS = {'start': 'int', 'limit': 'int', 'g': 'float'}


def _guarded():
    node = cst.parse_module(SOURCE).body[0]
    out, count = guard_integers(node, LOCALS, PARAMS)
    return cst.Module([]).code_for_node(out), count


def test_each_context_takes_its_slow_path():
    text, count = _guarded()
    wide = '< _mx_cy.cast(_mx_cy.longlong, 2305843009213693952)'
    narrow = '< _mx_cy.cast(_mx_cy.longlong, 2147483648)'
    # range bound and integer locals: the exact value converted to long long on the slow path
    assert f'range(1, (limit + 1 if ((limit ^ (limit >> 63))) {wide} else ' \
           '_mx_cy.cast(_mx_cy.longlong, _mx_cy.cast(object, limit) + 1)))' in text
    assert f'(k - 1) * 3 if ((k ^ (k >> 63))) {wide} else _mx_cy.cast(_mx_cy.longlong,' in text  # a literal factor
    assert f's = (start + off if ((off ^ (off >> 63)) | (start ^ (start >> 63))) {wide} else ' in text
    # comparison, true division and float(): the whole result guarded, C-typed on both branches
    assert f'(k * 3 <= g if ((k ^ (k >> 63))) {wide} else ' \
           '_mx_cy.cast(_mx_cy.bint, _mx_cy.cast(object, k) * 3 <= g))' in text
    assert "_mx_cy.cast(_mx_cy.double, ((_mx_cy.cast(object, s) - _mx_cy.cast(object, start)) / 12.0)))" in text
    assert f'(float(s * s) if ((s ^ (s >> 63))) {narrow} else ' \
           '_mx_cy.cast(_mx_cy.double, float(_mx_cy.cast(object, s) * _mx_cy.cast(object, s))))' in text
    assert f'k2 = (k2 + (1) if ((k2 ^ (k2 >> 63))) {wide} else ' in text   # an augmented integer local
    assert 'same = k2 is k' in text            # identity is not arithmetic
    assert count == 8


def test_the_bound_test_is_exact_for_any_int():
    for bound in (1 << 31, 1 << 61):
        for x in (-(1 << 70), -(1 << 63) - 1, -(1 << 63), -bound - 1, -bound, -1, 0, 1, bound - 1, bound,
                  (1 << 63) - 1, 1 << 70):
            assert ((x ^ (x >> 63)) < bound) == (-bound <= x < bound), (bound, x)


def test_values_are_exact_on_both_paths():
    text, _ = _guarded()
    scope = {}
    shim = 'class _mx_cy:\n    longlong = int\n    double = float\n    bint = bool\n    @staticmethod\n    def cast(kind, value):\n        return kind(value) if kind is not object else value\n'
    exec(shim + text, scope)
    exec(SOURCE.replace('def f(', 'def g('), scope)
    big = (1 << 62) + 5
    for start, limit, g in ((0, 5, 7.0), (big, 3, 1e30), (-big, 4, -1.0)):
        assert scope['f'](None, start, limit, g) == scope['g'](None, start, limit, g)


def test_subscripts_are_guarded_except_a_proven_raw_vector_buffer():
    """An index is guarded (exact value as long long on the slow path), except the raw buffer
    `v.data[...]` that W1.6 emits only inside a loop whose index range it has proven."""
    source = ('def f(self, n):\n'
              '    for i in range(n):\n'
              '        a = values[i + 1]\n'
              '        b = vec.data[i + 1]\n'
              '    return a + b\n')
    node = cst.parse_module(source).body[0]
    out, count = guard_integers(node, {'i': 'int'}, {'n': 'int'})
    text = cst.Module([]).code_for_node(out)
    assert 'values[(i + 1 if ((i ^ (i >> 63))) < _mx_cy.cast(_mx_cy.longlong, 2305843009213693952) else ' \
           '_mx_cy.cast(_mx_cy.longlong, _mx_cy.cast(object, i) + 1))]' in text
    assert 'vec.data[i + 1]' in text
    assert count == 1
