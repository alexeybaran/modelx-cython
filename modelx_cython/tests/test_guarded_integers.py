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
    # range bound and integer locals: the exact value converted to long long on the slow path
    assert 'range(1, (limit + 1 if -2305843009213693952 <= limit <= 2305843009213693952 else ' \
           '_mx_cy.cast(_mx_cy.longlong, _mx_cy.cast(object, limit) + 1)))' in text
    assert '(k - 1) * 3 if -2305843009213693952 <= k <= 2305843009213693952 else _mx_cy.cast(_mx_cy.longlong,' in text  # a literal factor keeps a wide bound
    assert 's = (start + off if ' in text
    # comparison, true division and float(): the whole result guarded, exact either way
    assert '(k * 3 <= g if -2305843009213693952 <= k <= 2305843009213693952 else _mx_cy.cast(object, k) * 3 <= g)' in text
    assert "(((s - start) / 12.0) if " in text and "((_mx_cy.cast(object, s) - _mx_cy.cast(object, start)) / 12.0))" in text
    assert '(float(s * s) if -2147483648 <= s <= 2147483648 else float(_mx_cy.cast(object, s) * _mx_cy.cast(object, s)))' in text
    # an augmented integer local
    assert 'k2 = (k2 + (1) if ' in text
    assert 'same = k2 is k' in text            # identity is not arithmetic
    assert count == 8


def test_values_are_exact_on_both_paths():
    text, _ = _guarded()
    scope = {}
    shim = 'class _mx_cy:\n    longlong = int\n    @staticmethod\n    def cast(kind, value):\n        return kind(value) if kind is not object else value\n'
    exec(shim + text, scope)
    exec(SOURCE.replace('def f(', 'def g('), scope)
    big = (1 << 62) + 5
    for start, limit, g in ((0, 5, 7.0), (big, 3, 1e30), (-big, 4, -1.0)):
        assert scope['f'](None, start, limit, g) == scope['g'](None, start, limit, g)
