"""diag-us2: later passes read a D76 guard as its native expression and drop it where redundant."""

import itertools

import libcst as cst

from modelx_cython.checked_arithmetic import CheckedArithmetic
from modelx_cython.entry_pipeline import _argument_range
from modelx_cython.guarded_integers import guard_integers, long_long_guard, native_branch
from modelx_cython.integer_boundaries import _ProvenGuards

SOURCE = '''def f(self, t, a, g):
    m = t - a
    n = a - 1 - t
    k = 0
    k += a
    x = self.cell(t - 1, a - t, a * a - t)
    y = float(a * t)
    z = (a - t) / 12.0
    w = t - 1 < a
    for i in range(1, a + 1):
        pass
    return [a - t]
'''
LOCALS = {'m': 'int', 'n': 'int', 'k': 'int', 'x': 'float', 'y': 'float', 'z': 'float', 'w': 'unknown',
          'i': 'int'}
PARAMS = {'t': 'int', 'a': 'int', 'g': 'float'}
LIMIT = 1 << 63


def _code(node):
    return cst.Module([]).code_for_node(node)


def _guarded():
    node = cst.parse_module(SOURCE).body[0]
    out, _ = guard_integers(node, LOCALS, PARAMS)
    return out


def _guards(node):
    found = []

    class Collect(cst.CSTVisitor):
        def visit_IfExp(self, item):
            found.append(item)
    node.visit(Collect())
    return found


def test_native_branch_reads_every_guard_as_its_expression():
    guards = _guards(_guarded())
    natives = sorted(_code(native_branch(g)) for g in guards)
    assert natives == sorted([
        't - a', 'a - 1 - t', 'k + (a)', 't - 1', 'a - t', 'a * a - t', 'float(a * t)', '(a - t) / 12.0',
        't - 1 < a', 'a + 1', 'a - t'])
    assert native_branch(cst.parse_expression('(t - 1 if g else t + 1)')) is None
    assert native_branch(cst.parse_expression('(t - 1 if (t ^ (t >> 63)) < 5 else t)')) is None


def test_long_long_contexts():
    kinds = {_code(native_branch(g)): long_long_guard(g) for g in _guards(_guarded())}
    for native in ('t - a', 'a - 1 - t', 'k + (a)', 't - 1', 'a * a - t', 'a + 1'):
        assert kinds[native] is True, native
    for native in ('float(a * t)', '(a - t) / 12.0', 't - 1 < a'):
        assert kinds[native] is False, native


class _Cell:
    params = ('t', 'a', 'g')


def _collapse(intervals):
    arithmetic = CheckedArithmetic(None, _Cell())
    arithmetic.parameters = {'t', 'a'}
    arithmetic.parameter_intervals = intervals
    node = _guarded()
    return node.with_changes(body=node.body.visit(_ProvenGuards(arithmetic)))


def test_proven_and_outer_operation_guards_are_dropped():
    text = _code(_collapse({'t': (0, 910)}))
    # proven from t's window, or one outer operation whose value is taken as long long
    assert 'm = (t - a)\n' in text
    assert 'self.cell((t - 1), (a - t), ' in text
    assert 'range(1, (a + 1))' in text
    assert 'k = (k + (a))\n' in text
    # an inner operation that is not proven keeps the guard; so do float and division of an
    # unproven product or difference, and a Python-object context (a list element)
    remaining = sorted(_code(native_branch(g)) for g in _guards(_collapse({'t': (0, 910)})))
    assert remaining == sorted(['a - 1 - t', 'a * a - t', 'float(a * t)', '(a - t) / 12.0', 'a - t'])
    assert 'w = (t - 1 < a)\n' in text      # a comparison whose integer side is proven
    # with a window on both names the inner operations are proven too
    remaining = sorted(_code(native_branch(g)) for g in _guards(_collapse({'t': (0, 910), 'a': (0, 11)})))
    assert remaining == []


def _evaluate(node, values):
    """(exact value, True if an intermediate of a dropped guard left the 64-bit range)."""
    if isinstance(node, cst.Integer):
        return node.evaluated_value, False
    if isinstance(node, cst.Name):
        return values[node.value], False
    if isinstance(node, cst.UnaryOperation):
        value, wide = _evaluate(node.expression, values)
        return -value, wide
    left, wide_left = _evaluate(node.left, values)
    right, wide_right = _evaluate(node.right, values)
    value = {cst.Add: lambda: left + right, cst.Subtract: lambda: left - right,
             cst.Multiply: lambda: left * right}[type(node.operator)]()
    return value, wide_left or wide_right or not -LIMIT <= value < LIMIT


def test_dropped_guards_are_exact_or_raise_on_both_paths():
    """Each dropped long long guard: the native evaluation either equals the exact value, or the
    exact value is itself outside the 64-bit range (the slow branch raises there too)."""
    edges = (-LIMIT, -LIMIT + 1, -(1 << 62), -912, -1, 0, 1, 11, 910, (1 << 62), LIMIT - 2, LIMIT - 1)
    collapsed = _collapse({'t': (0, 910)})
    original = {_code(native_branch(g)) for g in _guards(_guarded())}
    remaining = {_code(native_branch(g)) for g in _guards(collapsed)}
    dropped = [cst.parse_expression(text) for text in original - remaining
               if text not in ('float(a * t)', '(a - t) / 12.0', 't - 1 < a')]
    assert dropped
    for expression in dropped:
        for t, a, k in itertools.product(range(0, 911, 130), edges, edges):
            exact, wide = _evaluate(expression, {'t': t, 'a': a, 'k': k})
            final_outside = not -LIMIT <= exact < LIMIT
            # an intermediate overflow in C is only possible where the exact result is outside
            assert not wide or final_outside, (_code(expression), t, a, k)


def test_argument_range_reads_through_a_guard():
    guard = next(g for g in _guards(_guarded()) if _code(native_branch(g)) == 't - 1')
    assert _argument_range(guard, {'t': (1, 911)}, {'t': 'long long'}, 'long long') == (0, 910)
