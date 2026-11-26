"""Reductions over tuples and lists of numbers, and ``zip`` of them, read as C values when every item is an exact
float (or an int within 64 bits) and the element cannot run code; otherwise the loop as written over the objects.
Type, repr (the bits of a float, the sign of zero, nan) and exceptions are live's."""
import re

from reduction_cases import assert_equal_live, outcomes

CELLS = {
    'vals': 'lambda t: tuple(0.1 * (t + k) - 0.3 for k in range(4))',
    'more': 'lambda t: [1e16, -1e16, 0.7, 2.5 * t, -0.0][: t + 2]',
    'weights': 'lambda t: (0.5, -0.0, 1e16, 3.0)[: t]',
    'ints': 'lambda t: [t - 2, 5, 2 ** 62, 2 ** 62, -7][: t + 1]',
    'mixed': 'lambda t: (1.5, t, -0.0)',
    'flags': 'lambda t: (True, False, t > 1)',
    'nans': 'lambda t: (float("nan"), 1.0, -2.0 * t)',
    'zip_sum': 'lambda t: sum(a * b for a, b in zip(vals(t), weights(t)) if a != 0.0)',
    'zip_three': 'lambda t: sum(a * (1.0 - b) * c for a, b, c in zip(vals(t), weights(t), more(t)) if a != 0.0)',
    'zip_mixed': 'lambda t: sum(a * b for a, b in zip(mixed(t), vals(t)))',
    'float_sum': 'lambda t: sum(a for a in more(t))',
    'float_list_comp': 'lambda t: sum([a * 2.0 for a in more(t)])',
    'int_sum': 'lambda t: sum(a for a in ints(t))',
    'int_square': 'lambda t: sum(a * a for a in ints(t))',
    'int_max': 'lambda t: max(a for a in ints(t) if a < t - 2)',
    'int_against_float': 'lambda t: max(a for a in ints(t) if a <= 0.5 * t)',
    'float_min': 'lambda t: min(a for a in vals(t) if a > 0.0)',
    'nan_max': 'lambda t: max(a for a in nans(t))',
    'nan_min': 'lambda t: min(a for a in nans(t))',
    'mixed_sum': 'lambda t: sum(x for x in mixed(t))',
    'flag_sum': 'lambda t: sum(b for b in flags(t))',
    'flag_max': 'lambda t: max(b for b in flags(t))',
    'some': 'lambda t: any(a > 0.25 for a in vals(t))',
    'every': 'lambda t: all(a != 0.0 for a in weights(t))',
    'assigned': 'def assigned(t):\n    v = vals(t)\n    total = sum(a * a for a in v if a < 0.0)\n    return total + 1.0\n',
}
CHECKED = [name for name in CELLS if name not in ('vals', 'more', 'weights', 'ints', 'mixed', 'flags', 'nans')]


def test_tuples_and_lists_of_numbers(tmp_path):
    live = assert_equal_live(tmp_path, CELLS, [name for name in CHECKED if name != 'int_square'], range(5))
    assert live['int_max', 0] == ('raise', 'ValueError', live['int_max', 0][2])
    assert live['int_sum', 4][1] == repr(2 + 5 + 2 ** 63 - 7)
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    typed = {name for name in CHECKED if re.search(rf'def _f_{name}\(.*?(?=\n    @|\Z)', source, re.S)
             and '.seq_f64(' in _body(source, name) or '.seq_i64(' in _body(source, name)}
    assert typed >= {'zip_sum', 'zip_three', 'zip_mixed', 'float_sum', 'float_list_comp', 'int_sum', 'int_square',
                     'int_max', 'float_min', 'nan_max', 'nan_min', 'mixed_sum', 'flag_sum', 'some', 'every',
                     'assigned'}, typed
    # an int compared with a float is compared exactly in Python, not as C compares: no integer variant
    assert '.seq_i64(' not in _body(source, 'int_against_float')


def test_integer_items_beyond_64_bits_raise_naming_the_cell(tmp_path):
    """``a * a`` of C integer items is integer arithmetic like any other: past 64 bits the call raises, naming the
    Cell, where live gives a Python int; within 64 bits the values are live's. The sum itself never wraps: past 64
    bits it continues on Python ints as the built-in does (``int_sum`` above)."""
    result = outcomes(tmp_path, CELLS, ['int_square'], range(5))
    live, compiled = result['exported'], result['exported_cy']
    for key, value in live.items():
        if value[0] == 'int' and abs(int(value[1])) >= 2 ** 63:
            assert compiled[key] == ('raise', 'OverflowError', 'Projection.int_square: value too large '
                                     '(integer arithmetic beyond 64 bits)'), (key, compiled[key])
        else:
            assert compiled[key] == value, (key, value, compiled[key])
    assert sum(value[0] == 'int' and abs(int(value[1])) < 2 ** 63 for value in live.values()) >= 2


def _body(source, cell):
    start = source.find(f'def _f_{cell}(')
    end = source.find('\n    @', start)
    text = source[start:end if end > 0 else None]
    called = re.search(r'\b(_mx_\w*formula_\d+)\(', text)
    if called:
        begin = source.index(f'\ndef {called.group(1)}(')
        stop = re.search(r'\n(def |class |@|_mx_)', source[begin + 1:])
        text += source[begin:begin + 1 + stop.start()] if stop else source[begin:]
    return text
