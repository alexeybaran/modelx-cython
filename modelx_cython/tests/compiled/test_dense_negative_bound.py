"""D15: the public one-axis range check is overflow-free.

A signed ``key - lower`` overflows for a key far below a positive lower bound (or above a negative
one), so the offset is taken in unsigned arithmetic, which wraps by definition:
``cast(ulonglong, key) - cast(ulonglong, lower) < span``, and ``cast(ulonglong, key) < span`` for
lower 0. The check is compiled and run here at keys around and far from the bounds.
"""
import re

import pytest
import libcst as cst

from modelx_cython.cell_entries import DenseAxis, DenseEntry, EntryModule
from modelx_cython.config import CompilerPolicy
from modelx_cython.entry_pipeline import lower_entries


def _check_text(lower, stop):
    source = f'''class Policy:
    def value(self, t: _mx_cy.longlong) -> _mx_cy.double:
        if {lower} <= t < {stop}:
            return self._f_value(t)
        else:
            raise IndexError(t)
'''
    layout = DenseEntry('Policy', '_f_value', (DenseAxis('t', lower, stop, 'long long'),), False,
                        guard=cst.parse_expression(f'{lower} <= t < {stop}'))
    policy = CompilerPolicy.resolve({'automatic_optimizations': False, 'dense_one_axis': True})
    code = lower_entries({'m': EntryModule(cst.parse_module(source))},
                         {'m': {'entry': layout}}, {}, policy)['m'].syntax.code
    line, = (line.strip() for line in code.splitlines() if '_mx_cy.cast' in line)
    return line.removeprefix('if ').removesuffix(':')


@pytest.mark.parametrize('lower,stop,expected', [
    (0, 4, '_mx_cy.cast(_mx_cy.ulonglong, t) < 4'),
    (-9, 6, '_mx_cy.cast(_mx_cy.ulonglong, t) - _mx_cy.cast(_mx_cy.ulonglong, -9) < 15'),
    (3, 8, '_mx_cy.cast(_mx_cy.ulonglong, t) - _mx_cy.cast(_mx_cy.ulonglong, 3) < 5'),
])
def test_range_check_subtracts_in_unsigned_arithmetic(lower, stop, expected):
    assert _check_text(lower, stop) == expected


@pytest.mark.parametrize('lower,stop', [(-9, 6), (0, 4), (3, 8)])
def test_range_check_compiled(lower, stop):
    cython = pytest.importorskip('cython')
    # the emitted casts as Cython's own syntax: an inline module has no ``cython`` import
    check = re.sub(r'_mx_cy\.cast\(_mx_cy\.ulonglong, ([^)]*)\)', r'(<unsigned long long>(\1))',
                   _check_text(lower, stop))
    keys = [*range(lower - 4, stop + 4), -2 ** 63, -2 ** 63 + 1, 2 ** 63 - 1, 2 ** 63 - 2]
    body = ('cdef long long t\nresult = []\nfor t in keys:\n'
            '    result.append(bool(%s))\nreturn result\n') % check
    got = cython.inline(body, locals={'keys': keys}, quiet=True)
    assert got == [lower <= t < stop for t in keys]
