"""A private numeric cache is sized from the evaluations one receiver makes, not from its parameters' range.

Sizing the table from the declared or sampled range (twice its entries, a power of two) cost VA_US_S 1.38%
instructions per policy with item allocation, and MYGA SEC 17%: a Cell over a wide range that one policy
evaluates a few times got a table for the whole range, allocated with every item. The tree's choice is kept:
twice the formula evaluations one receiver made in the sample, a power of two, at most MAX_PRIMARY_ENTRIES,
whatever the range; no evaluation observed gives no inferred capacity.
"""
from types import SimpleNamespace

import pytest

from modelx_cython.automatic import MAX_PRIMARY_ENTRIES, primary_capacity
from modelx_cython.cell_entries import DenseAxis


def _cell(evaluations, *sizes):
    """A Cell with integer parameters over ``sizes`` (the declared or sampled range) and ``evaluations``
    formula evaluations per receiver in the sample."""
    axes = tuple(DenseAxis(f'p{i}', 0, size, 'long long') for i, size in enumerate(sizes))
    return SimpleNamespace(params=tuple(axis.key for axis in axes), associative_logical_axes=axes,
                           has_typeinfo=lambda: True,
                           _rt=SimpleNamespace(max_evaluations_per_receiver=evaluations))


@pytest.mark.parametrize('evaluations,sizes,capacity', [
    (3, (1000,), 8),                    # a wide range evaluated three times per policy: a small table
    (3, (120, 120), 8),
    (40, (1000,), 128),
    (1, (5000,), 2),
    (5000, (10,), MAX_PRIMARY_ENTRIES),
    (0, (1000,), 0),                    # nothing observed: no inferred capacity
])
def test_capacity_follows_the_evaluations_per_receiver_not_the_range(evaluations, sizes, capacity):
    assert primary_capacity(_cell(evaluations, *sizes)) == capacity
