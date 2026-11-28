"""Small installed-wheel example for ordinary Direct and Collect models."""

from pathlib import Path
import sys

import modelx as mx


collect = sys.argv[1] == 'collect'
model = mx.new_model('Portfolio')
projection = model.new_space('Projection', formula=lambda policy_id, scenario: None)
projection.policy_ids = (1, 2, 3)


@mx.defcells(projection)
def cashflow(month):
    return 1.0 * policy_id * scenario * (month + 1)


@mx.defcells(projection)
def extra(month):
    return cashflow(month) + 1.0


def outputs():
    return {
        'individual': {'cashflow': {month: cashflow(month) for month in range(4)}},
        'sum': {'pv': cashflow(0) + cashflow(1) + cashflow(2) + cashflow(3)},
    }


if collect:
    projection.set_collect(over={'policy_id': 'policy_ids'}, outputs=outputs)
    result = projection.collect(3)
    assert result.sum.pv == 180.0
    assert result.individual[2].cashflow(1) == result.individual[2].cashflow[1] == 12.0
assert projection[2, 3].extra(1) == 13.0
mx.export_model(model, 'portfolio_nomx')
model.close()

sample = """\
from portfolio_nomx import mx_model

p = mx_model.Projection
assert p[2, 3].extra(1) == 13.0
for i in (1, 2, 3):
    for month in range(4):
        assert p[i, 3].cashflow(month) == float(i * 3 * (month + 1))
"""
if collect:
    sample += """\
mx_model._mx_collect_workers = 4
direct = p[2, 3]
result = p.collect(3)
assert result.sum.pv == 180.0
assert result.individual[2].cashflow(1) == result.individual[2].cashflow[1] == 12.0
assert p[2, 3] is direct and direct.extra(1) == 13.0
"""
Path('sample.py').write_text(sample, encoding='utf-8')
Path('check.py').write_text('import sys\nassert not sys._is_gil_enabled()\n' +
                            sample.replace('portfolio_nomx import', 'portfolio_nomx_cy import'), encoding='utf-8')
