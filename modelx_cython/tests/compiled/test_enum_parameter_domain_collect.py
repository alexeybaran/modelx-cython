"""The collect half of ``test_enum_parameter_domain``: the enum domain on a model built from a collect sample,
evaluated through the compiled collect hook at one and four workers."""
import os
import subprocess
import sys
import textwrap
from http import HTTPStatus

import pytest
import modelx as mx

from test_enum_parameter_domain import PROBE, build

pytestmark = pytest.mark.skipif(
    not hasattr(mx.core.space.UserSpace, 'set_collect'), reason='modelx without collect')


def test_enum_parameter_domain_through_collect(tmp_path):
    rate = build(tmp_path, collect=True)
    top = max(int(member) for member in HTTPStatus)
    assert rate['dense_logical_domain'] == [[0, top + 1]], rate
    probe = PROBE + textwrap.dedent('''\
        total = sum(0.5 * 200 + 0.5 * 201 + 2 * k for k in (1, 2, 3))
        for workers in (1, 4):
            m._mx_collect_workers = workers
            m.Projection._mx_collect_cache.clear()
            assert m.Projection.collect().sum.total == total, workers
        ''')
    done = subprocess.run([sys.executable, '-B', '-c', probe], cwd=tmp_path, env=dict(os.environ),
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
