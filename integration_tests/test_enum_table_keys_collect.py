"""The collect half of ``test_enum_table_keys``: member-keyed table reads through the compiled collect hook at
one and four workers, one reused item per worker (its holds are bound again per policy), equal to live."""
import pytest
import modelx as mx

from test_enum_table_keys import build

pytestmark = pytest.mark.skipif(
    not hasattr(mx.core.space.UserSpace, 'set_collect'), reason='modelx without collect')

CHECK = '''\
import math
from importlib import import_module
from enum_keys_model import build
compiled, live = import_module('exported_cy').mx_model, build(True)
want = live.P.collect().sum.total
for workers in (1, 4):
    compiled._mx_collect_workers = workers
    compiled.P._mx_collect_cache.clear()
    got = compiled.P.collect().sum.total
    assert math.isclose(got, want, rel_tol=1e-10), (workers, got, want)
print('equal')
'''


def test_member_keys_through_collect(tmp_path):
    env, run = build(tmp_path, collect=True)
    (tmp_path / 'check.py').write_text(CHECK, encoding='utf-8')
    done = run('check.py')
    assert done.returncode == 0 and done.stdout.strip() == 'equal', done.stdout[-3000:] + done.stderr[-3000:]
