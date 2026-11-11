"""Check the compiled scalar collect route against the MYGA benchmark inputs."""

import importlib
import math
from pathlib import Path
import sys
import threading

from modelx_helpers.immutable_inputs import freeze
from lifelib.libraries.uslib.products.fixed_deferred_annuity import adjusted_myga


def check(build, inputs):
    sys.path.insert(0, str(build))
    model = importlib.import_module('exported_cy').mx_model
    runtime = importlib.import_module('exported_cy._mx_sys')
    projection = model.Projection
    data = adjusted_myga.load_inputs(inputs)
    rows = {row['point_id']: row for row in data['policies']}
    projection.policy_rows = freeze(rows)
    for name, value in data['references'].items():
        setattr(projection, name, freeze(value) if type(value) is dict else value)
    keys = tuple(rows)
    assert keys == tuple(model.policy_keys)
    check_coords = runtime._mx_collect_native_coords
    decisions = []

    def record(coords):
        answer = check_coords(coords)
        decisions.append(answer)
        return answer

    runtime._mx_collect_native_coords = record
    result = projection.collect()
    assert decisions == [True], decisions
    assert projection.collect() is result
    assert decisions == [True], decisions
    assert math.isclose(result.sum.cashflows, -65357750.19608184, rel_tol=0, abs_tol=1e-6)

    class PolicyKey(int):
        pass

    small = keys[:4]
    projection.policy_keys = tuple(PolicyKey(key) for key in small)
    fallback = projection.collect().sum.cashflows
    assert decisions[-1] is False
    assert fallback == projection._mx_evaluate_serial_collect(small).sum.cashflows
    assert decisions[-1] is True

    # Replaced References still run through the original compiled Cell.
    original = projection.lookup
    calls = []

    def replacement(*args):
        calls.append(args[1])
        return original(*args)

    projection.lookup = replacement
    projection.policy_keys = small[::-1]
    rebound = projection.collect().sum.cashflows
    assert decisions[-1] is True and calls, (decisions, calls)
    projection.lookup = original
    assert rebound == projection._mx_evaluate_serial_collect(small[::-1]).sum.cashflows

    # A successful first policy followed by a missing one exercises failure and retry.
    missing = max(keys) + 1
    projection.policy_keys = (small[0], missing)
    for _ in range(2):
        try:
            projection.collect()
        except KeyError as error:
            assert error.args == (missing,), error
        else:
            raise AssertionError('missing policy should fail')
    assert decisions[-2:] == [True, True]
    projection.policy_rows = freeze({**rows, missing: rows[small[1]]})
    retried = projection.collect().sum.cashflows
    direct = projection[small[0]].policy_cashflows() + projection[missing].policy_cashflows()
    assert retried == direct
    del projection[small[0]]
    del projection[missing]
    assert not sys._is_gil_enabled()
    print('PASS native route, aggregate cache, coordinate fallback, Reference rebinding, failure and retry')


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    threading.stack_size(0x10000000 - 1)
    check(*(Path(arg).resolve() for arg in sys.argv[1:]))
