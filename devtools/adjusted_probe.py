"""Fresh-process adjusted-model validation/timing; no compiler-specific API."""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time


def main(args):
    import psutil
    process = psutil.Process()
    if args.cpu is not None:
        process.cpu_affinity([args.cpu] if args.workers == 1 else [0, 2, 4, 6])
    import pandas as pd
    binaries = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (args.build / 'exported_cy').rglob('*.pyd')}
    if not args.validate:
        receipt = json.loads((args.build / 'validated.json').read_text())
        assert receipt['binaries'] == binaries, 'stale validation'
    started = time.perf_counter()
    sys.path.insert(0, str(args.build))
    gil = getattr(sys, '_is_gil_enabled', lambda: True)
    before = gil()
    model = importlib.import_module('exported_cy').mx_model
    assert gil() == before
    loaded = time.perf_counter()
    reports = ('result_cf', 'result_pols', 'result_av')
    if args.product == 'va':
        reports += ('result_bases',)
    # Established oracle for the policy subtotal Cell and ordered policy fold.
    expected = {'myga': -65357750.19608184, 'va': -5633271.794869637}[args.product]
    keys = tuple(model.policy_keys)
    assert len(keys) == 2500
    if args.validate:
        reference = importlib.import_module('exported').mx_model
        for key in keys:
            p, q = model.Projection[key], reference.Projection[key]
            assert math.isclose(p.policy_cashflows(), q.policy_cashflows(),
                                rel_tol=1e-11, abs_tol=1e-8), key
            for report in reports:
                pd.testing.assert_frame_equal(getattr(p, report)(), getattr(q, report)(),
                                              rtol=1e-10, atol=1e-8)
            del model.Projection[key]
            del reference.Projection[key]
        result = model.Projection.collect().sum.cashflows
        assert math.isclose(result, expected, rel_tol=0, abs_tol=1e-6), result
        write(args.build / 'validated.json', {'status': 'PASS', 'policies': len(keys),
              'reports': reports, 'binaries': binaries, 'sum': result})
        return
    if args.inputs:
        module = ('fixed_deferred_annuity.adjusted_myga' if args.product == 'myga'
                  else 'variable_annuity.adjusted_va')
        adapter = importlib.import_module('lifelib.libraries.uslib.products.' + module)
        payload = adapter.load_inputs(args.inputs)
        rows = {row['point_id']: row for row in payload['policies']}
        assert tuple(rows) == keys
        if type(model.Projection.policy_rows) is dict:
            adapt = lambda value: value
        else:
            from modelx_helpers.immutable_inputs import freeze
            adapt = lambda value: freeze(value) if type(value) is dict else value
        model.Projection.policy_rows = adapt(rows)
        for name, value in payload['references'].items():
            setattr(model.Projection, name, adapt(value))
    native_inputs = getattr(model.Projection, '_mx_typed_inputs', None)
    if native_inputs is not None:
        native_inputs()
    prepared = time.perf_counter()
    if args.profile:
        import cProfile
        profile = cProfile.Profile()
        profile.enable()
    start = time.perf_counter()
    if args.mode == 'collect':
        result = model.Projection.collect().sum.cashflows
    elif args.mode == 'direct':
        result = (model.Portfolio.direct_total() if args.workers == 1 else
                  model.Portfolio.direct_total_parallel())
    else:
        result = 0.0
        frames = []
        for key in keys:
            p = model.Projection[key]
            for report in reports:
                frames.append(getattr(p, report)())
            result += p.policy_cashflows()
    end = time.perf_counter()
    if args.profile:
        profile.disable()
        profile.dump_stats(str(args.profile))
    assert math.isclose(result, expected, rel_tol=0, abs_tol=1e-6), result
    memory = process.memory_info()
    retained = 0
    for key in keys:
        try:
            del model.Projection[key]
        except KeyError:
            pass
        else:
            retained += 1
    assert retained == (0 if args.mode == 'collect' else len(keys)), retained
    write(args.output, {'mode': args.mode, 'product': args.product,
          'calculation_s': end - start, 'import_through_result_s': end - started,
          'import_s': loaded - started, 'preparation_s': prepared - started,
          'csv_inputs_reloaded': bool(args.inputs), 'workers': args.workers,
          'gil_enabled': gil(), 'peak_memory_bytes': getattr(memory, 'peak_wset', None),
          'result': result, 'policies': len(keys), 'binaries': binaries})


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--product', choices=['myga', 'va'], required=True)
    parser.add_argument('--mode', choices=['direct', 'collect', 'reports'], default='collect')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    args.build = args.build.resolve()
    sys.setrecursionlimit(20000)
    threading.stack_size(0x10000000 - 1)
    errors = []
    def run():
        try:
            main(args)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
