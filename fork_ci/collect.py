"""Validate one GitHub Actions attempt from its API snapshots and JUnit artifacts."""

import argparse
import csv
import json
from pathlib import Path

from verify import (EXCLUSIONS, LEGACY_HELPER_SHA, PYTEST_VERSION, SHA, SOURCE_COVERAGE, SOURCES, SUITES, expected_lanes,
                    junit_counts, write_json)


REQUIRED_STEPS = ('Read reviewed pins', 'Verify pins and install', 'Unit tests', 'Integration tests',
                  'Upload test reports')
VERIFICATION_BRANCH = 'verify/round3-release'
OPTIONAL_COVERAGE = Path(__file__).resolve().parents[1] / 'OPTIONAL-COVERAGE.tsv'
ENUM_EXCLUSION_PREFIX = 'modelx_cython/tests/test_table_admission.py::test_awkward_keys_are_admitted[index2-'
ENUM_EXCLUSION_CURRENT = ENUM_EXCLUSION_PREFIX + '1]'
ENUM_EXCLUSION_PYTHON39 = ENUM_EXCLUSION_PREFIX + 'Sex.M]'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def coverage_ids(suite, lane):
    with OPTIONAL_COVERAGE.open(encoding='utf-8', newline='') as file:
        return {row['test id'] for row in csv.DictReader(file, delimiter='\t')
                if row['suite'] == suite and row['evidence lane'] == lane}


def check_runtime(info, runner, interpreter):
    platform = {'ubuntu': 'linux', 'macos': 'darwin', 'windows': 'win32'}[runner.split('-')[0]]
    require(info['platform'] == platform, 'recorded runtime platform does not match the runner')
    require(info['implementation'] == 'cpython'
            and info['version'].startswith(interpreter.rstrip('t') + '.'), 'incorrect interpreter')
    expected = interpreter.endswith('t')
    require(info['free_threaded'] is expected and info['gil_enabled'] is (not expected),
            'incorrect free-threaded ABI or GIL state')


def expected_scope(suite, profile, source):
    allowed = read_json(EXCLUSIONS)
    dependencies = ('modelx_helpers', 'lifelib') if profile == 'compatibility-floor' else ('modelx_helpers',)
    prefix = SUITES[suite] + '/'
    modules = {path: dependency for dependency in dependencies for path in allowed[dependency]['modules']
               if path.startswith(prefix)}
    targets = {path: dependency for dependency in dependencies for path in allowed[dependency]['ids']
               if path.startswith(prefix)}
    def reason(case):
        return next((dependency for pattern, dependency in targets.items()
                     if case == pattern or case.startswith(pattern + '[')), None)

    full = set(source['selected'])
    require(full and len(full) == len(source['selected']), f'{suite} source inventory is invalid')
    cases = {case: reason(case) for case in full if reason(case)}
    under_modules = {case for case in full if case.split('::', 1)[0] in modules}
    if profile == 'legacy-helper':
        expected = coverage_ids(suite, 'legacy helper')
        require(expected == cases.keys() | under_modules,
                f'{suite} legacy inventory differs from source and approved exclusions')
        return expected, {}, {}, []
    expected = full - cases.keys() - under_modules
    opt_in = sorted(case for case in source['opt_in'] if case.split('::', 1)[0] not in modules and not reason(case))
    if profile == 'compatibility-floor' and ENUM_EXCLUSION_CURRENT in cases:
        # Pytest renders the same IntEnum parameter differently on CPython 3.9.
        cases[ENUM_EXCLUSION_PYTHON39] = cases.pop(ENUM_EXCLUSION_CURRENT)
    return expected, modules, cases, opt_in


def check_coverage(receipt, suite, profile, source):
    expected, modules, cases, opt_in = expected_scope(suite, profile, source)
    selected = receipt['selected_ids']
    observed = set(selected)
    require(bool(expected) and observed == expected and len(observed) == len(selected),
            f'{suite} selected IDs differ: absent={sorted(expected - observed)[:10]}, '
            f'extra={sorted(observed - expected)[:10]}')
    require(receipt['excluded_modules'] == modules and receipt['excluded_ids'] == cases
            and receipt['unmatched_exclusions'] == [] and receipt['suite_deselections'] == opt_in,
            f'{suite} exclusions or opt-in deselections differ from the reviewed scope')


def check_job(lane, spec, job, artifact, run_id, attempt, pins, coverage):
    runner, interpreter, profile = spec
    require(job['status'] == 'completed' and job['conclusion'] == 'success',
            f"job {job['status']}, conclusion {job['conclusion']}")
    steps = {step['name']: step for step in job['steps']}
    for name in REQUIRED_STEPS:
        step = steps.get(name, {})
        require(step.get('status') == 'completed' and step.get('conclusion') == 'success',
                f'missing or nonpassing job step: {name}')
    environment = read_json(artifact / 'environment.json')
    require((environment['lane'], environment['run_id'], environment['run_attempt'], environment['pins'])
            == (lane, run_id, attempt, pins), 'job artifact identity differs from the requested attempt or pins')
    check_runtime(environment['runtime'], runner, interpreter)
    dependencies = environment['dependencies']
    versions = dependencies['versions']
    legacy = profile == 'legacy-helper'
    require(dependencies['modelx_helpers_present'] is legacy
            and dependencies['legacy_helper_commit'] == (LEGACY_HELPER_SHA if legacy else None),
            'job helper availability or source pin differs from its profile')
    floor = profile == 'compatibility-floor'
    require(dependencies['lifelib_installed'] is (not floor) and ('lifelib' in versions) is (not floor),
            'lifelib scope differs from the compatibility policy')
    require(set(dependencies['requires_python']) == set(SOURCES)
            and all(dependencies['requires_python'].values()), 'missing dependency Requires-Python evidence')
    require({'compiler', 'modelx', 'Cython', 'pandas', 'numpy', 'pytest', 'libcst', 'openpyxl', 'psutil'} <= versions.keys()
            and all(versions.values()), 'missing installed dependency versions')
    require(versions['pytest'] == PYTEST_VERSION, 'test ID provenance has a different pytest version')
    if profile == 'cython-floor':
        require(versions['Cython'] == '3.2.4', 'incorrect Cython floor version')
    if floor:
        require(versions['pandas'] == '2.2.2', 'incorrect pandas floor version')

    suites = {}
    for suite in SUITES:
        receipt = read_json(artifact / (suite + '.json'))
        require(receipt['suite'] == suite and receipt['exit_code'] == 0, f'{suite} did not complete successfully')
        for field in ('runtime_before', 'runtime_after'):
            check_runtime(receipt[field], runner, interpreter)
            require(receipt[field] == environment['runtime'], f'{suite} ran under a different runtime')
        check_coverage(receipt, suite, profile, coverage['suites'][suite])
        counts = junit_counts(artifact / (suite + '.xml'), receipt['selected_ids'])
        require(counts == receipt['counts'] and counts['passed'] > 0
                and counts['errors'] == 0 and counts['failures'] == 0,
                f'{suite} JUnit evidence is invalid or nonpassing')
        if legacy:
            require(counts['skipped'] == 0, f'{suite} compatibility suite skipped requested cases')
        suites[suite] = receipt
    scope = ('compiler and modelx compatibility' if floor else 'legacy helper compatibility'
             if legacy else 'helper-free compiler and lifelib')
    return {'job_id': job['id'], 'conclusion': job['conclusion'], 'environment': environment, 'suites': suites,
            'scope': scope}


def collect(run, pages, artifacts, expected_id, attempt, pins, coverage):
    if not isinstance(pages, list):
        pages = [pages]
    jobs = [job for page in pages for job in page['jobs']]
    matrix = expected_lanes()
    by_name = {job['name']: job for job in jobs}
    errors = []
    if coverage['compiler_pin'] != pins['compiler'] or coverage['pytest'] != PYTEST_VERSION:
        errors.append('source coverage comes from a different compiler or pytest pin')
    if ((run['id'], run['run_attempt'], run['head_sha'], run['status'], run['conclusion'],
         run['event'], run['head_branch'])
            != (expected_id, attempt, pins['verifier'], 'completed', 'success', 'push', VERIFICATION_BRANCH)):
        errors.append('run is a different attempt, wrong verifier revision, or has a nonpassing conclusion')
    if len(jobs) != len(by_name) or set(by_name) != set(matrix):
        errors.append('unexpected, missing or duplicate matrix jobs')
    if any(page.get('total_count', len(jobs)) != len(jobs) for page in pages):
        errors.append('the jobs snapshot is incomplete')
    results = {}
    for lane, spec in matrix.items():
        job = by_name.get(lane)
        if job is None:
            errors.append(f'{lane}: absent matrix job')
            continue
        try:
            require(job.get('run_id') == expected_id, 'job comes from a different run')
            if 'run_attempt' in job:
                require(job['run_attempt'] == attempt, 'job comes from a different attempt')
            directory = artifacts / f'junit-{lane}-attempt-{attempt}'
            results[lane] = check_job(lane, spec, job, directory, expected_id, attempt, pins, coverage)
        except (OSError, KeyError, ValueError, TypeError, SyntaxError) as error:
            errors.append(f'{lane}: {error}')
            results[lane] = {'job_id': job.get('id'), 'conclusion': job.get('conclusion'), 'verified': False}
    return {'run_id': expected_id, 'run_attempt': attempt, 'pins': pins, 'passing': not errors,
            'errors': errors, 'jobs': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-json', type=Path, required=True)
    parser.add_argument('--jobs-json', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--run-id', type=int, required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    for name in (*SOURCES, 'verifier'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    pins = {name: getattr(args, name).lower() for name in (*SOURCES, 'verifier')}
    if not all(SHA.fullmatch(sha) for sha in pins.values()):
        parser.error('all four pins require full Git commit SHAs')
    report = collect(read_json(args.run_json), read_json(args.jobs_json), args.artifacts,
                     args.run_id, args.attempt, pins, read_json(SOURCE_COVERAGE))
    write_json(args.output, report)
    print(f"{'PASS' if report['passing'] else 'FAIL'}: {len(report['jobs'])} matrix jobs")
    for error in report['errors']:
        print(error)
    return 0 if report['passing'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
