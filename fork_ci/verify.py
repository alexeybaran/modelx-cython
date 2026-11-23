"""Install pinned sources, run one suite and seal release verification evidence."""

import argparse
from collections import Counter
import hashlib
import importlib
from importlib.machinery import ModuleSpec
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname
from xml.etree import ElementTree


SOURCES = ('compiler', 'modelx', 'lifelib')
SUITES = {'unit': 'modelx_cython/tests', 'integration': 'integration_tests'}
EXCLUSIONS = Path(__file__).with_name('exclusions.json')
MANIFEST = Path(__file__).with_name('pins.json')
SOURCE_COVERAGE = Path(__file__).with_name('source-coverage.json')
SHA = re.compile(r'[0-9a-f]{40}')
PYTEST_VERSION = '8.4.2'
RETAINED_PROFILE = 'retained-pandas'
RETAINED_TESTS = {
    'unit': (
        'modelx_cython/tests/compiled/test_audit_runtime.py::test_scope_overrides_aliases_and_arithmetic',
        'modelx_cython/tests/compiled/test_public_object_return_spec.py',
        'modelx_cython/tests/test_object_return_annotation.py',
        'modelx_cython/tests/compiled/test_enum_by_type.py',
        'modelx_cython/tests/compiled/test_native_module_math.py',
        'modelx_cython/tests/compiled/test_native_generator_reductions.py',
    ),
    'integration': (
        'integration_tests/test_held_record_reads.py',
        'integration_tests/test_diag_kr_forms.py',
        'integration_tests/test_prepared_table_integer_keys.py',
        'integration_tests/test_table_cells_api.py',
        'integration_tests/test_table_safety_net.py',
    ),
}


def retained_cases(suite, source):
    matched = set()
    for target in RETAINED_TESTS[suite]:
        hits = {case for case in source if case == target or case.startswith((target + '::', target + '['))}
        if not hits:
            raise ValueError(f'retained-behavior target is absent from the source inventory: {target}')
        matched.update(hits)
    return matched


def expected_lanes():
    lanes = [(runner, python, 'release')
             for runner in ('ubuntu-latest', 'macos-latest', 'windows-latest')
             for python in ('3.13', '3.14', '3.14t')]
    lanes.extend((('ubuntu-24.04', '3.9', 'compatibility-floor'),
                  ('ubuntu-latest', '3.14', 'cython-floor'),
                  ('windows-latest', '3.14', 'cython-floor'),
                  ('ubuntu-latest', '3.14t', RETAINED_PROFILE)))
    return {f'{runner}-py{python}-{profile}': (runner, python, profile)
            for runner, python, profile in lanes}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def command(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def read_product_pins(path=MANIFEST):
    pins = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(pins, dict) or set(pins) != set(SOURCES):
        raise ValueError('pin manifest must contain compiler, modelx and lifelib')
    for name, sha in pins.items():
        if not isinstance(sha, str) or SHA.fullmatch(sha) is None:
            raise ValueError(f'{name} requires a full lower-case commit SHA')
    return pins


def verified_sources(workspace):
    expected = read_product_pins()
    expected['verifier'] = os.environ['GITHUB_SHA'].lower()
    for name, sha in expected.items():
        if SHA.fullmatch(sha) is None:
            raise ValueError(f'{name} requires a full commit SHA')
        root = workspace / name
        if command('git', 'rev-parse', 'HEAD', cwd=root) != sha:
            raise ValueError(f'{name} checkout differs from the requested SHA')
        if command('git', 'status', '--porcelain', '--untracked-files=no', cwd=root):
            raise ValueError(f'{name} checkout contains tracked modifications')
    return expected


def runtime(expected):
    version = '.'.join(map(str, sys.version_info[:2]))
    free_threaded = sysconfig.get_config_var('Py_GIL_DISABLED') in (1, '1')
    gil_enabled = getattr(sys, '_is_gil_enabled', lambda: True)()
    if (version != expected.rstrip('t') or free_threaded != expected.endswith('t')
            or gil_enabled == free_threaded or sys.implementation.name != 'cpython'):
        raise ValueError(f'{expected} runtime mismatch: {sys.version}, ABI={free_threaded}, GIL={gil_enabled}')
    return {'implementation': sys.implementation.name, 'version': platform.python_version(),
            'free_threaded': free_threaded, 'gil_enabled': gil_enabled, 'platform': sys.platform}


def source_requires_python(source):
    with (source / 'pyproject.toml').open(encoding='utf-8') as file:
        metadata = file.read()
    match = re.search(r'^requires-python\s*=\s*[\'\"]([^\'\"]+)[\'\"]', metadata, re.M)
    if match is None:
        raise ValueError('lifelib does not declare Requires-Python in pyproject.toml')
    return match.group(1)


def install(workspace, profile):
    sources = [workspace / 'modelx', workspace / 'compiler']
    if profile != 'compatibility-floor':
        sources.insert(1, workspace / 'lifelib')
    pandas = 'pandas==2.2.2' if profile == 'compatibility-floor' else 'pandas'
    cython = 'Cython==3.2.4' if profile == 'cython-floor' else 'Cython'
    subprocess.run([sys.executable, '-m', 'pip', 'install', *map(str, sources), pandas, cython,
                    f'pytest=={PYTEST_VERSION}', 'openpyxl', 'psutil'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)


def plain_distribution(name, module_name, source):
    distribution = importlib.metadata.distribution(name)
    direct = json.loads(distribution.read_text('direct_url.json') or '{}')
    url = urlsplit(direct.get('url', ''))
    if (url.scheme != 'file' or 'dir_info' not in direct or direct['dir_info'].get('editable')
            or Path(url2pathname(url.path)).resolve() != source.resolve()):
        raise ValueError(f'{name} is not a plain install from its pinned checkout')
    module_file = Path(importlib.import_module(module_name).__file__).resolve()
    if not module_file.is_relative_to(Path(distribution.locate_file('')).resolve()):
        raise ValueError(f'{name} was imported outside the installed distribution')
    return distribution


def dependencies(workspace, profile):
    from packaging.specifiers import SpecifierSet

    version = platform.python_version()
    core = {key: plain_distribution(name, module, workspace / key) for key, name, module in
            (('compiler', 'modelx-cython', 'modelx_cython'), ('modelx', 'modelx', 'modelx'))}
    requires_python = {key: dist.metadata.get('Requires-Python') for key, dist in core.items()}
    for name, requirement in requires_python.items():
        if requirement is None or not SpecifierSet(requirement).contains(version):
            raise ValueError(f'{name} does not declare support for Python {version}')
    available = importlib.util.find_spec('lifelib') is not None
    requires_python['lifelib'] = source_requires_python(workspace / 'lifelib')
    supported = SpecifierSet(requires_python['lifelib']).contains(version)
    if profile == 'compatibility-floor':
        if available or supported:
            raise ValueError('the scoped 3.9 job requires an incompatible, uninstalled lifelib pin')
    elif not supported or not available:
        raise ValueError('required lifelib installation is missing or incompatible')
    else:
        core['lifelib'] = plain_distribution('lifelib', 'lifelib', workspace / 'lifelib')
    legacy = importlib.util.find_spec('modelx_helpers')
    legacy_source = workspace / 'toolkit/parallel/20261113/round3/handoff/reproduction/modelx_helpers'
    helpers = legacy is not None and Path(legacy.origin).resolve() == legacy_source / '__init__.py'
    if not helpers:
        raise ValueError('the pinned legacy test fixture is unavailable')
    versions = {key: dist.version for key, dist in core.items()}
    for name, module in (('Cython', 'Cython'), ('pandas', 'pandas'), ('numpy', 'numpy'),
                         ('pytest', 'pytest'), ('libcst', 'libcst'), ('openpyxl', 'openpyxl'), ('psutil', 'psutil')):
        importlib.import_module(module)
        versions[name] = importlib.metadata.version(name)
    if profile == 'cython-floor' and versions['Cython'] != '3.2.4':
        raise ValueError('Cython floor must use 3.2.4 exactly')
    if profile == 'compatibility-floor' and versions['pandas'] != '2.2.2':
        raise ValueError('Python compatibility floor must use pandas 2.2.2 exactly')
    if versions['pytest'] != PYTEST_VERSION:
        raise ValueError(f'test IDs require pytest {PYTEST_VERSION}')
    return {'versions': versions, 'requires_python': requires_python,
            'lifelib_installed': available, 'modelx_helpers_present': helpers}


def junit_identity(nodeid):
    prefix, bracket, parameter = nodeid.partition('[')
    path, *names = prefix.split('::')
    classname = '.'.join((path.removesuffix('.py').replace('/', '.'), *names[:-1]))
    return classname, names[-1] + bracket + parameter


def junit_counts(path, selected=None):
    tree = ElementTree.parse(path).getroot()
    suites = [tree] if tree.tag == 'testsuite' else tree.findall('testsuite')
    if not suites:
        raise ValueError(f'no JUnit suites in {path.name}')
    counts = {key: sum(int(suite.attrib[key]) for suite in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    cases = [case for suite in suites for case in suite.findall('testcase')]
    actual = {key: sum(case.find(tag) is not None for case in cases)
              for key, tag in (('failures', 'failure'), ('errors', 'error'), ('skipped', 'skipped'))}
    if counts != {'tests': len(cases), **actual}:
        raise ValueError(f'JUnit counters and test cases disagree in {path.name}')
    if selected is not None:
        expected = Counter(map(junit_identity, selected))
        observed = Counter((case.attrib.get('classname'), case.attrib.get('name')) for case in cases)
        if observed != expected:
            raise ValueError(f'JUnit cases differ from the selected IDs in {path.name}: '
                             f'missing={list(expected - observed)[:5]}, extra={list(observed - expected)[:5]}')
    counts['passed'] = len(cases) - sum(actual.values())
    counts['skip_details'] = [
        {'case': case.get('classname', '') + '.' + case.get('name', ''), 'reason': skip.get('message', '')}
        for case in cases for skip in case.findall('skipped')]
    counts['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return counts


def exclusions_for(compiler, suite, profile):
    manifest = json.loads(EXCLUSIONS.read_text(encoding='utf-8'))
    result = {'modules': {}, 'ids': {}}
    for dependency in ('lifelib',) if profile == 'compatibility-floor' else ():
        for kind, targets in manifest[dependency].items():
            for target in targets:
                path = target.split('::', 1)[0]
                if path.startswith(SUITES[suite] + '/'):
                    if not (compiler / path).is_file():
                        raise ValueError(f'exclusion no longer identifies a tracked test: {path}')
                    result[kind][target] = dependency
    return result


class Inventory:
    def __init__(self, targets):
        self.targets = targets
        self.excluded = {}
        self.deselected = []
        self.opt_in = {}
        self.matches = set()
        self.selected = []

    def pytest_collection_modifyitems(self, config, items):
        retained, removed = [], []
        for item in items:
            match = next((target for target in self.targets
                          if item.nodeid == target or item.nodeid.startswith(target + '[')), None)
            if match is None:
                retained.append(item)
            else:
                removed.append(item)
                self.matches.add(match)
                self.excluded[item.nodeid] = self.targets[match]
        items[:] = retained
        if removed:
            config.hook.pytest_deselected(items=removed)

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)
        for item in items:
            markers = [name for name in ('fuzz_full', 'stress') if item.get_closest_marker(name)]
            if markers:
                self.opt_in[item.nodeid] = markers

    def pytest_collection_finish(self, session):
        self.selected = [item.nodeid for item in session.items]


def expose_test_sources(compiler):
    # The wheel excludes test files, but some tests import fixtures from one another.
    name = 'modelx_cython.tests'
    spec = ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(compiler / 'modelx_cython' / 'tests')]
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    importlib.import_module('modelx_cython').tests = package


def run_suite(workspace, suite, profile, interpreter):
    import pytest

    report = workspace / 'reports' / (suite + '.xml')
    compiler = workspace / 'compiler'
    exclusions = exclusions_for(compiler, suite, profile)
    arguments = ['-rfEs', '-q', '--rootdir=' + str(compiler), '--junitxml=' + str(report)]
    if profile == RETAINED_PROFILE:
        arguments.extend(RETAINED_TESTS[suite])
    else:
        arguments.append(SUITES[suite])
        arguments.extend('--ignore=' + path for path in exclusions['modules'])
    inventory = Inventory(exclusions['ids'])
    if sys.platform == 'win32':
        arguments.append('--basetemp=C:/t-' + suite)
    before = runtime(interpreter)
    expose_test_sources(compiler)
    os.chdir(compiler)
    try:
        code = int(pytest.main(arguments, plugins=[inventory]))
    finally:
        os.chdir(workspace)
    after = runtime(interpreter)
    unmatched = sorted(set(inventory.targets) - inventory.matches)
    write_json(report.with_suffix('.json'), {
        'suite': suite, 'exit_code': code, 'runtime_before': before, 'runtime_after': after,
        'counts': junit_counts(report, inventory.selected) if code == 0 else junit_counts(report),
        'excluded_modules': exclusions['modules'],
        'excluded_ids': inventory.excluded, 'unmatched_exclusions': unmatched,
        'suite_deselections': sorted(set(inventory.deselected) - inventory.excluded.keys()),
        'selected_ids': sorted(inventory.selected),
    })
    if unmatched:
        raise ValueError(f'{suite} exclusions did not match collected IDs: {unmatched}')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pins', 'install', *SUITES))
    args = parser.parse_args()
    if args.phase == 'pins':
        product_pins = read_product_pins()
        coverage = json.loads(SOURCE_COVERAGE.read_text(encoding='utf-8'))
        if coverage['compiler_pin'] != product_pins['compiler'] or coverage['pytest'] != PYTEST_VERSION:
            raise ValueError('source coverage was collected with a different compiler or pytest pin')
        for suite in SUITES:
            retained_cases(suite, coverage['suites'][suite]['selected'])
        for name, sha in product_pins.items():
            print(f'{name}={sha}')
        return 0
    workspace = Path(os.environ['GITHUB_WORKSPACE']).resolve()
    runner, interpreter, profile = (os.environ[key] for key in ('CI_RUNNER', 'CI_PYTHON', 'CI_PROFILE'))
    lane = f'{runner}-py{interpreter}-{profile}'
    if lane not in expected_lanes():
        raise ValueError(f'unknown matrix job {lane}')
    pins = verified_sources(workspace)
    runtime(interpreter)
    if args.phase == 'install':
        install(workspace, profile)
    installed = dependencies(workspace, profile)
    observed_runtime = runtime(interpreter)
    environment = {'lane': lane, 'run_id': int(os.environ['GITHUB_RUN_ID']),
                   'run_attempt': int(os.environ['GITHUB_RUN_ATTEMPT']), 'pins': pins,
                   'runtime': observed_runtime, 'dependencies': installed}
    path = workspace / 'reports' / 'environment.json'
    if args.phase == 'install':
        if profile != 'compatibility-floor':
            from regional_contract import verify_regional_contract
            write_json(path.with_name('regional-contract.json'), verify_regional_contract(workspace / 'lifelib'))
        write_json(path, environment)
        return 0
    if json.loads(path.read_text(encoding='utf-8')) != environment:
        raise ValueError('the test environment differs from the verified installation')
    return run_suite(workspace, args.phase, profile, interpreter)


if __name__ == '__main__':
    raise SystemExit(main())
