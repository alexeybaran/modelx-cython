import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from collect import REQUIRED_STEPS, VERIFICATION_BRANCH, collect, expected_scope
from verify import (LEGACY_HELPER_SHA, PYTEST_VERSION, SOURCE_COVERAGE, SOURCES, SUITES, expected_lanes, junit_counts, junit_identity,
                    read_product_pins, runtime, write_json)


PINS = dict(zip((*SOURCES, 'verifier'), (character * 40 for character in 'abcd')))
RUN = {'id': 123, 'run_attempt': 2, 'head_sha': PINS['verifier'], 'status': 'completed',
       'conclusion': 'success', 'event': 'push', 'head_branch': VERIFICATION_BRANCH}
COVERAGE = {**json.loads(SOURCE_COVERAGE.read_text(encoding='utf-8')), 'compiler_pin': PINS['compiler']}


class VerifierTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.artifacts = Path(temporary.name)
        jobs = []
        for lane, (runner, version, profile) in expected_lanes().items():
            has_lifelib = profile != 'compatibility-floor'
            legacy = profile == 'legacy-helper'
            info = {'implementation': 'cpython', 'version': version.rstrip('t') + '.8',
                    'platform': {'ubuntu': 'linux', 'macos': 'darwin', 'windows': 'win32'}[runner.split('-')[0]],
                    'free_threaded': version.endswith('t'), 'gil_enabled': not version.endswith('t')}
            versions = {'compiler': '0.1.1', 'modelx': '0.34.0', 'Cython': '3.2.4' if profile == 'cython-floor' else '3.3.1',
                        'pandas': '3.1.1' if has_lifelib else '2.2.2', 'numpy': '2.5.4', 'pytest': PYTEST_VERSION,
                        'libcst': '1.9.0', 'openpyxl': '3.1.5', 'psutil': '7.0.0'}
            if has_lifelib:
                versions['lifelib'] = '0.17.3'
            folder = self.artifacts / f'junit-{lane}-attempt-2'
            write_json(folder / 'environment.json', {'lane': lane, 'run_id': 123, 'run_attempt': 2, 'pins': PINS,
                       'runtime': info, 'dependencies': {'versions': versions, 'requires_python': {
                           'compiler': '>=3.9', 'modelx': '>=3.7', 'lifelib': '>=3.10'},
                           'modelx_helpers_present': legacy, 'lifelib_installed': has_lifelib,
                           'legacy_helper_commit': LEGACY_HELPER_SHA if legacy else None}})
            for suite in SUITES:
                xml = folder / (suite + '.xml')
                all_cases, modules, excluded_ids, opt_in = expected_scope(suite, profile, COVERAGE['suites'][suite])
                selected = sorted(all_cases)
                report = ElementTree.Element('testsuite', tests=str(len(selected)), failures='0', errors='0',
                                             skipped='0' if legacy else '1')
                for index, nodeid in enumerate(selected):
                    classname, name = junit_identity(nodeid)
                    case = ElementTree.SubElement(report, 'testcase', classname=classname, name=name)
                    if index == 1 and not legacy:
                        ElementTree.SubElement(case, 'skipped', message='known platform')
                ElementTree.ElementTree(report).write(xml, encoding='utf-8')
                write_json(folder / (suite + '.json'), {'suite': suite, 'exit_code': 0, 'runtime_before': info,
                           'runtime_after': info, 'counts': junit_counts(xml), 'excluded_modules': modules,
                           'excluded_ids': excluded_ids, 'unmatched_exclusions': [], 'suite_deselections': opt_in,
                           'selected_ids': selected})
            jobs.append({'id': 101 + len(jobs), 'name': lane, 'status': 'completed', 'conclusion': 'success',
                         'run_id': 123, 'run_attempt': 2,
                         'steps': [{'name': name, 'status': 'completed', 'conclusion': 'success'}
                                   for name in REQUIRED_STEPS]})
        self.jobs = {'total_count': len(jobs), 'jobs': jobs}

    def result(self, jobs=None, run=None):
        return collect(run or RUN, jobs or self.jobs, self.artifacts, 123, 2, PINS, COVERAGE)

    def test_accepts_all_exact_jobs_and_counts_scoped_floor(self):
        result = self.result()
        self.assertTrue(result['passing'], result['errors'])
        self.assertEqual(len(result['jobs']), 13)
        floor = result['jobs']['ubuntu-24.04-py3.9-compatibility-floor']
        self.assertEqual(floor['scope'], 'compiler and modelx compatibility')
        self.assertEqual(floor['suites']['unit']['counts']['passed'], 829)
        self.assertEqual(floor['suites']['unit']['counts']['skipped'], 1)
        self.assertEqual(result['jobs']['ubuntu-latest-py3.14t-legacy-helper']['scope'], 'legacy helper compatibility')
        ordinary = result['jobs']['ubuntu-latest-py3.14t-release']['suites']
        self.assertEqual(len(ordinary['integration']['suite_deselections']), 7)
        self.assertEqual(ordinary['integration']['counts']['tests'], 740)

    def test_cancelled_skipped_missing_and_wrong_attempt_are_rejected(self):
        for conclusion in ('cancelled', 'skipped', 'failure', None):
            jobs = copy.deepcopy(self.jobs)
            jobs['jobs'][0]['conclusion'] = conclusion
            self.assertFalse(self.result(jobs)['passing'])
        jobs = copy.deepcopy(self.jobs)
        jobs['jobs'].pop()
        self.assertFalse(self.result(jobs)['passing'])
        self.assertFalse(self.result(run={**RUN, 'run_attempt': 1})['passing'])

    def test_successful_job_with_skipped_suite_is_rejected(self):
        unit = next(step for step in self.jobs['jobs'][0]['steps'] if step['name'] == 'Unit tests')
        unit['conclusion'] = 'skipped'
        self.assertFalse(self.result()['passing'])

    def test_wrong_event_branch_and_malformed_manifest_are_rejected(self):
        self.assertFalse(self.result(run={**RUN, 'event': 'workflow_dispatch'})['passing'])
        self.assertFalse(self.result(run={**RUN, 'head_branch': 'main'})['passing'])
        manifest = self.artifacts / 'pins.json'
        expected = {name: PINS[name] for name in SOURCES}
        write_json(manifest, expected)
        self.assertEqual(read_product_pins(manifest), expected)
        for incomplete in ({**expected, 'compiler': 'HEAD'}, {**expected, 'other': PINS['verifier']}):
            write_json(manifest, incomplete)
            with self.assertRaises(ValueError):
                read_product_pins(manifest)

    def test_wrong_gil_wrong_pins_and_wrong_floors_are_rejected(self):
        options = [('ubuntu-latest-py3.14t-release', lambda env: env['runtime'].update(gil_enabled=True)),
                   ('macos-latest-py3.13-release', lambda env: env['pins'].update(modelx='z' * 40)),
                   ('windows-latest-py3.14-cython-floor', lambda env: env['dependencies']['versions'].update(Cython='3.3.1')),
                   ('windows-latest-py3.14-release', lambda env: env['dependencies']['versions'].update(pytest='9.2.0')),
                   ('ubuntu-24.04-py3.9-compatibility-floor', lambda env: env['dependencies']['versions'].update(pandas='2.3.1'))]
        for lane, operation in options:
            with self.subTest(lane=lane):
                path = self.artifacts / f'junit-{lane}-attempt-2' / 'environment.json'
                original = json.loads(path.read_text(encoding='utf-8'))
                changed = copy.deepcopy(original)
                operation(changed)
                write_json(path, changed)
                self.assertFalse(self.result()['passing'])
                write_json(path, original)

    def test_xml_tampering_and_unapproved_exclusion_are_rejected(self):
        folder = self.artifacts / 'junit-ubuntu-latest-py3.13-release-attempt-2'
        xml = folder / 'unit.xml'
        original = xml.read_text(encoding='utf-8')
        xml.write_text(original + '\n', encoding='utf-8')
        self.assertFalse(self.result()['passing'])
        xml.write_text(original, encoding='utf-8')
        receipt = folder / 'unit.json'
        altered = json.loads(receipt.read_text(encoding='utf-8'))
        altered['excluded_ids']['modelx_cython/tests/test_ready.py::test_required'] = 'modelx_helpers'
        write_json(receipt, altered)
        self.assertFalse(self.result()['passing'])

    def test_free_threaded_runtime_with_enabled_gil_is_rejected(self):
        with patch('verify.sys.version_info', (3, 14, 8)), patch('verify.sysconfig.get_config_var', return_value=1), \
                patch('verify.sys._is_gil_enabled', return_value=True, create=True):
            with self.assertRaises(ValueError):
                runtime('3.14t')

    def test_missing_legacy_id_and_unreported_deselection_are_rejected(self):
        receipt = self.artifacts / 'junit-ubuntu-latest-py3.14t-legacy-helper-attempt-2' / 'unit.json'
        altered = json.loads(receipt.read_text(encoding='utf-8'))
        original = altered['selected_ids'][0]
        altered['selected_ids'][0] = original.rsplit('::', 1)[0] + '::test_other'
        write_json(receipt, altered)
        self.assertFalse(self.result()['passing'])
        altered['selected_ids'][0] = original
        write_json(receipt, altered)
        receipt = self.artifacts / 'junit-ubuntu-latest-py3.13-release-attempt-2' / 'unit.json'
        altered = json.loads(receipt.read_text(encoding='utf-8'))
        altered['suite_deselections'] = ['modelx_cython/tests/test_ready.py::test_required']
        write_json(receipt, altered)
        self.assertFalse(self.result()['passing'])

    def test_partial_regular_suite_and_mismatched_xml_identity_are_rejected(self):
        folder = self.artifacts / 'junit-ubuntu-latest-py3.13-release-attempt-2'
        receipt = folder / 'unit.json'
        xml = folder / 'unit.xml'
        original = json.loads(receipt.read_text(encoding='utf-8'))
        original_xml = xml.read_bytes()
        report = ElementTree.parse(xml)
        report.getroot().find('testcase').set('name', 'different_test')
        report.write(xml, encoding='utf-8')
        write_json(receipt, {**original, 'counts': junit_counts(xml)})
        self.assertFalse(self.result()['passing'])
        xml.write_bytes(original_xml)
        write_json(receipt, {**original, 'selected_ids': original['selected_ids'][:2]})
        self.assertFalse(self.result()['passing'])


if __name__ == '__main__':
    unittest.main()
