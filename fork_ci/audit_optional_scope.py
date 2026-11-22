"""Compare the original full test inventory with the final helper-free and legacy scopes."""

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest

from verify import EXCLUSIONS, PYTEST_VERSION, Inventory, SUITES, command, expose_test_sources, write_json


MODULES = {
    'test_audit_runtime': 'Compiler audit of arithmetic, native resources and pickle behavior',
    'test_builtin_reduction_lowering': 'Builtin lowering and compatibility of legacy helper call targets',
    'test_native_scalar_math': 'Dispatch of the registered legacy numeric-math provider',
    'test_provider_contract': 'Native provider metadata, table and record bindings',
    'test_public_object_return_spec': 'Public object returns through the legacy Table and Record providers',
    'test_grouped_report_metadata': 'Legacy helper row, column, grouping and pandas metadata vocabulary',
    'test_raw_scalar_preparation': 'Legacy helper raw-scalar ownership and missing-value vocabulary',
    'test_table_operations': 'Legacy helper table operations and explicit compiled-provider compatibility',
    'test_complete_key_helper_modes': 'Explicit legacy Table key helper and native-provider path',
    'test_ordered_index_modes': 'Explicit legacy Table index ownership and pickle semantics',
    'test_pandas_preparation_modes': 'Explicit legacy helper pandas-to-Table construction and lookup',
    'test_table_keys_provider': 'Explicit legacy Table keys-provider dispatch',
    'test_tracer_table_schema': 'Tracer schema discovery through the explicit legacy Table provider',
    'test_grid_index_d86': 'Runtime indexing of explicit legacy helper Tables',
    'test_table_handles_w1_2': 'Explicit legacy Table handles, sharing, and concurrent reads',
    'test_bound_table_round2_keys': 'Explicit legacy bound Table key dispatch',
    'test_diag_jp_forms': 'Compiler numeric and analysis diagnostics in a legacy Table fixture',
    'test_held_record_reads': 'Values and errors through retained object-capable legacy Records',
    'test_native_table_cell_result': 'Retained native Table result provider and explicit frame conversion',
    'test_native_table_index_semantics': 'Retained native Table index contract with optional helper inputs',
    'test_reduction_lift_table_key': 'Compiler reduction placement in a legacy Table fixture',
    'test_table_forms_w1_9': 'Legacy helper operations through exported and compiled formulas',
    'test_enum_by_type': 'Compatibility with the optional legacy enum marker',
    'test_input_scenarios': 'Fresh-process and fresh-instance input replacement through collect',
    'test_table_admission': 'Retained explicit provider validation and pandas-to-helper conversion',
    'test_table_cells_api': 'Legacy helper table result API and unchanged Reference behavior',
    'test_table_matrix': 'Legacy helper table vocabulary and explicit native handles, checked in live, exported and compiled modes',
}

PORTS = {
    'test_pickle_restores_distinct_multidimensional_native_array_shapes': 'Standalone Cython multi-dimensional array pickle',
    'test_observed_wide_integers_and_buffer_formats_fall_back_to_objects': 'Wide integer and buffer format fallback',
    'test_non_tables_have_no_schema': 'Tracer behavior for the absence of any Table',
    'test_new_input_scenario_after_collect': 'Fresh input scenarios, refixtured with ordinary prepared pandas',
}
HELPER_VOCABULARY = {'test_grouped_report_metadata', 'test_raw_scalar_preparation', 'test_table_operations'}
MIXED_COMPILER = {'test_audit_runtime', 'test_builtin_reduction_lowering', 'test_public_object_return_spec',
                  'test_diag_jp_forms', 'test_reduction_lift_table_key', 'test_table_cells_api'}


def tests(source, suite):
    inventory = Inventory({})
    expose_test_sources(source)
    previous = Path.cwd()
    os.chdir(source)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as diagnostics:
            status = pytest.main([SUITES[suite], '--rootdir=' + str(source), '--collect-only', '-q',
                                  '-p', 'no:cacheprovider'], plugins=[inventory])
    finally:
        os.chdir(previous)
    if status:
        raise ValueError('\n'.join(diagnostics.getvalue().splitlines()[-40:]))
    if set(inventory.deselected) != set(inventory.opt_in):
        raise ValueError(f'{suite} has unclassified deselections: '
                         + ', '.join(sorted(set(inventory.deselected) - set(inventory.opt_in))))
    return inventory


def baseline_status(file):
    if file is None:
        return {}
    observed = {}
    with ZipFile(file) as archive:
        for xml in archive.namelist():
            if xml.endswith('.xml'):
                for case in ElementTree.fromstring(archive.read(xml)).findall('.//testcase'):
                    outcome = next((name for name in ('error', 'failure', 'skipped') if case.find(name) is not None),
                                   'passed')
                    observed[case.get('classname'), case.get('name')] = outcome
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiler', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--historical-junit', type=Path)
    parser.add_argument('--source-coverage', type=Path)
    args = parser.parse_args()
    if pytest.__version__ != PYTEST_VERSION:
        parser.error(f'the source inventory requires pytest {PYTEST_VERSION}')
    manifest = json.loads(EXCLUSIONS.read_text(encoding='utf-8'))
    affected = manifest['modelx_helpers']
    floor = manifest['lifelib']
    baseline = baseline_status(args.historical_junit)
    revision = command('git', 'rev-parse', 'HEAD', cwd=args.compiler)
    coverage = {'compiler_pin': revision, 'pytest': PYTEST_VERSION, 'suites': {}}
    rows = []
    for suite in SUITES:
        inventory = tests(args.compiler.resolve(), suite)
        coverage['suites'][suite] = {'selected': sorted(inventory.selected), 'opt_in': inventory.opt_in}
        for case in inventory.selected:
            path, node = case.split('::', 1)
            stem = Path(path).stem
            name = node.split('[', 1)[0]
            if stem in MODULES and name in PORTS:
                rows.append((suite, path, case, 'normal', 'active compiler', PORTS[name]))
            elif path in affected['modules'] or any(case == item or case.startswith(item + '[')
                                                    for item in affected['ids']):
                if stem in HELPER_VOCABULARY and name != 'test_compiled_selection_has_no_python_helper_pandas_or_owner_calls':
                    kind = 'legacy helper vocabulary'
                else:
                    kind = 'active compiler with legacy fixture' if stem in MIXED_COMPILER else 'retained optional provider'
                rows.append((suite, path, case, 'legacy helper', kind, MODULES[stem]))
            elif path in floor['modules'] or any(case == item or case.startswith(item + '[') for item in floor['ids']):
                rows.append((suite, path, case, 'release', 'lifelib model integration',
                             'Excluded only from the scoped Linux 3.9 lane because lifelib requires Python >=3.10'))
    with args.output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(('suite', 'module', 'test id', 'evidence lane', 'subject', 'reason', 'historical Linux 3.14t'))
        for row in rows:
            module = row[1].removesuffix('.py').replace('/', '.')
            historical = baseline.get((module, row[2].split('::', 1)[1]),
                                      'no matching case' if baseline else 'historical artifact not supplied')
            writer.writerow((*row, historical))
    if args.source_coverage:
        write_json(args.source_coverage, coverage)
    print(revision, len(rows), 'affected case identities')


if __name__ == '__main__':
    main()
