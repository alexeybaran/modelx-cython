"""Compare pandas read cases across live, exported, and compiled Cells."""

import ast
from collections import defaultdict
from functools import cache
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest

import table_matrix_formulas as examples
import table_matrix_frames as fixtures
from table_matrix_squares import (
    FORMS, KEY_AXIS_OPERATIONS, KEYS, MODES, OPERATIONS, applicable, declared,
)


READS = {
    'loc': ('k, column', 'return table.loc[k, column]'),
    'at': ('k, column', 'return table.at[k, column]'),
    'has': ('k,', 'return k in table.index'),
    'column': ('column, k', 'return table[column][k]'),
    'sub_loc': ('prefix, k, column', 'sub = table.loc[prefix]\nreturn sub.loc[k, column]'),
    'sub_has': ('prefix, k', 'sub = table.loc[prefix]\nreturn k in sub.index'),
}
EXPORT = 'matrix_export'
COMPILED = EXPORT + '_cy'
NATIVE_CASE = 'sq_C_1_int'
FALLBACK_CASE = 'loc_list'


def formula(name, parameter, body):
    return f'def {name}({parameter}):\n{textwrap.indent(body, "    ")}\n'


def _case_groups():
    """Keep each label in Python data so the Cell entry cannot coerce its type."""
    cases = {}
    cells = {}
    arguments = defaultdict(list)

    def add_read(read, variant, args):
        name = f'read_{read}_{variant}'
        values = f'matrix_arguments[{name!r}][i]'
        table = f'M_{variant}()'
        if name not in cells:
            if read == 'sub_cell':
                child = f'child_{variant}'
                cells[child] = formula(child, 'i', f'return {table}.loc[{values}[0]]')
                cells[name] = formula(name, 'i', f'_, k, column = {values}\n'
                                      f'sub = {child}(i)\nreturn sub is {child}(i), sub.loc[k, column]')
            else:
                params, body = READS[read]
                cells[name] = formula(name, 'i', f'{params} = {values}\ntable = {table}\n{body}')
        index = len(arguments[name])
        arguments[name].append(args)
        return name, index

    for square in applicable():
        if square in examples.SQUARES:
            form, operation, key = square
            name = f'sq_{form}_{operation}_{key}'
            body, indices = examples.SQUARES[square]
            cells[name] = formula(name, 'i', textwrap.dedent('    ' + body))
            cases[square] = [(name, index) for index in indices]
        else:
            cases[square] = [add_read(*read) for read in fixtures.reads(*square)]

    cases['C', 10, 'int'].extend(add_read(*read) for read in fixtures.numeric_boundaries())
    cells[FALLBACK_CASE] = formula(
        FALLBACK_CASE, 'i', 'return T_C().loc[[5, 7 + i], "q"]',
    )
    cases['C', 1, 'int'].extend((FALLBACK_CASE, index) for index in (0, 99))
    return cases, cells, dict(arguments)


CASES, CELLS, ARGUMENTS = _case_groups()
CALLS = tuple(call for group in CASES.values() for call in group)


def _table_model():
    frames = {f'T_{name}': frame for name, frame in examples.frames().items()}
    frames.update({f'M_{name}': frame for name, frame in fixtures.FRAMES.items()})
    refs = {f'frame_{name}': frame for name, frame in frames.items()}
    refs.update(matrix_arguments=ARGUMENTS, Sex=examples.Sex)
    formulas = {name: formula(name, '', f'return frame_{name}') for name in frames}
    for name, (root, prefix) in fixtures.LAYOUT.items():
        if prefix is not None:
            name = 'M_' + name
            formulas[name] = formula(name, '', f'return M_{root}().loc[{fixtures.label(prefix)!r}]')
    return refs, {**formulas, **examples.DERIVED, **CELLS}


def _oracle(refs, formulas):
    """Use the same expressions on pandas objects, preserving cached Cell identity."""
    namespace = dict(refs)
    for name, source in formulas.items():
        exec(source, namespace)
        namespace[name] = cache(namespace[name])
    return namespace


def _outcome(call, index):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        try:
            return 'value', call(index)
        except Exception as error:
            return 'raises', type(error).__name__, error.args


def outcomes(namespace):
    return {(name, index): _outcome(namespace[name], index) for name, index in CALLS}


def _assert_value(got, expected):
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(got, expected, check_exact=True)
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(got, expected, check_exact=True)
    elif isinstance(expected, pd.Index):
        pd.testing.assert_index_equal(got, expected, exact=True)
    elif isinstance(expected, np.ndarray):
        assert isinstance(got, np.ndarray)
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)
    elif isinstance(expected, (tuple, list)):
        assert isinstance(got, (tuple, list))
        assert len(got) == len(expected)
        for actual, value in zip(got, expected):
            _assert_value(actual, value)
    elif isinstance(expected, (float, np.floating)) and math.isnan(expected):
        assert math.isnan(got)
    else:
        assert got == expected


_RUN = '''\
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
from pathlib import Path
import pickle
import sys

from test_table_matrix import CELLS, outcomes

package, mode = sys.argv[1:]
if mode == 'compiled':
    assert isinstance(import_module(package + '._mx_classes').__loader__, ExtensionFileLoader)
space = import_module(package).mx_model.Matrix
results = outcomes({name: getattr(space, name) for name in CELLS})
Path(mode + '.pickle').write_bytes(pickle.dumps(results))
'''

_SAMPLE = '''\
from matrix_export import mx_model
from test_table_matrix import CALLS

for name, index in CALLS:
    try:
        getattr(mx_model.Matrix, name)(index)
    except Exception:
        pass
'''


def _attributes(root, name):
    functions = []
    for path in root.rglob('_mx_classes.py'):
        definitions = {node.name: node for node in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
                       if isinstance(node, ast.FunctionDef)}
        selected = [node for node in definitions.values() if node.name.endswith(name)]
        functions.extend(selected)
        functions.extend(definitions[call.func.id] for node in selected for call in ast.walk(node)
                         if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                         and call.func.id.startswith('_mx_dense_formula_'))
    assert functions, name
    return {attr.attr for node in functions for attr in ast.walk(node) if isinstance(attr, ast.Attribute)}


def _check_compiled_paths(root):
    report = json.loads((root / '_mx_table_reads.json').read_text(encoding='utf-8'))
    native = report[f'Matrix.{NATIVE_CASE}']
    assert any(read['form'] == 'L1' for read in native), native
    assert {'f64_1', 'f64_c'} & _attributes(root, NATIVE_CASE), native
    fallback = report[f'Matrix.{FALLBACK_CASE}']
    assert any(read['form'] == 'pandas' and 'selects several rows' in read['reason']
               for read in fallback), fallback
    attributes = _attributes(root, FALLBACK_CASE)
    assert 'loc' in attributes and not {'f64_1', 'f64_c', 'kcode'} & attributes, fallback


class Matrix:
    def __init__(self, root, model, expected):
        self.root = root
        self.model = model
        self.expected = expected
        self.process_env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, (
            str(Path(__file__).resolve().parent), os.environ.get('PYTHONPATH'),
        ))))
        self.results = {}

    def run(self, *args):
        result = subprocess.run([sys.executable, *args], cwd=self.root, env=self.process_env,
                                capture_output=True, text=True, timeout=1800)
        assert result.returncode == 0, f'{args}\n{result.stdout[-4000:]}\n{result.stderr[-8000:]}'

    def evaluate(self, mode):
        if mode not in self.results:
            if mode == 'live':
                space = self.model.Matrix
                self.results[mode] = outcomes({name: getattr(space, name) for name in CELLS})
            else:
                package = EXPORT
                if mode == 'compiled':
                    self.run('-m', 'modelx_cython', EXPORT, '--sample', 'sample.py', '--spec', 'spec.py')
                    package = COMPILED
                    _check_compiled_paths(self.root / package)
                self.run('run_matrix.py', package, mode)
                self.results[mode] = pickle.loads((self.root / f'{mode}.pickle').read_bytes())
        return self.results[mode]


@pytest.fixture(scope='module')
def matrix(tmp_path_factory):
    import modelx as mx

    root = tmp_path_factory.mktemp('pandas_matrix')
    (root / 'sample.py').write_text(_SAMPLE, encoding='utf-8')
    (root / 'run_matrix.py').write_text(_RUN, encoding='utf-8')
    (root / 'spec.py').write_text(repr({'strict_lowering': True}), encoding='utf-8')
    refs, formulas = _table_model()
    model = mx.new_model('PandasMatrix')
    previous = mx.use_formula_error()
    mx.use_formula_error(False)
    try:
        space = model.new_space('Matrix', refs=refs)
        for name, source in formulas.items():
            space.new_cells(name, source)
        model.export(root / EXPORT)
        yield Matrix(root, model, outcomes(_oracle(refs, formulas)))
    finally:
        mx.use_formula_error(previous)
        model.close()


def test_every_square_is_declared():
    rows = declared()
    assert set(rows) == {(form, operation) for form in FORMS for operation in OPERATIONS}
    for (form, operation), (codes, reason) in rows.items():
        assert len(codes) == len(KEYS) and set(codes) <= {'y', '-'}, (form, operation)
        if '-' in codes:
            assert reason != '-', f'{form} {operation}: a not-applicable square needs a reason'
        if operation not in KEY_AXIS_OPERATIONS:
            assert codes.count('y') <= 1, f'{form} {operation}: key axis on {KEY_AXIS_OPERATIONS} only'
    assert set(examples.SQUARES) <= set(CASES)
    assert set(CASES) == set(applicable()) and len(CASES) == 235
    assert all(CASES.values())


@pytest.mark.parametrize('form,operation,key,mode', [
    pytest.param(form, operation, key, mode, id=f'{form}-{operation}-{key}-{mode}')
    for form, operation, key in applicable() for mode in MODES
])
def test_square(matrix, form, operation, key, mode):
    results = matrix.evaluate(mode)
    for call in CASES[form, operation, key]:
        got, expected = results[call], matrix.expected[call]
        try:
            _assert_value(got, expected)
        except AssertionError as error:
            name, index = call
            args = ARGUMENTS[name][index] if name in ARGUMENTS else (index,)
            raise AssertionError(f'{name}{args!r}: {mode} returned {got!r}; pandas returned {expected!r}') from error
