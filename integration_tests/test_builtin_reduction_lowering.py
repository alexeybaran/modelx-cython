"""Compile the owned lowering/provider without editing shared pipeline wiring."""
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import libcst as cst
from lifelib import typed_inputs as operations

from modelx_cython.helpers import module_name, HelperCalls
from modelx_cython.helper_members import module_helpers
from modelx_cython.reductions import NativeReductions


def test_compiled_builtin_reduction_lowering(tmp_path):
    def info(result, params):
        return SimpleNamespace(params=params, has_formula_def=True,
            has_typeinfo=lambda: True, called_with_kwargs=False, body_has_closure=False,
            formula_allows_exact_receiver=True,
            get_rettype_expr=lambda **kw: result,
            get_argtype_expr=lambda *args, **kw: 'long long' if kw.get('c_style') else '_mx_cy.longlong')
    cls = SimpleNamespace(name='_c_Test',
        refs={'operations': SimpleNamespace(helper_members=module_helpers(operations))},
        cells={'predicate': info('bint', ('i',)), 'amount': info('double', ('i',))})
    alias = module_name('modelx_cython._native_iterables')
    source = Path(__file__).parents[1] / 'modelx_cython'
    for suffix in ('.pyx', '.pxd'):
        (tmp_path / (alias + suffix)).write_bytes((source / ('_native_iterables' + suffix)).read_bytes())
    import lifelib
    helper_alias = module_name('lifelib._native_inputs')
    for suffix in ('.pyx', '.pxd'):
        (tmp_path / (helper_alias + suffix)).write_bytes(
            (Path(lifelib.__file__).parent / ('_native_inputs' + suffix)).read_bytes())
    methods, cold = [], []
    for operation in ('all', 'any', 'min', 'max'):
        boolean = operation in ('all', 'any')
        result = 'bint' if boolean else 'long long'
        annotation = '_mx_cy.bint' if boolean else '_mx_cy.longlong'
        payload = 'self.predicate(i)' if boolean else 'i'
        text = (f'def {operation}_check(self, n: _mx_cy.longlong) -> {annotation}:\n'
                f'    return {operation}({payload} for i in range(n) if i != 1)\n')
        formula = cst.parse_module(text).body[0]
        pass_ = NativeReductions(cls, info(result, ('n',)), set(), formula,
            lambda op=operation: '_cold_' + op, True)
        rewritten = cst.Module([formula.visit(pass_)]).code
        assert pass_.count == 0
        assert rewritten == text
        methods.append('\n'.join('    ' + line if line else '' for line in rewritten.splitlines()))
        cold.extend(cst.Module([function]).code for function in pass_.fallbacks)
    for name, payload, annotation, result in (
            ('all_true', 'self.predicate(i)', '_mx_cy.bint', 'bint'),
            ('any_true', 'self.predicate(i)', '_mx_cy.bint', 'bint'),
            ('sum_f64', 'self.amount(i)', '_mx_cy.double', 'double'),
            ('max_value', 'self.amount(i)', '_mx_cy.double', 'double')):
        text = (f'def namespace_{name}(self, n: _mx_cy.longlong) -> {annotation}:\n'
                f'    return self.operations.{name}({payload} for i in range(n) if i != 1)\n')
        formula = cst.parse_module(text).body[0]
        pass_ = NativeReductions(cls, info(result, ('n',)), set(), formula,
                                lambda op=name: '_cold_' + op, True)
        rewritten_formula = formula.visit(pass_)
        assert pass_.count == (1 if name == 'sum_f64' else 0)
        rewritten = cst.Module([HelperCalls(cls.refs, set()).rewrite(rewritten_formula)]).code
        methods.append('\n'.join('    ' + line if line else '' for line in rewritten.splitlines()))
        cold.extend(cst.Module([function]).code for function in pass_.fallbacks)
    prefix = f'''import cython as _mx_cy
from cython.cimports import {alias}
from cython.cimports import {helper_alias}
from lifelib import typed_inputs as operations
@_mx_cy.cclass
class _c_Test:
    hook: object
    operations: object
    def __init__(self):
        self.hook = lambda i: i != 2
        self.operations = operations
    @_mx_cy.ccall
    def set_hook(self, function: object):
        self.hook = function
    @_mx_cy.ccall
    def predicate(self, i: _mx_cy.longlong) -> _mx_cy.bint:
        return self.hook(i)
    @_mx_cy.ccall
    def amount(self, i: _mx_cy.longlong) -> _mx_cy.double:
        return 1.0 + i * 0.5
    @_mx_cy.ccall
    def set_operations(self, value: object):
        self.operations = value
'''
    (tmp_path / 'probe.py').write_text(prefix + '\n'.join(methods) + '\n' + '\n'.join(cold))
    (tmp_path / 'setup.py').write_text('from setuptools import setup\nfrom Cython.Build import cythonize\n'
        f'setup(ext_modules=cythonize(["probe.py", "{alias}.pyx", "{helper_alias}.pyx"]))\n')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=tmp_path, check=True)
    c_source = (tmp_path / 'probe.c').read_text()
    assert '__pyx_gb_' in c_source
    for name in ('all_check','any_check','min_check','max_check',
                 'namespace_all_true','namespace_any_true','namespace_sum_f64','namespace_max_value'):
        assert re.search(r'^static[^\n]*__pyx_(?:pf|gb)_[^\n]*'+re.escape(name)+r'\(',c_source,re.MULTILINE),name
    subprocess.run([sys.executable, '-c', '''from probe import _c_Test
p = _c_Test()
assert p.all_check(0) is True and p.all_check(4) is False
assert p.any_check(0) is False and p.any_check(4) is True
assert p.min_check(4) == 0 and p.max_check(4) == 3
for name in ('min', 'max'):
    try:
        getattr(__import__('builtins'), name)(())
    except ValueError as error:
        expected = error.args
    try:
        getattr(p, name + '_check')(0)
    except ValueError as error:
        assert error.args == expected
    else:
        raise AssertionError('empty extrema did not raise')
seen = []
error = StopIteration('sentinel')
def broken(i):
    seen.append(i)
    raise error
p.set_hook(broken)
for name in ('all', 'any'):
    try:
        getattr(p, name + '_check')(4)
    except RuntimeError as actual:
        assert actual.__cause__ is error and actual.__context__ is error
    else:
        raise AssertionError('body StopIteration not converted')
assert seen == [0, 0]
p.set_hook(lambda i: i != 2)
assert p.namespace_all_true(4) is False
assert p.namespace_any_true(4) is True
assert p.namespace_sum_f64(3) == 3.0
assert p.namespace_max_value(3) == 2.0
assert p.namespace_all_true(0) is True
assert p.namespace_sum_f64(0) == 0.0
from types import SimpleNamespace
held = []
def replacement(values):
    assert iter(values) is values
    held.append(values)
    return True
p.set_operations(SimpleNamespace(all_true=replacement))
assert p.namespace_all_true(4) is True
assert list(held[0]) == [True, False, True]
assert list(held[0]) == []
'''], cwd=tmp_path, check=True)
