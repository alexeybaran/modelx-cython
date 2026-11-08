"""Compile the owned lowering/provider without editing shared pipeline wiring."""
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import libcst as cst

from modelx_cython.helpers import module_name
from modelx_cython.reductions import NativeReductions


def test_compiled_builtin_reduction_lowering(tmp_path):
    def info(result, params):
        return SimpleNamespace(params=params, has_formula_def=True,
            has_typeinfo=lambda: True, called_with_kwargs=False, body_has_closure=False,
            formula_allows_exact_receiver=True,
            get_rettype_expr=lambda **kw: result,
            get_argtype_expr=lambda *args, **kw: 'long long' if kw.get('c_style') else '_mx_cy.longlong')
    cls = SimpleNamespace(name='_c_Test', refs={}, cells={'predicate': info('bint', ('i',))})
    alias = module_name('modelx_cython._native_iterables')
    source = Path(__file__).parents[1] / 'modelx_cython'
    for suffix in ('.pyx', '.pxd'):
        (tmp_path / (alias + suffix)).write_bytes((source / ('_native_iterables' + suffix)).read_bytes())
    methods, cold = [], []
    for operation in ('all', 'any', 'min', 'max'):
        boolean = operation in ('all', 'any')
        result = 'bint' if boolean else 'long long'
        annotation = '_mx_cy.bint' if boolean else '_mx_cy.longlong'
        payload = 'self.predicate(i)' if boolean else 'i'
        text = (f'@_mx_cy.ccall\ndef {operation}_check(self, n: _mx_cy.longlong) -> {annotation}:\n'
                f'    return {operation}({payload} for i in range(n) if i != 1)\n')
        formula = cst.parse_module(text).body[0]
        pass_ = NativeReductions(cls, info(result, ('n',)), set(), formula,
            lambda op=operation: '_cold_' + op, True, builtin_reductions=(operation,))
        rewritten = cst.Module([formula.visit(pass_)]).code
        assert pass_.count == 1
        methods.append('\n'.join('    ' + line if line else '' for line in rewritten.splitlines()))
        cold.extend(cst.Module([function]).code for function in pass_.fallbacks)
    prefix = f'''import cython as _mx_cy
from cython.cimports import {alias}
@_mx_cy.cclass
class _c_Test:
    hook: object
    def __init__(self):
        self.hook = lambda i: i != 2
    @_mx_cy.ccall
    def set_hook(self, function: object):
        self.hook = function
    @_mx_cy.ccall
    def predicate(self, i: _mx_cy.longlong) -> _mx_cy.bint:
        return self.hook(i)
'''
    (tmp_path / 'probe.py').write_text(prefix + '\n'.join(methods) + '\n' + '\n'.join(cold))
    (tmp_path / 'setup.py').write_text('from setuptools import setup\nfrom Cython.Build import cythonize\n'
        f'setup(ext_modules=cythonize(["probe.py", "{alias}.pyx"]))\n')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=tmp_path, check=True)
    c_source = (tmp_path / 'probe.c').read_text()
    for operation in ('all', 'any', 'min', 'max'):
        pattern = r'^static [^\n]*__pyx_f_[^\n]*_' + operation + r'_check\([^\n]*\) \{'
        signatures = list(re.finditer(pattern, c_source, re.MULTILINE))
        assert signatures
        for signature in signatures:
            body = c_source[signature.start():c_source.index('\n}\n', signature.end())]
            assert 'for (' in body
            assert not any(token in body for token in
                           ('PyObject_GetIter(', '__Pyx_PyObject_GetIterNextFunc(', 'PyIter_Next('))
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
'''], cwd=tmp_path, check=True)
