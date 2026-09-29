"""Compile a source-preserved Space formula with explicit reference bindings."""

import ast
import hashlib

from .contracts import ContractError


def emit_space_preparer(contract):
    """Only add a bindings argument; leave the author's formula body intact."""
    source = contract['source']
    if hashlib.sha256(source.encode('utf-8')).hexdigest() != contract['sha256']:
        raise ContractError('Space formula source hash mismatch')
    parsed = ast.parse(source)
    if len(parsed.body) != 1:
        raise ContractError('Space formula must contain one function or lambda')
    node = parsed.body[0]
    if isinstance(node, ast.FunctionDef):
        fn = node
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Lambda):
        fn = ast.FunctionDef(name='prepare', args=node.value.args,
                             body=[ast.Return(value=node.value.body)],
                             decorator_list=[], returns=None, type_comment=None)
    else:
        raise ContractError('Space formula must contain one function or lambda')
    if fn.decorator_list or any(arg.arg == '_mxcf_bindings' for arg in
                                (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)):
        raise ContractError('Prepared Space formula has unsupported binding syntax')
    names = {node.id for node in ast.walk(fn) if isinstance(node, ast.Name)}
    if '_mxcf_bindings' in names:
        raise ContractError('Prepared Space formula reserves _mxcf_bindings')
    refs = contract['references']
    if (not isinstance(refs, list) or len(set(refs)) != len(refs)
            or any(not isinstance(name, str) or not name.isidentifier() for name in refs)):
        raise ContractError('Prepared Space formula references must be named uniquely')
    fn.name = 'prepare'
    (fn.args.posonlyargs if fn.args.posonlyargs else fn.args.args).insert(
        0, ast.arg(arg='_mxcf_bindings'))
    assignments = [ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())],
        value=ast.Subscript(value=ast.Name(id='_mxcf_bindings', ctx=ast.Load()),
                            slice=ast.Constant(value=name), ctx=ast.Load()))
        for name in refs]
    start = int(bool(fn.body and isinstance(fn.body[0], ast.Expr)
                     and isinstance(fn.body[0].value, ast.Constant)
                     and isinstance(fn.body[0].value.value, str)))
    fn.body[start:start] = assignments
    return '# cython: language_level=3\n' + ast.unparse(ast.fix_missing_locations(fn)) + '\n'
