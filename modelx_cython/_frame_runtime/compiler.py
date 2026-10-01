"""Small, opt-in scalar-kernel emitter.

It binds declared names/signatures, emits memo wrappers and bulk output loops,
then delegates formula bodies to Cython. No pandas, category-flow, range, axis,
or expression-type interpreter. Compilation is explicit and fail-closed.
"""
from pathlib import Path
import ast
import copy
import hashlib
import io
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
import tokenize
from enum import IntEnum
from .contracts import ContractError, TableKey
from .lookups import (Curve, CurveTable, IndexedFloat1, IndexedFloat2,
                      IndexedInt1, IndexedInt2, power)

_INDEXED_ARRAYS = (IndexedInt1, IndexedFloat1, IndexedInt2, IndexedFloat2)


_TYPES = {float: 'double', int: 'long long', bool: 'bint'}
_BUILTINS = {'min', 'max', 'abs', 'int', 'float', 'bool', 'range', 'len', 'round',
             'ValueError', 'KeyError', 'RuntimeError', 'IndexError', 'OverflowError',
             'ArithmeticError', 'ZeroDivisionError'}
_MATH = {'floor', 'ceil', 'sqrt', 'exp', 'log', 'log1p', 'expm1', 'fabs', 'isfinite'}


def type_info(typ):
    if isinstance(typ, TableKey):
        return {"kind":"table_key", "ctype":"long long", "table":typ.table, "default":typ.default}
    if typ is None:
        typ = type(None)
    if typ in (object, inspect.Signature.empty, str, bytes, tuple, list, dict, set, type(None)):
        return {'kind': 'object', 'ctype': 'object',
                'python_type': 'object' if typ is inspect.Signature.empty else typ.__name__}
    if typ in _TYPES:
        return {'kind': typ.__name__, 'ctype': _TYPES[typ]}
    if typ in (Curve, CurveTable):
        return {'kind': 'curve' if typ is Curve else 'curve_table', 'ctype': typ.__name__}
    if typ in _INDEXED_ARRAYS:
        return {'kind': 'indexed_array', 'ctype': typ.__name__}
    if inspect.isclass(typ) and issubclass(typ, IntEnum):
        values = [int(x) for x in typ]
        if values != list(range(len(values))) or len(values) != len(typ.__members__):
            raise ContractError('Kernel IntEnums require unique contiguous values starting at zero')
        from .enum_binding import enum_origin
        info = {'kind': 'enum', 'ctype': 'long long', 'name': typ.__name__,
                'members': {k: int(v) for k, v in typ.__members__.items()}}
        origin = enum_origin(typ)
        if origin is not None:
            info.update(origin)
        return info
    raise ContractError(f'Explicit scalar/Curve type required; received {typ!r}')


def function_ast(fn):
    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    node.decorator_list = []
    return node


class FrontendRewrite(ast.NodeTransformer):
    """Lower the small author-facing ``lookup`` vocabulary to declared native inputs.

    This is intentionally not a pandas interpreter. Every accepted table/column/helper
    combination is declared in ``KernelSpec.frontend`` and maps to an already typed
    physical input. Formula control flow remains untouched and is compiled by Cython.
    """
    def __init__(self, frontend, inputs, *, integer_names=(), integer_cells=(),
                 global_names=()):
        self.frontend = frontend or {}
        self.inputs = inputs
        self.integer_names = set(integer_names)
        self.integer_cells = set(integer_cells)
        self.global_names = set(global_names)
        self.native_fields = self.frontend.get('native_mapping_fields', {})
        self.native_direct = self.frontend.get('native_direct_inputs', {})

    def _integer_index(self, node):
        """Admit only typed integral indexing; float-to-int casts change NumPy semantics."""
        if isinstance(node, ast.Constant):
            return type(node.value) is int
        if isinstance(node, ast.Name):
            return node.id in self.integer_names
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return self._integer_index(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (
                ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            return self._integer_index(node.left) and self._integer_index(node.right)
        if isinstance(node, ast.IfExp):
            return self._integer_index(node.body) and self._integer_index(node.orelse)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            return ((name in self.integer_cells and all(self._integer_index(a) for a in node.args))
                    or (name in ('min', 'max') and bool(node.args)
                        and all(self._integer_index(a) for a in node.args))
                    or (name == 'int' and len(node.args) == 1))
        return False

    @staticmethod
    def _data_table(node):
        if (isinstance(node, ast.Call) and not node.args and not node.keywords
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'data'):
            return node.func.attr
        return None

    @staticmethod
    def _literal_string(node, what):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            raise ContractError(what + ' must be a literal string in native formulas')
        return node.value

    @staticmethod
    def _lookup_name(node):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == 'lookup'):
            return node.attr
        return None

    def _physical_call(self, input_name, method, args, source):
        if input_name not in self.inputs or self.inputs[input_name] not in (Curve, CurveTable):
            raise ContractError(f'{source}: frontend mapping must target a Curve/CurveTable input')
        return ast.copy_location(ast.Call(
            ast.Attribute(ast.Name(input_name, ast.Load()), method, ast.Load()),
            [self.visit(a) for a in args], []), source)

    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load):
            if (isinstance(node.value, ast.Subscript)
                    and isinstance(node.value.value, ast.Name)
                    and isinstance(node.value.slice, ast.Constant)
                    and type(node.value.slice.value) is str):
                owner = node.value.value.id
                field = node.value.slice.value
                physical = self.native_fields.get(owner, {}).get(field)
                if physical is not None and owner in self.global_names:
                    if self.inputs[physical] is not IndexedFloat1:
                        raise ContractError('Only a declared numeric array can be indexed')
                    if not self._integer_index(node.slice):
                        raise ContractError('Native numeric arrays need an integral index')
                    return ast.copy_location(ast.Call(
                        ast.Attribute(ast.Name(physical, ast.Load()), 'get', ast.Load()),
                        [self.visit(node.slice)], []), node)
            if isinstance(node.value, ast.Name):
                owner = node.value.id
                if owner in self.global_names and isinstance(node.slice, ast.Constant):
                    physical = self.native_fields.get(owner, {}).get(node.slice.value)
                    if physical is not None and self.inputs[physical] is float:
                        return ast.copy_location(ast.Name(physical, ast.Load()), node)
                physical = self.native_direct.get(owner)
                if physical is not None and owner in self.global_names:
                    if self.inputs[physical] is not IndexedFloat1:
                        raise ContractError('Only a declared numeric array can be indexed')
                    if not self._integer_index(node.slice):
                        raise ContractError('Native numeric arrays need an integral index')
                    return ast.copy_location(ast.Call(
                        ast.Attribute(ast.Name(physical, ast.Load()), 'get', ast.Load()),
                        [self.visit(node.slice)], []), node)
            table = self._data_table(node.value)
            physical = self.frontend.get('arrays', {}).get(table)
            if physical is not None:
                typ = self.inputs[physical]
                axes = 2 if typ in (IndexedInt2, IndexedFloat2) else 1
                indices = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
                if len(indices) != axes or not all(self._integer_index(i) for i in indices):
                    raise ContractError('Native indexed Data arrays require typed integral indices on every axis')
                return ast.copy_location(ast.Call(
                    ast.Attribute(ast.Name(physical, ast.Load()), 'get', ast.Load()),
                    [self.visit(index) for index in indices], []), node)
        return self.generic_visit(node)

    def visit_Name(self, node):
        if (isinstance(node.ctx, ast.Load) and node.id in self.global_names
                and node.id in self.native_direct):
            physical = self.native_direct[node.id]
            if self.inputs[physical] is float:
                return ast.copy_location(ast.Name(physical, ast.Load()), node)
        return node

    def visit_Call(self, node):
        helper = self._lookup_name(node.func)
        if helper is None:
            return self.generic_visit(node)
        if node.keywords:
            raise ContractError('Native lookup helpers use positional arguments only')
        if not node.args:
            raise ContractError('Malformed lookup helper call')
        table = self._data_table(node.args[0])
        if table is None:
            raise ContractError('Native lookup helper first argument must be data.<table>()')

        if helper in ('field', 'category'):
            if len(node.args) < 3:
                raise ContractError(helper + ' requires table, key, column')
            if self.frontend.get('point_parameter') and not (isinstance(node.args[1], ast.Name) and node.args[1].id == self.frontend['point_parameter']):
                raise ContractError('Nonlocal row selection uses the Python calculator')
            column = self._literal_string(node.args[2], 'lookup column')
            try:
                physical = self.frontend['fields'][table][column]
            except KeyError as e:
                raise ContractError(f'Undeclared model-point field lookup: {table}.{column}') from e
            if physical not in self.inputs:
                raise ContractError('Frontend field maps to undeclared input: ' + physical)
            return ast.copy_location(ast.Name(physical, ast.Load()), node)

        if helper == 'key':
            if len(node.args) != 2:
                raise ContractError('lookup.key requires table and label')
            if table not in self.frontend.get('keys', {}):
                raise ContractError('Undeclared table-key domain: ' + table)
            return ast.copy_location(self.visit(node.args[1]), node)

        if helper not in ('at', 'get', 'previous', 'previous_group', 'linear'):
            raise ContractError('Unsupported native lookup helper: ' + helper)
        column_index = 3 if helper == 'previous_group' else 2
        if len(node.args) <= column_index:
            raise ContractError('Malformed lookup.' + helper + ' call')
        column = self._literal_string(node.args[column_index], 'lookup column')
        try:
            cfg = self.frontend['lookups'][table][column]
        except KeyError as e:
            raise ContractError(f'Undeclared native lookup: {helper}({table}.{column})') from e
        if cfg.get('helper') != helper:
            raise ContractError(f'{table}.{column}: expected lookup.{cfg.get("helper")}, got lookup.{helper}')

        # A fixed literal group may lower to a dedicated Curve for maximum hot-path
        # efficiency while the model still reads a normal MultiIndex table.
        if 'fixed_groups' in cfg:
            key = node.args[1]
            if not isinstance(key, ast.Tuple) or len(key.elts) < 2:
                raise ContractError('Fixed-group lookup requires a literal-group tuple key')
            group = self._literal_string(key.elts[cfg.get('group_index', 0)], 'lookup group')
            try:
                physical = cfg['fixed_groups'][group]
            except KeyError as e:
                raise ContractError(f'Undeclared fixed lookup group {group!r}') from e
            key_expr = key.elts[cfg.get('key_index', 1)]
            extra = [node.args[3]] if helper == 'get' and len(node.args) > 3 else []
            return self._physical_call(physical, helper, [key_expr] + extra, node)

        physical = cfg['input']
        typ = self.inputs.get(physical)
        if helper == 'previous_group':
            args = [node.args[1], node.args[2]]
            if len(node.args) > 4:
                args.append(node.args[4])
            return self._physical_call(physical, 'previous', args, node)
        if helper in ('at', 'get'):
            key = node.args[1]
            if isinstance(key, ast.Tuple):
                order = tuple(cfg.get('tuple_order', range(len(key.elts))))
                if sorted(order) != list(range(len(key.elts))):
                    raise ContractError('Invalid tuple_order in frontend mapping')
                args = [key.elts[i] for i in order]
            else:
                args = [key]
            if helper == 'get' and len(node.args) > 3:
                args.append(node.args[3])
            expected = CurveTable if len(args) >= 2 and (helper == 'at' or len(args) >= 3) else Curve
            if typ not in (Curve, CurveTable):
                raise ContractError('Lookup mapping targets a non-lookup physical input')
            return self._physical_call(physical, helper, args, node)
        if helper == 'previous':
            args = [node.args[1]]
            if len(node.args) > 3:
                args.append(node.args[3])
            return self._physical_call(physical, 'previous', args, node)
        if helper == 'linear':
            return self._physical_call(physical, 'linear', [node.args[1]], node)
        raise AssertionError(helper)


class BindNames(ast.NodeTransformer):
    """Only structural name binding and explicitly declared enum constants."""
    def __init__(self, cells, inputs, enums, locals_, native_lookups=False,
                 frame_native=False, frame_casts=None):
        self.cells, self.inputs, self.enums, self.locals = cells, inputs, enums, locals_
        self.native_lookups = native_lookups
        self.frame_native = frame_native
        self.frame_casts = frame_casts or {}

    def visit_Call(self, node):
        if (self.frame_native and isinstance(node.func, ast.Name)
                and node.func.id in self.cells and node.func.id not in self.locals):
            return ast.copy_location(ast.Call(
                ast.Name('_cell_' + node.func.id, ast.Load()),
                [ast.Name('state', ast.Load()), *[self.visit(a) for a in node.args]],
                [ast.keyword(k.arg, self.visit(k.value)) for k in node.keywords]), node)
        if (self.native_lookups and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in self.inputs and node.func.value.id not in self.locals
                and self.inputs[node.func.value.id] in (Curve, CurveTable)
                and node.func.attr in ('at','get','previous','linear')):
            typ = self.inputs[node.func.value.id]
            return ast.copy_location(ast.Call(ast.Name('_lookup_'+typ.__name__+'_'+node.func.attr, ast.Load()),
                [self.visit(node.func.value)]+[self.visit(a) for a in node.args],
                [ast.keyword(k.arg,self.visit(k.value)) for k in node.keywords]),node)
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.enums and node.value.id not in self.locals:
            enum = self.enums[node.value.id]
            if node.attr not in enum.__members__:
                raise ContractError('Unknown enum member: ' + ast.unparse(node))
            return ast.copy_location(ast.Name('_E_' + node.value.id + '_' + node.attr, ast.Load()), node)
        if isinstance(node.value, ast.Name) and node.value.id == 'math' and node.value.id not in self.locals:
            if node.attr not in _MATH:
                raise ContractError('Undeclared native math function: ' + node.attr)
            return ast.copy_location(ast.Name('_math_' + node.attr, ast.Load()), node)
        return self.generic_visit(node)

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load) or node.id in self.locals:
            return node
        if node.id in self.cells:
            if self.frame_native:
                return ast.copy_location(ast.Name('_cell_' + node.id, ast.Load()), node)
            return ast.copy_location(ast.Attribute(ast.Name('self', ast.Load()), '_cell_' + node.id, ast.Load()), node)
        if node.id in self.inputs:
            if self.frame_native:
                if node.id in self.frame_casts:
                    return ast.copy_location(ast.Name('__MX_FRAME_CAST_' + node.id + '__', ast.Load()), node)
                return ast.copy_location(ast.Attribute(ast.Name('state', ast.Load()), node.id, ast.Load()), node)
            return ast.copy_location(ast.Attribute(ast.Name('self', ast.Load()), node.id, ast.Load()), node)
        if node.id == '_power_compatible':
            return node
        if node.id == 'power':
            return ast.copy_location(ast.Name('_power', ast.Load()), node)
        if node.id == 'round':
            return ast.copy_location(ast.Name('_round_compatible', ast.Load()), node)
        if node.id in _MATH:
            return ast.copy_location(ast.Name('_math_' + node.id, ast.Load()), node)
        if node.id not in _BUILTINS and not node.id.startswith('_E_'):
            raise ContractError('Undeclared kernel name: ' + node.id)
        return node


def _expression(source, binder):
    n = binder.visit(ast.parse(source, mode='eval').body)
    return ast.unparse(ast.fix_missing_locations(n))


def _frame_cast_names(source, casts):
    """Insert Cython casts only at generated AST names, never in formula literals."""
    if not casts:
        return source
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    changes = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME and token.string in casts:
            start = offsets[token.start[0] - 1] + token.start[1]
            end = offsets[token.end[0] - 1] + token.end[1]
            changes.append((start, end, casts[token.string]))
    for start, end, replacement in reversed(changes):
        source = source[:start] + replacement + source[end:]
    return source


def _frame_input_casts(input_info):
    object_types = {'object', 'Curve', 'CurveTable', 'IndexedInt1',
                    'IndexedFloat1', 'IndexedInt2', 'IndexedFloat2',
                    'dict', 'set'}
    return {key: info['ctype'] for key, info in input_info.items()
            if info['ctype'] in object_types}


def _frame_expression(source, binder, cast_names):
    return _frame_cast_names(_expression(source, binder), cast_names)


def _cell_entry_lines(name, row, tail, call, native_suffix, diagnostics,
                      *, input_info):
    """Emit one typed Cell cache entry for the physical frame."""
    receiver = 'state'
    signature = '_Frame* state' + tail
    ret = row['result']['ctype']
    def field(key):
        return receiver + '.' + key
    args = 'state' + (', ' + call if call else '')
    formula = f'_calc_{name}({args})'
    if row['direct_input'] is not None:
        exception = 'noexcept' if native_suffix else 'except? -1'
        value = field(row['direct_input'])
        info = input_info.get(row['direct_input'], {})
        if info.get('ctype') in {'object', 'Curve', 'CurveTable',
                                 'IndexedInt1', 'IndexedFloat1',
                                 'IndexedInt2', 'IndexedFloat2',
                                 'dict', 'set'}:
            value = f"(<{info['ctype']}>{value})"
        return [f'cdef inline {ret} _cell_{name}({signature}) {exception}{native_suffix}:',
                f'    return {value}']
    if not row['cached']:
        return [f'cdef inline {ret} _cell_{name}({signature}) except? -1{native_suffix}:',
                f'    cdef {ret} _mx_value',
                f'    _mx_value = {formula}',
                '    return _mx_value']

    params = row['params']
    result = ['@cython.boundscheck(False)', '@cython.wraparound(False)',
              '@cython.overflowcheck(False)', '@cython.initializedcheck(False)',
              f'cdef inline {ret} _cell_{name}({signature}) except? -1{native_suffix}:',
              '    cdef Py_ssize_t _mx_key', f'    cdef {ret} _mx_value']
    if diagnostics:
        result.append(f'    {field("_calls")} += 1')
    if row['hashed']:
        result += ['    cdef unsigned long long _mx_hashcode = 1469598103934665603',
                   '    cdef _DoubleBits _mx_bits',
                   '    cdef Py_ssize_t _mx_probe, _mx_bucket']
        for p in params:
            arg = p['name']
            if p['type']['kind'] == 'float':
                result += [f'    if not _math_isfinite({arg}): _cache_error(2, b"{name}.{arg}")',
                           f'    _mx_bits.value = 0.0 if {arg} == 0.0 else {arg}',
                           '    _mx_hashcode = (_mx_hashcode ^ _mx_bits.bits) * <unsigned long long>1099511628211']
            else:
                result.append(f'    _mx_hashcode = (_mx_hashcode ^ <unsigned long long>{arg}) * <unsigned long long>1099511628211')
        result += ['    _mx_hashcode = (_mx_hashcode ^ (_mx_hashcode >> 30)) * <unsigned long long>13787848793156543929',
                   '    _mx_hashcode = (_mx_hashcode ^ (_mx_hashcode >> 27)) * <unsigned long long>10723151780598845931',
                   '    _mx_hashcode = _mx_hashcode ^ (_mx_hashcode >> 31)',
                   f"    _mx_bucket = _mx_hashcode & {row['slots']-1}",
                   f"    for _mx_probe in range({row['slots']}):",
                   f"        _mx_key = {row['offset']} + _mx_bucket"]
        if row['recursive']:
            result += [f'        if {field("_seen")}[_mx_key] != {field("_epoch")} and {field("_seen")}[_mx_key] != {field("_epoch")} - 1:',
                       '            break']
        else:
            result += [f'        if {field("_seen")}[_mx_key] != {field("_epoch")}:',
                       '            break']
        equal = ' and '.join(f'{field("_keys_" + name + "_" + p["name"])}[_mx_bucket] == {p["name"]}'
                             for p in params) or 'True'
        result += [f'        if {equal}: break',
                   f"        _mx_bucket = (_mx_bucket + 1) & {row['slots']-1}",
                   '    else:', f'        _cache_error(3, b"{name}")']
    else:
        index = '0'
        for p in params:
            pnm, lower, size = p['name'], p['lower'], p['size']
            result += [f'    if {pnm} < {lower} or {pnm} >= {lower + size}:',
                       f'        _cache_error(0, b"{name}.{pnm}")']
            index = f'(({index}) * {size} + ({pnm} - ({lower})))'
        result.append(f"    _mx_key = {row['offset']} + {index}")
    store = '_floats' if row['result']['kind'] == 'float' else '_ints'
    result += [f'    if {field("_seen")}[_mx_key] == {field("_epoch")}:',
               f'        return {field(store)}[_mx_key]']
    if row['recursive']:
        result += [f'    if {field("_seen")}[_mx_key] == {field("_epoch")} - 1:',
                   f'        _cache_error(1, b"{name}")',
                   f'    {field("_seen")}[_mx_key] = {field("_epoch")} - 1']
    if row['hashed']:
        for p in params:
            result.append(f'    {field("_keys_" + name + "_" + p["name"])}[_mx_bucket] = {p["name"]}')
    if diagnostics:
        result.append(f'    {field("_misses")} += 1')
    result += [f'    _mx_value = {formula}',
               f'    {field(store)}[_mx_key] = _mx_value',
               f'    {field("_seen")}[_mx_key] = {field("_epoch")}',
               '    return _mx_value']
    return result



def _needs_checked_formula(node, result_info, params, inputs, cells):
    """Whether this formula can directly create a checked Cython error state.

    This is intentionally shallow. Projection-index arithmetic is bounded by
    KernelSpec.steps; only integer-result formulas retain overflow capture. Calls to
    other Cells own their own boundary. Division by a literal nonzero denominator is
    native-safe and needs no Python error check.
    """
    integral_result = result_info['kind'] in ('int', 'enum', 'bool')
    def literal_nonzero(expr):
        return isinstance(expr, ast.Constant) and type(expr.value) in (int,float) and expr.value != 0
    for n in ast.walk(node):
        if isinstance(n, ast.Raise):
            return True
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                if n.func.id in cells:
                    continue
                if n.func.id in {'power', '_power_compatible', 'int', 'round'}:
                    return True
            if (isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                    and n.func.value.id in inputs and inputs[n.func.value.id] in (Curve, CurveTable, *_INDEXED_ARRAYS)
                    and n.func.attr in ('at', 'get', 'previous', 'linear')):
                return True
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)) and not literal_nonzero(n.right):
                return True
            if integral_result and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.LShift, ast.RShift)):
                return True
        if isinstance(n, ast.AugAssign):
            if isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)) and not literal_nonzero(n.value):
                return True
            if integral_result and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.LShift, ast.RShift)):
                return True
        if integral_result and isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return True
    return False


def _native_rejection(node, selected, spec, enums, native_suffix):
    try:
        for statement in ast.walk(node):
            if isinstance(statement, (ast.List, ast.Dict, ast.Set, ast.Subscript,
                                      ast.Match, ast.Assert, ast.Delete)):
                raise ContractError('Object/container operations use the Python calculator')
            if isinstance(statement, ast.Tuple):
                membership = any(isinstance(parent, ast.Compare) and statement in parent.comparators
                                 and any(isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops)
                                 for parent in ast.walk(node))
                if not membership:
                    raise ContractError('Tuple values use the Python calculator')
        # Cython owns control-flow/local inference; closures and first-class Cells
        # are deliberately outside this flat scalar-kernel contract.
        for n in ast.walk(node):
            if isinstance(n, (ast.GeneratorExp, ast.ListComp, ast.DictComp, ast.SetComp)):
                raise ContractError("Comprehensions use the Python calculator")
            if isinstance(n, ast.Name) and (n.id.startswith('_mx_') or n.id == 'self'):
                raise ContractError('Use ordinary free functions; self and the _mx_ prefix are reserved')
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == 'round'):
                if (len(n.args) != 2 or n.keywords
                        or not isinstance(n.args[1], ast.Constant)
                        or type(n.args[1].value) is not int
                        or (node.name in spec.cells and
                            spec.cells[node.name].__globals__.get('round', round) is not round)):
                    raise ContractError('Native round requires the builtin and a literal integer ndigits')
            if isinstance(n, ast.arg) and n.arg.startswith('_mx_'):
                raise ContractError('The _mx_ prefix is reserved for generated names')
            if isinstance(n, ast.Attribute) and (isinstance(n.ctx, ast.Store) or n.attr.startswith('_')):
                raise ContractError('Scalar formulas use read-only public helper operations')
            if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store):
                raise ContractError('Scalar formulas cannot mutate indexed storage')
            if isinstance(n, (ast.With, ast.AsyncWith)):
                raise ContractError('Context/GIL managers are outside the numeric kernel contract')
            if native_suffix and isinstance(n, ast.Raise):
                exc = n.exc
                if (n.cause is not None or not isinstance(exc,ast.Call) or not isinstance(exc.func,ast.Name)
                        or exc.func.id not in _BUILTINS or exc.keywords
                        or any(not isinstance(a,ast.Constant) or type(a.value) not in (str,int,float,bool,type(None)) for a in exc.args)):
                    raise ContractError('Native raises require a named exception and literal arguments; error paths may not call Cells')
            if isinstance(n, (ast.Try, ast.TryStar)):
                raise ContractError('Native kernel errors abort the evaluation epoch; use explicit lookup defaults instead of exception recovery')
            if n is not node and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Yield, ast.YieldFrom)):
                raise ContractError('Closures/generators are not admitted in native scalar kernels')
        local = {p.arg for p in node.args.args} | {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        binder = BindNames(selected, spec.inputs, enums, local, native_lookups=True)
        for statement in node.body:
            binder.visit(copy.deepcopy(statement))
    except ContractError as exc:
        return str(exc)
    return None


def emit(spec, *, diagnostics=False, python_cells=()):
    native_suffix = ' nogil'
    cells = spec.cells
    if any(n.startswith('_mx_') for n in cells):
        raise ContractError('The _mx_ prefix is reserved for generated names')
    enums = spec.enums
    input_info = {k: type_info(v) for k, v in spec.inputs.items()}
    for name, info in input_info.items():
        if info['kind'] == 'table_key' and input_info.get(info['table'], {}).get('kind') != 'curve_table':
            raise ContractError(name + ': TableKey must refer to a declared CurveTable input')
    raw = {k: function_ast(v) for k, v in cells.items()}
    parsed, python_reasons = {}, {}
    if set(python_cells) - set(cells):
        raise ContractError('Unknown Python calculator override')
    for k, node in raw.items():
        try:
            index_names = {name for name, typ in spec.inputs.items() if typ is int}
            index_names.update(p.name for p in inspect.signature(cells[k]).parameters.values()
                               if p.annotation is int)
            index_cells = {name for name, fn in cells.items()
                           if inspect.signature(fn).return_annotation is int}
            rewrite = FrontendRewrite(
                spec.frontend, spec.inputs, integer_names=index_names,
                integer_cells=index_cells, global_names=cells[k].__code__.co_names)
            # This is index-type propagation, not evaluation of the formula.
            # Cython remains responsible for its local types and arithmetic.
            changed = True
            while changed:
                changed = False
                for assignment in ast.walk(node):
                    if (isinstance(assignment, ast.Assign)
                            and len(assignment.targets) == 1
                            and isinstance(assignment.targets[0], ast.Name)
                            and rewrite._integer_index(assignment.value)
                            and assignment.targets[0].id not in rewrite.integer_names):
                        rewrite.integer_names.add(assignment.targets[0].id)
                        changed = True
            parsed[k] = ast.fix_missing_locations(rewrite.visit(copy.deepcopy(node)))
        except ContractError as exc:
            parsed[k] = node
            python_reasons[k] = str(exc)
        if k in python_cells:
            python_reasons[k] = 'Explicit build/test calculator override'
        if k not in python_reasons:
            reason = _native_rejection(parsed[k], cells, spec, enums, native_suffix)
            if reason:
                python_reasons[k] = reason

    exprs = [spec.length] + [v for table in spec.outputs.values() for v in table.values()] + [k + '()' for k in spec.scalar_outputs]
    roots = set(spec.cell_outputs)
    if roots - cells.keys():
        raise ContractError('Unknown declared Cell output')
    for s in exprs:
        roots |= {n.func.id for n in ast.walk(ast.parse(s, mode='eval'))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in cells}
    from .python_calculator import cell_dependencies
    all_dependencies = {name: cell_dependencies(fn, cells) for name, fn in cells.items()}
    selected = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        pending.extend(all_dependencies[name])
    if not selected:
        raise ContractError('Outputs must call at least one declared Cell')
    # Storage follows the declared boundary contract, not sampled return values.
    # An object consumer remains Python-semantic; native ancestors can still call
    # its scalar entry directly. No object pointer crosses the numeric C++ guard.
    object_cells = {k for k in selected if any(
        type_info(t)['kind'] == 'object' for t in
        [inspect.signature(cells[k]).return_annotation] +
        [p.annotation for p in inspect.signature(cells[k]).parameters.values()])}
    for k in selected:
        if k in object_cells:
            python_reasons[k] = 'Python object argument/result contract'
        elif all_dependencies[k] & object_cells:
            python_reasons[k] = 'Consumes a Python-object Cell contract'
        elif any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                 and n.id in cells[k].__code__.co_names
                 and n.id in spec.inputs
                 and type_info(spec.inputs[n.id])['kind'] == 'object'
                 for n in ast.walk(parsed[k])):
            python_reasons[k] = 'Consumes a Python-object input'
    for expr in exprs:
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id in object_cells for n in ast.walk(ast.parse(expr, mode='eval'))):
            raise ContractError('Numeric outputs need a scalar Cell boundary; use cell_outputs for objects')
    python_reasons = {k: v for k, v in python_reasons.items() if k in selected}
    def native_closure(root):
        pending, seen = [root], set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            if name in python_reasons:
                return False
            seen.add(name)
            pending.extend(all_dependencies[name])
        return True
    native_scalar_outputs = sorted(name for name in spec.scalar_outputs
                                   if native_closure(name))
    from .power_codegen import lower_numeric_powers
    numeric_power_rewrites = {}
    for name in selected - python_reasons.keys():
        parsed[name], count = lower_numeric_powers(parsed[name], cells, spec.inputs)
        if count:
            numeric_power_rewrites[name] = count
    mixed = bool(python_reasons)
    rows = {}
    slot = 0
    dependencies = {k: all_dependencies[k] for k in selected}

    def _recursive(name):
        pending=list(dependencies[name]); visited=set()
        while pending:
            dep=pending.pop()
            if dep == name:
                return True
            if dep in visited:
                continue
            visited.add(dep); pending.extend(dependencies.get(dep, ()))
        return False

    def _hash_capacity():
        # Exact native hash memo for continuous numeric keys. The model author does
        # not choose this policy. A low load factor keeps probes short over the
        # declared projection bound; filling still fails explicitly.
        target=max(16, 4 * (spec.steps + 1))
        return 1 << (target - 1).bit_length()

    for name in sorted(selected):
        fn, node = cells[name], parsed[name]
        sig = inspect.signature(fn)
        ret = type_info(sig.return_annotation)
        if ret['kind'] in ('curve', 'curve_table', 'indexed_array'):
            raise ContractError('Cell results must be scalar; native arrays are input types')
        params = []
        hashed = False
        count = 1
        for p in sig.parameters.values():
            if p.name.startswith('_mx_'):
                raise ContractError('The _mx_ prefix is reserved for generated names')
            if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                raise ContractError('Variadic and keyword-only Cell parameters are not supported')
            ti = type_info(p.annotation)
            if ti['kind'] in ('curve', 'curve_table', 'indexed_array'):
                raise ContractError('Bind native arrays as inputs, not Cell keys')
            lower, size = (0, len(ti['members'])) if ti['kind'] == 'enum' else (-1, spec.steps + 2)
            if ti['kind'] == 'bool':
                lower, size = 0, 2
            if ti['kind'] == 'float':
                hashed = True
            default = None
            if p.default is not p.empty:
                if ti['kind'] == 'object':
                    try:
                        default = repr(ast.literal_eval(repr(p.default)))
                    except (ValueError, SyntaxError):
                        raise ContractError('Object defaults must be literal constants') from None
                elif ti['kind'] in ('enum', 'int', 'bool'):
                    if not isinstance(p.default, (int, IntEnum)) or not lower <= int(p.default) < lower + size:
                        raise ContractError('Default outside declared cache key domain')
                    default = repr(int(p.default))
                elif type(p.default) in (float, int):
                    default = repr(float(p.default))
                else:
                    raise ContractError('Defaults must be scalar constants')
            params.append(dict(name=p.name, type=ti, lower=lower, size=size, default=default))
            if mixed:
                params[-1]['positional_only'] = p.kind == p.POSITIONAL_ONLY
            if not hashed:
                count *= size
        cached = getattr(fn, '__kernel_cached__', True)
        python_key = any(p['type']['kind'] == 'object' for p in params)
        object_storage = name in object_cells
        if object_storage and (python_key or hashed):
            count = 0  # Python dictionaries grow with actual keys, not guessed bounds.
        elif hashed:
            count = _hash_capacity()
        if not cached:
            count, hashed = 0, False
        if count > 20_000_000:
            raise ContractError('Declared cache exceeds the per-Cell allocation limit')
        row = dict(name=name, result=ret, params=params, cached=cached, hashed=hashed, last=False,
                   recursive=name in python_reasons or _recursive(name), checked_formula=name in python_reasons or _needs_checked_formula(node, ret, params, spec.inputs, cells),
                   offset=slot, slots=count)
        body = [n for n in node.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
        direct = None
        if len(body)==1 and isinstance(body[0],ast.Return) and isinstance(body[0].value,ast.Name):
            field=body[0].value.id
            if field in input_info and field not in sig.parameters and input_info[field]['ctype']==ret['ctype']:
                direct=field
        row['direct_input'] = None if name in python_reasons else direct
        if mixed:
            row['calculator'] = 'python' if name in python_reasons else 'native'
            row['native_rejection'] = python_reasons.get(name)
            row['dependencies'] = sorted(dependencies[name])
        if object_storage:
            row.update(storage='python_dict' if python_key or hashed else 'object_slots',
                       key_abi='python' if python_key else 'native',
                       allow_none=(spec.allow_none or {}).get(name, True),
                       offset=0, hashed=False)
        rows[name] = row
        if not object_storage:
            slot += count
    if slot > 30_000_000:
        raise ContractError('Declared aggregate cache exceeds the allocation limit')
    all_rows = rows
    object_rows = {k: r for k, r in rows.items() if k in object_cells}
    rows = {k: r for k, r in rows.items() if k not in object_cells}
    if sum(r['slots'] for r in object_rows.values()) > 30_000_000:
        raise ContractError('Object slot allocation exceeds the limit')
    lines = ['# cython: language_level=3, infer_types=True, cdivision=False, cpow=False, overflowcheck=True',
             'import numpy as np', 'cimport cython',
             'cdef union _DoubleBits:', '    double value', '    unsigned long long bits',
             'from ._lookups cimport Curve, CurveTable, IndexedInt1, IndexedFloat1, IndexedInt2, IndexedFloat2',
             'from libc.math cimport pow as real_pow, floor, isfinite',
             'from libc.math cimport ' + ', '.join(f'{x} as _math_{x}' for x in sorted(_MATH))]
    from .lookup_codegen import inline_lookup_source
    lines += [inline_lookup_source()]
    power_body = function_ast(power).body
    lines += ['cdef inline double _power(double base, double exponent) except? -1 nogil:']
    for statement in power_body:
        lines += textwrap.indent(ast.unparse(statement), '    ').splitlines()
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'round'
           for name in selected - python_reasons.keys() for n in ast.walk(parsed[name])):
        lines += [
            'cdef inline double _round_compatible(double value, long long ndigits) except? -1 nogil:',
            '    with gil:',
            '        return <double>round(value, ndigits)',
        ]
    if numeric_power_rewrites:
        lines += ['cdef inline double _power_compatible(double base, double exponent) except? -1 nogil:',
                  '    cdef double result',
                  '    if base > 0.0 and isfinite(base) and isfinite(exponent):',
                  '        result = real_pow(base, exponent)',
                  '        if isfinite(result):',
                  '            return result',
                  '    return base ** exponent']
    lines += ['cdef int _cache_error(int kind, const char* where) except -1 nogil:',
              '    with gil:', '        location = where.decode()',
              '        if kind == 0: raise IndexError("Outside declared cache bound: " + location)',
              '        if kind == 1: raise RuntimeError("Cell cycle: " + location)',
              '        if kind == 2: raise ValueError("Non-finite hash key: " + location)',
              '        raise MemoryError("Hash memo capacity exhausted: " + location)']
    for name, typ in enums.items():
        type_info(typ)
        lines += ['cdef enum:'] + [f'    _E_{name}_{k} = {int(v)}' for k, v in typ.__members__.items()]
    frame_header = tuple(lines)
    frame_field_types = {
        '_epoch': 'unsigned int', '_floats': 'double[::1]',
        '_ints': 'long long[::1]', '_seen': 'unsigned int[::1]',
        '_calls': 'unsigned long long', '_misses': 'unsigned long long',
    }
    lines += ['@cython.final', 'cdef class Kernel:', '    cdef object _records', '    cdef unsigned int _epoch',
              '    cdef double[::1] _floats', '    cdef long long[::1] _ints',
              '    cdef unsigned int[::1] _seen',
              '    cdef unsigned long long _calls, _misses']
    if mixed:
        lines += ['    cdef object _python_scope']
        frame_field_types['_python_scope'] = 'object'
    for k, ti in input_info.items():
        ct = ti['ctype']
        lines += [f'    cdef {ct} {k}']
        frame_field_types[k] = ct
    for k, r in rows.items():
        if r['hashed']:
            for param in r['params']:
                dt = 'double' if param['type']['kind'] == 'float' else 'long long'
                lines += [f"    cdef {dt}[::1] _keys_{k}_{param['name']}"]
                frame_field_types[f"_keys_{k}_{param['name']}"] = f'{dt}[::1]'
    frame_fields = tuple(lines[len(frame_header) + 2:])
    allocated = max(1, slot)
    lines += ['    def __init__(self, records):', '        self._records = records', '        self._epoch = 0',
              f'        self._floats = np.empty({allocated}, dtype=np.float64)',
              f'        self._ints = np.empty({allocated}, dtype=np.int64)',
              f'        self._seen = np.zeros({allocated}, dtype=np.uint32)',
              '        self._calls = self._misses = 0']
    if mixed:
        lines += ['        self._python_scope = None']
    for k, r in rows.items():
        if r['hashed']:
            for param in r['params']:
                dt = 'np.float64' if param['type']['kind'] == 'float' else 'np.int64'
                lines += [f"        self._keys_{k}_{param['name']} = np.empty({r['slots']}, dtype={dt})"]
    lines += ['    cdef void _advance(self) except *:',
              '        if self._epoch >= 4294967294:',
              '            np.asarray(self._seen).fill(0)',
              '            self._epoch = 0', '        self._epoch += 2']
    lines += ['    def reset(self):', '        self._advance()',
              '    def stats(self):', f"        return {{'enabled': {diagnostics}, 'calls': self._calls, 'misses': self._misses, 'cache_slots': {slot}, 'epoch': self._epoch}}",
              '    def replace_records(self, records):', '        self._records = records', '        self._advance()',
              '    cdef void _select(self, tuple record) except *:']
    for i, (k, ti) in enumerate(input_info.items()):
        lines += [f'        self.{k} = record[{i}]']
    lines += ['        self._advance()']
    frame_runtime_start = len(lines)
    frame_methods = []
    frame_casts = _frame_input_casts(input_info)
    cast_names = ({'__MX_FRAME_CAST_' + key + '__': f'(<{ctype}>state.{key})'
                   for key, ctype in frame_casts.items()})
    for name, row in rows.items():
        params = row['params']
        decl = ', '.join(p['type']['ctype'] + ' ' + p['name'] + ((' = ' + p['default']) if p['default'] is not None else '') for p in params)
        tail = (', ' + decl) if decl else ''
        calc_decl = ', '.join(p['type']['ctype'] + ' ' + p['name'] for p in params)
        calc_tail = (', ' + calc_decl) if calc_decl else ''
        call = ', '.join(p['name'] for p in params)
        ret = row['result']['ctype']
        frame_entry = tuple(_cell_entry_lines(name, row, tail, call,
                                              native_suffix, diagnostics,
                                              input_info=input_info))
        if row['direct_input'] is not None:
            frame_methods.append(frame_entry)
            continue
        node = copy.deepcopy(parsed[name])
        loc = ({'_mx_value', '_mx_key', '_mx_object_key', 'self'} if row['cached'] else set()) | {p['name'] for p in params} | {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if name in python_reasons:
            tuple_expr = '(' + call + (',' if params else '') + ')'
            frame_body = ['    with gil:',
                          '        return (<object>state._python_scope).calculate('
                          + repr(name) + ', ' + tuple_expr + ')']
        else:
            frame_binder = BindNames(selected, spec.inputs, enums, loc,
                                     native_lookups=True, frame_native=True,
                                     frame_casts=frame_casts)
            frame_body = []
            for statement in node.body:
                if (isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Constant)
                        and isinstance(statement.value.value, str)):
                    continue
                source = ast.unparse(ast.fix_missing_locations(
                    frame_binder.visit(copy.deepcopy(statement))))
                source = _frame_cast_names(source, cast_names)
                frame_body.extend(textwrap.indent(source, '    ').splitlines())
        frame_methods.extend((frame_entry,
                              tuple([f'cdef inline {ret} _calc_{name}(_Frame* state{calc_tail}) except? -1{native_suffix}:']
                                    + frame_body)))
    frame_runtime_end = len(lines)
    frame_runtime_prefix_end = len(lines)
    for table in spec.outputs:
        if not table.isidentifier() or table.startswith('_'):
            raise ContractError('Output tables require public identifier names')
    for name in spec.scalar_outputs:
        if name not in rows or rows[name]['params']:
            raise ContractError('Scalar outputs must be declared zero-argument Cells')
    manifest = dict(schema='mxcf.scalar_kernel.v1', inputs=input_info, outputs=spec.outputs,
                     steps=spec.steps, length=spec.length, parameters=spec.parameters, scalar_outputs=spec.scalar_outputs, output_dtype='float64', cells=all_rows, cache_slots=slot,
                     native_scalar_outputs=native_scalar_outputs,
                    memo='dense', lookups='native', diagnostics=diagnostics, nogil_formulas=bool(native_suffix), frontend_helpers=bool(spec.frontend),
                    numeric_power_rewrites=numeric_power_rewrites,
                    limitations=['flat scalar kernel', 'explicit contiguous IntEnums', 'bounded integral cache keys',
                                 'continuous numeric Cell keys use compiler-owned bounded exact hash memo', 'real-valued arithmetic', 'serial independent policies', 'errors abort the evaluation epoch'])
    if object_rows:
        manifest.update(object_cells=sorted(object_rows), cell_outputs=spec.cell_outputs,
                        object_cache_slots=sum(r['slots'] for r in object_rows.values()))
    if mixed:
        from .python_calculator import bundle_source
        python_source, required, used_inputs = bundle_source(cells, python_reasons, spec.inputs, enums)
        manifest.update(python_cells=python_reasons, python_source=python_source,
                        python_references=required, python_inputs=used_inputs,
                        python_enums={k: type_info(v) for k, v in enums.items()})
    if spec.model is not None:
        manifest['model'] = spec.model
        manifest['model_sha256'] = hashlib.sha256(json.dumps(spec.model, sort_keys=True).encode()).hexdigest()
    output_binder = BindNames(selected, spec.inputs, enums, {'t'},
                              native_lookups=True, frame_native=True,
                              frame_casts=frame_casts)
    frame_outputs = {
        'length': (_frame_expression(spec.length, output_binder, cast_names)
                   if spec.outputs else '0'),
        'tables': {table: [_frame_expression(expr, output_binder, cast_names)
                           for expr in columns.values()]
                   for table, columns in spec.outputs.items()},
    }
    return {
        'header': frame_header,
        'fields': frame_fields,
        'field_types': frame_field_types,
        'methods': tuple(frame_methods),
        'frame_outputs': frame_outputs,
        'runtime_prefix': tuple(lines[len(frame_header) + 2 + len(frame_fields):frame_runtime_start])
                          + tuple(lines[frame_runtime_end:frame_runtime_prefix_end]),
        'runtime': tuple(lines[len(frame_header) + 2 + len(frame_fields):frame_runtime_start])
                   + tuple(lines[frame_runtime_end:]),
    }, manifest


def build(spec, destination, *, diagnostics=False, python_cells=(),
          compile_native=True):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Build destination must be empty')
    destination.mkdir(parents=True, exist_ok=True)
    parts, manifest = emit(spec, diagnostics=diagnostics, python_cells=python_cells)
    if spec.cython_objects:
        manifest["cython_objects"] = True
    from .frame_codegen import to_c_frame
    source = to_c_frame(parts, manifest)
    if spec.cython_objects:
        from .gil_codegen import augment
        source = augment(source, manifest, spec, python_cells)
    if spec.cython_objects:
        lines=source.splitlines()
        directives=[x for x in lines if x.startswith('# cython:')]
        source='\n'.join(directives+[x for x in lines if not x.startswith('# cython:')])+'\n'
    if spec.model is not None:
        from .model_codegen import add_model_entries
        source = add_model_entries(source, manifest)
        if spec.cython_objects:
            from .collect_map_codegen import augment as add_collect_map
            source = add_collect_map(source, manifest, spec)
    from .coverage import record_formula_coverage
    record_formula_coverage(manifest, spec, python_cells)
    import Cython
    build_flags = {k: os.environ.get(k, '') for k in ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS')}
    build_flags.update(cython=Cython.__version__, python=sys.version)
    helper_bytes = b''.join(Path(__file__).with_name('lookups'+suffix).read_bytes() for suffix in ('.py','.pxd'))
    if spec.cython_objects:
        helper_bytes += b''.join(Path(__file__).with_name('table_lookup'+suffix).read_bytes() for suffix in ('.py','.pxd'))
    python_source = manifest.pop('python_source', '')
    space_source = ''
    if spec.model is not None and spec.model.get('space_formula'):
        from .space_formula_codegen import emit_space_preparer
        space_source = emit_space_preparer(spec.model['space_formula'])
        manifest['space_formula_source_sha256'] = hashlib.sha256(space_source.encode()).hexdigest()
    digest = hashlib.sha256(source.encode() + python_source.encode() + space_source.encode()
                            + helper_bytes + manifest.get('model_sha256', '').encode()
                            + json.dumps(build_flags,sort_keys=True).encode()).hexdigest()
    package = 'mxk_' + digest[:16]
    pkg = destination / package
    pkg.mkdir()
    (pkg / '__init__.py').write_text('', encoding='utf-8', newline='\n')
    (pkg / '_kernel.pyx').write_text(source, encoding='utf-8', newline='\n')
    if python_source:
        (pkg / '_python.py').write_text(python_source, encoding='utf-8', newline='\n')
        manifest['python_source_sha256'] = hashlib.sha256(python_source.encode()).hexdigest()
    if space_source:
        (pkg / '_space_formula.pyx').write_text(space_source, encoding='utf-8', newline='\n')
    for suffix in ('.py', '.pxd'):
        (pkg / ('_lookups' + suffix)).write_bytes(Path(__file__).with_name('lookups' + suffix).read_bytes())
    if spec.cython_objects:
        for suffix in ('.py','.pxd'):
            (pkg / ('_table_lookup'+suffix)).write_text(Path(__file__).with_name('table_lookup'+suffix).read_text().replace('from .lookups ', 'from ._lookups '))
    manifest.update(package=package, artifact_signature=digest, build_flags=build_flags, source_sha256=hashlib.sha256(source.encode()).hexdigest(), python=sys.version, source_file=inspect.getsourcefile(spec.module))
    if spec.model is not None:
        from .package_codegen import add_public_package
        add_public_package(destination, manifest, spec.initial_tables, spec.model_name)
    extensions = []
    if space_source:
        extensions.append(
            f"    Extension('{package}._space_formula', ['{package}/_space_formula.pyx']),")
    if spec.cython_objects:
        extensions.append(
            f"    Extension('{package}._table_lookup', ['{package}/_table_lookup.py']),")
    extensions.extend((
        f"    Extension('{package}._lookups', ['{package}/_lookups.py']),",
        f"    Extension('{package}._kernel', ['{package}/_kernel.pyx'], language='c++'),",
    ))
    setup = ("from setuptools import setup, Extension\n"
             "from Cython.Build import cythonize\n"
             "setup(ext_modules=cythonize([\n"
             + "\n".join(extensions) + "\n"
             "], compiler_directives={'language_level':3,'infer_types':True}, annotate=True))\n")
    (destination / 'setup.py').write_text(setup)
    manifest['setup_sha256'] = hashlib.sha256((destination / 'setup.py').read_bytes()).hexdigest()
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if compile_native:
        compile_generated(destination)
    return destination


def compile_generated(destination):
    """Compile a translated artifact after checking its recorded Cython sources."""
    destination = Path(destination).resolve()
    manifest = json.loads((destination / 'manifest.json').read_text(encoding='utf-8'))
    package = manifest.get('package')
    if (not isinstance(package, str) or not package.startswith('mxk_')
            or not package[4:].isalnum()):
        raise ContractError('Invalid generated package identity')
    source = destination / package / '_kernel.pyx'
    if hashlib.sha256(source.read_text(encoding='utf-8').encode()).hexdigest() != manifest.get('source_sha256'):
        raise ContractError('Generated Cython source differs from translation receipt')
    for name, digest in (('_python.py', manifest.get('python_source_sha256')),
                         ('_space_formula.pyx', manifest.get('space_formula_source_sha256'))):
        if digest is not None:
            path = destination / package / name
            if hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest() != digest:
                raise ContractError('Generated source differs from translation receipt: ' + name)
    if not (destination / 'setup.py').is_file():
        raise ContractError('Translated artifact has no setup.py')
    if ('setup_sha256' in manifest and
            hashlib.sha256((destination / 'setup.py').read_bytes()).hexdigest()
            != manifest['setup_sha256']):
        raise ContractError('Generated setup.py differs from translation receipt')
    with (destination / 'build.log').open('w') as log:
        result = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=destination,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Cython build failed; see ' + str(destination / 'build.log'))
    return destination
