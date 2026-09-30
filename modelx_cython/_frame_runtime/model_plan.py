"""Derive physical bindings from the small lookup vocabulary and pandas schemas.

Only helper signatures, literal columns, index layouts and declared types are read.
Formula control flow and ordinary Python/pandas execution are never interpreted.
The resulting JSON model plan is sufficient to bind an artifact without its source.
"""
from enum import IntEnum
import ast
import hashlib
import inspect
import json
import types
from .contracts import ContractError, KernelSpec, TableKey
from .compiler import FrontendRewrite, function_ast, type_info
from .lookups import (Curve, CurveTable, IndexedFloat1, IndexedFloat2,
                      IndexedInt1, IndexedInt2)
from .python_calculator import referenced_names


def _name(*parts):
    text = '__'.join(str(p) for p in parts)
    safe = ''.join(c if c.isalnum() or c == '_' else '_' for c in text)
    return 'input_' + safe + (('_' + hashlib.sha256(text.encode()).hexdigest()[:8]) if safe != text else '')


def _enum(typ):
    return inspect.isclass(typ) and issubclass(typ, IntEnum)


def _dtype(series):
    from pandas.api.types import is_bool_dtype, is_integer_dtype, is_float_dtype
    dtype = series.dtype
    return bool if is_bool_dtype(dtype) else int if is_integer_dtype(dtype) else float if is_float_dtype(dtype) else object


def derive_spec(module, tables, *, references, space_formula, outputs, steps, length,
                scalar_outputs=(), cell_outputs=(), data_space='Data', projection='Projection',
                allow_none=None, model_name=None, point_table=None, cython_objects=False,
                row_references=None, point_parameter=None, parameter_initial_values=None,
                reference_types=None, static_projection=False):
    """Build a kernel contract without author INPUTS/FRONTEND maps.

    Supported structure: one Data Space with zero-argument DataFrame Cells and
    one independent-policy dynamic Space. Direct dynamic Spaces can instead
    bind their selector on each ItemSpace without a finite policy table.
    Grouped lookup tables have exactly two named index levels, one numeric axis.
    Ambiguous numeric/numeric indexes use the explicit (group, x) convention.
    """
    import pandas as pd
    reserved = {'artifact', 'manifest', 'reset', 'close'}
    if model_name is None:
        stem = module.__name__.rsplit('.', 1)[-1]
        model_name = ''.join(part[:1].upper() + part[1:] for part in stem.split('_') if part) or 'mx_model'
    if (not isinstance(model_name, str) or not model_name.isidentifier() or model_name.startswith('_')
            or model_name in {'mx_model'}):
        raise ContractError('model_name must be a public identifier distinct from mx_model')
    if (data_space == projection or any(not n.isidentifier() or n.startswith('_') or n in reserved
                                        for n in (data_space, projection))):
        raise ContractError('Distinct non-reserved Data and Projection Space names required')
    signature = inspect.signature(space_formula)
    parameters = list(signature.parameters.values())
    if not parameters or any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters):
        raise ContractError('A policy parameter and ordinary positional Space parameters are required')
    point = point_parameter or parameters[0].name
    if point not in signature.parameters:
        raise ContractError('The policy selector is not a Space parameter')
    if signature.parameters[point].default is not inspect.Parameter.empty:
        raise ContractError('The point selector must be a required Space parameter')
    initial = dict(parameter_initial_values or {})
    required = {p.name for p in parameters if p.name != point and p.default is p.empty}
    if set(initial) != required:
        raise ContractError('Required non-policy Space parameters need initial values')
    defaults = {p.name: (initial[p.name] if p.name in required else p.default)
                for p in parameters if p.name != point}
    for value in [*references.values(), *defaults.values()]:
        try: json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            raise ContractError('Manifest defaults must be JSON values; bind complex Data through Data Cells') from None
    if set(references) & set(signature.parameters):
        raise ContractError('Space parameters and references must have distinct names')
    reference_types = dict(reference_types or {})
    if set(reference_types) - set(references) or any(
            value is not object for value in reference_types.values()):
        raise ContractError('Only declared object reference types are supported')
    import numpy as np
    for table, frame in tables.items():
        allowed = (pd.DataFrame, pd.Series, np.ndarray) if cython_objects else (pd.DataFrame,)
        if not table.isidentifier() or not isinstance(frame, allowed):
            raise ContractError('Data must be named DataFrames, or opaque Series/ndarrays in Cython-object mode')
    cells = {n: f for n, f in vars(module).items() if inspect.isfunction(f) and hasattr(f, '__kernel_cell__')}
    trees = {n: function_ast(f) for n, f in cells.items()}
    calls = [(name, node) for name, tree in trees.items() for node in ast.walk(tree)
             if isinstance(node, ast.Call) and FrontendRewrite._lookup_name(node.func)]
    # Object-valued Cells keep their checked Cython/Python-semantic bodies, but
    # they must not disable native bindings for independent numeric helper
    # chains in the same Space. The emitter decides eligibility per Cell.
    frontend = dict(fields={}, keys={}, lookups={}, arrays={})
    inputs, bindings, key_demands, fields, point_tables = {}, {}, {}, {}, set()
    # Plans retain original index order, so replacing a table cannot silently alter keys.
    schemas = {n: dict(index_names=list(t.index.names) if isinstance(t, (pd.DataFrame, pd.Series)) else [],
                       columns={}, object_kind=type(t).__name__) for n, t in tables.items()}

    def validate_native(table):
        frame = tables[table]
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ContractError('Duplicate table keys/columns: ' + table)
        names = list(frame.index.names)
        if len(names) not in (1, 2) or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
            raise ContractError('Native indexes need one or two uniquely named levels: ' + table)

    def table_of(node):
        table = FrontendRewrite._data_table(node)
        if table not in tables: raise ContractError('Lookup requires a known data.<table>()')
        return table

    def column_of(node, table):
        column = FrontendRewrite._literal_string(node, 'lookup column')
        if column not in tables[table]: raise ContractError(f'Missing column: {table}.{column}')
        return column

    def add(name, typ, binding):
        if name in bindings and bindings[name] != binding:
            raise ContractError('Inconsistent lookup contract: ' + name)
        inputs[name], bindings[name] = typ, binding
        return name

    # A direct numeric Data-array read keeps the original formula's index
    # expressions. Only the array owner and checked element access are replaced.
    # Slices, advanced indexing and unrecognised dtypes stay on the Python path.
    array_types = {(1, 'int64'): IndexedInt1, (1, 'float64'): IndexedFloat1,
                   (2, 'int64'): IndexedInt2, (2, 'float64'): IndexedFloat2}
    for tree in trees.values():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Load):
                continue
            table = FrontendRewrite._data_table(node.value)
            value = tables.get(table)
            if not isinstance(value, np.ndarray):
                continue
            typ = array_types.get((value.ndim, str(value.dtype)))
            if typ is None:
                continue
            indices = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            if len(indices) != value.ndim or any(isinstance(i, ast.Slice) for i in indices):
                continue
            physical = add(_name(table), typ, dict(kind='indexed_array', table=table,
                                                   array_type=typ.__name__))
            frontend['arrays'][table] = physical

    def expr_type(expr, owner):
        if isinstance(expr, ast.Name):
            p = inspect.signature(cells[owner]).parameters.get(expr.id)
            return p.annotation if p else None
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in cells:
            return inspect.signature(cells[expr.func.id]).return_annotation
        return None

    # A key helper declares a label domain, not a policy/scenario selection.
    for owner, call in calls:
        helper = FrontendRewrite._lookup_name(call.func)
        if call.keywords: raise ContractError('Native lookup syntax uses positional helper arguments')
        if helper == 'key':
            if len(call.args) != 2: raise ContractError('lookup.key requires two arguments')
            table = table_of(call.args[0]); arg = call.args[1]
            if tables[table].index.nlevels != 2:
                raise ContractError('lookup.key requires a grouped table')
            if isinstance(arg, ast.Name) and arg.id in defaults:
                key_demands[('parameter', arg.id)] = table
            elif (isinstance(arg, ast.Call) and FrontendRewrite._lookup_name(arg.func) == 'field'):
                ft = table_of(arg.args[0]); col = column_of(arg.args[2], ft)
                key = ('field', ft, col)
                if key in key_demands and key_demands[key] != table:
                    raise ContractError('One field cannot use incompatible key domains')
                key_demands[key] = table
            elif not (isinstance(arg, ast.Constant) or isinstance(arg, ast.Name)):
                raise ContractError('lookup.key labels must be a field, parameter or literal')

    # Curves first, so field/parameter domain contracts refer to a stable input.
    domains = {}
    for owner, call in calls:
        helper = FrontendRewrite._lookup_name(call.func)
        if helper in ('field', 'category', 'key'): continue
        if helper not in ('at', 'get', 'previous', 'previous_group', 'linear'):
            raise ContractError('Unknown lookup helper: ' + helper)
        table = table_of(call.args[0]); ci = 3 if helper == 'previous_group' else 2
        validate_native(table)
        column = column_of(call.args[ci], table); frame = tables[table]
        if _dtype(frame[column]) not in (float, int):
            raise ContractError('Native curve columns must be numeric: ' + table + '.' + column)
        schemas[table]['columns'][column] = 'numeric'
        cfgs = frontend['lookups'].setdefault(table, {})
        grouped = frame.index.nlevels == 2
        binding = dict(kind='curve', table=table, column=column)
        if not grouped:
            physical = add(_name(table, column), Curve, binding)
            cfg = dict(helper=helper, input=physical)
        else:
            if helper == 'previous_group':
                group_level, group_expr = 0, call.args[1]
            elif isinstance(call.args[1], ast.Tuple) and len(call.args[1].elts) == 2:
                from pandas.api.types import is_numeric_dtype
                numeric = [is_numeric_dtype(frame.index.get_level_values(i).dtype) for i in range(2)]
                group_level = 1 if numeric == [True, False] else 0
                group_expr = call.args[1].elts[group_level]
            else:
                raise ContractError('Grouped lookups need a two-element tuple or previous_group')
            x_level = 1 - group_level
            if isinstance(group_expr, ast.Constant):
                label = group_expr.value
                physical = add(_name(table, column, label), Curve,
                               dict(binding, group_level=group_level, fixed_group=label))
                cfg = dict(helper=helper, fixed_groups={label: physical}, key_index=x_level, group_index=group_level)
            else:
                enum = expr_type(group_expr, owner)
                codes = {k: int(v) for k, v in enum.__members__.items()} if _enum(enum) else None
                physical = add(_name(table, column), CurveTable,
                               dict(binding, kind='curve_table', group_level=group_level, codes=codes))
                cfg = dict(helper=helper, input=physical, tuple_order=(group_level, x_level))
                domains.setdefault(table, physical)
            if column in cfgs and 'fixed_groups' in cfgs[column] and 'fixed_groups' in cfg:
                cfgs[column]['fixed_groups'].update(cfg['fixed_groups']); continue
        if column in cfgs and cfgs[column] != cfg:
            raise ContractError('Conflicting helper/layout for column: ' + table + '.' + column)
        cfgs[column] = cfg
    for table in set(key_demands.values()):
        if table not in domains:
            raise ContractError('A key domain needs a grouped numeric lookup from the same table: ' + table)
        frontend['keys'][table] = domains[table]

    for owner, call in calls:
        helper = FrontendRewrite._lookup_name(call.func)
        if helper not in ('field', 'category'): continue
        table = table_of(call.args[0]); column = column_of(call.args[2], table)
        if not isinstance(call.args[1], ast.Name) or call.args[1].id != point:
            # Never substitute the current policy for an arbitrary lookup key.
            continue  # compiler keeps this formula in Python, against the Data binding
        validate_native(table)
        if tables[table].index.nlevels != 1: raise ContractError('Policy fields need a one-level index')
        point_tables.add(table)
        domain = key_demands.get(('field', table, column))
        if helper == 'category':
            if len(call.args) != 4 or not isinstance(call.args[3], ast.Name):
                raise ContractError('category requires a declared enum')
            typ = getattr(module, call.args[3].id)
            if not _enum(typ): raise ContractError('category requires an IntEnum')
        elif domain:
            typ = TableKey(domains[domain])
        else:
            typ = _dtype(tables[table][column])
        name = add(_name(table, column), typ, dict(kind='field', table=table, column=column))
        previous = fields.setdefault((table, column), type_info(typ))
        if previous != type_info(typ): raise ContractError('Incompatible field contracts')
        schemas[table]['columns'][column] = 'field'
        frontend['fields'].setdefault(table, {})[column] = name
    if point_table is not None:
        if point_table not in tables:
            raise ContractError('Unknown declared policy table: ' + point_table)
        if point_tables and point_tables != {point_table}:
            raise ContractError('Declared policy table conflicts with lookup.field/category')
        if tables[point_table].index.nlevels != 1 or not tables[point_table].index.is_unique:
            raise ContractError('A policy table needs a unique single-level index')
    elif point_tables:
        if len(point_tables) != 1:
            raise ContractError('Exactly one policy table must be identifiable from lookup.field/category')
        point_table = next(iter(point_tables))
    used = set().union(*(referenced_names(f) for f in cells.values()))
    row_references = dict(row_references or {})
    if row_references and (point_table is None or not isinstance(tables[point_table], pd.DataFrame)):
        raise ContractError('Per-policy references require a DataFrame policy table')
    if set(row_references) & (set(references) | set(defaults) | set(cells)
                              | {point, 'data', 'lookup'}):
        raise ContractError('Per-policy reference names must not shadow references, parameters or Cells')
    for name, column in row_references.items():
        if (not isinstance(name, str) or not name.isidentifier() or name.startswith('_')
                or not isinstance(column, str) or column not in tables[point_table].columns):
            raise ContractError('Per-policy reference needs a named policy-table column: ' + str(name))
        if name in used:
            add(name, object, dict(kind='row_reference', table=point_table, column=column))
            schemas[point_table]['columns'][column] = 'object'
    # Declared references are model data, not inferred actuarial assumptions.
    if point_table is None and not static_projection and point in used:
        # A direct dynamic Space has no finite portfolio domain. Its selector is
        # an object-valued input in the single template record; each ItemSpace
        # receives its own checked override and independently owned cache.
        add(point, object, dict(kind='point'))
    if frontend['arrays'] and point in used and point_table is not None:
        from pandas.api.types import is_integer_dtype
        index = tables[point_table].index
        if is_integer_dtype(index.dtype) and all(
                not isinstance(value, bool) and -(1 << 63) <= int(value) < (1 << 63)
                for value in index):
            add(point, int, dict(kind='point'))
    for name, value in {**references, **defaults}.items():
        if name not in used: continue
        domain = key_demands.get(('parameter', name))
        typ = (TableKey(domains[domain], value if type(value) is int and value < 0 else None)
               if domain else reference_types.get(name, type(value)))
        add(name, typ, dict(kind='parameter' if name in defaults else 'reference', name=name))
    # Domain annotations name public Data Cells, not generated storage names.
    clone = types.ModuleType(module.__name__)
    clone.__dict__.update(vars(module))
    for name, fn in cells.items():
        annotations = {k: TableKey(domains[v.table], v.default) if isinstance(v, TableKey) and v.table in domains else v
                       for k, v in fn.__annotations__.items()}
        if annotations != fn.__annotations__:
            bound = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
            bound.__dict__.update(fn.__dict__); bound.__annotations__ = annotations
            bound.__module__ = fn.__module__; clone.__dict__[name] = bound
    # Stable generic ordering; the model author never names the physical inputs.
    for schema in schemas.values(): schema['native'] = bool(schema['columns'])
    inputs = dict(sorted(inputs.items()))
    plan = dict(schema='mxcf.model.v1', data_space=data_space, projection=projection,
                point_parameter=point, point_table=point_table, parameter_defaults=defaults,
                static_projection=static_projection,
                required_parameters=sorted(required),
                space_parameters=[(p.name, p.kind.name) for p in parameters],
                reference_defaults=dict(references), bindings=bindings, tables=schemas,
                row_references=row_references, output_index='t')
    frontend['point_parameter'] = point
    # Freeze the author-time Data snapshot into the build contract. The generated
    # compiled package owns this initial data exactly as an exported model package does.
    initial_tables = {name: frame.copy() if isinstance(frame, np.ndarray) else frame.copy(deep=True)
                      for name, frame in tables.items()}
    return KernelSpec(clone, inputs, outputs, steps, length, scalar_outputs=tuple(scalar_outputs),
                      parameters=tuple(n for n in defaults if n in inputs), frontend=frontend,
                      cell_outputs=tuple(cell_outputs), allow_none=allow_none, model=plan,
                      model_name=model_name, initial_tables=initial_tables, cython_objects=cython_objects)
