"""Pack complete declared tables, without moving group selection out of formulas."""
from dataclasses import dataclass
from enum import IntEnum
import numbers
from .contracts import ContractError
from .lookups import (IndexedFloat1, IndexedFloat2, IndexedInt1,
                      IndexedInt2, curve, curve_table)


@dataclass(frozen=True)
class PreparedModelRows:
    """Internal row set with fixed values established by ``prepare_inputs``.

    Only internal resource preparation constructs this marker. Public
    CompiledKernel.bind records continue to take the complete validation path.
    ``fixed_names`` contains
    values known to be identical and safe to convert once across this binding.
    """

    rows: tuple[dict, ...]
    fixed_names: frozenset[str]


def _enum_codes(values, members):
    labels = set(values)
    result = {}
    for name, code in members.items():
        candidates = [label for label in labels if (isinstance(label, str) and label.upper() == name)
                      or (isinstance(label, numbers.Integral) and int(label) == code)]
        if len(candidates) > 1: raise ContractError('Ambiguous enum labels: ' + name)
        for label in candidates: result[label] = code
    if labels - result.keys(): raise ContractError('Table contains undeclared enum labels')
    return result


def pack_tables(model_plan, tables):
    """Validate stable schema and pack each used column once per input revision."""
    import pandas as pd
    from pandas.api.types import is_numeric_dtype
    if set(tables) != set(model_plan['tables']):
        raise ContractError('Data table names differ from the artifact schema')
    for name, schema in model_plan['tables'].items():
        frame = tables[name]
        import numpy as np
        expected = {'DataFrame': pd.DataFrame, 'Series': pd.Series, 'ndarray': np.ndarray}.get(schema.get('object_kind', 'DataFrame'))
        if expected is None or not isinstance(frame, expected):
            raise ContractError(name + ' has a different Data object kind')
        if not schema['native']: continue  # Opaque pandas stays Python's responsibility.
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ContractError('Duplicate Data keys/columns: ' + name)
        if list(frame.index.names) != schema['index_names']:
            raise ContractError('Changed Data index layout: ' + name)
        if set(schema['columns']) - set(frame.columns):
            raise ContractError('Missing Data columns: ' + name)
        for column, role in schema['columns'].items():
            if role == 'numeric' and not is_numeric_dtype(frame[column].dtype):
                raise ContractError('Numeric curve column required: ' + name + '.' + column)
    packed = {}
    for name, binding in model_plan['bindings'].items():
        kind = binding['kind']
        if kind == 'indexed_array':
            values = tables[binding['table']]
            constructors = {'IndexedInt1': IndexedInt1, 'IndexedFloat1': IndexedFloat1,
                            'IndexedInt2': IndexedInt2, 'IndexedFloat2': IndexedFloat2}
            # The declared ABI, not a newly supplied array's dtype, selects the
            # wrapper. A rebinding with a changed numeric kind fails atomically.
            packed[name] = constructors[binding['array_type']](values)
            continue
        if kind not in ('curve', 'curve_table'): continue
        series = tables[binding['table']][binding['column']]
        if 'fixed_group' in binding:
            try:
                series = series.xs(binding['fixed_group'], level=binding['group_level'])
            except KeyError:
                # Empty exact curves preserve lookup.get's declared default;
                # required lookup.at still fails on an actual call, not at binding.
                series = pd.Series([], index=pd.Index([], dtype=float), dtype=float)
        if kind == 'curve':
            packed[name] = curve(series)
        else:
            codes = binding['codes']
            if codes:
                codes = _enum_codes(series.index.get_level_values(binding['group_level']), codes)
            packed[name] = curve_table(series, group=series.index.names[binding['group_level']], codes=codes)
    return packed


def prepare_inputs(manifest, tables, *, references=None, packed=None,
                   prepared=False):
    """Internal binding only. Runtime validation is authoritative for scalar values."""
    plan = manifest['model']
    refs = {**plan['reference_defaults'], **dict(references or {})}
    if set(refs) != set(plan['reference_defaults']): raise ContractError('Unknown Projection reference')
    packed = pack_tables(plan, tables) if packed is None else packed
    point_table = plan['point_table']
    points = tables[point_table] if point_table is not None else None
    point_index = points.index if points is not None else (0 if plan.get('static_projection') else None,)
    if not len(point_index): raise ContractError('At least one policy is required')
    # A policy table is normally read repeatedly for every bound scalar. Select
    # each aligned column once, while retaining the original label lookup for
    # other layouts (including duplicated or differently ordered indexes).
    field_columns = {}
    field_bindings = []
    point_bindings = []
    fixed = dict(packed)
    for name, info in manifest['inputs'].items():
        binding = plan['bindings'][name]
        kind = binding['kind']
        if kind in ('curve', 'curve_table', 'indexed_array'):
            continue
        if kind in ('field', 'row_reference', 'row_reference_numeric'):
            table = tables[binding['table']]
            if (points is not None and table.index.is_unique and points.index.is_unique
                    and table.index.equals(points.index)):
                column_key = (binding['table'], binding['column'])
                if column_key not in field_columns:
                    field_columns[column_key] = table[binding['column']].to_numpy(copy=False)
                column = field_columns[column_key]
            else:
                column = None
            field_bindings.append((name, info, binding, column))
        elif kind == 'point':
            point_bindings.append(name)
        elif kind == 'parameter':
            fixed[name] = plan['parameter_defaults'][binding['name']]
        else:
            fixed[name] = refs[binding['name']]
    rows = []
    for position, point in enumerate(point_index):
        row = fixed.copy()
        for name in point_bindings:
            row[name] = point
        for name, info, binding, column in field_bindings:
            value = (column[position] if column is not None else
                     tables[binding['table']].at[point, binding['column']])
            if binding['kind'] == 'row_reference_numeric':
                field = binding['field']
                if field is not None:
                    try:
                        value = value[field]
                    except (KeyError, IndexError, TypeError) as exc:
                        raise ContractError(name + ': missing numeric row field') from exc
                # PreparedModelRows marks this as an internal binding. Its
                # artifact-specific native wrapper is built once in runtime,
                # avoiding a second array snapshot during validation.
            if info['kind'] == 'enum':
                if isinstance(value, str):
                    try: value = info['members'][value.upper()]
                    except KeyError: raise ContractError(name + ': unknown category') from None
                elif isinstance(value, IntEnum): value = int(value)
            row[name] = value
        rows.append(row)
    if not prepared:
        return rows
    fixed_names = frozenset(
        name for name, value in fixed.items()
        if (manifest['inputs'][name]['kind'] in ('curve', 'curve_table', 'indexed_array')
            or (manifest['inputs'][name]['kind'] != 'table_key'
                and type(value) in (int, float, bool, str, type(None)))))
    return PreparedModelRows(tuple(rows), fixed_names)
