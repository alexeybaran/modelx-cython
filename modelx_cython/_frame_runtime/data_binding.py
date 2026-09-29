"""Pack complete declared tables, without moving group selection out of formulas."""
from enum import IntEnum
import numbers
from .contracts import ContractError
from .lookups import curve, curve_table


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


def prepare_inputs(manifest, tables, *, references=None, packed=None):
    """Internal binding only. Runtime validation is authoritative for scalar values."""
    plan = manifest['model']
    refs = {**plan['reference_defaults'], **dict(references or {})}
    if set(refs) != set(plan['reference_defaults']): raise ContractError('Unknown Projection reference')
    packed = pack_tables(plan, tables) if packed is None else packed
    points = tables[plan['point_table']]
    if not len(points): raise ContractError('At least one policy is required')
    rows = []
    for point in points.index:
        row = dict(packed)
        for name, info in manifest['inputs'].items():
            binding = plan['bindings'][name]; kind = binding['kind']
            if kind in ('curve', 'curve_table'): continue
            if kind == 'field':
                value = tables[binding['table']].at[point, binding['column']]
                if info['kind'] == 'enum':
                    if isinstance(value, str):
                        try: value = info['members'][value.upper()]
                        except KeyError: raise ContractError(name + ': unknown category') from None
                    elif isinstance(value, IntEnum): value = int(value)
            elif kind == 'point':
                value = point
            elif kind == 'parameter':
                value = plan['parameter_defaults'][binding['name']]
            else:
                value = refs[binding['name']]
            row[name] = value
        rows.append(row)
    return rows
