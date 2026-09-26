"""Small author-facing lookup vocabulary over ordinary pandas tables.

These functions are intentionally boring Python. Live modelx/exported Python calls them
against DataFrame inputs. The native compiler recognizes the same calls and lowers them
to prepared Curve/CurveTable operations declared in KernelSpec.frontend.
"""
from __future__ import annotations
import math
from enum import IntEnum


def _normalize_key(table, key):
    """Translate typed enum keys back to ordinary pandas labels when necessary."""
    if isinstance(key, tuple):
        values = []
        for level, value in enumerate(key):
            if isinstance(value, IntEnum):
                labels = table.index.get_level_values(level)
                if value.name in labels:
                    value = value.name
                elif value.name.lower() in labels:
                    value = value.name.lower()
            values.append(value)
        return tuple(values)
    if isinstance(key, IntEnum):
        labels = table.index
        if key.name in labels:
            return key.name
        if key.name.lower() in labels:
            return key.name.lower()
    return key


def field(table, key, column):
    return table.loc[_normalize_key(table, key), column]


def key(table, value):
    """Validate and return a public table-group label without selecting a group."""
    if getattr(table.index, 'nlevels', 1) < 2:
        raise ValueError('key() requires a grouped table')
    labels = table.index.get_level_values(0)
    normalized = _normalize_key(table, value)
    if normalized not in labels:
        raise KeyError(value)
    return normalized


def category(table, key, column, enum_type):
    value = table.loc[_normalize_key(table, key), column]
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        return enum_type[value.upper()]
    return enum_type(int(value))


def at(table, key, column):
    return float(table.loc[_normalize_key(table, key), column])


def get(table, key, column, default=0.0):
    key = _normalize_key(table, key)
    return float(table.loc[key, column]) if key in table.index else float(default)


def previous(table, key, column, below=math.nan):
    """Step lookup on an ungrouped numeric index."""
    candidates = [x for x in table.index if x <= key]
    if not candidates:
        if math.isnan(below):
            raise KeyError(key)
        return float(below)
    return float(table.loc[max(candidates), column])


def previous_group(table, group, key, column, below=math.nan):
    sub = table.loc[group]
    return previous(sub, key, column, below)


def linear(table, key, column):
    """Linear interpolation on an ungrouped numeric index; hold above range."""
    if len(table.index) == 0:
        raise KeyError(key)
    lo = [x for x in table.index if x <= key]
    if not lo:
        raise KeyError(key)
    a = max(lo)
    if a == table.index.max():
        return float(table.loc[a, column])
    hi = min(x for x in table.index if x > key)
    y0, y1 = float(table.loc[a, column]), float(table.loc[hi, column])
    return y0 + (y1-y0) * (key-a) / (hi-a)


def max_key(table, group=None):
    """Maximum final-axis key of one group, or of an ungrouped table."""
    return table.index.max() if group is None else table.loc[group].index.max()


def max_level(table, level):
    """Maximum of a named or positional index level across the entire table."""
    return table.index.get_level_values(level).max()


def keys(table, group=None):
    """Full/group keys in original index order; no policy-dependent preselection."""
    return list(table.index if group is None else table.loc[group].index)


def group_curve(table, group, column, axis=-1):
    """Select an owned numeric curve in a model formula, not in a run script.

    The selected group contains every value along `axis` (an integer level
    position, or an unambiguous index-level name). Python and compiled formulas
    use the same Curve.at/get/previous semantics and finite float64 key contract.
    """
    import pandas as pd
    from .lookups import Curve
    if not isinstance(axis,int):
        if list(table.index.names).count(axis)!=1:raise ValueError('Missing or ambiguous axis')
        axis=list(table.index.names).index(axis)
    axis=axis % table.index.nlevels
    selected=[]
    for key,row in zip(table.index,table.itertuples(index=False,name=None)):
        parts=(key,) if table.index.nlevels==1 else key
        rem=parts[:axis]+parts[axis+1:]
        actual=None if not rem else rem[0] if len(rem)==1 else rem
        if actual==group:selected.append((parts[axis],row[table.columns.get_loc(column)]))
    if not selected:raise KeyError(group)
    selected.sort(key=lambda row:row[0])
    return Curve([x for x,y in selected],[y for x,y in selected])
