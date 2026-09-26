"""Fixed experimental lookup adapter. Full inputs are prepared before selection.

Prepared data are owned immutable snapshots. Rebuild the compiled model after
changing source tables, just as cached table Cells require reconstruction. This
module is not a pandas interpreter and does not change author-facing helpers.
"""
import math
from enum import IntEnum
import numpy as np
import pandas as pd
import cython
if cython.compiled:
    from cython.cimports._mx_curves import Curve, CurveTable  # relocated by builder
else:
    from ._mx_curves import Curve, CurveTable


def _normal(value, labels):
    if isinstance(value, IntEnum):
        if value.name in labels:
            return value.name
        if value.name.lower() in labels:
            return value.name.lower()
    return value


class FieldTable:
    def __init__(self, table, column):
        if not isinstance(table,pd.DataFrame) or table.index.nlevels != 1 or not table.index.is_unique:
            raise ValueError('field/category preparation requires a unique one-level table')
        self._values = dict(table[column].items())

    def at(self, key):
        return self._values[_normal(key,self._values)]

    def category(self, key, enum_type):
        value=self.at(key)
        if isinstance(value,enum_type): return value
        if isinstance(value,str): return enum_type[value.upper()]
        return enum_type(int(value))


class GroupKeys:
    """Validate original public first-axis labels without repeated pandas work."""
    def __init__(self, table):
        if table.index.nlevels < 2:
            raise ValueError('key() requires a grouped table')
        self._labels = dict.fromkeys(table.index.get_level_values(0).unique())

    def key(self, value):
        normalized = _normal(value, self._labels)
        if normalized not in self._labels:
            raise KeyError(value)
        return normalized


def prepare(table, column, kind, axis):
    if not isinstance(table,pd.DataFrame) or not table.index.is_unique:
        raise ValueError('Prepared lookups require a DataFrame with unique keys')
    if kind=='field': return FieldTable(table,column)
    if kind=='labels': return GroupKeys(table)
    values=np.asarray(table[column])
    if values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
        raise ValueError('Numeric lookups require finite numeric values')
    levels=table.index.nlevels
    if levels not in (1,2) or axis not in range(levels):
        raise ValueError('Expected a one- or two-level numeric curve')
    raw=np.asarray(table.index.get_level_values(axis))
    if raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ValueError('Curve keys must be finite numeric values')
    if raw.dtype.kind in 'iu' and any(int(x)!=float(x) for x in raw):
        raise ValueError('Integer curve keys must be exactly representable in float64')
    if levels==1: return Curve(raw,values)
    groups=list(table.index.get_level_values(1-axis))
    labels=[]
    rows={}
    for group,x,y in zip(groups,raw,values):
        if isinstance(group,np.integer): group=int(group)
        if type(group) not in (str,int): raise ValueError('Group labels must be strings or integers')
        if group not in rows: labels.append(group);rows[group]=[]
        rows[group].append((x,y))
    offsets=[0];xx=[];yy=[]
    for label in labels:
        pairs=sorted(rows[label])
        xx.extend(x for x,y in pairs);yy.extend(y for x,y in pairs);offsets.append(len(xx))
    return CurveTable(np.arange(len(labels),dtype=np.int64),np.asarray(offsets,dtype=np.int64),xx,yy,labels)
