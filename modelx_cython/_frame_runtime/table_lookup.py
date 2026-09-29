"""Prepared full-table indexes for Python-semantic lookup calls.

This is a fixed helper implementation, not a pandas expression interpreter. All
rows/groups remain available; formulas, not preparation, choose their labels.
"""
from bisect import bisect_right
from enum import IntEnum
import math
import numpy as np
import cython
from .lookups import CurveTable, Curve


def _normalize_enum(value, labels):
    # Mirror lookup._normalize_key for the prepared-table Python/GIL path.
    # IntEnum is a typed formula domain; pandas tables retain their ordinary
    # string labels, so convert only when that name actually exists.
    if isinstance(value, IntEnum):
        if value.name in labels:
            return value.name
        lower=value.name.lower()
        if lower in labels:
            return lower
    return value


def _normalize_key(key, labels, levels):
    if levels==1:
        return _normalize_enum(key,labels[0])
    if isinstance(key,tuple) and len(key)==levels:
        values=[]
        for i,value in enumerate(key):
            values.append(_normalize_enum(value,labels[i]))
        return tuple(values)
    return key


class IndexedTable:
    def __init__(self, table):
        if not table.index.is_unique or not table.columns.is_unique:
            raise ValueError('Prepared lookup requires unique keys and columns')
        self.columns=tuple(table.columns)
        self.rows={key:tuple(row) for key,row in zip(table.index,table.itertuples(index=False,name=None))}
        self.positions={c:i for i,c in enumerate(self.columns)}
        self.levels=table.index.nlevels
        self.names=tuple(table.index.names)
        self.labels=tuple(frozenset(table.index.get_level_values(i)) for i in range(self.levels))
        self.maximum=table.index.max()
        self.maxima=tuple(table.index.get_level_values(i).max() for i in range(self.levels))
        self.groups={}
        if self.levels==1:
            self.groups[None]=list(table.index)
        else:
            for key in table.index:
                group=key[0] if self.levels==2 else key[:-1]
                self.groups.setdefault(group,[]).append(key[-1])
        self.ordered={}
        self.native={}
        self.curves={}
        self.codes={}
        self.numeric_axis=-1
        # Order only the final lookup axis. Opaque keys still support exact lookups.
        for group,keys in self.groups.items():
            try:self.ordered[group]=tuple(sorted(keys))
            except TypeError:pass

        # Reuse the existing full-table numeric representation. No policy,
        # scenario or lookup label is selected during preparation.
        axes=[i for i in range(self.levels) if table.index.get_level_values(i).dtype.kind in 'iuf']
        for axis in reversed(axes):
            raw=table.index.get_level_values(axis).to_numpy()
            if len(raw) and np.isfinite(raw).all() and np.all(np.abs(raw)<=2**53):
                self.numeric_axis=axis;break
        if self.numeric_axis>=0:
            axis=self.numeric_axis
            grouped={}
            for key,row in self.rows.items():
                parts=(key,) if self.levels==1 else key
                rem=parts[:axis]+parts[axis+1:]
                group=None if not rem else rem[0] if len(rem)==1 else rem
                grouped.setdefault(group,[]).append((parts[axis],row))
            self.codes={g:i for i,g in enumerate(grouped)}
            ordered=[sorted(v,key=lambda item:item[0]) for v in grouped.values()]
            offsets=[0]
            for rows in ordered:offsets.append(offsets[-1]+len(rows))
            xs=[x for rows in ordered for x,_ in rows]
            for column in self.columns:
                col=self.positions[column]
                if table[column].dtype.kind not in 'biuf':continue
                ys=[row[col] for rows in ordered for _,row in rows]
                if np.isfinite(ys).all():
                    for group,items in zip(grouped,ordered):
                        self.curves[(group,column)]=Curve([x for x,row in items],[row[col] for x,row in items])
                    self.native[column]=CurveTable(np.arange(len(ordered),dtype=np.int64),
                        np.asarray(offsets,dtype=np.int64),xs,ys)

    def _full(self, group, key):
        if self.levels==1:return key
        return (*group,key) if self.levels>2 else (group,key)

    @cython.locals(curve=CurveTable, group_id=cython.longlong)
    def at(self,key,column):
        key=_normalize_key(key,self.labels,self.levels)
        if self.native:
            parts=(key,) if self.levels==1 else key
            if isinstance(parts,tuple) and len(parts)==self.levels:
                x=parts[self.numeric_axis]
                if type(x) in (int,float) and -2**53<=x<=2**53:
                    curve=self.native.get(column)
                    if curve is not None:
                        rem=parts[:self.numeric_axis]+parts[self.numeric_axis+1:]
                        group=None if not rem else rem[0] if len(rem)==1 else rem
                        group_id=self.codes[group]
                        return curve.at(group_id,x)
        return float(self.rows[key][self.positions[column]])

    def get(self,key,column,default=0.0):
        key=_normalize_key(key,self.labels,self.levels)
        if key not in self.rows:return float(default)
        return self.at(key,column)

    def previous(self,key,column,below=math.nan):
        return self.previous_group(None,key,column,below)

    @cython.locals(curve=CurveTable, group_id=cython.longlong)
    def previous_group(self,group,key,column,below=math.nan):
        if self.levels>1:
            group=_normalize_enum(group,self.labels[0])
        if self.numeric_axis==self.levels-1 and type(key) in (int,float) and -2**53<=key<=2**53:
            curve=self.native.get(column)
            if curve is not None:
                group_id=self.codes[group]
                return curve.previous(group_id,key,below)
        axis=self.ordered[group]  # Unknown group always raises, even with a below default.
        pos=-1 if key != key else bisect_right(axis,key)-1
        if pos<0:
            if math.isnan(below):raise KeyError(key)
            return float(below)
        return self.at(self._full(group,axis[pos]),column)

    def max_key(self,group=None):
        if group is None:return self.maximum
        if self.levels>1:group=_normalize_enum(group,self.labels[0])
        axis=self.ordered[group]
        if not axis:raise KeyError(group)
        return axis[-1]

    def max_level(self,level):
        if not isinstance(level,int):
            if self.names.count(level)!=1:raise ValueError('Index level name is missing or ambiguous')
            level=self.names.index(level)
        return self.maxima[level]

    def keys(self,group=None):
        if group is not None and self.levels>1:group=_normalize_enum(group,self.labels[0])
        return list(self.rows) if group is None else list(self.groups[group])

    def group_curve(self,group,column,axis=-1):
        if self.levels>1:group=_normalize_enum(group,self.labels[0])
        if not isinstance(axis,int):
            if self.names.count(axis)!=1:raise ValueError('Missing or ambiguous axis')
            axis=self.names.index(axis)
        axis=axis % self.levels
        if axis!=self.numeric_axis:raise ValueError('Bound curves require the prepared numeric axis')
        return self.curves[(group,column)]
