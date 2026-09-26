"""Opt-in adapter for the retained experimental lookup vocabulary.

This is deliberately NOT a model frontend, formula evaluator or pandas lowering
engine. It recognizes fixed helper identities and static table/column bindings
inside existing modelx-cython Cells, then calls ordinary compiled helper classes.
Typing, Cell discovery, formula compilation and caching remain upstream-owned.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field
import hashlib
import inspect
from pathlib import Path
import shutil
import libcst as cst

_MODULE=cst.Module(body=())
def code(node): return _MODULE.code_for_node(node)

def _attr(node):
    if isinstance(node,cst.Attribute) and isinstance(node.value,cst.Name) and node.value.value=='self':
        return node.attr.value
    return None

@dataclass
class Binding:
    name: str
    type: str
    table: str
    column: str
    kind: str
    axis: int
    raw_table: str
    table_source: str

@dataclass
class HelperInfo:
    bindings: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    calls: dict = field(default_factory=dict)
    inventory: list = field(default_factory=list)

    def source(self, expression):
        if expression not in self.sources:
            self.sources[expression]=f'_mx_h_source{len(self.sources)}'
        return self.sources[expression]

    def declarations(self):
        return [(v,'object') for v in self.sources.values()] + [(b.name,b.type) for b in self.bindings.values()]

    def initialize(self):
        lines=[f'self.{name} = {expr}' for expr,name in self.sources.items()]
        lines += [f'self.{b.name} = _mx_helper_preparation.prepare({b.raw_table}, {b.column!r}, {b.kind!r}, {b.axis})'
                  for b in self.bindings.values()]
        return lines

    def copy(self):
        return [f'self.{name} = base_.{name}' for name,_ in self.declarations()]


def _table_sample(expr, cls, classes):
    # Only no-argument getters on traced Space refs. No evaluation of expressions.
    if not isinstance(expr,cst.Call) or expr.args or not isinstance(expr.func,cst.Attribute):
        return None
    owner=expr.func.value
    chain=[]
    while isinstance(owner,cst.Attribute):
        chain.append(owner.attr.value); owner=owner.value
    if not isinstance(owner,cst.Name) or owner.value!='self': return None
    current=cls
    for part in reversed(chain):
        ref=current.refs.get(part)
        if ref is None or ref.mx_class not in classes: return None
        current=classes[ref.mx_class]
    cell=current.cells.get(expr.func.attr.value)
    if cell is None or cell.params or not cell._rt: return None
    # Preparing must not warm a table Cell before the user can set its inputs.
    # Resolve only a trivial reference getter; arbitrary table formulas stay Python.
    raw_ref=None
    for node in ast.parse(current.visitor.source).body:
        if isinstance(node,ast.ClassDef) and node.name==current.name:
            for fn in node.body:
                if isinstance(fn,ast.FunctionDef) and fn.name=='_f_'+cell.name:
                    body=[stmt for stmt in fn.body if not (isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Constant) and isinstance(stmt.value.value,str))]
                    if len(body)==1 and isinstance(body[0],ast.Return):
                        v=body[0].value
                        if isinstance(v,ast.Attribute) and isinstance(v.value,ast.Name) and v.value.id=='self':raw_ref=v.attr
    if raw_ref is None:return None
    raw_table=code(expr.func.value)+'.'+raw_ref
    return getattr(cell._rt,'sample_value',None),raw_table


class _Discover(cst.CSTVisitor):
    def __init__(self,cls,cell,classes,info):
        self.cls,self.cell,self.classes,self.info=cls,cell,classes,info

    def visit_Call(self,node):
        func=node.func
        name=None; refname=None
        if isinstance(func,cst.Attribute) and _attr(func.value):
            refname=_attr(func.value); name=func.attr.value
        elif _attr(func):
            refname=_attr(func);name='power'
        if refname is None: return
        rt=self.cls.logger.ref_info.get(self.cls.fqname+'.'+refname)
        ref=getattr(rt,'value',None)
        is_power=(name=='power' and getattr(ref,'__module__',None) in ('mxcf_kernel.lookups', 'lifelib.helpers.lookups')
                  and getattr(ref,'__name__',None)=='power')
        is_lookup=getattr(ref,'__name__',None) in ('mxcf_kernel.lookup', 'lifelib.helpers.lookup')
        if not (is_power or is_lookup): return
        item={'cell':self.cell.fqname,'helper':name,'expression':code(node)}
        self.info.inventory.append(item)
        if any(a.keyword or a.star for a in node.args):
            item.update(status='ordinary_object_path',reason='keyword/star helper call');return
        if is_power and len(node.args)==2:
            source=self.info.source('self.'+refname)
            self.info.calls[(self.cell.name,code(node))]=('power',None,source,None)
            item.update(status='native_fixed_helper',binding_guard=True);return
        if name=='key' and len(node.args)==2:
            table_expr=node.args[0].value
            sample=_table_sample(table_expr,self.cls,self.classes)
            table,raw_table=sample if sample is not None else (None,None)
            import pandas as pd
            if not isinstance(table,pd.DataFrame) or table.index.nlevels<2 or not table.index.is_unique:
                item.update(status='ordinary_object_path',reason='group-label table not observed');return
            key=(code(table_expr),'','labels',0)
            if key not in self.info.bindings:
                self.info.bindings[key]=Binding(f'_mx_h{len(self.info.bindings)}',
                    '_mx_helpers.GroupKeys',code(table_expr),'','labels',0,raw_table,self.info.source(raw_table))
            b=self.info.bindings[key]
            owner=code(table_expr.func.value)
            self.info.calls[(self.cell.name,code(node))]=(name,b,self.info.source('self.'+refname),
                                                       (owner,self.info.source(owner)))
            item.update(status='native_fixed_helper',binding=b.name,kind='labels',axis=0,
                        binding_guard=True,ownership='full first-axis labels snapshot; public label preserved');return
        if name not in ('field','category','at','get','previous','previous_group','linear'):
            item.update(status='ordinary_object_path',reason='helper outside fixed numeric/field adapter');return
        args=[a.value for a in node.args]
        col_pos=3 if name=='previous_group' else 2
        if len(args)<=col_pos:
            item.update(status='ordinary_object_path',reason='incomplete helper call');return
        try: column=ast.literal_eval(code(args[col_pos]))
        except (ValueError,SyntaxError):
            item.update(status='ordinary_object_path',reason='dynamic column');return
        if not isinstance(column,str):
            item.update(status='ordinary_object_path',reason='non-string column');return
        sample=_table_sample(args[0],self.cls,self.classes)
        table,raw_table=sample if sample is not None else (None,None)
        import pandas as pd
        if not isinstance(table,pd.DataFrame) or column not in table.columns or not table.index.is_unique:
            item.update(status='ordinary_object_path',reason='table/getter schema not observed or non-unique');return
        levels=table.index.nlevels
        kind='field' if name in ('field','category') else 'curve'
        axis=0
        if kind=='field':
            if levels!=1:
                item.update(status='ordinary_object_path',reason='field table must have one index level');return
            tp='_mx_helpers.FieldTable'
        else:
            if table[column].dtype.kind not in 'fiu':
                item.update(status='ordinary_object_path',reason='non-numeric lookup result');return
            numeric=[i for i in range(levels) if table.index.get_level_values(i).dtype.kind in 'fiu']
            if levels==1 and numeric==[0]: tp='_mx_curves.Curve'
            elif levels==2:
                if name=='previous_group' and 1 in numeric: axis=1
                elif len(numeric)==1 and name in ('at','get'): axis=numeric[0]
                else:
                    item.update(status='ordinary_object_path',reason='ambiguous grouped numeric axis');return
                tp='_mx_curves.CurveTable'
                if name!='previous_group' and not (isinstance(args[1],cst.Tuple) and len(args[1].elements)==2):
                    item.update(status='ordinary_object_path',reason='grouped key not a two-element tuple');return
            else:
                item.update(status='ordinary_object_path',reason='unsupported index axes');return
        table_expr=code(args[0]); key=(table_expr,column,kind,axis)
        if key not in self.info.bindings:
            self.info.bindings[key]=Binding(f'_mx_h{len(self.info.bindings)}',tp,table_expr,column,kind,axis,raw_table,self.info.source(raw_table))
        b=self.info.bindings[key]
        source=self.info.source('self.'+refname)
        owner=code(args[0].func.value)
        owner_source=self.info.source(owner)
        self.info.calls[(self.cell.name,code(node))]=(name,b,source,(owner,owner_source))
        item.update(status='native_fixed_helper',binding=b.name,kind=kind,axis=axis,
                    binding_guard=True,ownership='full-table immutable snapshot at model construction')


class _Rewrite(cst.CSTTransformer):
    def __init__(self,info,cell): self.info,self.cell=info,cell
    def leave_Call(self,original_node,updated_node):
        found=self.info.calls.get((self.cell.name,code(original_node)))
        if found is None:return updated_node
        name,b,source,owner=found
        args=[code(a.value) for a in updated_node.args]
        if name=='power':
            target=f'_mx_curves.power({args[0]}, {args[1]})'
            ref=_attr(original_node.func)
        else:
            ref=_attr(original_node.func.value)
            h='self.'+b.name
            if name=='key': target=f'{h}.key({args[1]})'
            elif name=='field': target=f'{h}.at({args[1]})'
            elif name=='category': target=f'{h}.category({args[1]}, {args[3]})'
            elif b.type.endswith('CurveTable'):
                if name=='previous_group': group,x=args[1:3];op='previous';extra=args[4:]
                else:
                    pair=updated_node.args[1].value.elements
                    x=code(pair[b.axis].value);group=code(pair[1-b.axis].value);op=name;extra=args[3:]
                target=f'{h}.{op}({h}.code({group}), {x}'+(' ,'+', '.join(extra) if extra else '')+')'
            else:
                extra=args[3:]
                target=f'{h}.{name}({args[1]}'+(', '+', '.join(extra) if extra else '')+')'
        guard=f'self.{ref} is self.{source}'
        if owner:
            guard+=f' and {owner[0]} is self.{owner[1]}'
            # Calling the real cached table Cell here preserves its cache and input
            # update semantics. A changed identity selects the ordinary helper.
            guard+=f' and {b.table} is self.{b.table_source}'
        return cst.parse_expression(f'({target} if {guard} else {code(updated_node)})')


def configure(units):
    classes={c.fqname:c for u in units for c in u.module_info.classes.values()}
    for u in units:
        for cls in u.module_info.classes.values():
            info=HelperInfo()
            for node in cst.parse_module(u.source).body:
                if not isinstance(node,cst.ClassDef) or node.name.value!=cls.name:continue
                for fn in node.body.body:
                    if not isinstance(fn,cst.FunctionDef):continue
                    cell=cls.cells.get(fn.name.value.removeprefix('_f_'))
                    if cell is None or getattr(cell, "python_fallback", False):continue
                    if cell.has_formula_def and not fn.name.value.startswith('_f_'):continue
                    fn.visit(_Discover(cls,cell,classes,info))
            cls.helper_info=info
        u.module_info.helper_enabled=any(c.helper_info.calls for c in u.module_info.classes.values())


def rewrite(node,cls,cell):
    info=getattr(cls,'helper_info',None)
    return node.visit(_Rewrite(info,cell)) if info and info.calls else node


def emit_runtime(model_path,units):
    if not any(getattr(u.module_info,'helper_enabled',False) for u in units):return []
    src=Path(__file__).with_name('experimental_runtime')
    for name in ('_mx_curves.py','_mx_curves.pxd','_mx_helpers.py','_mx_helpers.pxd'):
        text=(src/name).read_text()
        if name=='_mx_helpers.py':
            text=text.replace('from cython.cimports._mx_curves import',f'from cython.cimports.{model_path.name}._mx_curves import')
        (model_path/name).write_text(text)
    return [Path(model_path.name)/'_mx_curves.py',Path(model_path.name)/'_mx_helpers.py']
