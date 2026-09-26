"""Guarded binding of immutable, ordinary Enum members; values remain objects.

This is reference binding, not enum-to-integer lowering or formula evaluation.
Custom Enum metaclasses stay on Python attribute access. Changing a public class
reference selects its normal attribute path. Existing Cell cache semantics apply.
"""
from dataclasses import dataclass, field
from enum import EnumMeta
import libcst as cst

@dataclass
class EnumBindings:
    refs: dict = field(default_factory=dict)
    members: dict = field(default_factory=dict)

    def declarations(self):
        return [(name, 'object') for name in (*self.refs.values(), *self.members.values())]

    def initialize(self):
        return ([f'self.{n} = self.{ref}' for ref,n in self.refs.items()] +
                [f'self.{n} = self.{ref}.{member}' for (ref,member),n in self.members.items()])

    def copy(self):
        return [f'self.{name} = base_.{name}' for name,_ in self.declarations()]

class _Discover(cst.CSTVisitor):
    def __init__(self, cls, info): self.cls,self.info=cls,info
    def visit_Attribute(self, node):
        ref=node.value
        if not (isinstance(ref,cst.Attribute) and isinstance(ref.value,cst.Name)
                and ref.value.value=='self'): return
        rt=self.cls.logger.ref_info.get(self.cls.fqname+'.'+ref.attr.value)
        value=getattr(rt,'value',None)
        if type(value) is not EnumMeta or node.attr.value not in value.__members__: return
        self.info.refs.setdefault(ref.attr.value, '_mx_enum_ref_'+str(len(self.info.refs)))
        self.info.members.setdefault((ref.attr.value,node.attr.value),
                                     '_mx_enum_member_'+str(len(self.info.members)))

def configure(units):
    for unit in units:
        for node in cst.parse_module(unit.source).body:
            if not isinstance(node,cst.ClassDef) or node.name.value not in unit.module_info.classes: continue
            cls=unit.module_info.classes[node.name.value]; info=EnumBindings()
            for fn in node.body.body:
                if not isinstance(fn,cst.FunctionDef): continue
                cell=cls.cells.get(fn.name.value.removeprefix('_f_'))
                if cell is None or getattr(cell, "python_fallback", False) or cell.formula_is_generator: continue
                if cell.has_formula_def and not fn.name.value.startswith('_f_'): continue
                fn.visit(_Discover(cls,info))
            cls.enum_info=info

class _Rewrite(cst.CSTTransformer):
    def __init__(self,info): self.info=info
    def leave_Attribute(self,old,new):
        ref=old.value
        if not (isinstance(ref,cst.Attribute) and isinstance(ref.value,cst.Name)
                and ref.value.value=='self'):return new
        member=self.info.members.get((ref.attr.value,old.attr.value))
        if member is None:return new
        return cst.IfExp(
            test=cst.Comparison(ref,[cst.ComparisonTarget(cst.Is(),
                cst.Attribute(cst.Name('self'),cst.Name(self.info.refs[ref.attr.value])))]),
            body=cst.Attribute(cst.Name('self'),cst.Name(member)), orelse=new,
            lpar=[cst.LeftParen()],rpar=[cst.RightParen()])

def rewrite(node, cls):
    info=getattr(cls,'enum_info',None)
    return node.visit(_Rewrite(info)) if info and info.members else node
