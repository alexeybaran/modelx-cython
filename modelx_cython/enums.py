"""Map explicitly authored enum definitions to private Cython constants."""
import libcst as cst

from .scopes import CstFacts, FormulaTransformer, Names, lexical_nodes
from .syntax import walk
from .consts import MX_SYS_MOD


def storage(name):
    return '_mx_definition_' + name


def prepare(module):
    module.code_enum_symbols = {}
    identifiers = Names(module.visitor.wrapper.module)
    for cls in module.classes.values():
        markers = {name: ref.enum_definitions for name,ref in cls.refs.items()
                   if getattr(ref, 'enum_definitions', ())}
        names = tuple(dict.fromkeys(n for declaration in markers.values() for n in declaration))
        cls.readonly_refs = tuple(dict.fromkeys((*names, *markers)))
        for name in names:
            ref = cls.refs.get(name)
            if ref is None or not getattr(ref, 'enum', None):
                raise ValueError(f'{cls.fqname}.{name}: enum definition must bind an IntEnum class')
            if storage(name) in cls.refs:
                raise ValueError(f'{cls.fqname}: reserved enum storage name {storage(name)}')
            if not ref.enum_safe:
                continue
            for member, value in ref.enum[2].items():
                if -(1 << 31) <= value < (1 << 31):
                    symbol = identifiers.reserve(f'_mx_enum_const_{len(module.code_enum_symbols)}')
                    module.code_enum_symbols[cls.name, name, member] = symbol, value


def declarations(module):
    if not getattr(module, 'code_enum_symbols', {}):
        return ''
    return 'cdef enum:\n' + ''.join(f'    {symbol} = {value}\n'
        for symbol,value in module.code_enum_symbols.values())


class Constants(FormulaTransformer):
    def __init__(self, cls, cell, symbols):
        super().__init__()
        self.cls, self.cell, self.symbols = cls, cell, symbols
        self.locals = set()
        self.count = 0
        self.comparisons = 0

    def visit_FunctionDef(self, node):
        assignments = {}
        invalid = set()
        nodes = list(lexical_nodes(node.body.body))
        facts = CstFacts(node.body.body)
        if 'self' in facts.bindings(nodes):
            return False
        for part in nodes:
            if isinstance(part, cst.Assign):
                for target in part.targets:
                    if isinstance(target.target, cst.Name):
                        assignments.setdefault(target.target.value, []).append(part.value)
            elif isinstance(part, (cst.AugAssign, cst.AnnAssign, cst.NamedExpr, cst.For, cst.CompFor)):
                invalid.update(n.value for n in walk(part.target) if isinstance(n, cst.Name))
            elif not isinstance(part, (cst.Name, cst.Assign)):
                invalid.update(facts.bindings((part,)))
        for name, values in assignments.items():
            if name not in invalid and all(isinstance(v, cst.Call) and isinstance(v.func, cst.Attribute)
                    and isinstance(v.func.value, cst.Name) and v.func.value.value == 'self'
                    and v.func.attr.value in self.cls.cells
                    and not self.cls.cells[v.func.attr.value].called_with_kwargs
                    and not self.cls.cells[v.func.attr.value].body_has_closure
                    and self.cls.cells[v.func.attr.value].get_rettype_expr(c_style=True)
                    in ('long long','bint') for v in values):
                self.locals.add(name)

    def member(self, node):
        if (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Attribute)
                and isinstance(node.value.value, cst.Name) and node.value.value.value == 'self'):
            entry = self.symbols.get((self.cls.name, node.value.attr.value, node.attr.value))
            if entry:
                self.count += 1
                return cst.Name(entry[0],lpar=node.lpar,rpar=node.rpar)
        return node

    def integer(self, node):
        if isinstance(node, cst.Name):
            return node.value in self.locals or (node.value in self.cell.params and
                self.cell.get_argtype_expr(node.value, c_style=True) in ('long long','bint'))
        if (isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute)
                and isinstance(node.func.value, cst.Name) and node.func.value.value == 'self'):
            cell = self.cls.cells.get(node.func.attr.value)
            return (cell is not None and not cell.called_with_kwargs and not cell.body_has_closure
                    and cell.get_rettype_expr(c_style=True) in ('long long','bint'))
        return False

    def leave_Comparison(self, before, after):
        if len(after.comparisons) == 1 and self.integer(after.left):
            comp = after.comparisons[0]
            if isinstance(comp.operator, (cst.Equal,cst.NotEqual)):
                member = self.member(comp.comparator)
                if member is not comp.comparator:
                    self.comparisons += 1
                return after.with_changes(comparisons=(comp.with_changes(comparator=member),))
        return after

    def leave_Call(self, before, after):
        func = after.func
        if (not isinstance(func, cst.Attribute) or not isinstance(func.value,cst.Name)
                or func.value.value != 'self' or any(a.keyword or a.star for a in after.args)):
            return after
        cell = self.cls.cells.get(func.attr.value)
        if cell is None or cell.called_with_kwargs or cell.body_has_closure:
            return after
        args = list(after.args)
        for i, param in enumerate(cell.params):
            if i < len(args) and cell.get_argtype_expr(param,c_style=True) in ('long long','bint'):
                args[i] = args[i].with_changes(value=self.member(args[i].value))
        return after.with_changes(args=tuple(args))

    def leave_Return(self, before, after):
        if (self.cell.get_rettype_expr(c_style=True) in ('long long','bint','double')
                and (self.cell.has_formula_def or not self.cell.called_with_kwargs)):
            return after.with_changes(value=self.member(after.value))
        return after


class Bindings(cst.CSTTransformer):
    def __init__(self, names):
        self.names = names

    def leave_AssignTarget(self, before, after):
        target = after.target
        if (isinstance(target, cst.Attribute) and isinstance(target.value,cst.Name)
                and target.value.value == 'self' and target.attr.value in self.names):
            return after.with_changes(target=target.with_changes(attr=cst.Name(storage(target.attr.value))))
        return after


def bind_properties(cls, body):
    names = getattr(cls, 'readonly_refs', ())
    result = []
    for statement in body:
        if names and isinstance(statement,cst.FunctionDef) and statement.name.value=='__init__':
            statements=list(statement.body.body)
            docstring=(statements and isinstance(statements[0],cst.SimpleStatementLine)
                and len(statements[0].body)==1 and isinstance(statements[0].body[0],cst.Expr)
                and isinstance(statements[0].body[0].value,(cst.SimpleString,cst.ConcatenatedString)))
            check=cst.parse_statement(f'{MX_SYS_MOD}._mx_check_enum_schema(self, {cls.name}, {names!r})\n')
            statements.insert(int(bool(docstring)),check)
            statement=statement.with_changes(body=statement.body.with_changes(body=tuple(statements)))
        if isinstance(statement, cst.FunctionDef) and statement.name.value in ('_mx_assign_refs','_mx_copy_refs'):
            statement = statement.visit(Bindings(names))
        result.append(statement)
    for name in names:
        result.extend(cst.parse_module(f'@property\ndef {name}(self):\n    return self.{storage(name)}\n').body)
    return tuple(result)
