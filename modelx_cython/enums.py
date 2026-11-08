"""Map explicitly authored enum definitions to private Cython constants."""
import ast

import libcst as cst

from .scopes import FormulaTransformer, binding_names, lexical_nodes
from .cell_entries import exact_receiver


def storage(name):
    return '_mx_definition_' + name


def prepare(module):
    module.code_enum_symbols = {}
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
                    symbol = f'_mx_enum_const_{len(module.code_enum_symbols)}'
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
        self._guarded_comparisons = {}

    def visit_FunctionDef(self, node):
        tree = ast.parse(cst.Module([]).code_for_node(node))
        if any((isinstance(n, ast.Name) and n.id.startswith('_mx_enum_const_'))
               or (isinstance(n, ast.arg) and n.arg.startswith('_mx_enum_const_'))
               for n in ast.walk(tree)):
            return False
        assignments = {}
        invalid = set()
        nodes = list(lexical_nodes(tree.body[0].body))
        for part in nodes:
            if isinstance(part, ast.Assign):
                for target in part.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(part.value)
            elif isinstance(part, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
                invalid.update(n.id for n in ast.walk(part.target) if isinstance(n, ast.Name))
            elif isinstance(part, (ast.For, ast.comprehension)):
                invalid.update(n.id for n in ast.walk(part.target) if isinstance(n, ast.Name))
            elif not isinstance(part, (ast.Name, ast.Assign)):
                invalid.update(binding_names((part,)))
        for name, values in assignments.items():
            if name not in invalid and all(isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                    and isinstance(v.func.value, ast.Name) and v.func.value.id == 'self'
                    and v.func.attr in self.cls.cells
                    and not self.cls.cells[v.func.attr].called_with_kwargs
                    and not self.cls.cells[v.func.attr].body_has_closure
                    and self.cls.cells[v.func.attr].get_rettype_expr(c_style=True)
                    in ('long long','bint') for v in values):
                self.locals.add(name)

    def member(self, node):
        if (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Attribute)
                and isinstance(node.value.value, cst.Name) and node.value.value.value == 'self'):
            entry = self.symbols.get((self.cls.name, node.value.attr.value, node.attr.value))
            if entry and getattr(self.cell, 'formula_allows_exact_receiver', False):
                self.count += 1
                return cst.IfExp(test=cst.parse_expression(exact_receiver(self.cls.name)),
                    body=cst.Name(entry[0]), orelse=node,
                    lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))
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
                    # Guard the comparison, not its operand: an object-valued
                    # fallback operand would box even the exact native branch.
                    result = member.with_changes(
                        body=after.with_changes(comparisons=(comp.with_changes(
                            comparator=member.body),)), orelse=after)
                    self._guarded_comparisons[id(result)] = result
                    return result
        return after

    def truth_test(self, node, compound=False):
        if compound and isinstance(node, cst.BooleanOperation):
            return node.with_changes(left=self.truth_test(node.left, True),
                                     right=self.truth_test(node.right, True))
        if compound and isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Not):
            return node.with_changes(expression=self.truth_test(node.expression, True))
        if self._guarded_comparisons.get(id(node)) is not node:
            return node
        # Truth contexts already coerce comparison results. Make that coercion
        # explicit only in the object branch so Cython keeps a native bint join.
        fallback = node.orelse.with_changes(lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))
        return node.with_changes(orelse=cst.UnaryOperation(cst.Not(),
            cst.UnaryOperation(cst.Not(), fallback)))

    def leave_If(self, before, after):
        return after.with_changes(test=self.truth_test(after.test, True))

    def leave_While(self, before, after):
        return after.with_changes(test=self.truth_test(after.test, True))

    def leave_IfExp(self, before, after):
        return after.with_changes(test=self.truth_test(after.test, True))

    def leave_Assert(self, before, after):
        return after.with_changes(test=self.truth_test(after.test, True))

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
        if self.cell.get_rettype_expr(c_style=True) == 'bint':
            after = after.with_changes(value=self.truth_test(after.value))
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
        if isinstance(statement, cst.FunctionDef) and statement.name.value in ('_mx_assign_refs','_mx_copy_refs'):
            statement = statement.visit(Bindings(names))
        result.append(statement)
    for name in names:
        result.extend(cst.parse_module(f'@property\ndef {name}(self):\n    return self.{storage(name)}\n').body)
    return tuple(result)
