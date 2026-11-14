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
        self.untyped = []           # expression forms left as Python (admission report)
        self.rebinds_self = False
        self.member_fields = 0
        self.memberships = 0        # ``in`` / ``not in`` over member tuples lowered
        self.lines = None

    def visit_FunctionDef(self, node):
        from .reductions import formula_lines
        self.lines = formula_lines(node)
        assignments = {}
        invalid = set()
        nodes = list(lexical_nodes(node.body.body))
        facts = CstFacts(node.body.body)
        self.rebinds_self = 'self' in facts.bindings(nodes)
        if self.rebinds_self:
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

    def leave_FunctionDef(self, before, after):
        if before is not self._formula_root:
            return after          # a nested def: its own receiver, left as written
        self.cell.enum_untyped = list(self.untyped)
        self.cell.enum_memberships = self.memberships
        if self.lines is None or self.rebinds_self:
            return after
        # W1.4: a member left as an object (an object position) is read from the definition's
        # C field, not through the Space property: self._mx_definition_<Ref>.<member>.
        fields = _MemberFields(self.cls.name, self.symbols)
        after = after.with_changes(body=after.body.visit(fields))
        self.member_fields = fields.count
        return after

    def member(self, node):
        if (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Attribute)
                and isinstance(node.value.value, cst.Name) and node.value.value.value == 'self'):
            entry = self.symbols.get((self.cls.name, node.value.attr.value, node.attr.value))
            if entry:
                self.count += 1
                return cst.Name(entry[0],lpar=node.lpar,rpar=node.rpar)
        return node

    def integer(self, node):
        if isinstance(node, cst.Integer):
            return node.evaluated_value < (1 << 63)
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
            return self.integer(node.expression)
        if isinstance(node, cst.BinaryOperation) and isinstance(
                node.operator, (cst.Add, cst.Subtract, cst.Multiply, cst.FloorDivide, cst.Modulo)):
            return self.integer(node.left) and self.integer(node.right)
        if isinstance(node, cst.Name):
            return node.value in self.locals or (node.value in self.cell.params and
                self.cell.get_argtype_expr(node.value, c_style=True, formula=True) in ('long long','bint'))
        if (isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute)
                and isinstance(node.func.value, cst.Name) and node.func.value.value == 'self'):
            cell = self.cls.cells.get(node.func.attr.value)
            return (cell is not None and not cell.called_with_kwargs and not cell.body_has_closure
                    and cell.get_rettype_expr(c_style=True) in ('long long','bint'))
        return False

    def members(self, node):
        """Every element of a literal tuple as a build-time constant, else None.

        ``x in (E.A, E.B)`` on a typed integer ``x`` then compiles to a chain of
        C ``==`` with ``x`` evaluated once (Cython's literal-tuple membership).
        IntEnum members compare as their int values (``enum_safe``), so the
        result is the one the tuple of member objects gives.
        """
        if not isinstance(node, cst.Tuple) or not node.elements:
            return None
        count, elements = self.count, []
        for element in node.elements:
            member = self.member(element.value) if isinstance(element, cst.Element) else None
            if member is None or member is element.value:
                self.count = count
                return None
            elements.append(element.with_changes(value=member))
        return node.with_changes(elements=elements)

    def leave_Comparison(self, before, after):
        if len(after.comparisons) != 1:
            return after
        comp = after.comparisons[0]
        if self.integer(after.left):
            if isinstance(comp.operator, (cst.Equal,cst.NotEqual)):
                member = self.member(comp.comparator)
                if member is not comp.comparator:
                    self.comparisons += 1
                return after.with_changes(comparisons=(comp.with_changes(comparator=member),))
            if isinstance(comp.operator, (cst.In, cst.NotIn)):
                members = self.members(comp.comparator)
                if members is None:
                    if self.member_tuple(comp.comparator):
                        self.decline(before, 'a tuple element is not a member of a declared enum')
                    return after
                self.comparisons += len(members.elements)
                self.memberships += 1
                return after.with_changes(comparisons=(comp.with_changes(comparator=members),))
        elif isinstance(comp.operator, (cst.Equal, cst.NotEqual)) and self.integer(comp.comparator):
            # ``E.A == x``: a member on the left has no evaluation to reorder.
            member = self.member(after.left)
            if member is not after.left:
                self.comparisons += 1
                return after.with_changes(left=member)
        elif isinstance(comp.operator, (cst.In, cst.NotIn)) and self.member_tuple(comp.comparator):
            self.decline(before, 'the left operand is not a typed integer (an object or untraced value)')
        return after

    def member_tuple(self, node):
        """A literal tuple holding at least one ``self.<Ref>.<name>`` of a declared enum."""
        return isinstance(node, cst.Tuple) and any(
            isinstance(element, cst.Element) and isinstance(element.value, cst.Attribute)
            and isinstance(element.value.value, cst.Attribute)
            and isinstance(element.value.value.value, cst.Name) and element.value.value.value.value == 'self'
            and any(key[:2] == (self.cls.name, element.value.value.attr.value) for key in self.symbols)
            for element in node.elements)

    def decline(self, node, reason):
        from .reductions import untyped_entry
        self.untyped.append(untyped_entry(self.lines, node, 'in', reason))

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


class _MemberFields(cst.CSTTransformer):
    """``self.<Ref>.<member>`` of a declared enum -> ``self.<storage>.<member>`` (lexical scope)."""
    def __init__(self, cls_name, symbols):
        super().__init__()
        self.refs = {ref for owner, ref, _ in symbols if owner == cls_name}
        self.count = 0

    def on_visit(self, node):
        from .scopes import OPAQUE_SCOPES
        return not isinstance(node, OPAQUE_SCOPES) and super().on_visit(node)

    def leave_Attribute(self, before, after):
        inner = after.value
        if (isinstance(inner, cst.Attribute) and isinstance(inner.value, cst.Name)
                and inner.value.value == 'self' and inner.attr.value in self.refs):
            self.count += 1
            return after.with_changes(value=inner.with_changes(attr=cst.Name(storage(inner.attr.value))))
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
