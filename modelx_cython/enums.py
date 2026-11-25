"""IntEnum members in model text: integer literals before any other pass, and fixed definitions.

An enum definition is a Reference whose traced value is an enum-safe IntEnum class (selected by
type, no declaration). ``literals`` replaces each ``self.<Ref>.<member>`` by the member's value,
so the compiler sees integers everywhere a member is used; the definition is read-only in the
compiled model (``prepare``, ``bind_properties``), because its values are now in the code.
"""
import builtins
import re

import libcst as cst

from .scopes import CstFacts, lexical_nodes
from .syntax import walk as cst_walk
from .consts import MX_SYS_MOD


def storage(name):
    return '_mx_definition_' + name


def prepare(module):
    for cls in module.classes.values():
        names = tuple(name for name, ref in cls.refs.items()
                      if name not in getattr(cls, 'spaces', ())
                      and getattr(ref, 'enum', None) and getattr(ref, 'enum_safe', False))
        cls.readonly_refs = names
        for name in names:
            if storage(name) in cls.refs:
                raise ValueError(f'{cls.fqname}: reserved enum storage name {storage(name)}')


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


# ---------------------------------------------------------------- members as integer literals
def literals(module, source, ref_info, cells_info=None):
    """``source`` with every member of an IntEnum class held in a Reference of its Space
    (``self.<Ref>.<member>``, the Reference's traced value an enum-safe IntEnum class) replaced by
    the member's integer literal, before any other pass reads the module. Comparisons, call
    arguments, locals, loops, ``in`` and table keys then take the paths that integers take.

    A value known to be of an IntEnum class (a member, a parameter or Cell result traced as one, a
    local bound only to members of one class) keeps ``.name`` (its name, looked up by value) and
    ``.value`` (the integer); any other attribute of it, and an identity test with a member
    (``x is E.m``), is a build error: compiled code holds such values as integers."""
    members = {}            # (class, Reference) -> {member: value}, joined as builder._init_refs does
    for fqname, info in ref_info.items():
        owner, _, ref = fqname.rpartition('.')
        enum = getattr(info, 'enum', None)
        if owner.rpartition('.')[0] == module and enum and getattr(info, 'enum_safe', False):
            members[owner.rpartition('.')[2], ref] = {
                name: value for name, value in enum[2].items() if -(1 << 63) <= value < (1 << 63)}
    traced = {key: _traced_names(info) for key, info in (cells_info or {}).items()}     # all modules: child Spaces
    if not members and not any(any(names.values()) for names in traced.values()):
        return source
    return cst.parse_module(source).visit(_Literals(module, members, traced)).code


class _Table(dict):
    plain = True            # str() and format() of a member give its integer's


def _names(cls):
    """{value: name} of an IntEnum class that compares as its integers, else None."""
    import enum
    if (isinstance(cls, type) and issubclass(cls, enum.IntEnum) and type(cls) is type(enum.IntEnum)
            and cls.__eq__ is int.__eq__ and cls.__ne__ is int.__ne__):
        table = _Table((int(member), member.name) for member in cls)
        table.plain = all(getattr(cls, m) is getattr(enum.IntEnum, m) for m in ('__str__', '__format__'))
        return table


_code = cst.Module([]).code_for_node
_SHOWN = ('repr', 'ascii', 'type', 'isinstance', 'getattr', 'hasattr', 'id', 'str', 'format')   # the class shows


def _traced_names(info):
    """{parameter or None (the result): {value: name}} of a Cell's values traced as one IntEnum class."""
    ret = getattr(info, 'ret_type', None)
    found = {None: _names(getattr(ret, 'value_type', None)) if not getattr(ret, 'is_array', True) else None}
    found.update((name, _names(tp)) for name, tp in (getattr(info, 'arg_types', None) or {}).items())
    found.update(enumerate(_names(tp) for tp in (getattr(info, 'arg_types', None) or {}).values()))   # by position
    return {key: names for key, names in found.items() if names}


class _Literals(cst.CSTTransformer):
    def __init__(self, module, members, traced):
        super().__init__()
        self.module, self.members, self.traced = module, members, traced
        self.cls, self.method, self.scopes = None, None, []     # scopes: (receiver is the method's, {name: names})
        self.in_place = set()       # ids of displays iterated in place or right of `in`
        self.kept = set()           # ids of members that stay objects: their class shows, or a stored key

    def where(self):
        """The Cell as the model names it (``Space.cell``, as ``builder`` ``model_path``)."""
        spaces = [part[3:] for part in self.module.split('.')[1:-1]]
        return '.'.join((*spaces, (self.cls or '').removeprefix('_c_'), (self.method or '').removeprefix('_f_')))

    def visit_ClassDef(self, node):
        if self.cls is not None or self.scopes:
            return False            # a nested class keeps its own names
        self.cls = node.name.value

    def leave_ClassDef(self, original, updated):
        self.cls = None
        return updated

    def _enter(self, node, bound):
        if not self.scopes:         # a method: `self` is its first parameter and is never rebound
            body = node.body.body if isinstance(node.body, cst.IndentedBlock) else (node.body,)
            bindings = CstFacts(body).bindings(lexical_nodes(body))
            own = bound[:1] == ['self'] and 'self' not in bindings and self.cls is not None
            self.method = node.name.value if isinstance(node, cst.FunctionDef) else None
            self.scopes.append((own, {}))
            traced = self.traced.get(f'{self.module}.{self.cls}.{self.method}', {}) if own else {}
            typed = {name: names for name, names in traced.items() if name in bound and name not in bindings}
            typed.update(self._member_locals(body, typed) if own else {})
            self.scopes[-1] = (own, typed)
        else:                       # a nested scope: the method's `self` and names unless it binds its own
            own, typed = self.scopes[-1]
            self.scopes.append((own and 'self' not in bound,
                                {k: v for k, v in typed.items() if k not in bound}))

    def _member_locals(self, body, typed):
        """{local: names} of the locals bound (by ``=`` or ``for`` over a literal tuple) only to members of one class."""
        found, counts, facts = {}, {}, CstFacts(body)
        for part in lexical_nodes(body):
            if isinstance(part, cst.Assign) and len(part.targets) == 1 and isinstance(part.targets[0].target, cst.Name):
                name, values = part.targets[0].target.value, [part.value]
            elif isinstance(part, cst.For) and isinstance(part.target, cst.Name) and isinstance(part.iter, cst.Tuple):
                name, values = part.target.value, [e.value for e in part.iter.elements]
            else:
                continue
            names = {self._names_key(value) for value in values}
            found.setdefault(name, set()).update(names)
            counts[name] = counts.get(name, 0) + 1
        return {name: dict(keys.pop()) for name, keys in found.items() if len(keys) == 1 and None not in keys
                and name not in typed and facts.stores[name] == counts[name]}

    def _names_key(self, node):
        if isinstance(node, cst.IfExp):
            body, orelse = self._names_key(node.body), self._names_key(node.orelse)
            return body if body == orelse else None
        names = self.names(node)
        return tuple(sorted(names.items())) if names else None

    def visit_FunctionDef(self, node):
        params = node.params
        self._enter(node, [p.name.value for p in (*params.posonly_params, *params.params, *params.kwonly_params,
                                                  *(p for p in (params.star_arg, params.star_kwarg)
                                                    if isinstance(p, cst.Param)))])

    def visit_Lambda(self, node):
        if self.scopes:
            self.visit_FunctionDef(node)
        else:
            self.scopes.append((False, {}))     # a lambda outside a method has no receiver

    def _comprehension(self, node):
        if self.scopes:
            self._enter(node, [n.value for part in cst_walk(node) if isinstance(part, cst.CompFor)
                               for n in cst_walk(part.target) if isinstance(n, cst.Name)])
        else:
            self.scopes.append((False, {}))

    def _left(self, original, updated):
        self.scopes.pop()
        return updated

    visit_GeneratorExp = visit_ListComp = visit_SetComp = visit_DictComp = _comprehension
    leave_GeneratorExp = leave_ListComp = leave_SetComp = leave_DictComp = _left
    leave_FunctionDef = leave_Lambda = _left

    def value(self, node):
        """The integer of ``self.<Ref>.<member>`` in a method of the class holding the Reference, else None."""
        if (self.scopes and self.scopes[-1][0] and isinstance(node, cst.Attribute)
                and isinstance(node.value, cst.Attribute) and isinstance(node.value.value, cst.Name)
                and node.value.value.value == 'self'):
            return self.members.get((self.cls, node.value.attr.value), {}).get(node.attr.value)
        return None

    def names(self, node):
        """{value: name} of the IntEnum class whose value ``node`` is known to hold (original node), else None."""
        if not self.scopes or not self.scopes[-1][0]:
            return None
        if isinstance(node, cst.Name):
            return self.scopes[-1][1].get(node.value)
        if self.value(node) is not None:
            names = {}
            for name, value in self.members[self.cls, node.value.attr.value].items():
                names.setdefault(value, name)       # the first name of a value; later ones are aliases
            return names
        return self._traced_call(node).get(None)

    def _traced_call(self, node):
        """The traced names of ``self.<cell>(...)`` or ``self.<child Space>.<cell>(...)``, else {}."""
        func = node.func if isinstance(node, cst.Call) and self.scopes and self.scopes[-1][0] else None
        owner = func.value if isinstance(func, cst.Attribute) else None
        if isinstance(owner, cst.Attribute) and isinstance(owner.value, cst.Name) and owner.value.value == 'self':
            owner = f'{self.module.rpartition(".")[0]}._m_{self.cls[3:]}._mx_classes._c_{owner.attr.value}'
        elif isinstance(owner, cst.Name) and owner.value == 'self':
            owner = f'{self.module}.{self.cls}'
        return isinstance(owner, str) and (self.traced.get(f'{owner}._f_{func.attr.value}')
                                           or self.traced.get(f'{owner}.{func.attr.value}')) or {}

    def _shown(self, node, through_str):
        """A member where its class shows stays an object; an enum value held as an integer is a build error.
        ``through_str``: only a class with its own __str__/__format__ shows."""
        if self.value(node) is not None or self.names(node) is None:
            return self.kept.update(() if through_str or self.value(node) is None else (id(node),))
        if not (through_str and getattr(self.names(node), 'plain', True)):
            raise TypeError(f'{self.where()}: `{_code(node)}` holds an IntEnum value as an integer where its class '
                            'shows (repr, type, str), so the result would differ from live modelx')

    def visit_Call(self, node):
        if isinstance(node.func, cst.Name) and node.func.value in _SHOWN and node.args:
            self._shown(node.args[0].value, node.func.value in ('str', 'format'))
        traced = self._traced_call(node)
        for position, arg in enumerate(node.args):
            expected, given = traced.get(arg.keyword.value if arg.keyword else position), self.names(arg.value)
            if expected and given is not None and not arg.star and dict(given) != dict(expected):
                raise TypeError(f'{self.where()}: `{_code(node)}` passes a value of one IntEnum class to a parameter '
                                'traced as another IntEnum class, so its .name would differ from live modelx')

    def visit_FormattedStringExpression(self, node):
        self._shown(node.expression, node.conversion not in ('r', 'a'))

    def visit_BinaryOperation(self, node):
        if isinstance(node.operator, cst.Modulo) and isinstance(node.left, (cst.SimpleString, cst.ConcatenatedString)):
            through_str = not re.search(r'%[-#0 +\d.*]*[ra]', _code(node.left))
            right = node.right
            for operand in ([e.value for e in right.elements] if isinstance(right, cst.Tuple) else [right]):
                self._shown(operand, through_str)

    def leave_Comparison(self, original, updated):
        # as live: identity with None, a bool, a builtin type, or a member of a Reference that is not substituted
        fixed = lambda n: isinstance(n, cst.Name) and (n.value in ('None', 'True', 'False')
                                                       or isinstance(getattr(builtins, n.value, None), type))
        foreign = lambda n: (isinstance(n, cst.Attribute) and isinstance(n.value, cst.Attribute)
                             and isinstance(n.value.value, cst.Name) and n.value.value.value == 'self'
                             and (self.cls, n.value.attr.value) not in self.members)
        formula, operands = self.method and not self.method.startswith(('_mx_', '__')), [original.left]
        for left, item in zip(operands, original.comparisons):
            operands.append(right := item.comparator)
            if isinstance(item.operator, (cst.Is, cst.IsNot)) and not (fixed(left) or fixed(right)) and (
                    any(self.value(n) is not None or self.names(n) is not None for n in (left, right))
                    or self.members and formula and not (foreign(left) or foreign(right))):
                raise TypeError(f'{self.where()}: `{_code(original)}` tests by identity in a model whose IntEnum '
                                'members compile to integers, so the result would differ from live modelx; '
                                'compare with == or != instead')
        return updated

    # members held in a container display stay objects (so `.name` works where the container goes), unless the
    # display is iterated in place, is the right side of `in` or is a subscript key
    def visit_For(self, node):
        self.in_place.add(id(node.iter))

    def visit_Index(self, node):
        if id(node.value) not in self.kept:
            self.in_place.add(id(node.value))

    def visit_AssignTarget(self, node):         # a stored key stays a member: the container keeps it
        if isinstance(node.target, cst.Subscript):
            self.kept.update(id(e.slice.value) for e in node.target.slice if isinstance(e.slice, cst.Index))

    visit_CompFor = visit_For

    def visit_Comparison(self, node):
        self.in_place.update(id(c.comparator) for c in node.comparisons if isinstance(c.operator, (cst.In, cst.NotIn)))

    def _display(self, original, updated):
        if id(original) in self.in_place:
            return updated
        keep = lambda o, u: o if self.value(o) is not None else u
        if isinstance(original, cst.Dict):
            return updated.with_changes(elements=[u.with_changes(key=keep(o.key, u.key), value=keep(o.value, u.value))
                                                  if isinstance(o, cst.DictElement) else u
                                                  for o, u in zip(original.elements, updated.elements)])
        return updated.with_changes(elements=[u.with_changes(value=keep(o.value, u.value))
                                              for o, u in zip(original.elements, updated.elements)])

    leave_Tuple = leave_List = leave_Set = leave_Dict = _display

    def leave_Attribute(self, original, updated):
        value = self.value(original)
        if value is not None and id(original) not in self.kept:
            literal = cst.Integer(str(abs(value)))
            if value < 0:
                literal = cst.UnaryOperation(cst.Minus(), literal, lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
            return literal.with_changes(lpar=(*original.lpar, *literal.lpar), rpar=(*literal.rpar, *original.rpar))
        names, attr, inner, receiver = self.names(original.value), original.attr.value, updated.value, original.value
        if (names is None and attr in ('name', 'value') and isinstance(receiver, cst.Subscript)
                and isinstance(receiver.value, (cst.List, cst.Tuple)) and any(
                    self.value(e.value) is None and self.names(e.value) is not None for e in receiver.value.elements)):
            raise TypeError(f'{self.where()}: `{_code(original)}` holds an IntEnum value as an integer in a container, '
                            f'so .{attr} would fail where live modelx works')
        if names is None:
            return updated
        if attr == 'value':
            return inner.with_changes(lpar=(*original.lpar, cst.LeftParen()), rpar=(cst.RightParen(), *original.rpar))
        if attr != 'name':
            raise TypeError(f'{self.where()}: `{_code(original)}` reads .{attr} of an IntEnum value. '
                            'Compiled code holds such values as integers and supports only .name and .value')
        if self.value(original.value) is not None:
            return cst.SimpleString(repr(names[self.value(original.value)]), lpar=original.lpar, rpar=original.rpar)
        table = '{' + ', '.join(f'{v}: {n!r}' for v, n in sorted(names.items())) + '}'
        index = _code(inner.with_changes(lpar=(), rpar=()) if hasattr(inner, 'lpar') else inner)
        return cst.parse_expression(f"{table}.get({index}) or getattr({index}, 'name')").with_changes(   # live's error
            lpar=(*original.lpar, cst.LeftParen()), rpar=(cst.RightParen(), *original.rpar))
