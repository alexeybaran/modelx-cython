"""IntEnum members in model text: integers where that makes the speed, members everywhere else.

An enum definition is a Reference whose traced value is an enum-safe IntEnum class (selected by
type, no declaration). ``literals`` replaces ``self.<Ref>.<member>`` by the member's value only in an
integer context: an operand of a comparison or of arithmetic, an argument of a Cell whose parameter
is a number, a subscript that is read, a loop over a display of members whose variable is used only
in such ways. Every other use keeps the member object. A parameter or Cell result that the tracer
typed as a C integer because every traced value was a member of one class is converted back through
its class (``Class(value)``, which raises for an integer that is not a member) wherever the object
is needed. The definition is read-only in the compiled model (``prepare``, ``bind_properties``),
because its values are now in the code.
"""
import numbers

import libcst as cst

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


# ---------------------------------------------------------------- members as integers, by use
_code = cst.Module([]).code_for_node
_NUMERIC = ('int', 'float', 'bool', 'abs', 'round', 'range', 'divmod', 'pow', 'hash', 'len')    # give a plain number
_EXTREMES = ('min', 'max')          # give one of their arguments
_DISPLAYS = (cst.Tuple, cst.List, cst.Set)


def enum_safe(cls):
    """Whether ``cls`` is an IntEnum class whose members behave as their integers in every operation of ``int``
    and whose ``value`` is that integer: its integer can then stand for a member wherever only the number counts."""
    import enum
    return (isinstance(cls, type) and issubclass(cls, enum.IntEnum) and type(cls) is type(enum.IntEnum)
            and all(getattr(cls, name, None) is getattr(enum.IntEnum, name, None)
                    for name in {*vars(int), '__str__'} - {'__doc__'} if name.startswith('__'))
            and all(type(member.value) is int and member.value == int(member) for member in cls))


def _kind(tp):
    """How a traced type is held: (module, qualname) of an enum-safe IntEnum class (a C integer that stands for a
    member), 'number' for another numeric type, None for an object."""
    import enum
    if enum_safe(tp):
        return tp.__module__, tp.__qualname__
    if tp in (numbers.Integral, numbers.Real) or (isinstance(tp, type) and issubclass(tp, numbers.Real)
                                                  and not issubclass(tp, enum.Enum)):
        return 'number'
    return None


def _signature(info):
    """(parameter names, {name: kind}, result kind) of a traced Cell."""
    kinds = {name: _kind(tp) for name, tp in (getattr(info, 'arg_types', None) or {}).items()}
    ret = getattr(info, 'ret_type', None)
    result = _kind(getattr(ret, 'value_type', None)) if not getattr(ret, 'is_array', True) else None
    return tuple(kinds), kinds, result


def _target_names(target):
    """The names a target binds (subscripts and attributes bind none)."""
    if isinstance(target, cst.Name):
        return [target.value]
    if isinstance(target, (cst.Tuple, cst.List)):
        return [name for element in target.elements for name in _target_names(element.value)]
    return []


class _Model:
    """What the trace knows of the whole model: enum definitions, Space References and Cell signatures."""

    def __init__(self, ref_info, cells_info):
        self.members, self.reach, self.spaces = {}, {}, {}
        for fqname, info in ref_info.items():
            owner, _, ref = fqname.rpartition('.')
            enum = getattr(info, 'enum', None)
            if enum and getattr(info, 'enum_safe', False):
                self.members[owner, ref] = (enum[:2], {name: value for name, value in enum[2].items()
                                                       if -(1 << 63) <= value < (1 << 63)})
                self.reach.setdefault((owner, enum[:2]), ref)
            if getattr(info, 'mx_class', ''):
                self.spaces[owner, ref] = info.mx_class
        self.cells = {key: _signature(info) for key, info in (cells_info or {}).items()}
        self.typed = any(isinstance(kind, tuple) for _, kinds, result in self.cells.values()
                         for kind in (*kinds.values(), result))

    def concerns(self, module):
        return self.typed or any(owner.rpartition('.')[0] == module for owner, _ in self.members)

    def cell(self, owner, name):
        return self.cells.get(f'{owner}._f_{name}') or self.cells.get(f'{owner}.{name}')


class _Bindings(cst.CSTVisitor):
    """{name: [values of each binding, or None for a binding that is not `name = value` or `for name in display`]}
    of a method's own scope."""

    def __init__(self):
        self.found, self.depth = {}, 0

    def other(self, target):
        for name in _target_names(target):
            self.found.setdefault(name, []).append(None)

    def visit_Assign(self, node):
        if self.depth:
            return
        if len(node.targets) == 1 and isinstance(node.targets[0].target, cst.Name):
            self.found.setdefault(node.targets[0].target.value, []).append([node.value])
        else:
            for target in node.targets:
                self.other(target.target)

    def visit_For(self, node):
        if self.depth:
            return
        if (isinstance(node.target, cst.Name) and isinstance(node.iter, _DISPLAYS)
                and all(isinstance(element, cst.Element) for element in node.iter.elements)):
            self.found.setdefault(node.target.value, []).append([element.value for element in node.iter.elements])
        else:
            self.other(node.target)

    def visit_AugAssign(self, node):
        if not self.depth:
            self.other(node.target)

    visit_AnnAssign = visit_Del = visit_AugAssign

    def visit_NamedExpr(self, node):        # binds in the function, also from inside a comprehension
        self.other(node.target)

    def visit_AsName(self, node):
        if not self.depth:
            self.other(node.name)

    def visit_ImportAlias(self, node):
        if not self.depth and node.asname is None:
            root = node.name
            while isinstance(root, cst.Attribute):
                root = root.value
            self.other(root)

    def visit_Global(self, node):
        for item in node.names:
            self.found.setdefault(item.name.value, []).append(None)

    visit_Nonlocal = visit_Global

    def visit_FunctionDef(self, node):
        if not self.depth:
            self.other(node.name)
        self.depth += 1

    visit_ClassDef = visit_FunctionDef

    def leave_FunctionDef(self, node):
        self.depth -= 1

    leave_ClassDef = leave_FunctionDef

    def visit_Lambda(self, node):
        self.depth += 1

    def leave_Lambda(self, node):
        self.depth -= 1

    def visit_MatchAs(self, node):
        if not self.depth and node.name is not None:
            self.other(node.name)

    def visit_MatchStar(self, node):
        self.visit_MatchAs(node)

    def visit_MatchMapping(self, node):
        if not self.depth and node.rest is not None:
            self.other(node.rest)


class _Formula:
    """One method of a Space class: its signature, bindings and, after ``analyse``, its uses."""

    def __init__(self, model, module, cls, node):
        self.model, self.module, self.cls, self.node = model, module, cls, node
        self.owner = f'{module}.{cls}'
        self.key = f'{self.owner}.{node.name.value}'
        names, kinds, self.result = model.cells.get(self.key) or ((), {}, None)
        params = node.params
        self.params = [p.name.value for p in (*params.posonly_params, *params.params, *params.kwonly_params,
                                              *(p for p in (params.star_arg, params.star_kwarg)
                                                if isinstance(p, cst.Param)))]
        finder = _Bindings()
        node.body.visit(finder)
        self.bindings = finder.found
        self.own = self.params[:1] == ['self'] and 'self' not in self.bindings
        self.kinds = {name: kinds.get(name) for name in self.params[1:]}

    def where(self):
        """The Cell as the model names it (``Space.cell``, as ``builder`` ``model_path``)."""
        spaces = [part[3:] for part in self.module.split('.')[1:-1]]
        return '.'.join((*spaces, self.cls.removeprefix('_c_'), self.node.name.value.removeprefix('_f_')))

    def member(self, node):
        """(class, integer) of ``self.<Ref>.<member>``, else None."""
        if (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Attribute)
                and isinstance(node.value.value, cst.Name) and node.value.value.value == 'self'):
            cls, members = self.model.members.get((self.owner, node.value.attr.value), (None, {}))
            if node.attr.value in members:
                return cls, members[node.attr.value]
        return None

    def space(self, node):
        """The class path of the Space that ``self.<name>`` holds (a child Space or a Reference to a Space), else None."""
        if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name) and node.value.value == 'self':
            name = node.attr.value
            found = self.model.spaces.get((self.owner, name))
            if found is not None:           # the model may have been renamed since the trace
                return '.'.join((self.module.partition('.')[0], *found.split('.')[1:]))
            child = f'{self.module.rpartition(".")[0]}._m_{self.cls[3:]}._mx_classes._c_{name}'
            if name in self.children or any(key.startswith(child + '.') for key in self.model.cells):
                return child
        return None

    def callee(self, node):
        """(Cell key, signature) of ``self.<cell>(...)`` or ``self.<Space>.<cell>(...)``, else None."""
        func = node.func
        if not isinstance(func, cst.Attribute):
            return None
        if isinstance(func.value, cst.Name) and func.value.value == 'self':
            owner = self.owner
        else:
            owner = self.space(func.value)
        for key in (f'{owner}._f_{func.attr.value}', f'{owner}.{func.attr.value}') if owner else ():
            if key in self.model.cells:
                return key, self.model.cells[key]
        return None

    def reach(self, cls):
        """The expression that gives the class ``cls`` in this method."""
        ref = self.model.reach.get((self.owner, cls))
        if ref is not None:
            return f'self.{ref}'
        return f"__import__('importlib').import_module({cls[0]!r}).{cls[1]}"

    def analyse(self):
        banned = set()
        while True:
            uses = _Uses(self, banned)
            self.node.body.visit(uses)
            if uses.objects <= banned:
                self.uses = uses
                return
            banned |= uses.objects


class _Uses(cst.CSTVisitor):
    """The contexts of one method. ``literal``: members to write as integers; ``convert``: integer-held values to
    convert to their member; ``objects``: locals and loop variables that must hold objects (they have a use that
    needs the member), found under the assumption that those in ``banned`` already do."""

    def __init__(self, formula, banned):
        self.f, self.banned = formula, banned
        self.int_ok, self.elements, self.nonload, self.stores = set(), set(), set(), set()
        self.literal, self.convert, self.objects = {}, {}, set()
        self.depth = self.foreign = 0
        # a held name: (class, key, whether it may be turned into an object by banning it)
        scope = {name: (kind if name not in formula.bindings else None, ('param', name), False)
                 for name, kind in formula.kinds.items() if isinstance(kind, tuple)}
        self.scopes = [scope]
        grown = True
        while grown:
            grown = False
            for name, bindings in formula.bindings.items():
                if name in scope or name in formula.params or ('local', name) in banned or None in bindings:
                    continue
                classes = {self.classify(value) for values in bindings for value in values}
                if len(classes) == 1 and None not in classes:
                    scope[name] = (classes.pop(), ('local', name), True)
                    grown = True

    # ---- names and classes
    def resolve(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def classify(self, node):
        """The class of an expression that is an enum value held as an integer (or a member written here)."""
        if isinstance(node, cst.IfExp):
            body, orelse = self.classify(node.body), self.classify(node.orelse)
            return body if body == orelse else None
        if isinstance(node, cst.Name):
            entry = self.resolve(node.value)
            return entry[0] if entry else None
        if self.foreign:
            return None
        member = self.f.member(node)
        if member is not None:
            return member[0]
        callee = self.f.callee(node) if isinstance(node, cst.Call) else None
        return callee[1][2] if callee and isinstance(callee[1][2], tuple) else None

    def numeric(self, node):
        """Whether ``node`` is known to be a number (so that ``node % x`` is arithmetic, not formatting)."""
        if isinstance(node, (cst.Integer, cst.Float)) or self.classify(node) is not None:
            return True
        if isinstance(node, cst.Name):
            return self.resolve(node.value) is None and self.f.kinds.get(node.value) == 'number' \
                and node.value not in self.f.bindings
        if isinstance(node, cst.UnaryOperation):
            return self.numeric(node.expression)
        if isinstance(node, cst.BinaryOperation):
            return not isinstance(node.operator, cst.Modulo) or self.numeric(node.left)
        if isinstance(node, cst.Call):
            callee = self.f.callee(node) if not self.foreign else None
            return callee is not None and callee[1][2] is not None or self.builtin(node) in _NUMERIC
        return False

    def builtin(self, node):
        name = node.func.value if isinstance(node.func, cst.Name) else None
        return name if name not in self.f.bindings and name not in self.f.params and self.resolve(name) is None \
            else None

    def target(self, node):
        if isinstance(node, cst.Name):
            self.nonload.add(id(node))
        elif isinstance(node, (cst.Tuple, cst.List)):
            for element in node.elements:
                self.target(element.value)
        elif isinstance(node, cst.StarredElement):
            self.target(node.value)
        elif isinstance(node, cst.Subscript):
            self.stores.add(id(node))

    def needs_member(self, node, cls):
        if cls is None:
            raise TypeError(f'{self.f.where()}: `{_code(node)}` needs the IntEnum member of a parameter that the '
                            'formula rebinds; compiled code holds the parameter as an integer. Bind another name')
        if self.foreign:
            raise TypeError(f'{self.f.where()}: `{_code(node)}` needs an IntEnum member inside a function that has '
                            'its own `self`; compiled code holds the value as an integer')
        self.convert[id(node)] = self.f.reach(cls)

    # ---- uses
    def visit_Name(self, node):
        if id(node) in self.nonload or id(node) in self.int_ok:
            return
        entry = self.resolve(node.value)
        if entry is None:
            return
        cls, key, bannable = entry
        if bannable:
            self.objects.add(key)
        else:
            self.needs_member(node, cls)

    def visit_Attribute(self, node):
        self.nonload.add(id(node.attr))
        member = None if self.foreign else self.f.member(node)
        if member is not None:
            if id(node) in self.int_ok:
                self.literal[id(node)] = member[1]
            return False

    def visit_Call(self, node):
        callee = None if self.foreign else self.f.callee(node)
        builtin = self.builtin(node)
        starred = False
        for position, arg in enumerate(node.args):
            starred = starred or bool(arg.star)
            if arg.keyword is not None:
                self.nonload.add(id(arg.keyword))
            if arg.star or starred and arg.keyword is None:
                continue
            if callee is not None:
                names, kinds, _ = callee[1]
                name = arg.keyword.value if arg.keyword is not None else (names[position:position + 1] or (None,))[0]
                if kinds.get(name) is not None:
                    self.int_ok.add(id(arg.value))
            elif builtin in _NUMERIC or builtin in _EXTREMES and id(node) in self.int_ok:
                self.int_ok.add(id(arg.value))
        if callee is not None and isinstance(callee[1][2], tuple) and id(node) not in self.int_ok:
            self.needs_member(node, callee[1][2])

    def visit_Comparison(self, node):
        operands = [node.left, *(item.comparator for item in node.comparisons)]
        operators = [item.operator for item in node.comparisons]
        for position, operand in enumerate(operands):
            near = operators[max(position - 1, 0):position + 1]
            if any(isinstance(operator, (cst.Is, cst.IsNot)) for operator in near):
                continue
            self.int_ok.add(id(operand))
            if position and isinstance(operators[position - 1], (cst.In, cst.NotIn)) and isinstance(operand, _DISPLAYS):
                self.elements.add(id(operand))

    def visit_BinaryOperation(self, node):
        if not isinstance(node.operator, cst.Modulo) or self.numeric(node.left):
            self.int_ok.update((id(node.left), id(node.right)))

    def visit_UnaryOperation(self, node):
        self.int_ok.add(id(node.expression))

    def visit_BooleanOperation(self, node):
        if id(node) in self.int_ok:
            self.int_ok.update((id(node.left), id(node.right)))

    def visit_IfExp(self, node):
        self.int_ok.add(id(node.test))
        if id(node) in self.int_ok:
            self.int_ok.update((id(node.body), id(node.orelse)))

    def visit_If(self, node):
        self.int_ok.add(id(node.test))

    visit_While = visit_Assert = visit_CompIf = visit_If

    def visit_Subscript(self, node):
        if id(node) in self.stores or not self.foreign and self.f.space(node.value) is not None:
            return                      # a stored key and the argument of an item Space stay objects
        for element in node.slice:
            part = element.slice
            if isinstance(part, cst.Index):
                self.int_ok.add(id(part.value))
                if isinstance(part.value, cst.Tuple):
                    self.elements.add(id(part.value))
            else:
                self.int_ok.update(id(bound) for bound in (part.lower, part.upper, part.step) if bound is not None)

    def visit_Tuple(self, node):
        if id(node) in self.elements:
            self.int_ok.update(id(element.value) for element in node.elements if isinstance(element, cst.Element))

    visit_List = visit_Set = visit_Tuple

    def visit_MatchValue(self, node):
        self.int_ok.add(id(node.value))

    def visit_Return(self, node):
        if node.value is not None and not self.depth and self.f.result is not None:
            self.int_ok.add(id(node.value))

    def held_local(self, target):
        entry = self.resolve(target.value) if isinstance(target, cst.Name) else None
        return entry is not None and entry[2] and entry[1] == ('local', target.value)

    def visit_Assign(self, node):
        for target in node.targets:
            self.target(target.target)
        if len(node.targets) == 1 and self.held_local(node.targets[0].target):
            self.int_ok.add(id(node.value))

    def visit_AugAssign(self, node):
        self.target(node.target)
        if not isinstance(node.operator, cst.ModuloAssign):
            self.int_ok.add(id(node.value))

    def visit_AnnAssign(self, node):
        self.target(node.target)

    visit_Del = visit_AnnAssign

    def visit_NamedExpr(self, node):
        self.target(node.target)

    def visit_AsName(self, node):
        self.target(node.name)

    def visit_For(self, node):
        self.target(node.target)
        if self.held_local(node.target) and isinstance(node.iter, _DISPLAYS):
            self.elements.add(id(node.iter))

    # ---- scopes
    def _comprehension(self, node):
        parts, part = [], node.for_in
        while part is not None:
            parts.append(part)
            part = part.inner_for_in
        bound = [name for part in parts for name in _target_names(part.target)]
        scope = dict.fromkeys(bound)
        for part in parts:
            self.target(part.target)
            if (isinstance(part.target, cst.Name) and isinstance(part.iter, _DISPLAYS) and part.iter.elements
                    and bound.count(part.target.value) == 1 and ('loop', id(part)) not in self.banned
                    and all(isinstance(element, cst.Element) for element in part.iter.elements)):
                classes = {self.classify(element.value) for element in part.iter.elements}
                if len(classes) == 1 and None not in classes:
                    scope[part.target.value] = (classes.pop(), ('loop', id(part)), True)
                    self.elements.add(id(part.iter))
        self.scopes.append(scope)

    def _leave(self, node):
        self.scopes.pop()

    visit_GeneratorExp = visit_ListComp = visit_SetComp = visit_DictComp = _comprehension
    leave_GeneratorExp = leave_ListComp = leave_SetComp = leave_DictComp = _leave

    def _function(self, node, bound):
        finder = _Bindings()
        node.body.visit(finder)
        bound = {*bound, *finder.found}
        self.scopes.append(dict.fromkeys(bound))
        self.depth += 1
        self.foreign += 'self' in bound

    def visit_FunctionDef(self, node):
        self.nonload.add(id(node.name))
        params = node.params
        self._function(node, [p.name.value for p in (*params.posonly_params, *params.params, *params.kwonly_params,
                                                     *(p for p in (params.star_arg, params.star_kwarg)
                                                       if isinstance(p, cst.Param)))])

    def visit_Lambda(self, node):
        params = node.params
        self._function(node, [p.name.value for p in (*params.posonly_params, *params.params, *params.kwonly_params,
                                                     *(p for p in (params.star_arg, params.star_kwarg)
                                                       if isinstance(p, cst.Param)))])

    def leave_FunctionDef(self, node):
        self.foreign -= 'self' in self.scopes.pop()
        self.depth -= 1

    leave_Lambda = leave_FunctionDef

    def visit_Param(self, node):
        self.nonload.add(id(node.name))

    def visit_Annotation(self, node):
        return False

    visit_ClassDef = visit_Import = visit_ImportFrom = visit_Global = visit_Nonlocal = visit_Annotation

    def visit_MatchAs(self, node):
        if node.name is not None:
            self.nonload.add(id(node.name))

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node):
        if node.rest is not None:
            self.nonload.add(id(node.rest))

    def visit_MatchKeywordElement(self, node):
        self.nonload.add(id(node.key))


class _Rewrite(cst.CSTTransformer):
    def __init__(self, literal, convert):
        super().__init__()
        self.literal, self.convert = literal, convert

    def leave_Attribute(self, original, updated):
        value = self.literal.get(id(original))
        if value is None:
            return updated
        literal = cst.Integer(str(abs(value)))
        if value < 0:
            literal = cst.UnaryOperation(cst.Minus(), literal, lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        return literal.with_changes(lpar=(*original.lpar, *literal.lpar), rpar=(*literal.rpar, *original.rpar))

    def leave_Name(self, original, updated):
        cls = self.convert.get(id(original))
        if cls is None:
            return updated
        return cst.Call(cst.parse_expression(cls), [cst.Arg(updated.with_changes(lpar=(), rpar=()))],
                        lpar=updated.lpar, rpar=updated.rpar)

    leave_Call = leave_Name


def plan(sources, ref_info, cells_info=None):
    """{module: source} with IntEnum members and values written for their uses (see the module docstring), for all
    modules of a model at once: a Cell's parameters and result are known wherever it is called."""
    model = _Model(ref_info, cells_info)
    result = dict(sources)
    for module, source in sources.items():
        if not model.concerns(module):
            continue
        tree = cst.parse_module(source)
        literal, convert = {}, {}
        for cls in tree.body:
            if not isinstance(cls, cst.ClassDef):
                continue
            methods = [node for node in cls.body.body if isinstance(node, cst.FunctionDef)]
            children = {name for node in methods if node.name.value == '__init__'
                        for line in node.body.body if isinstance(line, cst.SimpleStatementLine)
                        for small in line.body if isinstance(small, cst.Assign) and isinstance(small.value, cst.Call)
                        and _code(small.value.func).rpartition('.')[2].startswith('_c_')
                        for target in small.targets if isinstance(target.target, cst.Attribute)
                        for name in [target.target.attr.value]}
            for node in methods:
                if node.name.value.startswith(('_mx_', '__')):
                    continue
                formula = _Formula(model, module, cls.name.value, node)
                formula.children = children
                if formula.own:
                    formula.analyse()
                    literal.update(formula.uses.literal)
                    convert.update(formula.uses.convert)
        if literal or convert:
            result[module] = tree.visit(_Rewrite(literal, convert)).code
    return result


def literals(module, source, ref_info, cells_info=None):
    """``plan`` for one module."""
    return plan({module: source}, ref_info, cells_info)[module]
