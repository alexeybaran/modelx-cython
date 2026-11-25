"""Enum members as integers where that makes the speed, members everywhere else.

A *family* is an IntEnum class whose values are 0..n-1, or a StrEnum class (Python 3.11+), whose members compare
and hash as their values. The tracer types a Cell parameter given members of one family, or their values, as a C
integer: the member's position in its class (an IntEnum member's value), with a cache of the size of the class.
``plan`` writes the formulas of a model for that, by use. A held name (such a parameter, or a loop variable over a
display of members) is an integer in three uses only: compared by ``==``, ``!=`` or ``in`` with members, their
values or a held name of its class; given to a Cell parameter held for its class; iterated as a display of
members. Members written in those uses become their positions; every other use sees the member
(``_mx_enum_at``). A value from elsewhere reaches a held parameter through ``_mx_enum_code``: a member of its
class; also an object equal to a member where the Cell uses only the position (not *strict*); anything else
raises at the Cell. A parameter is held only where a Reference of its Space holds its class, which the compiled
model then keeps read-only (``prepare``, ``bind_properties``): its members are in the code.
"""
import enum
import functools

import libcst as cst
from libcst.metadata import MetadataWrapper, ParentNodeProvider, ScopeProvider

from .consts import MX_SYS_MOD

_code = cst.Module([]).code_for_node
_DISPLAYS = (cst.Tuple, cst.List, cst.Set)


def storage(name):
    return '_mx_definition_' + name


def prepare(module):
    for cls in module.classes.values():
        cls.readonly_refs = tuple(name for name, ref in cls.refs.items()
                                  if name not in getattr(cls, 'spaces', ()) and getattr(ref, 'enum_safe', False))
        for name in cls.readonly_refs:
            if storage(name) in cls.refs:
                raise ValueError(f'{cls.fqname}: reserved enum storage name {storage(name)}')


class _Stored(cst.CSTTransformer):
    def __init__(self, names):
        self.names = names

    def leave_AssignTarget(self, before, after):
        target = after.target
        if (isinstance(target, cst.Attribute) and isinstance(target.value, cst.Name)
                and target.value.value == 'self' and target.attr.value in self.names):
            return after.with_changes(target=target.with_changes(attr=cst.Name(storage(target.attr.value))))
        return after


def bind_properties(cls, body):
    names = getattr(cls, 'readonly_refs', ())
    result = []
    for statement in body:
        if names and isinstance(statement, cst.FunctionDef) and statement.name.value == '__init__':
            lines = list(statement.body.body)
            first = lines[0].body[0] if lines and isinstance(lines[0], cst.SimpleStatementLine) else None
            docstring = isinstance(first, cst.Expr) and isinstance(first.value, (cst.SimpleString,
                                                                                 cst.ConcatenatedString))
            lines.insert(int(docstring), cst.parse_statement(
                f'{MX_SYS_MOD}._mx_check_enum_schema(self, {cls.name}, {names!r})\n'))
            statement = statement.with_changes(body=statement.body.with_changes(body=tuple(lines)))
        if isinstance(statement, cst.FunctionDef) and statement.name.value in ('_mx_assign_refs', '_mx_copy_refs'):
            statement = statement.visit(_Stored(names))
        result.append(statement)
    for name in names:
        result.extend(cst.parse_module(f'@property\ndef {name}(self):\n    return self.{storage(name)}\n').body)
    return tuple(result)


# ---------------------------------------------------------------- families
@functools.cache
def family(cls):
    """The type of the values of ``cls``, ``int`` or ``str``, if it is a family, else None."""
    for base, kind in ((enum.IntEnum, int), (getattr(enum, 'StrEnum', None), str)):
        if (base is not None and isinstance(cls, type) and issubclass(cls, base) and type(cls) is type(base)
                and len(cls)
                and all(getattr(cls, name) is getattr(base, name) for name in ('__eq__', '__ne__', '__hash__'))
                and [m._value_ for m in cls] == (list(range(len(cls))) if kind is int else [str(m._value_) for m in cls])
                and all(type(m._value_) is kind and (kind is int or str.__str__(m) == m._value_) for m in cls)):
            return kind
    return None


@functools.cache
def positions(cls):
    """({member name: position}, {member value: position}) of a family, by declaration order."""
    order = {member: position for position, member in enumerate(cls)}
    return ({name: order[member] for name, member in cls.__members__.items()},
            {member._value_: position for member, position in order.items()})


def release(info, name):
    """Type a parameter that the tracer held for a family as an object after all."""
    info.arg_types[name] = object
    for table in ('enum_codes', 'enum_args', 'max_args', 'min_args', 'observed_min_args', 'observed_max_args'):
        getattr(info, table, {}).pop(name, None)


def _path(key):
    """The Cell of a key (``module._c_Space._f_cell``) as the model names it (``Space.cell``)."""
    module, cls, name = key.rsplit('.', 2)
    return '.'.join((*(part[3:] for part in module.split('.')[1:-1]), cls[3:], name.removeprefix('_f_')))


# ---------------------------------------------------------------- the pass
class _Model:
    """What the trace knows of the whole model: References holding a family, Spaces and Cell parameters."""

    def __init__(self, ref_info, cells_info):
        self.refs, self.spaces, self.cells, self.strict = {}, {}, {}, set()
        for fqname, info in ref_info.items():
            owner, _, ref = fqname.rpartition('.')
            if getattr(info, 'enum_class', None) is not None:
                self.refs[owner, ref] = info.enum_class
            if getattr(info, 'mx_class', ''):
                self.spaces[owner, ref] = info.mx_class
        self.infos = cells_info
        self.held()

    def held(self):
        """{Cell key: (parameter names, {held parameter: (class, Reference)})}; a parameter whose Space has no
        Reference to its class is an object."""
        for key, info in self.infos.items():
            owner, held = key.rpartition('.')[0], {}
            for name, cls in sorted((getattr(info, 'enum_codes', None) or {}).items()):
                ref = min((ref for (where, ref), value in self.refs.items() if where == owner and value is cls),
                          default=None)
                if ref is None:
                    release(info, name)
                else:
                    held[name] = (cls, ref)
            self.cells[key] = (tuple(getattr(info, 'arg_types', None) or ()), held)

    def cell(self, owner, name):
        for key in (f'{owner}._f_{name}', f'{owner}.{name}'):
            if key in self.cells:
                return key, *self.cells[key]
        return None


class _Formula:
    """One method of a Space class: its held names and their uses."""

    def __init__(self, model, module, cls, node, scopes, parents):
        self.model, self.module, self.cls, self.node, self.parents = model, module, cls, node, parents
        self.owner = f'{module}.{cls}'
        self.key = f'{self.owner}.{node.name.value}'
        self.scopes, scope = scopes, scopes[node.body]
        params = [*node.params.posonly_params, *node.params.params, *node.params.kwonly_params]
        self.params = {p.name.value: p for p in params}
        first = params[0].name.value if params else None
        mine = [a for a in scope.assignments[first] if a.node is params[0]] if first == 'self' else []
        self.me = mine[0] if mine else None
        self.selves = {id(a.node) for a in (self.me.references if self.me else ())
                       if set(scopes[a.node]['self']) == {self.me}}
        self.held = dict(model.cells.get(self.key, ((), {}))[1])
        self.calls, self.loops = [], []
        for found in _Found(self).collect(node.body):
            (self.calls if isinstance(found, cst.Call) else self.loops).append(found)

    def refs(self, node):
        """The uses of the name a parameter or loop target binds, if this is its only binding."""
        name = node.name.value if isinstance(node, cst.Param) else node.value
        bound = self.scopes[node].assignments[name]
        return [a.node for a in next(iter(bound)).references] if len(bound) == 1 else None

    def own(self, node):
        """Whether ``node``, a use, sees this method's ``self``."""
        return set(self.scopes[node]['self']) == {self.me}

    def member(self, node):
        """(class, position) of ``self.<Ref>.<member>``, else None."""
        if (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Attribute)
                and id(node.value.value) in self.selves):
            cls = self.model.refs.get((self.owner, node.value.attr.value))
            if cls is not None:
                return cls, positions(cls)[0].get(node.attr.value)
        return None

    def callee(self, call):
        """(key, parameter names, held parameters) of ``self.<cell>(...)`` or ``self.<Space>.<cell>(...)``."""
        func = call.func
        if not isinstance(func, cst.Attribute):
            return None
        if id(func.value) in self.selves:
            return self.model.cell(self.owner, func.attr.value)
        if isinstance(func.value, cst.Attribute) and id(func.value.value) in self.selves:
            name = func.value.attr.value
            found = self.model.spaces.get((self.owner, name))       # the model may be renamed since the trace
            child = '.'.join(filter(None, (self.module.rpartition('.')[0], f'_m_{self.cls[3:]}', '_mx_classes')))
            for owner in ('.'.join((self.module.partition('.')[0], *found.split('.')[1:])) if found else None,
                          f'{child}._c_{name}'):
                if owner and self.model.cell(owner, func.attr.value):
                    return self.model.cell(owner, func.attr.value)
        return None

    def slots(self, call, callee):
        """[(argument, held parameter)] of a call; None where a starred argument hides them."""
        names, held, found = callee[1], callee[2], []
        for position, arg in enumerate(call.args):
            if arg.star:
                return None if held else []
            name = arg.keyword.value if arg.keyword else (names[position] if position < len(names) else None)
            if name in held:
                found.append((arg, name))
        return found

    def code(self, node, cls, held):
        """What ``node`` stands for beside a held value of ``cls``: the position of a member or of a value
        literal (-1 for one of no member), True for a held name of ``cls``, else None."""
        member = self.member(node)
        if member is not None:
            return member[1] if member[0] is cls and member[1] is not None else None
        if held.get(id(node)) is cls:
            return True
        literal = node.evaluated_value if isinstance(node, (cst.SimpleString, cst.Integer)) else None
        return None if literal is None or not isinstance(literal, family(cls)) else positions(cls)[1].get(literal, -1)

    def use(self, node, cls, held):
        """The literals to write as positions where the use of a held ``node`` takes the integer: a list of
        (node, position), with the parameter it is given to (key, name) or None; else None (an object use)."""
        parent = self.parents[node]
        comparison = self.parents[parent] if isinstance(parent, cst.ComparisonTarget) else parent
        if isinstance(comparison, cst.Comparison) and len(comparison.comparisons) == 1:
            target = comparison.comparisons[0]
            sides = [target.comparator if comparison.left is node else comparison.left]
            if isinstance(target.operator, (cst.In, cst.NotIn)):
                display = target.comparator
                if comparison.left is not node or not isinstance(display, _DISPLAYS) or not all(
                        isinstance(element, cst.Element) for element in display.elements):
                    return None
                sides = [element.value for element in display.elements]
            elif not isinstance(target.operator, (cst.Equal, cst.NotEqual)):
                return None
            codes = [self.code(side, cls, held) for side in sides]
            return None if None in codes else ([(s, c) for s, c in zip(sides, codes) if c is not True], None)
        call = self.parents.get(parent)
        if isinstance(parent, cst.Arg) and parent.value is node and isinstance(call, cst.Call):
            callee = self.callee(call)
            for arg, name in (self.slots(call, callee) or ()) if callee else ():
                if arg is parent and callee[2][name][0] is cls:
                    return [], (callee[0], name)
        return None


class _Found(cst.CSTVisitor):
    """The calls of Cells with held parameters and the loops over displays of members in a method."""

    def __init__(self, formula):
        self.f, self.found = formula, []

    def collect(self, node):
        node.visit(self)
        return self.found

    def visit_Call(self, node):
        callee = self.f.callee(node)
        if callee and callee[2]:
            self.found.append(node)

    def loop(self, node):
        elements = node.iter.elements if isinstance(node.iter, _DISPLAYS) else ()
        members = [self.f.member(e.value) if isinstance(e, cst.Element) else None for e in elements]
        if (isinstance(node.target, cst.Name) and members and None not in members
                and len({cls for cls, _ in members}) == 1 and None not in {position for _, position in members}):
            self.found.append(node)

    visit_For = visit_CompFor = loop


def plan(sources, ref_info, cells_info=None):
    """{module: source} with the formulas of all modules of a model written for held enum members (see the
    module docstring). Sets ``enum_gate``, {parameter: (Reference, strict)}, on the info of each Cell with a held
    parameter, for its calls from Python."""
    model = _Model(ref_info, cells_info or {})
    names = {key.rpartition('.')[2].removeprefix('_f_') for key, (_, held) in model.cells.items() if held}
    formulas, wrappers = [], {}
    for module, source in sources.items():
        if not names or not any(name + '(' in source for name in names):
            continue
        wrappers[module] = wrapper = MetadataWrapper(cst.parse_module(source), unsafe_skip_copy=True)
        scopes, parents = wrapper.resolve(ScopeProvider), wrapper.resolve(ParentNodeProvider)
        for cls in wrapper.module.body:
            methods = {node.name.value: node for node in (cls.body.body if isinstance(cls, cst.ClassDef) else ())
                       if isinstance(node, cst.FunctionDef)}
            for name, node in methods.items():      # not the model's own methods, nor the cache of a formula
                if not name.startswith(('_mx_', '__')) and '_f_' + name not in methods:
                    formulas.append(_Formula(model, module, cls.name.value, node, scopes, parents))
    seen = {f.key for f in formulas}
    for key, (_, held) in model.cells.items():      # a formula that is not read here cannot hold its parameters
        for name in held if key not in seen else ():
            release(model.infos[key], name)
    while True:
        edits = {module: {} for module in wrappers}     # {id of a node: [functions of the rewritten node]}
        for f in formulas:
            _settle(f, model, edits[f.module])
        model.held()
        flows, objects = [], set()
        for f in formulas:
            _uses(f, model, edits[f.module], flows, objects)
        grown = True
        while grown:            # a parameter given on to a strict one is strict
            grown = not objects.issuperset(caller for caller, given in flows if given in objects)
            objects.update(caller for caller, given in flows if given in objects)
        texts = {(key, name) for key, name in objects if name in getattr(model.infos[key], 'enum_plain', ())}
        if not texts:           # a strict parameter that the sample gave a plain value is an object
            break
        for key, name in texts:
            release(model.infos[key], name)
    model.strict = objects
    for key, (_, held) in model.cells.items():
        if held:
            model.infos[key].enum_gate = {name: (ref, (key, name) in objects) for name, (_, ref) in held.items()}
    result = dict(sources)
    for module, wrapper in wrappers.items():
        if edits[module]:
            result[module] = wrapper.module.visit(_Rewrite(edits[module])).code
    return result


def _settle(f, model, edits):
    """Release the parameters that cannot be held: bound again, seen where `self` is another, a default that is
    no member's value, given by a starred argument."""
    for name, (cls, _) in list(f.held.items()):
        param = f.params.get(name)
        refs = f.refs(param) if param is not None else None
        default = None if param is None or param.default is None else f.code(param.default, cls, {})
        if refs is None or not all(map(f.own, refs)) or param.default is not None and default in (None, -1):
            release(model.infos[f.key], name)
        elif default is not None:
            edits.setdefault(id(param.default), []).append(_literal(default))
    for call in f.calls:
        callee = f.callee(call)
        if f.slots(call, callee) is None:
            for name in callee[2]:
                release(model.infos[callee[0]], name)


def _uses(f, model, edits, flows, objects):
    """The edits of one formula by use; ``flows``: (its parameter, the parameter of a Cell it is given to);
    ``objects``: its parameters that have a use of the member."""
    def edit(node, function):
        edits.setdefault(id(node), []).append(function)

    f.held = dict(model.cells.get(f.key, ((), {}))[1])
    refs = {name: f.refs(f.params[name]) for name in f.held}
    loops = {id(loop): loop for loop in f.loops}
    while True:                 # a loop variable is held while every use takes the integer
        spans = {i: (f.member(loop.iter.elements[0].value)[0], f.refs(loop.target)) for i, loop in loops.items()}
        held = {id(n): cls for name, (cls, _) in f.held.items() for n in refs[name]}
        held.update((id(n), cls) for cls, uses in spans.values() for n in uses or ())
        dropped = [i for i, (cls, uses) in spans.items() if uses is None or any(f.use(n, cls, held) is None
                                                                                for n in uses)]
        for i in dropped:
            del loops[i]
        if not dropped:
            break
    for i, loop in loops.items():
        for element in loop.iter.elements:
            edit(element.value, _literal(f.member(element.value)[1]))
        for n in spans[i][1]:
            for side, position in f.use(n, spans[i][0], held)[0]:
                edit(side, _literal(position))
    for name, (cls, ref) in f.held.items():
        for n in refs[name]:
            found = f.use(n, cls, held)
            if found is None:
                objects.add((f.key, name))
                edit(n, _call('_mx_enum_at', f'self.{ref}'))
                continue
            for side, position in found[0]:
                edit(side, _literal(position))
            if found[1]:
                flows.append(((f.key, name), found[1]))
    for call in f.calls:
        callee = f.callee(call)
        for arg, name in f.slots(call, callee):
            cls, ref = callee[2][name]
            position = f.code(arg.value, cls, held)
            if position is not True:
                edit(arg.value, _literal(position) if isinstance(position, int) and position >= 0 else _call(
                    '_mx_enum_code', f'{_code(call.func.value)}.{ref}', (callee[0], name), model))


def _literal(position):
    return lambda node: cst.parse_expression(str(position)).with_changes(lpar=node.lpar, rpar=node.rpar)


def _call(function, cls, target=None, model=None):
    """``node`` given to ``function`` with the class ``cls``; for ``_mx_enum_code``, whether the parameter it is
    given to is strict (known once all formulas are read) and its Cell."""
    def edit(node):
        extra = (str(target in model.strict), repr(_path(target[0]))) if target else ()
        return cst.Call(cst.parse_expression(f'{MX_SYS_MOD}.{function}'),
                        [cst.Arg(value) for value in (cst.parse_expression(cls), node, *map(cst.parse_expression, extra))])
    return edit


class _Rewrite(cst.CSTTransformer):
    def __init__(self, edits):
        super().__init__()
        self.edits = edits

    def on_leave(self, original, updated):
        updated = super().on_leave(original, updated)
        for edit in self.edits.get(id(original), ()):
            updated = edit(updated)
        return updated


def literals(module, source, ref_info, cells_info=None):
    """``plan`` for one module."""
    return plan({module: source}, ref_info, cells_info)[module]
