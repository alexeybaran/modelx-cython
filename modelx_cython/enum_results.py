"""Results of IntEnum-valued Cells as members where they reach Python.

A compiled Cell whose every traced result is a member of one admitted IntEnum class (``enum_cells``) holds
that result as its C integer. Where the value reaches Python as an object, it is the live member: an
object use in compiled code (an attribute of it, an element of a container, a comprehension's element,
an argument of a Python call, the return of an object-valued function) is wrapped in ``enum_member``,
and a Python caller of the Cell gets the member (``__getattribute__`` gives the Cell's method wrapped).
Compiled callers keep the C integer.

``finish`` is the last pass over a generated module, after entry lowering and the integer guards. A module
without such Cells is returned unchanged.
"""
import libcst as cst
from libcst.metadata import ParentNodeProvider

from .consts import CY_MOD, MODULE_PREF, MX_SPACE_MOD, MX_SYS_MOD, SPACE_PREF

MEMBER = '_mx_enum_member'
ENTRY = '_mx_enum_entry'
CALL = '_mx_enum_call'

# The member of the IntEnum ``module.qualname`` whose value is ``value``: what a Python caller, an
# object-valued use or a collected output receives, as in live modelx. A value that is no member's is an
# integer the formula computed (``E.A - 1``), which live gives as the integer.
FUNCTIONS = f'''\
_mx_enums = {{}}


def {MEMBER}(module, qualname, value):
    cls = _mx_enums.get((module, qualname))
    if cls is None:
        import importlib
        cls = importlib.import_module(module)
        for part in qualname.split('.'):
            cls = getattr(cls, part)
        _mx_enums[module, qualname] = cls
    member = cls._value2member_map_.get(value)
    return member if member is not None else value


@{CY_MOD}.cclass
class {CALL}:
    # a Cell read from Python: called as it is, its C integer result given as the member
    method: object
    module: object
    qualname: object

    def __call__(self, *args, **kwargs):
        return {MEMBER}(self.module, self.qualname, self.method(*args, **kwargs))

    @property
    def __wrapped__(self):
        return self.method

    def __getattr__(self, name):
        return getattr(self.method, name)

    def __repr__(self):
        return repr(self.method)


@{CY_MOD}.cfunc
def {ENTRY}(receiver: object, owner: object, cells: dict, name: object, value: object) -> object:
    enum = cells.get(name)
    if enum is None or not (type(receiver) is owner or getattr(type(receiver), name) is getattr(owner, name)):
        return value
    entry: {CALL} = {CALL}.__new__({CALL})
    entry.method, entry.module, entry.qualname = value, enum[0], enum[1]
    return entry
'''


def enum_cells(cls_info):
    """``{name: (module, qualname)}`` of the cells whose C result is the value of an admitted IntEnum."""
    from modelx_cython.enums import _traced_enum
    result = {}
    for name, cells in cls_info.cells.items():
        if cells.is_special() or cells.formula_is_generator or not cells.has_typeinfo():
            continue
        if cells.get_rettype_expr(c_style=True) != 'long long':
            continue
        enum = _traced_enum(cells, None)
        if enum is not None:
            result[name] = enum
    return result


class EnumResults(cst.CSTTransformer):
    """Object uses of an enum cell's C result wrapped in ``enum_member``; the Python entry of each class with
    enum cells. Receivers: ``self`` in a method of a generated class, a parameter or local annotated or cast
    with a generated class (``self: _mx_private_receiver[_c_P]``), and a Reference to a Space declared with
    that class, recursively."""

    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self, module, others=None, origins=None):
        super().__init__()
        self.module = module
        self.cells = {cls_name: enum_cells(cls_info) for cls_name, cls_info in module.classes.items()}
        self.cells.update(others or {})     # (module, class) of other modules' Spaces (child Spaces)
        self.origins = origins or {}        # module function -> (class, Cell) it is the entry of
        self.parent = module.fqname.rpartition('.')[0]
        self._values = []                   # per function: {parameter or local: enum} of enum values held as C integers
        self._owners = []                   # per function: (class, Cell) whose formula or entry it is, or None
        self.shown = set()                  # (class, Cell, parameter) whose value is used as an object
        self.foreign = []                   # (class, Cell, parameter, caller): a member of another class passed
        constants = {}
        for (cls_name, ref, member), (symbol, value) in getattr(module, 'code_enum_symbols', {}).items():
            enum = getattr(module.classes[cls_name].refs.get(ref), 'enum', None)
            if enum:
                constants[symbol] = tuple(enum[:2])
        self.constants = constants          # enum constant -> (module, qualname) of its class
        self.wrapped = 0
        self.gated = []
        self._classes = []
        self._locals = []
        self._functions = []

    # ---------------------------------------------------------------- receivers
    @staticmethod
    def _class_name(expr):
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            return expr.attr.value
        if isinstance(expr, cst.Subscript) and len(expr.slice) == 1:
            element = expr.slice[0].slice
            if isinstance(element, cst.Index):
                return EnumResults._class_name(element.value)
        if isinstance(expr, cst.SimpleString):
            return expr.value.strip('\'"').rsplit('.', 1)[-1]
        return None

    def _known(self, name):
        return name if name in self.module.classes else None

    def _ref_class(self, cls_name, attr):
        cls_info = self.module.classes.get(cls_name) if isinstance(cls_name, str) else None
        if cls_info is not None and attr in getattr(cls_info, 'spaces', ()):
            child = (f'{self.parent}.{MODULE_PREF}{cls_name[len(SPACE_PREF):]}.{MX_SPACE_MOD}', SPACE_PREF + attr)
            return child if child in self.cells else None
        ref = cls_info.refs.get(attr) if cls_info is not None else None
        decl = getattr(ref, 'decl_type_expr', None)
        return self._known(decl.rsplit('.', 1)[-1]) if decl else None

    def _cast_class(self, call):
        func = call.func
        if (isinstance(func, cst.Attribute) and func.attr.value == 'cast'
                and isinstance(func.value, cst.Name) and func.value.value == CY_MOD and call.args):
            return self._known(self._class_name(call.args[0].value))
        return None

    def _receiver(self, expr):
        if isinstance(expr, cst.Name):
            if self._locals and expr.value in self._locals[-1]:
                return self._locals[-1][expr.value]
            if expr.value == 'self' and self._classes:
                return self._classes[-1]
            return None
        if isinstance(expr, cst.Attribute):
            owner = self._receiver(expr.value)
            return self._ref_class(owner, expr.attr.value) if owner else None
        if isinstance(expr, cst.Call):
            return self._cast_class(expr)
        return None

    # ---------------------------------------------------------------- scopes
    def visit_ClassDef(self, node):
        self._classes.append(node.name.value if node.name.value in self.module.classes else None)

    def leave_ClassDef(self, original, updated):
        name = self._classes.pop()
        if name is not None and not self._classes and self.cells.get(name):
            updated = self._python_entry(name, updated)
        return updated

    def visit_FunctionDef(self, node):
        types_ = {}
        for param in (*node.params.params, *node.params.posonly_params, *node.params.kwonly_params):
            if param.annotation is not None:
                cls = self._known(self._class_name(param.annotation.annotation))
                if cls:
                    types_[param.name.value] = cls
        lines = node.body.body if isinstance(node.body, cst.IndentedBlock) else (node.body,)
        for line in lines:
            if not isinstance(line, (cst.SimpleStatementLine, cst.SimpleStatementSuite)):
                continue
            for small in line.body:
                if isinstance(small, cst.AnnAssign) and isinstance(small.target, cst.Name):
                    cls = self._known(self._class_name(small.annotation.annotation))
                    if cls is None and isinstance(small.value, cst.Call):
                        cls = self._cast_class(small.value)
                    if cls:
                        types_[small.target.value] = cls
                elif (isinstance(small, cst.Assign) and len(small.targets) == 1
                      and isinstance(small.targets[0].target, cst.Name) and isinstance(small.value, cst.Call)):
                    cls = self._cast_class(small.value)
                    if cls:
                        types_[small.targets[0].target.value] = cls
        self._locals.append(types_)
        self._functions.append(node)
        self._values.append(self._enum_values(node))

    def leave_FunctionDef(self, original, updated):
        self._locals.pop()
        self._functions.pop()
        self._values.pop()
        self._owners.pop()
        return updated

    def _where(self):
        owner = next((o for o in reversed(self._owners) if o), None)
        return '.'.join((*[p[len(MODULE_PREF):] for p in self.module.fqname.split('.')[1:-1]],
                         owner[0][len(SPACE_PREF):], owner[1])) if owner else self.module.fqname

    def _enum_values(self, node):
        """{name: enum} of the C-integer parameters traced as one IntEnum class, and of the locals every binding of
        which is an assignment of an enum cell's C result (of one class): where their value is used as an object,
        it is the member, as live holds it."""
        from .enums import _traced_enum
        from .scopes import CstFacts, lexical_nodes
        found = {}
        owner = self.origins.get(node.name.value)
        if owner is None and self._classes and self._classes[-1]:
            name = node.name.value
            owner = (self._classes[-1], name[3:] if name.startswith('_f_') else name)
        cls_info = self.module.classes.get(owner[0]) if owner else None
        name = owner[1][3:] if owner and owner[1].startswith('_f_') else owner and owner[1]
        cell = cls_info.cells.get(name) if cls_info is not None else None
        self._owners.append((owner[0], name) if cell is not None else None)
        for param in node.params.params:
            annotation = param.annotation and cst.Module([]).code_for_node(param.annotation.annotation)
            if cell is not None and annotation == f'{CY_MOD}.longlong' and param.name.value in cell.params:
                enum = _traced_enum(cell, param.name.value)
                if enum is not None:
                    found[param.name.value] = enum
        body = node.body.body if isinstance(node.body, cst.IndentedBlock) else ()
        facts = CstFacts(body)
        assigned = {}
        for part in lexical_nodes(body):
            if isinstance(part, cst.Assign) and len(part.targets) == 1 and isinstance(part.targets[0].target, cst.Name):
                assigned.setdefault(part.targets[0].target.value, []).append(self._enum_of(part.value))
        for name, enums in assigned.items():
            kinds = {e[1] if e else None for e in enums}
            if None not in kinds and len(kinds) == 1 and facts.stores[name] == len(enums) and name not in found:
                found[name] = kinds.pop()
        return found

    def leave_Name(self, original, updated):
        enum = self._values[-1].get(original.value) if self._values else None
        if enum is None:
            return updated
        parent = self._parent(original)
        if isinstance(parent, (cst.AssignTarget, cst.Param)) or (isinstance(parent, cst.Attribute)
                                                                  and parent.attr is original):
            return updated
        if not self._object_use(original):
            return updated
        owner = self._owners[-1] if self._owners else None
        if owner is not None and original.value in self.module.classes[owner[0]].cells[owner[1]].params:
            self.shown.add((*owner, original.value))
        return self._member(enum, updated)

    # ---------------------------------------------------------------- object uses
    def leave_Call(self, original, updated):
        func = original.func
        callee = self._receiver(func.value) if isinstance(func, cst.Attribute) else None
        cell = self.module.classes[callee].cells.get(func.attr.value) if isinstance(callee, str) else None
        if cell is not None and not any(a.keyword or a.star for a in original.args):
            from .enums import _traced_enum
            for arg, param in zip(original.args, cell.params):
                value = arg.value
                while (isinstance(value, cst.Call) and isinstance(value.func, cst.Attribute) and value.args
                       and value.func.attr.value == '_mx_integer_result'):
                    value = value.args[0].value         # the integer guard's checked argument
                given = self.constants.get(value.value) if isinstance(value, cst.Name) else None
                expected = _traced_enum(cell, param) if given else None
                if expected is not None and given != expected:
                    self.foreign.append((callee, cell.name, param, self._where()))
        if (isinstance(func, cst.Attribute) and func.attr.value in self.cells.get(self._receiver(func.value), {})
                and self._object_use(original) and not isinstance(self._parent(original), cst.IfExp)):
            return self._member(self.cells[self._receiver(func.value)][func.attr.value], updated)
        return updated

    def leave_IfExp(self, original, updated):
        # an inline entry: `(entry(self, t) if exact else cast(T, self.cell(t)))`
        found = self._enum_of(original.orelse)
        if found is not None and self._object_use(original):
            return self._member(found[1], updated)
        return updated

    def _parent(self, node):
        return self.get_metadata(ParentNodeProvider, node, None)

    def _enum_of(self, expr):
        """(name, enum) of the enum cell whose C value ``expr`` is (a call of it, possibly cast), else None."""
        if isinstance(expr, cst.Call):
            func = expr.func
            if (isinstance(func, cst.Attribute) and func.attr.value == 'cast'
                    and isinstance(func.value, cst.Name) and func.value.value == CY_MOD and len(expr.args) == 2):
                return self._enum_of(expr.args[1].value)
            if (isinstance(func, cst.Attribute)
                    and func.attr.value in self.cells.get(self._receiver(func.value), {})):
                return func.attr.value, self.cells[self._receiver(func.value)][func.attr.value]
        if isinstance(expr, cst.IfExp):
            return self._enum_of(expr.orelse)
        return None

    def _member(self, enum, expr):
        module, qualname = enum
        self.wrapped += 1
        return cst.Call(cst.Name(MEMBER), args=(
            cst.Arg(cst.SimpleString(repr(module))), cst.Arg(cst.SimpleString(repr(qualname))),
            cst.Arg(expr.with_changes(lpar=(), rpar=()))))

    _TYPED_BUILTINS = {'int', 'float', 'bool', 'abs', 'min', 'max', 'round', 'range', 'len'}

    def _object_use(self, node):
        """Whether the value of ``node`` reaches Python as an object."""
        parent = self._parent(node)
        if isinstance(parent, cst.Attribute):
            return parent.value is node
        if isinstance(parent, cst.Element) and isinstance(self._parent(parent), cst.Tuple):
            # a tuple is usually a key (table, cache): its elements follow the tuple's own use
            return self._object_use(self._parent(parent))
        if isinstance(parent, (cst.Element, cst.StarredElement, cst.DictElement, cst.StarredDictElement,
                               cst.Yield)):
            return True
        if isinstance(parent, (cst.ListComp, cst.SetComp, cst.GeneratorExp)):
            return parent.elt is node
        if isinstance(parent, cst.DictComp):
            return parent.key is node or parent.value is node
        if isinstance(parent, cst.IfExp):
            return parent.test is not node and self._object_use(parent)
        if isinstance(parent, cst.Return):
            return self._returns_object(parent)
        if isinstance(parent, cst.Arg):
            call = self._parent(parent)
            return isinstance(call, cst.Call) and not self._typed_call(call)
        return False

    def _typed_call(self, call):
        func = call.func
        if isinstance(func, cst.Name):
            return func.value.startswith('_mx_') or func.value in self._TYPED_BUILTINS
        if isinstance(func, cst.Attribute):
            if func.attr.value.startswith('_mx_'):
                return True
            root = func.value
            while isinstance(root, (cst.Attribute, cst.Call, cst.Subscript)):
                root = root.func if isinstance(root, cst.Call) else root.value
            if isinstance(root, cst.Name) and (root.value.startswith('_mx_') or root.value == CY_MOD):
                return True     # a generated helper, a provider handle or a cast of one
            return self._receiver(func.value) is not None      # a cell or method of a compiled Space
        return False

    def _returns_object(self, node):
        while node is not None and not isinstance(node, (cst.FunctionDef, cst.Lambda)):
            node = self._parent(node)
        if not isinstance(node, cst.FunctionDef) or node.returns is None:
            return True
        annotation = cst.Module(body=[]).code_for_node(node.returns.annotation)
        return annotation == 'object'

    # ---------------------------------------------------------------- the Python entry
    def _python_entry(self, cls_name, node):
        """``__getattribute__`` gives a Python caller each enum cell's method wrapped (its result the member);
        an existing ``__getattribute__`` (the integer arguments' entry) keeps its body, its results wrapped."""
        field = f'_mx_enum_cells_{cls_name}'
        self.gated.append((field, {name: self.cells[cls_name][name] for name in sorted(self.cells[cls_name])}))
        owner = f'{CY_MOD}.cast(object, {cls_name})'

        class Results(cst.CSTTransformer):
            def leave_Return(self, original, updated):
                call = cst.parse_expression(f'{ENTRY}(self, {owner}, {field}, name, 0)')
                return updated.with_changes(value=call.with_changes(args=(*call.args[:4], cst.Arg(updated.value))))

            def visit_FunctionDef(self, node):
                return node.name.value == '__getattribute__'

        body, found = [], False
        for statement in node.body.body:
            if isinstance(statement, cst.FunctionDef) and statement.name.value == '__getattribute__':
                statement = statement.visit(Results())
                found = True
            body.append(statement)
        if not found:
            body.append(cst.parse_statement(
                f'def __getattribute__(self, name):\n'
                f'    return {ENTRY}(self, {owner}, {field}, name, {MX_SYS_MOD}._mx_generic_getattr(self, name))\n'))
        return node.with_changes(body=node.body.with_changes(body=tuple(body)))


def finish(syntax, module, others=None, origins=None):
    """The generated module with enum results as members where they reach Python; unchanged without enum cells.
    ``others``: ``{(module, class): enum cells}`` of the model's other modules; ``origins``: the Cell of each module
    function that is a Cell's entry."""
    from .enums import _traced_enum
    if not others and not any(enum_cells(cls_info) or any(
            _traced_enum(cell, param) for cell in cls_info.cells.values() if cell.has_typeinfo()
            for param in cell.params) for cls_info in module.classes.values()):
        return syntax
    results = EnumResults(module, others, origins)
    syntax = cst.MetadataWrapper(syntax, unsafe_skip_copy=True).visit(results)
    for cls_name, cell, param, where in results.foreign:
        if (cls_name, cell, param) in results.shown:
            raise TypeError(f'{where}: passes a member of one IntEnum class to the parameter {param!r} of '
                            f'{cell}, traced as another IntEnum class and used there as an object, so its value '
                            'would differ from live modelx')
    if not (results.wrapped or results.gated):
        return syntax
    added = cst.parse_module(FUNCTIONS).body
    added += tuple(cst.parse_statement(f'{field} = {cells!r}\n') for field, cells in results.gated)
    return syntax.with_changes(body=(*syntax.body, *added))
