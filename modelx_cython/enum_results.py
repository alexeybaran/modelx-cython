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

from .consts import CY_MOD, MX_SYS_MOD

MEMBER = '_mx_enum_member'
ENTRY = '_mx_enum_entry'
CALL = '_mx_enum_call'

# The member of the IntEnum ``module.qualname`` whose value is ``value``: what a Python caller, an
# object-valued use or a collected output receives, as in live modelx.
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
    return member if member is not None else cls(value)


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

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.cells = {cls_name: enum_cells(cls_info) for cls_name, cls_info in module.classes.items()}
        self.enums = {}
        for cells in self.cells.values():
            self.enums.update(cells)
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
        cls_info = self.module.classes.get(cls_name)
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

    def leave_FunctionDef(self, original, updated):
        self._locals.pop()
        self._functions.pop()
        return updated

    # ---------------------------------------------------------------- object uses
    def leave_Call(self, original, updated):
        func = original.func
        if (isinstance(func, cst.Attribute) and func.attr.value in self.cells.get(self._receiver(func.value), {})
                and self._object_use(original) and not isinstance(self._parent(original), cst.IfExp)):
            return self._member(func.attr.value, updated)
        return updated

    def leave_IfExp(self, original, updated):
        # an inline entry: `(entry(self, t) if exact else cast(T, self.cell(t)))`
        name = self._enum_of(original.orelse)
        if name is not None and self._object_use(original):
            return self._member(name, updated)
        return updated

    def _parent(self, node):
        return self.get_metadata(ParentNodeProvider, node, None)

    def _enum_of(self, expr):
        """The enum cell whose C value ``expr`` is (a call of it, possibly cast), else None."""
        if isinstance(expr, cst.Call):
            func = expr.func
            if (isinstance(func, cst.Attribute) and func.attr.value == 'cast'
                    and isinstance(func.value, cst.Name) and func.value.value == CY_MOD and len(expr.args) == 2):
                return self._enum_of(expr.args[1].value)
            if (isinstance(func, cst.Attribute)
                    and func.attr.value in self.cells.get(self._receiver(func.value), {})):
                return func.attr.value
        if isinstance(expr, cst.IfExp):
            return self._enum_of(expr.orelse)
        return None

    def _member(self, name, expr):
        module, qualname = self.enums[name]
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


def finish(syntax, module):
    """The generated module with enum results as members where they reach Python; unchanged without enum cells."""
    if not any(enum_cells(cls_info) for cls_info in module.classes.values()):
        return syntax
    results = EnumResults(module)
    syntax = cst.MetadataWrapper(syntax, unsafe_skip_copy=True).visit(results)
    added = cst.parse_module(FUNCTIONS).body
    added += tuple(cst.parse_statement(f'{field} = {cells!r}\n') for field, cells in results.gated)
    return syntax.with_changes(body=(*syntax.body, *added))
