"""Recognize factory computations whose private receiver cannot be published."""

import libcst as cst
from libcst.metadata import ExpressionContext, MetadataWrapper, ScopeProvider

from .scopes import CstFacts, lexical_nodes
from .syntax import arguments, dotted_name, literal_value, same_expression, statements, walk


_BUILTINS = {'abs', 'bool', 'float', 'int', 'max', 'min', 'round', 'str', 'ValueError'}
_REFLECTIVE = {'locals', 'globals', 'vars', 'eval', 'exec', 'super', 'getattr', 'setattr',
               '_getframe', 'currentframe', 'f_back', 'f_locals'}

_APPLY = """def _mx_apply_formula(self, *args, **kwargs):
    result = self._parent._mx_space_formula(*args, **kwargs)
    if result is None:
        return
    if not isinstance(result, dict) or result.keys() - {'refs'}:
        raise ValueError('Exported Space formula must return only dynamic References')
    refs = result.get('refs')
    if refs is None:
        return
    if type(refs) is not dict or refs.keys() - ALIASES:
        raise ValueError('Unexpected dynamic Reference')
    for name, value in refs.items():
        setattr(self, name, value)
"""


def _methods(node):
    return {method.name.value: method for method in node.body.body
            if isinstance(method, cst.FunctionDef)}


def _constant(value):
    return (isinstance(value, (cst.Integer, cst.Float, cst.Imaginary, cst.SimpleString, cst.Ellipsis))
            or isinstance(value, cst.ConcatenatedString) and value.evaluated_value is not None
            or isinstance(value, cst.Name) and value.value in ('None', 'True', 'False'))


def _attribute(node, receiver):
    if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name) and node.value.value == receiver:
        return node.attr.value
    return None


def standard_lifecycle(node, aliases, *, remap_references=False):
    """Recognize exporter-owned construction, reference copy and alias assignment."""
    methods = _methods(node)
    if any(name in methods for name in ('__new__', '__del__', '__getattribute__', '__setattr__', '__getattr__')):
        return False
    apply = methods.get('_mx_apply_formula')
    if apply is None or apply.decorators:
        return False
    template = cst.parse_statement(_APPLY.replace('ALIASES', repr(tuple(aliases))))
    if not same_expression(apply, template):
        return False
    for name in ('__init__', '_mx_copy_refs'):
        method = methods.get(name)
        if method is None or method.decorators:
            return False
        for stmt in statements(method.body):
            if not isinstance(stmt, cst.Assign) or len(stmt.targets) != 1:
                return False
            dest = _attribute(stmt.targets[0].target, 'self')
            if dest is None:
                return False
            value = stmt.value
            if name == '_mx_copy_refs':
                if not (_attribute(value, 'base') == dest
                        or remap_references and _remapped_reference(value, dest)):
                    return False
            elif not (_constant(value)
                      or isinstance(value, (cst.Dict, cst.List)) and not value.elements
                      or (dest, dotted_name(value)) in
                      (('_space', 'self'), ('_parent', 'parent'), ('_model', 'parent._model'))):
                return False
    return True


def _remapped_reference(value, name):
    if not isinstance(value, cst.IfExp) or _attribute(value.orelse, 'base') != name:
        return False
    args = arguments(value.test, f'base.{name}._mx_is_in', 1)
    if args is None or dotted_name(args[0]) != 'base_root':
        return False
    target = value.body
    while isinstance(target, cst.Attribute) and target.attr.value != '_parent':
        target = target.value
    return isinstance(target, cst.Attribute) and isinstance(target.value, cst.Name) and target.value.value == 'self'


def unshadowed_builtins(module):
    if any(isinstance(item, cst.ImportStar) for item in lexical_nodes(module.body)):
        return False
    scope = MetadataWrapper(module, unsafe_skip_copy=True).resolve(ScopeProvider)[module]
    protected = _BUILTINS | {'setattr', 'type', 'isinstance', 'dict'}
    return not any(assignment.name in protected for assignment in scope.assignments)


def _nodes_with_parents(suite):
    def visit(node, parent=None):
        yield node, parent
        # Attribute names and keyword labels are syntax, not variable uses.
        if isinstance(node, (cst.Attribute, cst.Arg)):
            yield from visit(node.value, node)
        else:
            for child in node.children:
                yield from visit(child, node)

    for stmt in statements(suite):
        yield from visit(stmt)


def detached_factory_graph(node, cls, cells, *, forbidden_refs=(), forbidden_cells=(), include_factory=True):
    """Check that the policy receiver cannot become an input or an output.

    The opt-in permits calls on fixed external dependencies that do not inspect
    caller frames. A bound policy method may only be the target of a direct call.
    """
    methods = _methods(node)
    pending = ['_f_' + name for name in cells]
    if include_factory:
        pending.append('_mx_space_formula')
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        method = methods.get(name)
        if method is None or name in forbidden_cells:
            return False
        contexts = CstFacts(statements(method.body)).contexts
        for expr, parent in _nodes_with_parents(method.body):
            if isinstance(expr, (cst.Global, cst.Nonlocal, cst.Yield, cst.Await,
                                 cst.Lambda, cst.FunctionDef, cst.ClassDef)):
                return False
            if isinstance(expr, (cst.Attribute, cst.Subscript)) and contexts.get(expr) is not ExpressionContext.LOAD:
                return False
            if isinstance(expr, cst.Name):
                if expr.value in _REFLECTIVE:
                    return False
                if expr.value == 'self' and (contexts.get(expr) is not ExpressionContext.LOAD
                                             or not isinstance(parent, cst.Attribute)):
                    return False
            if isinstance(expr, cst.Attribute):
                attribute = expr.attr.value
                if attribute in _REFLECTIVE or attribute.startswith('__'):
                    return False
                if _attribute(expr, 'self') is None:
                    continue
                if attribute in cls.cells:
                    if not isinstance(parent, cst.Call) or parent.func is not expr:
                        return False
                    pending.append('_f_' + attribute)
                elif attribute not in cls.refs or attribute.startswith('_') or attribute in forbidden_refs:
                    return False
    return True


def _strings(elements, element_type):
    names = []
    for element in elements:
        if not isinstance(element, element_type):
            return None
        value = element.value if element_type is cst.Element else element.key
        if not isinstance(value, (cst.SimpleString, cst.ConcatenatedString)) or not isinstance(value.evaluated_value, str):
            return None
        names.append(value.evaluated_value)
    return tuple(names)


def declared_aliases(node):
    """Read the aliases from the exporter-owned apply method."""
    method = _methods(node).get('_mx_apply_formula')
    if method is not None:
        for expr in walk(method.body):
            if (isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, cst.Subtract)
                    and isinstance(expr.right, cst.Tuple)):
                names = _strings(expr.right.elements, cst.Element)
                if names is not None:
                    return names
    return ()


def fixed_factory_refs(node):
    """Return the insertion order of a factory's identical literal result schemas."""
    factory = _methods(node).get('_mx_space_formula')
    if factory is None or factory.decorators or not isinstance(statements(factory.body)[-1], cst.Return):
        return ()
    aliases = set(declared_aliases(node))
    schema = None
    for expr in walk(factory.body):
        if not isinstance(expr, cst.Return):
            continue
        result = expr.value
        if (not isinstance(result, cst.Dict) or _strings(result.elements, cst.DictElement) != ('refs',)
                or not isinstance(refs := result.elements[0].value, cst.Dict)):
            return ()
        names = _strings(refs.elements, cst.DictElement)
        if (names is None or len(set(names)) != len(names) or not set(names) <= aliases
                or schema is not None and names != schema):
            return ()
        schema = names
    return schema or ()


def scalar_factory_graph(node, cls, shape):
    """Prove the reachable graph uses numeric inputs and direct calls to local Cells.

    An unrecognized graph keeps fresh private policy instances.
    """
    if any(kind not in ('double', 'long long') for _, kind in shape.extra):
        return False
    methods = _methods(node)
    pending = ['_mx_space_formula', *('_f_' + leaf.cell for leaf in shape.native)]
    checked = set()
    while pending:
        name = pending.pop()
        if name in checked:
            continue
        method = methods.get(name)
        if method is None:
            return False
        checked.add(name)
        sig = method.params
        params = {param.name.value for param in sig.params}
        if isinstance(sig.star_arg, cst.Param) or sig.star_kwarg or sig.posonly_params or sig.kwonly_params or params & _BUILTINS:
            return False
        for param in sig.params:
            if param.default is not None:
                try:
                    if type(literal_value(param.default)) not in (int, float, bool):
                        return False
                except (TypeError, ValueError):
                    return False
        for expr, parent in _nodes_with_parents(method.body):
            if isinstance(expr, (cst.StarredDictElement, cst.StarredElement)):
                return False
            if isinstance(expr, cst.Call):
                if arguments(expr) is None:
                    return False
                if isinstance(expr.func, cst.Attribute):
                    attribute = _attribute(expr.func, 'self')
                    if attribute not in cls.cells:
                        return False
                    pending.append('_f_' + attribute)
                elif not isinstance(expr.func, cst.Name) or expr.func.value not in _BUILTINS:
                    return False
            elif isinstance(expr, cst.Attribute):
                attribute = _attribute(expr, 'self')
                if attribute in cls.cells:
                    if not isinstance(parent, cst.Call) or parent.func is not expr:
                        return False
                elif (attribute not in cls.refs
                      or cls.refs[attribute].get_type_expr(c_style=True) not in ('long long', 'double', 'bint')):
                    return False
            elif isinstance(expr, cst.Name) and not _constant(expr):
                if expr.value == 'self':
                    if not isinstance(parent, cst.Attribute) or parent.value is not expr:
                        return False
                elif expr.value not in params and expr.value not in _BUILTINS:
                    return False
            elif isinstance(expr, (cst.BaseExpression, cst.BaseSmallStatement, cst.BaseCompoundStatement)):
                if not (_constant(expr) or isinstance(expr, (cst.Return, cst.Expr, cst.If, cst.IfExp,
                        cst.Raise, cst.Dict, cst.Tuple, cst.BinaryOperation, cst.UnaryOperation,
                        cst.BooleanOperation, cst.Comparison))):
                    return False
    return True
