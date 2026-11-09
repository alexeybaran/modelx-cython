"""Preserve object power semantics outside the proved native domain.

Boxing unproved power operands retains Python numeric types and exceptions.
Builtin extrema remain ordinary calls, eligible for Cython's native intrinsic.
Their inputs are captured at build time, without converting numbers to objects.
"""
import libcst as cst
from libcst.metadata import ScopeProvider
from libcst.metadata.scope_provider import BuiltinAssignment


class _OrderedArguments(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ScopeProvider,)

    def __init__(self, kind=None, native_result=False):
        super().__init__()
        self.kind = kind
        self.opaque_depth = 0
        self.native_result = native_result

    def on_visit(self, node):
        from .scopes import OPAQUE_SCOPES
        if isinstance(node, OPAQUE_SCOPES):
            self.opaque_depth += 1
        return super().on_visit(node)

    def on_leave(self, before, after):
        result = super().on_leave(before, after)
        from .scopes import OPAQUE_SCOPES
        if isinstance(before, OPAQUE_SCOPES):
            self.opaque_depth -= 1
        return result

    def leave_BinaryOperation(self, before, after):
        if not isinstance(before.operator, cst.Power):
            return after
        # The qualified real-power pass has already handled its proved C
        # domain. Preserve Python's remaining integer/mixed/complex power
        # result and error semantics through Cython's ordinary object operator.
        scope = self.get_metadata(ScopeProvider, before)
        bindings = tuple(scope['object'])
        if not bindings or not all(isinstance(binding, BuiltinAssignment) for binding in bindings):
            return after
        if tuple(scope['_mx_cy']):
            return after
        from .powers import KIND_FLOAT, KIND_INT
        if (self.kind is not None and self.opaque_depth == 1
                and not tuple(scope['_mx_sys'])
                and self.kind.kind(before.left) == KIND_FLOAT
                and self.kind.kind(before.right) in (KIND_FLOAT, KIND_INT)):
            capture = cst.Call(cst.parse_expression('_mx_sys._mx_capture_power_base'),
                               (cst.Arg(after.left),))
            return cst.Call(cst.parse_expression('_mx_sys._mx_object_power'),
                            (cst.Arg(capture), cst.Arg(after.right)),
                            lpar=after.lpar, rpar=after.rpar)
        def boxed(value):
            return cst.Call(cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')),
                            (cst.Arg(cst.Name('object')), cst.Arg(value)))
        return after.with_changes(left=boxed(after.left), right=boxed(after.right))

    def leave_Call(self, before, after):
        # A numeric Cell already normalizes its result through its native ABI.
        # Object results/child scopes must keep the selected operand's Python
        # scalar class: Cython promotes mixed native min/max inputs to double.
        # This fallback is about the result contract, not first-error order.
        if self.native_result and self.opaque_depth == 1:
            return after
        if not isinstance(before.func, cst.Name) or before.func.value not in ('min', 'max'):
            return after
        if any(argument.star for argument in after.args):
            return after
        positional = tuple(argument for argument in after.args if argument.keyword is None)
        if len(positional) < 2:
            return after
        scope = self.get_metadata(ScopeProvider, before.func)
        bindings = tuple(scope[before.func.value])
        if not bindings or not all(isinstance(binding, BuiltinAssignment) for binding in bindings):
            return after
        values = cst.Tuple(tuple(cst.Element(argument.value) for argument in positional))
        keywords = tuple(argument for argument in after.args if argument.keyword is not None)
        return after.with_changes(args=(cst.Arg(values, star='*'), *keywords))


def preserve_argument_order(formula, kind=None, native_result=False):
    """Keep unproved powers on Cython's ordinary Python-object operator."""
    wrapper = cst.MetadataWrapper(cst.Module(body=(formula,)))
    return wrapper.visit(_OrderedArguments(kind, native_result)).body[0]
