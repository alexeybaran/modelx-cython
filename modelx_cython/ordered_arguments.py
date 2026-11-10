"""Preserve object power semantics outside the proved native domain.

Numeric Cells keep c8's power and extrema mechanics unchanged. Object-result
Cells and child scopes use ordinary object operators when Cython's native
promotion would change their scalar-class contract.
"""
import libcst as cst
from libcst.metadata import ScopeProvider, ParentNodeProvider
from libcst.metadata.scope_provider import BuiltinAssignment


class _OrderedArguments(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider)

    def __init__(self, native_result=False, operand_kind=None):
        super().__init__()
        self.opaque_depth = 0
        self.native_result = native_result
        self.operand_kind = operand_kind

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
        if self.native_result and self.opaque_depth == 1:
            # Proven floating powers were lowered earlier. An integral exponent
            # retains c8's arithmetic; other unproved powers need the existing
            # object operator so Cython cannot infer an unordered complex value.
            from .powers import KIND_INT
            if self.operand_kind is None or self.operand_kind.kind(before.right) == KIND_INT:
                return after
            # A native return already supplies Cython's scalar conversion.
            # Keep that existing arithmetic path; intermediate values and
            # comparisons lack this conversion boundary and must stay objects.
            parent = self.get_metadata(ParentNodeProvider, before)
            while isinstance(parent, (cst.BinaryOperation, cst.UnaryOperation)):
                parent = self.get_metadata(ParentNodeProvider, parent)
            if isinstance(parent, cst.Return):
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


def preserve_argument_order(formula, native_result=False, operand_kind=None):
    """Keep unproved powers on Cython's ordinary Python-object operator."""
    wrapper = cst.MetadataWrapper(cst.Module(body=(formula,)))
    return wrapper.visit(_OrderedArguments(native_result, operand_kind)).body[0]
