"""Keep Python argument order for Cython's min/max numeric intrinsics.

Packing arguments before ordinary star expansion prevents intrinsic emission
from moving an effectful argument. The actual builtin remains the consumer;
there is no numeric approximation, formula analysis or additional public API.
"""
import libcst as cst
from libcst.metadata import ScopeProvider
from libcst.metadata.scope_provider import BuiltinAssignment


class _OrderedArguments(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ScopeProvider,)

    def leave_Call(self, before, after):
        function = before.func
        if not isinstance(function, cst.Name) or function.value not in ('min', 'max'):
            return after
        if any(argument.star for argument in after.args):
            return after
        positional = tuple(argument for argument in after.args if argument.keyword is None)
        if len(positional) < 2:
            return after
        scope = self.get_metadata(ScopeProvider, function)
        bindings = tuple(scope[function.value])
        if not bindings or not all(isinstance(binding, BuiltinAssignment) for binding in bindings):
            return after
        values = cst.Tuple(tuple(cst.Element(argument.value) for argument in positional))
        keywords = tuple(argument for argument in after.args if argument.keyword is not None)
        return after.with_changes(args=(cst.Arg(values, star='*'), *keywords))


def preserve_argument_order(formula):
    """Normalize declared builtin calls using the existing concrete syntax."""
    wrapper = cst.MetadataWrapper(cst.Module(body=(formula,)))
    return wrapper.visit(_OrderedArguments()).body[0]
