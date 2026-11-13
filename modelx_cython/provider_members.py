"""Lower explicitly declared operations on Cells with native owner bindings."""

import libcst as cst

from .helpers import module_name
from .scopes import FormulaTransformer
from .usage import RESOLVED, _extract_self_chain


def _integer_literal(value):
    sign = 1
    if isinstance(value, cst.UnaryOperation) and isinstance(value.operator, (cst.Plus, cst.Minus)):
        sign = -1 if isinstance(value.operator, cst.Minus) else 1
        value = value.expression
    return (isinstance(value, cst.Integer)
            and -(1 << 63) <= sign * value.evaluated_value < (1 << 63))


class ProviderMemberCalls(FormulaTransformer):
    def __init__(self, resolver, space, providers):
        super().__init__()
        self.resolver = resolver
        self.space = space
        self.providers = providers
        self.operations = set()

    def leave_Call(self, original, updated):
        member = original.func
        if not (isinstance(member, cst.Attribute) and isinstance(member.value, cst.Call)):
            return updated
        chain = _extract_self_chain(member.value.func)
        # Only a non-empty self-rooted chain can name a provider result; an
        # unresolved base (an attribute chain not rooted at self) or a
        # non-attribute callee is not a provider member call.
        if not isinstance(chain, list) or not chain:
            return updated
        resolved, name = self.resolver.resolve_chain(self.space, chain)
        binding = self.resolver.provider_results.get(name) if resolved == RESOLVED else None
        if binding is None:
            return updated
        operation = binding.declaration.members.get(member.attr.value)
        if (operation is None or operation.signature != 'int64_label_to_float64'
                or len(updated.args) != 1 or updated.args[0].keyword or updated.args[0].star
                or not _integer_literal(updated.args[0].value)):
            return updated
        provider = binding.manifest.provider
        self.providers.add(provider)
        self.operations.add(f'{provider}.{operation.native}')
        return updated.with_changes(
            func=cst.Attribute(cst.Name(module_name(provider)), cst.Name(operation.native)),
            args=(cst.Arg(updated.func.value), *updated.args),
        )
