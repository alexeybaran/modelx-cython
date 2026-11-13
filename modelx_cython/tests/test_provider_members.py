import libcst as cst

from modelx_cython.provider_members import ProviderMemberCalls
from modelx_cython.usage import CellsResolver

CLS = "Model_nomx._mx_classes._c_Space1"


def test_member_call_on_a_call_not_rooted_at_self_is_left_alone():
    """A method called on the result of a call whose callee is an attribute
    chain not rooted at self (``pd.Series(t).sum()``) is not a provider
    member call: the transformer must leave it unchanged rather than try to
    resolve the chain."""
    resolver = CellsResolver(
        cells_by_class={CLS: {"rate": CLS + "._f_rate"}},
        refs_by_class={CLS: {"pd": ""}},
        spaces_by_class={CLS: {}},
    )
    resolver.provider_results = {CLS + "._f_rate": object()}
    source = "def _f_target(self, t):\n    return pd.Series(t).sum() + self.pd.Series(t).sum()\n"
    func = cst.parse_statement(source)
    providers = set()
    members = ProviderMemberCalls(resolver, CLS, providers)
    out = func.visit(members)
    assert cst.Module([]).code_for_node(out) == source
    assert not providers and not members.operations
