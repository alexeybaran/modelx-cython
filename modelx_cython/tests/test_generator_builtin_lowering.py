"""The retired recognizer cannot silently reactivate AST lowering."""
import ast
import libcst as cst

from modelx_cython.reductions import NativeReductions


def test_retired_pass_preserves_source_and_never_parses_ast(monkeypatch):
    source = "def formula(self, n):\n    return all(self.term(t) for t in range(n))\n"
    formula = cst.parse_module(source)
    def forbidden(*args, **kwargs):
        raise AssertionError("retired generator AST parser")
    with monkeypatch.context() as patch:
        patch.setattr(ast, "parse", forbidden)
        retired = NativeReductions(None)
        result = formula.visit(retired).code
    assert result == source
    assert retired.count == 0 and retired.fallbacks == []
