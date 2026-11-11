from __future__ import annotations

import ast
from types import SimpleNamespace

from modelx_graph.category_abi import CategoryDomain
from modelx_graph.finite_domain_category import (
    FiniteValue,
    TruthValue,
    assign_finite_value,
    evaluate_predicate,
    finite_value,
)
from modelx_graph.modelx_cython_category_abi import (
    _prove_parameter_accepts_category_label,
)


def _expr(src: str) -> ast.expr:
    node = ast.parse(src, mode="eval").body
    assert isinstance(node, ast.expr)
    return node


def _formula(src: str, parameters: tuple[str, ...]):
    fn = ast.parse(src).body[0]
    assert isinstance(fn, ast.FunctionDef)
    return SimpleNamespace(function=fn, parameters=parameters)


def test_opaque_and_finite_false_collapses_to_false_without_category_rejection():
    env = {"cause": FiniteValue.one("burn")}
    result = evaluate_predicate(
        _expr("broad_def() and cause in ('cerebral', 'cardiac')"), env
    )
    assert result.truth == TruthValue.FALSE
    assert result.category_related
    assert not result.unsupported_category


def test_opaque_and_finite_true_remains_unknown_and_explores_both_paths():
    env = {"cause": FiniteValue.one("cerebral")}
    result = evaluate_predicate(
        _expr("broad_def() and cause in ('cerebral', 'cardiac')"), env
    )
    assert result.truth == TruthValue.UNKNOWN
    assert result.category_related
    assert not result.unsupported_category


def test_boolean_not_or_and_none_membership_are_exact():
    env = {"cause": FiniteValue.one(None)}
    result = evaluate_predicate(
        _expr("not (cause is not None) or cause in ('cancer', 'burn')"), env
    )
    assert result.truth == TruthValue.TRUE
    assert result.category_related
    assert not result.unsupported_category


def test_unsupported_category_operation_remains_fail_closed():
    env = {"cause": FiniteValue.one("burn")}
    result = evaluate_predicate(_expr("cause.startswith('b')"), env)
    assert result.truth == TruthValue.UNKNOWN
    assert result.category_related
    assert result.unsupported_category


def test_simple_assignment_and_conditional_expression_preserve_finite_values():
    env = {"cause": FiniteValue.one("burn")}
    assign_finite_value(env, "alias", _expr("cause"))
    assert env["alias"].values == frozenset({"burn"})
    assign_finite_value(env, "mapped", _expr("'A' if flag() else 'B'"))
    assert env["mapped"].values == frozenset({"A", "B"})
    assert finite_value(_expr("alias if flag() else cause"), env).values == frozenset({"burn"})


def test_p34_style_nonterminal_category_guard_accepts_fallthrough_labels():
    base = CategoryDomain.from_labels(("cancer", "cerebral", "cardiac"), "observed incidence table")
    downstream = CategoryDomain.from_labels(
        ("burn", "fracture", "liability", "cancer", "cerebral", "cardiac"),
        "complete downstream table",
    )
    formulas = {
        "inc_rate": _formula(
            "def inc_rate(t, cause):\n"
            "    if not born(t):\n"
            "        return 0.0\n"
            "    rate = inc_rate_at(t, cause)\n"
            "    if broad_def() and cause in ('cerebral', 'cardiac'):\n"
            "        rate = rate * 1.2\n"
            "    return rate\n",
            ("t", "cause"),
        ),
        "inc_rate_at": _formula(
            "def inc_rate_at(t, cause):\n"
            "    return 0.01\n",
            ("t", "cause"),
        ),
    }
    domains = {base.domain_id: base, downstream.domain_id: downstream}
    kwargs = dict(
        formulas=formulas,
        selected=set(formulas),
        uid="inc_rate",
        parameter="cause",
        domains=domains,
        parameter_domains={
            ("inc_rate", "cause"): base.domain_id,
            ("inc_rate_at", "cause"): downstream.domain_id,
        },
        result_domains={},
        exhaustive_dispatch={},
        cache={},
        active=set(),
    )
    for label in ("burn", "fracture", "liability"):
        assert _prove_parameter_accepts_category_label(label=label, **kwargs) == (
            True,
            "selected_source_forwarding_accepts",
        )


def test_category_completion_can_end_at_python_semantic_parameter_boundary():
    base = CategoryDomain.from_labels(("cancer",), "observed native caller")
    formulas = {
        "caller": _formula(
            "def caller(cause):\n"
            "    return helper(cause)\n",
            ("cause",),
        ),
        "helper": _formula(
            "def helper(cause):\n"
            "    return table.loc[cause]\n",
            ("cause",),
        ),
    }
    assert _prove_parameter_accepts_category_label(
        formulas=formulas,
        selected=set(formulas),
        uid="caller",
        parameter="cause",
        label="burn",
        domains={base.domain_id: base},
        parameter_domains={("caller", "cause"): base.domain_id},
        result_domains={},
        exhaustive_dispatch={},
        cache={},
        active=set(),
        python_semantic_parameters=frozenset({("helper", "cause")}),
    ) == (True, "selected_source_forwarding_accepts")


def test_untyped_non_python_boundary_still_fails_closed():
    base = CategoryDomain.from_labels(("cancer",), "observed native caller")
    formulas = {
        "caller": _formula("def caller(cause):\n    return helper(cause)\n", ("cause",)),
        "helper": _formula("def helper(cause):\n    return 0.0\n", ("cause",)),
    }
    result = _prove_parameter_accepts_category_label(
        formulas=formulas,
        selected=set(formulas),
        uid="caller",
        parameter="cause",
        label="burn",
        domains={base.domain_id: base},
        parameter_domains={("caller", "cause"): base.domain_id},
        result_domains={},
        exhaustive_dispatch={},
        cache={},
        active=set(),
    )
    assert result == (False, "category_forwarded_to_untyped_parameter")


def test_direct_call_literal_evidence_tracks_bounded_literal_for_loop():
    from modelx_graph.modelx_cython_category_abi import _direct_call_literal_args

    formulas = {
        "caller": _formula(
            "def caller():\n"
            "    total = 0.0\n"
            "    for cause in ('cancer', 'minor_cancer', 'cerebral', 'cardiac'):\n"
            "        total += callee(cause)\n"
            "    return total\n",
            (),
        ),
        "callee": _formula("def callee(cause):\n    return 0.0\n", ("cause",)),
    }
    assert _direct_call_literal_args(formulas)[("callee", 0)] == (
        "cancer", "cardiac", "cerebral", "minor_cancer"
    )
