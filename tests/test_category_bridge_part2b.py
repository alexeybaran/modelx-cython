from __future__ import annotations

import ast
from types import SimpleNamespace

from modelx_graph.category_abi import CategoryDomain
from modelx_graph.modelx_cython_category_abi import (
    _prove_parameter_accepts_category_label,
    _selected_category_forward_demands,
)


def _formula(src: str, parameters: tuple[str, ...]):
    fn = ast.parse(src).body[0]
    assert isinstance(fn, ast.FunctionDef)
    return SimpleNamespace(function=fn, parameters=parameters)


def test_part2b_category_acceptance_is_path_sensitive_after_terminal_dispatch():
    base = CategoryDomain.from_labels(("F", "M"), "observed table index")
    formulas = {
        "mort_rate_at_age": _formula(
            "def mort_rate_at_age(table_sex, x):\n"
            "    if x >= 120:\n"
            "        return 1.0\n"
            "    if table_sex == 'mix':\n"
            "        return 0.5\n"
            "    if table_sex not in ('M', 'F'):\n"
            "        raise ValueError('bad sex')\n"
            "    return mort_table.loc[(table_sex, x), 'q']\n",
            ("table_sex", "x"),
        )
    }
    kwargs = dict(
        formulas=formulas,
        selected={"mort_rate_at_age"},
        uid="mort_rate_at_age",
        parameter="table_sex",
        domains={base.domain_id: base},
        parameter_domains={("mort_rate_at_age", "table_sex"): base.domain_id},
        result_domains={},
        exhaustive_dispatch={},
        cache={},
        active=set(),
    )

    assert _prove_parameter_accepts_category_label(label="mix", **kwargs) == (
        True,
        "selected_source_terminal_paths_accept",
    )
    rejected = _prove_parameter_accepts_category_label(
        label="X", **{**kwargs, "cache": {}, "active": set()}
    )
    assert rejected == (False, "source_path_rejects_label")


def test_part2b_direct_category_result_creates_callee_parameter_demand():
    source = CategoryDomain.from_labels(("F", "M", "mix"), "source result")
    target = CategoryDomain.from_labels(("F", "M"), "observed callee input")
    formulas = {
        "sex": _formula("def sex(life):\n    return 'M'\n", ("life",)),
        "mort_rate_at_age": _formula(
            "def mort_rate_at_age(table_sex, x):\n    return 0.0\n",
            ("table_sex", "x"),
        ),
        "mort_rate": _formula(
            "def mort_rate(life, x):\n    return mort_rate_at_age(sex(life), x)\n",
            ("life", "x"),
        ),
    }
    demands = _selected_category_forward_demands(
        formulas,
        output_uid="mort_rate",
        domains={source.domain_id: source, target.domain_id: target},
        parameter_domains={("mort_rate_at_age", "table_sex"): target.domain_id},
        result_domains={"sex": source.domain_id},
    )

    assert set(demands[("mort_rate_at_age", "table_sex")]) == {"F", "M", "mix"}
    assert demands[("mort_rate_at_age", "table_sex")]["mix"] == {
        ("mort_rate", "result:sex")
    }
