from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from modelx_graph.execution_classes import (
    N0,
    N1,
    P,
    NATIVE_SCALAR_VALUE_ABI,
    UNSUPPORTED_VALUE_ABI,
    ExecutionClassCell,
    ExecutionClassPlan,
    FallbackPolicyContextContract,
    NativeMemoContract,
)
from modelx_graph.mixed_recursive_native import (
    MixedRecursiveNativeError,
    _core_cache_cython_p_candidate,
    _mixed_recursive_uids,
    _python_key_helper_candidates,
)
from modelx_graph.native_formula import NativeFormulaCell


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _cell(
    uid: str,
    src: str,
    parameters: tuple[str, ...],
    dtypes: tuple[str, ...],
    *,
    dtype: str = "float64",
    result_category_domain: str | None = None,
    parameter_category_domains: tuple[str | None, ...] = (),
) -> NativeFormulaCell:
    return NativeFormulaCell(
        uid=uid,
        source_name=uid,
        source_fullname=f"Test.Space.{uid}",
        dtype=dtype,
        parameters=parameters,
        parameter_dtypes=dtypes,
        function=_fn(src),
        reduction=None,
        result_category_domain=result_category_domain,
        parameter_category_domains=parameter_category_domains,
    )


def _row(
    cell: NativeFormulaCell,
    cls: str,
    deps: tuple[str, ...],
    *,
    bridge: bool,
    value_abi: str,
    blockers: tuple[tuple[str, str], ...] = (),
) -> ExecutionClassCell:
    return ExecutionClassCell(
        uid=cell.uid,
        source_name=cell.source_name,
        source_fullname=cell.source_fullname,
        execution_class=cls,
        dependencies=deps,
        direct_blockers=blockers,
        classification_reasons=("test",),
        native_scalar_abi=bridge and value_abi == NATIVE_SCALAR_VALUE_ABI,
        value_abi=value_abi,
        bridgeable_python_fallback=bridge,
        parameter_dtypes=cell.parameter_dtypes,
        result_dtype=cell.dtype,
        memo=NativeMemoContract(
            eligible=bridge,
            kind="scalar" if not cell.parameters and bridge else "dense" if bridge else "none",
            physicalization_ready=bridge,
            reason="test",
            value_abi=value_abi,
        ),
    )


def _plan(*, native_calls_helper: bool = False, categorical_leaf: bool = False):
    leaf = _cell(
        "leaf",
        "def leaf(t):\n    return t * 0.25\n",
        ("t",),
        ("int64",),
        result_category_domain="grade" if categorical_leaf else None,
    )
    helper = _cell(
        "helper",
        "def helper(t, grade):\n"
        "    if t <= 0:\n"
        "        return leaf(0)\n"
        "    return helper(t - 1, grade) + leaf(t)\n",
        ("t", "grade"),
        ("int64", "object"),
    )
    outer = _cell(
        "outer",
        "def outer(t):\n    return helper(t, 'A') + leaf(t)\n",
        ("t",),
        ("int64",),
    )
    cells = {x.uid: x for x in (leaf, helper, outer)}

    outer_cls = N1 if native_calls_helper else P
    rows = {
        "leaf": _row(leaf, N0, (), bridge=True, value_abi=NATIVE_SCALAR_VALUE_ABI),
        "helper": _row(
            helper,
            P,
            ("helper", "leaf"),
            bridge=False,
            value_abi=UNSUPPORTED_VALUE_ABI,
            blockers=(("parameter_abi", "object key"),),
        ),
        "outer": _row(
            outer,
            outer_cls,
            ("helper", "leaf"),
            bridge=True,
            value_abi=NATIVE_SCALAR_VALUE_ABI,
            blockers=(()) if native_calls_helper else (("python_semantics", "detached body"),),
        ),
    }
    if native_calls_helper:
        rows["outer"] = _row(
            outer,
            N1,
            ("helper", "leaf"),
            bridge=True,
            value_abi=NATIVE_SCALAR_VALUE_ABI,
        )
    classes = ExecutionClassPlan(
        schema="test",
        output_uid="outer",
        cells=rows,
        python_backed_misses={},
        fallback_context=FallbackPolicyContextContract(),
    )
    return SimpleNamespace(
        execution_class_plan=classes,
        formula_source_functions={uid: cell.function for uid, cell in cells.items()},
        source_transparent_cell_catalog=cells,
        parameter_ranges={
            "leaf": ((0, 8),),
            "outer": ((0, 8),),
            "helper": ((0, 8), None),
        },
        module_info=None,
        module_name="",
    ), cells


def test_part2a_forms_python_key_helper_closure_and_keeps_it_outside_native_closure():
    plan, cells = _plan()
    helpers, domains, refs = _python_key_helper_candidates(plan, cells)
    assert tuple(helpers) == ("helper",)
    assert domains["helper"] == ()
    assert refs["helper"] == ()

    outer = _core_cache_cython_p_candidate(
        plan, "outer", cells, python_key_catalog=helpers
    )
    assert outer is not None
    transformed, _domains, _refs = outer
    assert "helper(t, 'A')" in ast.unparse(transformed.function)

    (
        recursive,
        gil_backed,
        unsupported,
        core_cells,
        python_key_cells,
        _domains,
        _refs,
    ) = _mixed_recursive_uids(plan, cells)
    assert set(recursive) == {"outer", "leaf"}
    assert set(gil_backed) == {"outer"}
    assert unsupported == ()
    assert set(core_cells) == {"outer"}
    assert set(python_key_cells) == {"helper"}


def test_part2a_rejects_direct_native_to_python_key_helper_edge():
    plan, cells = _plan(native_calls_helper=True)
    with pytest.raises(MixedRecursiveNativeError, match="direct native N1 -> Python-key helper"):
        _mixed_recursive_uids(plan, cells)


def test_part2b_admits_category_boundary_inside_python_key_helper():
    plan, cells = _plan(categorical_leaf=True)
    helpers, domains, _refs = _python_key_helper_candidates(plan, cells)
    assert tuple(helpers) == ("helper",)
    assert domains["helper"] == ("grade",)


def test_detached_python_local_binding_discovery_includes_imports():
    import ast
    from modelx_graph.mixed_recursive_native import _assigned_names

    fn = ast.parse(
        "def f(x):\n"
        "    import math\n"
        "    import numpy.linalg as la\n"
        "    from statistics import mean as avg\n"
        "    return math.exp(x) + la.norm([x]) + avg([x])\n"
    ).body[0]
    assert isinstance(fn, ast.FunctionDef)
    names = _assigned_names(fn)
    assert {"x", "math", "la", "avg"} <= names
