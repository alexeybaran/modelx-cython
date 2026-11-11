from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import numpy as np
import pyximport

from modelx_graph.dense_recursive_native import build_dense_recursive_program
from modelx_graph.execution_classes import (
    N0,
    P,
    NATIVE_SCALAR_VALUE_ABI,
    UNSUPPORTED_VALUE_ABI,
    ExecutionClassCell,
    ExecutionClassPlan,
    FallbackPolicyContextContract,
    NativeMemoContract,
)
from modelx_graph.mixed_recursive_native import (
    FallbackTargetPlan,
    MixedRecursiveCythonGenerator,
    MixedRecursiveProgram,
)
from modelx_graph.native_formula import NativeFormulaCell, NativeFormulaProgram
from modelx_graph.object_memo_native import build_object_memo_plan
from modelx_graph.python_key_memo_native import build_python_key_memo_plan
from modelx_graph.source_preserving_native import build_source_preserving_compile_plan


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _cell(
    uid: str,
    src: str,
    parameters: tuple[str, ...] = (),
    dtypes: tuple[str, ...] = (),
) -> NativeFormulaCell:
    return NativeFormulaCell(
        uid=uid,
        source_name=uid,
        source_fullname=f"Test.Space.{uid}",
        dtype="float64",
        parameters=parameters,
        parameter_dtypes=dtypes,
        function=_fn(src),
        reduction=None,
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
            kind="scalar" if bridge and not cell.parameters else "dense" if bridge else "none",
            physicalization_ready=bridge,
            reason="test",
            value_abi=value_abi,
        ),
    )


def test_part2a_generated_module_executes_recursive_python_key_helper(tmp_path: Path):
    leaf = _cell(
        "leaf", "def leaf(n):\n    return n * 0.25\n", ("n",), ("int64",)
    )
    outer = _cell(
        "outer", "def outer():\n    return helper(5, 'A') + leaf(5)\n"
    )
    helper = _cell(
        "helper",
        "def helper(n, tag):\n"
        "    if n <= 0:\n"
        "        return leaf(0)\n"
        "    return helper(n - 1, tag) + leaf(n)\n",
        ("n", "tag"),
        ("int64", "object"),
    )
    outer_stub = _cell("outer", "def outer():\n    return 0.0\n")

    formula = NativeFormulaProgram(
        schema="test",
        uid="test",
        output_uid="outer",
        cells={"outer": outer_stub, "leaf": leaf},
        input_order=(),
        inputs=(),
    )
    compile_plan = build_source_preserving_compile_plan(formula, input_bounds={})
    assert compile_plan.supported
    dense = build_dense_recursive_program(
        compile_plan.formula_program,
        parameter_ranges={"outer": (), "leaf": ((0, 10),)},
    )
    assert dense.supported

    rows = {
        "outer": _row(
            outer,
            P,
            ("helper", "leaf"),
            bridge=True,
            value_abi=NATIVE_SCALAR_VALUE_ABI,
            blockers=(("python_semantics", "detached body"),),
        ),
        "leaf": _row(
            leaf, N0, (), bridge=True, value_abi=NATIVE_SCALAR_VALUE_ABI
        ),
        "helper": _row(
            helper,
            P,
            ("helper", "leaf"),
            bridge=False,
            value_abi=UNSUPPORTED_VALUE_ABI,
            blockers=(("parameter_abi", "object key"),),
        ),
    }
    execution = ExecutionClassPlan(
        schema="test",
        output_uid="outer",
        cells=rows,
        python_backed_misses={},
        fallback_context=FallbackPolicyContextContract(),
    )
    object_memo = build_object_memo_plan({}, parameter_ranges={})
    python_key_memo = build_python_key_memo_plan(
        {"helper": helper}, selected_uids=("helper",)
    )
    program = MixedRecursiveProgram(
        schema="test.part2a",
        formula_program=compile_plan.formula_program,
        compile_plan=compile_plan,
        dense_program=dense,
        execution_class_plan=execution,
        object_memo_plan=object_memo,
        python_key_memo_plan=python_key_memo,
        fallback_target_plan=FallbackTargetPlan("test", "Test.Space", {}),
        python_backed_uids=("outer",),
        object_backed_uids=(),
        ranked_backed_uids=(),
        gil_backed_uids=("outer",),
        modelx_fallback_uids=(),
        gil_backed_cells={"outer": outer},
        python_key_cells={"helper": helper},
        python_key_backed_uids=("helper",),
        supported=True,
        core_cache_cython_p_uids=("outer",),
    )

    module_name = "_python_key_helper_codegen_part2a"
    source = MixedRecursiveCythonGenerator(program).source(module_name)
    assert "cdef object pycalc_helper" in source
    assert "return r_helper(__mx_policy_i, n - 1, tag)" in source
    source += (
        "\ncpdef object probe():\n"
        "    global _memo_error\n"
        "    _memo_error = 0\n"
        "    return r_outer(0)\n"
    )
    pyx_path = tmp_path / f"{module_name}.pyx"
    pyx_path.write_text(source)

    pyximport.install(
        language_level=3,
        build_dir=str(tmp_path / "build"),
        inplace=False,
        setup_args={"include_dirs": [np.get_include()]},
    )
    sys.path.insert(0, str(tmp_path))
    try:
        sys.modules.pop(module_name, None)
        mod = importlib.import_module(module_name)
    finally:
        sys.path.remove(str(tmp_path))

    assert mod.probe() == 5.0
    stats = mod.python_key_memo_stats()["helper"]
    assert stats["misses"] == 6
    assert stats["entries"] == 6
    assert stats["active"] == 0
