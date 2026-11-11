from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pyximport

from modelx_graph.native_formula import NativeFormulaCell
from modelx_graph.python_key_memo_native import (
    PythonKeyMemoCythonEmitter,
    PythonKeyMemoNativeError,
    build_python_key_memo_plan,
)


def _cell(uid: str, parameters: tuple[str, ...], dtypes: tuple[str, ...]) -> NativeFormulaCell:
    args = ", ".join(parameters)
    fn = ast.parse(f"def {uid}({args}):\n    return None\n").body[0]
    assert isinstance(fn, ast.FunctionDef)
    return NativeFormulaCell(
        uid=uid,
        source_name=uid,
        source_fullname=f"Test.{uid}",
        dtype="object",
        parameters=parameters,
        parameter_dtypes=dtypes,
        function=fn,
        reduction=None,
    )


def _cells() -> dict[str, NativeFormulaCell]:
    return {
        "zero": _cell("zero", (), ()),
        "fib": _cell("fib", ("n",), ("object",)),
        "echo": _cell("echo", ("x",), ("object",)),
        "pair": _cell("pair", ("sex", "smoker", "column"), ("str", "str", "str")),
        "cycle": _cell("cycle", ("x",), ("object",)),
    }


def test_plan_is_explicit_and_preserves_python_key_shapes():
    cells = _cells()
    plan = build_python_key_memo_plan(
        cells,
        selected_uids=("zero", "echo", "pair"),
    )

    assert plan.cell("zero").key_shape == "scalar_sentinel"
    assert plan.cell("echo").key_shape == "direct"
    assert plan.cell("pair").key_shape == "tuple"
    assert plan.cell("pair").parameter_dtypes == ("str", "str", "str")
    assert plan.manifest()["storage"]["live_modelx_dependency"] is False


def test_plan_rejects_unknown_or_non_formula_cells():
    cells = _cells()
    try:
        build_python_key_memo_plan(cells, selected_uids=("missing",))
    except PythonKeyMemoNativeError as exc:
        assert "missing from catalog" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown uid must fail closed")

    helper = cells["echo"]
    cells["echo"] = NativeFormulaCell(
        **{**helper.__dict__, "execution_role": "materialized_helper"}
    )
    try:
        build_python_key_memo_plan(cells, selected_uids=("echo",))
    except PythonKeyMemoNativeError as exc:
        assert "requires a formula Cell" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("materialized helper must fail closed")


def _compile_probe(tmp_path: Path):
    cells = _cells()
    selected = ("zero", "fib", "echo", "pair", "cycle")
    plan = build_python_key_memo_plan(cells, selected_uids=selected)
    emitter = PythonKeyMemoCythonEmitter(plan, cells)

    source = "\n".join(
        [
            "# cython: language_level=3",
            "cdef unsigned int _epoch = 1",
            "cdef long long _memo_error = 0",
            emitter.declaration_source(),
            "cdef object pycalc_zero(Py_ssize_t policy_i):",
            "    return ('zero', policy_i)",
            "",
            "cdef object pycalc_fib(Py_ssize_t policy_i, object n):",
            "    if n <= 1:",
            "        return n",
            "    return r_fib(policy_i, n - 1) + r_fib(policy_i, n - 2)",
            "",
            "cdef object pycalc_echo(Py_ssize_t policy_i, object x):",
            "    return (type(x).__name__, x)",
            "",
            "cdef object pycalc_pair(Py_ssize_t policy_i, object sex, object smoker, object column):",
            "    return (sex, smoker, column, policy_i)",
            "",
            "cdef object pycalc_cycle(Py_ssize_t policy_i, object x):",
            "    return r_cycle(policy_i, x)",
            "",
            emitter.wrapper_source(),
            emitter.clear_source(),
            emitter.stats_source(),
            emitter.reset_stats_source(),
            "cpdef object eval_zero(Py_ssize_t policy_i=0):",
            "    global _memo_error",
            "    _memo_error = 0",
            "    return r_zero(policy_i)",
            "",
            "cpdef object eval_fib(object n):",
            "    global _memo_error",
            "    _memo_error = 0",
            "    return r_fib(0, n)",
            "",
            "cpdef object eval_echo(object x):",
            "    global _memo_error",
            "    _memo_error = 0",
            "    return r_echo(0, x)",
            "",
            "cpdef object eval_pair(object sex, object smoker, object column):",
            "    global _memo_error",
            "    _memo_error = 0",
            "    return r_pair(7, sex, smoker, column)",
            "",
            "cpdef object eval_cycle(object x):",
            "    global _memo_error",
            "    _memo_error = 0",
            "    return r_cycle(0, x)",
            "",
            "cpdef int memo_error():",
            "    return <int>_memo_error",
            "",
            "cpdef unsigned int epoch():",
            "    return _epoch",
            "",
            "cpdef advance_epoch():",
            "    global _epoch, _memo_error",
            "    _epoch += 1",
            "    _memo_error = 0",
            "",
            "cpdef clear_memos():",
            "    global _memo_error",
            "    _clear_python_key_memos()",
            "    _memo_error = 0",
            "",
        ]
    )

    mod_name = "_python_key_memo_probe"
    pyx_path = tmp_path / f"{mod_name}.pyx"
    pyx_path.write_text(source)
    build_dir = tmp_path / "build"
    pyximport.install(language_level=3, build_dir=str(build_dir), inplace=False)
    sys.path.insert(0, str(tmp_path))
    try:
        sys.modules.pop(mod_name, None)
        return importlib.import_module(mod_name)
    finally:
        sys.path.remove(str(tmp_path))


def test_compiled_python_key_runtime_recursion_epoch_and_exact_dict_semantics(tmp_path):
    mod = _compile_probe(tmp_path)

    # Recursive calls with different keys are legal and memoized.
    assert mod.eval_fib(10) == 55
    stats = mod.python_key_memo_stats()
    assert stats["fib"]["misses"] == 11
    assert stats["fib"]["entries"] == 11
    first_hits = stats["fib"]["hits"]
    assert first_hits > 0

    # Repeating the same Cell call in one policy epoch is a cache hit.
    assert mod.eval_fib(10) == 55
    stats = mod.python_key_memo_stats()
    assert stats["fib"]["misses"] == 11
    assert stats["fib"]["hits"] == first_hits + 1

    # Python dict semantics are exact: 65 and 65.0 are the same key.
    assert mod.eval_echo(65) == ("int", 65)
    assert mod.eval_echo(65.0) == ("int", 65)
    stats = mod.python_key_memo_stats()
    assert stats["echo"]["entries"] == 1
    assert stats["echo"]["misses"] == 1
    assert stats["echo"]["hits"] == 1

    # Multi-argument labels use their ordinary Python tuple key.
    expected = ("M", "NS", "i_ci", 7)
    assert mod.eval_pair("M", "NS", "i_ci") == expected
    assert mod.eval_pair("M", "NS", "i_ci") == expected
    stats = mod.python_key_memo_stats()
    assert stats["pair"]["entries"] == 1
    assert stats["pair"]["misses"] == 1
    assert stats["pair"]["hits"] == 1

    # Same-key recursive re-entry is detected instead of recursing forever.
    assert mod.eval_cycle("x") is None
    assert mod.memo_error() == 2
    stats = mod.python_key_memo_stats()
    assert stats["cycle"]["active"] == 0
    assert stats["cycle"]["entries"] == 0

    # A new policy epoch lazily invalidates only when the Cell is next touched.
    old_misses = stats["fib"]["misses"]
    mod.advance_epoch()
    assert mod.epoch() == 2
    assert mod.eval_fib(5) == 5
    stats = mod.python_key_memo_stats()
    assert stats["fib"]["epoch"] == 2
    assert stats["fib"]["entries"] == 6
    assert stats["fib"]["misses"] == old_misses + 6

    # Explicit clear supports model binding/reset and future epoch-wrap wiring.
    mod.clear_memos()
    stats = mod.python_key_memo_stats()
    assert all(row["entries"] == 0 for row in stats.values())
    assert all(row["active"] == 0 for row in stats.values())
    assert all(row["epoch"] == 0 for row in stats.values())
