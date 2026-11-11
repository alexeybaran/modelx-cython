from __future__ import annotations

"""Compiler-owned memoization for detached Cells with Python-semantic keys.

This module is the first half of the detached-P helper architecture.  It adds a
runtime primitive only; planner selection is intentionally unchanged.

The existing native recursive ABI uses scalar/dense/native-hash storage and
requires a native key ABI.  Some legitimate modelx Cells instead take values
such as strings or general Python objects.  ``PythonKeyMemoCythonEmitter``
emits an alternative physical implementation of the same logical ``r_<uid>``
Cell-call ABI:

* keys use normal Python ``dict`` semantics;
* one Cell owns one compiler-created cache and active-key set;
* the cache is lazily invalidated when the surrounding module's ``_epoch``
  changes, so only the current policy epoch is retained;
* recursive calls with different keys are supported;
* re-entry on the same key is detected as a cycle and sets ``_memo_error = 2``;
* cache misses call a compiler-owned ``pycalc_<uid>`` function under the GIL;
* no live modelx Cell or Space object is required by the memo runtime.

Part 2 will decide which detached P Cells use this ABI and will rewrite their
owner-local Cell calls to these wrappers.  Keeping that wiring separate makes
this runtime independently testable and prevents census behaviour changing in
Part 1.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .native_formula import NativeFormulaCell


class PythonKeyMemoNativeError(RuntimeError):
    pass


@dataclass(frozen=True)
class PythonKeyMemoCell:
    """One Cell backed by exact Python dictionary key semantics."""

    uid: str
    arity: int
    parameters: tuple[str, ...]
    parameter_dtypes: tuple[str, ...]
    key_shape: str  # scalar_sentinel | direct | tuple
    cycle_guard: bool = True
    decision_reason: str = "python_semantic_key"

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "arity": self.arity,
            "parameters": list(self.parameters),
            "parameter_dtypes": list(self.parameter_dtypes),
            "kind": "python_dict",
            "key_shape": self.key_shape,
            "key_abi": "python_object",
            "value_abi": "python_object",
            "cycle_guard": self.cycle_guard,
            "decision_reason": self.decision_reason,
        }


@dataclass(frozen=True)
class PythonKeyMemoPlan:
    schema: str
    cells: tuple[PythonKeyMemoCell, ...]

    def cell(self, uid: str) -> PythonKeyMemoCell:
        for row in self.cells:
            if row.uid == uid:
                return row
        raise PythonKeyMemoNativeError(f"python-key memo cell {uid!r} not found")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cells": [row.manifest() for row in self.cells],
            "storage": {
                "keys": "python_dict_exact_semantics",
                "values": "python_object",
                "active_keys": "python_set_cycle_guard",
                "generation": "lazy_per_cell_epoch",
                "cache_lifetime": "current_policy_epoch",
                "live_modelx_dependency": False,
            },
        }


def _key_shape(arity: int) -> str:
    if arity == 0:
        return "scalar_sentinel"
    if arity == 1:
        return "direct"
    return "tuple"


def build_python_key_memo_plan(
    cells: Mapping[str, NativeFormulaCell],
    *,
    selected_uids: Sequence[str] | None = None,
) -> PythonKeyMemoPlan:
    """Build a Python-key memo plan without changing execution classification.

    ``selected_uids`` is explicit on purpose.  Part 1 provides the runtime
    primitive but does not decide which P Cells should be admitted to it.  Part
    2 will supply the detached-helper closure selected by the mixed planner.
    """

    selected = tuple(sorted(cells)) if selected_uids is None else tuple(selected_uids)
    if len(set(selected)) != len(selected):
        raise PythonKeyMemoNativeError("selected_uids contains duplicates")

    rows: list[PythonKeyMemoCell] = []
    for uid in selected:
        if uid not in cells:
            raise PythonKeyMemoNativeError(f"python-key memo cell {uid!r} missing from catalog")
        cell = cells[uid]
        if cell.execution_role != "formula":
            raise PythonKeyMemoNativeError(
                f"{uid}: python-key memo requires a formula Cell, got {cell.execution_role!r}"
            )
        if len(cell.parameters) != len(cell.parameter_dtypes):
            raise PythonKeyMemoNativeError(
                f"{uid}: parameter dtype count does not match arity"
            )
        rows.append(
            PythonKeyMemoCell(
                uid=uid,
                arity=cell.arity,
                parameters=tuple(cell.parameters),
                parameter_dtypes=tuple(cell.parameter_dtypes),
                key_shape=_key_shape(cell.arity),
            )
        )

    return PythonKeyMemoPlan(
        "modelx_graph.python_key_memo_native.v1_exact_python_keys",
        tuple(rows),
    )


class PythonKeyMemoCythonEmitter:
    """Emit GIL-held ``r_<uid>`` wrappers backed by Python dicts.

    The surrounding Cython module must provide::

        cdef unsigned int _epoch
        cdef long long _memo_error

    Every planned Cell must provide::

        cdef object pycalc_<uid>(Py_ssize_t policy_i, object arg0, ...)

    All parameters are deliberately accepted as ``object`` even when tracing
    observed a narrower type.  This path exists to preserve Python semantics;
    native parameter conversion is the responsibility of the native ABI at an
    explicit boundary in Part 2.
    """

    def __init__(self, plan: PythonKeyMemoPlan, cells: Mapping[str, NativeFormulaCell]):
        self.plan = plan
        self.cells = dict(cells)
        missing = sorted(row.uid for row in plan.cells if row.uid not in self.cells)
        if missing:
            raise PythonKeyMemoNativeError(
                "python-key memo cells missing from catalog: " + ",".join(missing)
            )

    @staticmethod
    def _safe(uid: str) -> str:
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in uid)

    def _cache(self, uid: str) -> str:
        return f"_pkm_{self._safe(uid)}_cache"

    def _active(self, uid: str) -> str:
        return f"_pkm_{self._safe(uid)}_active"

    def _epoch_name(self, uid: str) -> str:
        return f"_pkm_{self._safe(uid)}_epoch"

    def _hits(self, uid: str) -> str:
        return f"_pkm_{self._safe(uid)}_hits"

    def _misses(self, uid: str) -> str:
        return f"_pkm_{self._safe(uid)}_misses"

    def declaration_source(self) -> str:
        lines: list[str] = []
        for row in self.plan.cells:
            lines += [
                f"cdef dict {self._cache(row.uid)} = {{}}",
                f"cdef set {self._active(row.uid)} = set()",
                f"cdef unsigned int {self._epoch_name(row.uid)} = 0",
                f"cdef unsigned long long {self._hits(row.uid)} = 0",
                f"cdef unsigned long long {self._misses(row.uid)} = 0",
            ]
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _signature(cell: NativeFormulaCell, prefix: str) -> str:
        args = ["Py_ssize_t policy_i"]
        args.extend(f"object {name}" for name in cell.parameters)
        return f"cdef object {prefix}_{cell.uid}({', '.join(args)})"

    @staticmethod
    def _key_expr(row: PythonKeyMemoCell) -> str:
        if row.arity == 0:
            # A private module-level sentinel is unnecessary: this dictionary
            # belongs to one Cell, so None is an exact and collision-free key.
            return "None"
        if row.arity == 1:
            return row.parameters[0]
        if row.arity == 2:
            return f"({row.parameters[0]}, {row.parameters[1]})"
        return "(" + ", ".join(row.parameters) + ",)"

    def _wrapper(self, cell: NativeFormulaCell, row: PythonKeyMemoCell) -> list[str]:
        cache = self._cache(row.uid)
        active = self._active(row.uid)
        cell_epoch = self._epoch_name(row.uid)
        hits = self._hits(row.uid)
        misses = self._misses(row.uid)
        args = ", ".join(cell.parameters)
        suffix = "" if not args else ", " + args
        key_expr = self._key_expr(row)

        return [
            self._signature(cell, "r") + ":",
            "    cdef object key",
            "    cdef object val",
            f"    global _memo_error, {cell_epoch}, {hits}, {misses}",
            f"    if {cell_epoch} != _epoch:",
            f"        {cache}.clear()",
            f"        {active}.clear()",
            f"        {cell_epoch} = _epoch",
            f"    key = {key_expr}",
            f"    if key in {cache}:",
            f"        {hits} += 1",
            f"        return {cache}[key]",
            f"    if key in {active}:",
            "        _memo_error = 2",
            "        return None",
            f"    {misses} += 1",
            f"    {active}.add(key)",
            "    try:",
            f"        val = pycalc_{cell.uid}(policy_i{suffix})",
            "    except BaseException:",
            f"        {active}.discard(key)",
            "        raise",
            f"    {active}.discard(key)",
            "    if _memo_error != 0:",
            "        return val",
            f"    {cache}[key] = val",
            "    return val",
            "",
        ]

    def wrapper_source(self) -> str:
        lines: list[str] = []
        for row in self.plan.cells:
            lines += self._wrapper(self.cells[row.uid], row)
        return "\n".join(lines) + ("\n" if lines else "")

    def clear_source(self, function_name: str = "_clear_python_key_memos") -> str:
        lines = [f"cdef void {function_name}():"]
        globals_: list[str] = []
        for row in self.plan.cells:
            globals_.append(self._epoch_name(row.uid))
        if globals_:
            lines.append("    global " + ", ".join(globals_))
        if not self.plan.cells:
            lines.append("    pass")
        for row in self.plan.cells:
            lines += [
                f"    {self._cache(row.uid)}.clear()",
                f"    {self._active(row.uid)}.clear()",
                f"    {self._epoch_name(row.uid)} = 0",
            ]
        lines.append("")
        return "\n".join(lines)

    def stats_source(self, function_name: str = "python_key_memo_stats") -> str:
        lines = [f"cpdef dict {function_name}():", "    return {"]
        for row in self.plan.cells:
            lines.append(
                f"        {row.uid!r}: {{'kind': 'python_dict', "
                f"'hits': int({self._hits(row.uid)}), "
                f"'misses': int({self._misses(row.uid)}), "
                f"'entries': len({self._cache(row.uid)}), "
                f"'active': len({self._active(row.uid)}), "
                f"'epoch': int({self._epoch_name(row.uid)}), "
                "'key_abi': 'python_object', 'value_abi': 'python_object'},"
            )
        lines += ["    }", ""]
        return "\n".join(lines)

    def reset_stats_source(self, function_name: str = "reset_python_key_memo_stats") -> str:
        names: list[str] = []
        for row in self.plan.cells:
            names.extend([self._hits(row.uid), self._misses(row.uid)])
        lines = [f"cpdef {function_name}():"]
        if not names:
            lines += ["    return None", ""]
            return "\n".join(lines)
        lines.append("    global " + ", ".join(names))
        for name in names:
            lines.append(f"    {name} = 0")
        lines += ["    return None", ""]
        return "\n".join(lines)


__all__ = [
    "PythonKeyMemoNativeError",
    "PythonKeyMemoCell",
    "PythonKeyMemoPlan",
    "build_python_key_memo_plan",
    "PythonKeyMemoCythonEmitter",
]
