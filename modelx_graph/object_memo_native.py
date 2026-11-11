from __future__ import annotations

"""Compiler-owned native-key memoization for Python object results.

This module is deliberately value-representation neutral.  It reuses the same
memo *key* policies as the scalar native cache (scalar, bounded dense, exact
native hash), but each value slot owns a ``PyObject*``.  NumPy arrays, pandas
objects and arbitrary detached-P results remain ordinary Python objects; their
contents are never converted merely for caching.

The emitter is shared by scalar and rank-1 mixed programs. Execution-class
planning decides whether a cached object represents an opaque Python value or
a ranked array while this module remains neutral to the object's contents.
"""

import ast
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .dense_recursive_native import _dense_strides, _native_hash_capacity, _native_hash_seed
from .native_formula import NativeFormulaCell


class ObjectMemoNativeError(RuntimeError):
    pass


_NATIVE_KEY_DTYPES = frozenset({"bool", "int64", "float64"})


@dataclass(frozen=True)
class ObjectMemoCell:
    uid: str
    arity: int
    kind: str  # scalar | dense | hash
    bounds: tuple[tuple[int, int], ...] = ()
    strides: tuple[int, ...] = ()
    size: int = 0
    hash_capacity: int = 0
    cycle_guard: bool = True
    decision_reason: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "arity": self.arity,
            "kind": self.kind,
            "value_abi": "owned_pyobject",
            "bounds": [list(x) for x in self.bounds],
            "strides": list(self.strides),
            "size": self.size,
            "hash_capacity": self.hash_capacity,
            "cycle_guard": self.cycle_guard,
            "decision_reason": self.decision_reason,
        }


@dataclass(frozen=True)
class ObjectMemoPlan:
    schema: str
    cells: tuple[ObjectMemoCell, ...]

    def cell(self, uid: str) -> ObjectMemoCell:
        for row in self.cells:
            if row.uid == uid:
                return row
        raise ObjectMemoNativeError(f"object memo cell {uid!r} not found")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "value_abi": "owned_pyobject",
            "cells": [row.manifest() for row in self.cells],
            "storage": {
                "keys": "native_typed_c_slots",
                "values": "owned_PyObject_pointer_slots",
                "generation": "uint32_epoch",
                "cycle_state": "busy_bit_or_hash_busy_byte",
                "content_conversion": False,
            },
        }


def _call_uids(cell: NativeFormulaCell, known: set[str]) -> set[str]:
    roots: list[ast.AST] = []
    if cell.function is not None:
        roots.append(cell.function)
    if cell.ranked_function is not None:
        roots.append(cell.ranked_function)
    if cell.reduction is not None:
        r = cell.reduction
        roots.extend([r.init, *r.range_args, r.body_expr, *r.filters])
    out: set[str] = set()
    for root in roots:
        for node in ast.walk(root):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in known:
                    out.add(node.func.id)
    return out


def _recursive_uids(cells: Mapping[str, NativeFormulaCell]) -> set[str]:
    known = set(cells)
    graph = {uid: _call_uids(cell, known) for uid, cell in cells.items()}
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    recursive: set[str] = set()

    def visit(uid: str) -> None:
        nonlocal index
        indices[uid] = low[uid] = index
        index += 1
        stack.append(uid)
        on_stack.add(uid)
        for dep in graph[uid]:
            if dep not in indices:
                visit(dep)
                low[uid] = min(low[uid], low[dep])
            elif dep in on_stack:
                low[uid] = min(low[uid], indices[dep])
        if low[uid] != indices[uid]:
            return
        component: list[str] = []
        while stack:
            item = stack.pop()
            on_stack.remove(item)
            component.append(item)
            if item == uid:
                break
        if len(component) > 1:
            recursive.update(component)
        elif uid in graph[uid]:
            recursive.add(uid)

    for uid in sorted(graph):
        if uid not in indices:
            visit(uid)
    return recursive


def build_object_memo_plan(
    cells: Mapping[str, NativeFormulaCell],
    *,
    parameter_ranges: Mapping[str, Sequence[tuple[int, int] | None]] | None = None,
    native_hash_capacity_by_uid: Mapping[str, int] | None = None,
    max_dense_slots_per_cell: int = 1_000_000,
) -> ObjectMemoPlan:
    """Choose native key storage for object-valued detached Cells.

    Result dtype is intentionally ignored: object identity is preserved in
    ``PyObject*`` slots.  Only the key ABI participates in memo planning.
    Unbounded/non-integral keys require explicit complete-domain hash capacity;
    there is no Python-dict fallback in this production object-cache plan.
    """

    parameter_ranges = parameter_ranges or {}
    native_hash_capacity_by_uid = native_hash_capacity_by_uid or {}
    recursive = _recursive_uids(cells)
    rows: list[ObjectMemoCell] = []
    for uid in sorted(cells):
        cell = cells[uid]
        if cell.execution_role != "formula":
            continue
        if any(dtype not in _NATIVE_KEY_DTYPES for dtype in cell.parameter_dtypes):
            bad = [x for x in cell.parameter_dtypes if x not in _NATIVE_KEY_DTYPES]
            raise ObjectMemoNativeError(
                f"{uid}: object memo key ABI is not native-typed: {bad!r}"
            )
        if cell.arity == 0:
            rows.append(ObjectMemoCell(
                uid=uid, arity=0, kind="scalar", size=1,
                cycle_guard=uid in recursive,
                decision_reason="zero_arity_object_scalar",
            ))
            continue

        requested = tuple(parameter_ranges.get(uid, ()))
        if len(requested) != cell.arity:
            requested = tuple(None for _ in range(cell.arity))
        dense_bounds: list[tuple[int, int]] = []
        dense_safe = all(dtype in {"int64", "bool"} for dtype in cell.parameter_dtypes)
        if dense_safe:
            for dtype, item in zip(cell.parameter_dtypes, requested):
                if item is None and dtype == "bool":
                    item = (0, 1)
                if item is None:
                    dense_safe = False
                    break
                lo, hi = int(item[0]), int(item[1])
                if lo > hi:
                    raise ObjectMemoNativeError(f"{uid}: invalid parameter range {(lo, hi)!r}")
                dense_bounds.append((lo, hi))
        if dense_safe and len(dense_bounds) == cell.arity:
            strides, size = _dense_strides(dense_bounds)
            if size <= int(max_dense_slots_per_cell):
                rows.append(ObjectMemoCell(
                    uid=uid, arity=cell.arity, kind="dense",
                    bounds=tuple(dense_bounds), strides=strides, size=size,
                    cycle_guard=uid in recursive,
                    decision_reason="bounded_native_key_object_dense",
                ))
                continue

        requested_capacity = native_hash_capacity_by_uid.get(uid)
        if requested_capacity is None:
            raise ObjectMemoNativeError(
                f"{uid}: object memo requires complete-domain native hash capacity "
                "for non-dense key ABI"
            )
        rows.append(ObjectMemoCell(
            uid=uid, arity=cell.arity, kind="hash",
            bounds=tuple(dense_bounds), size=0,
            hash_capacity=_native_hash_capacity(int(requested_capacity)),
            cycle_guard=uid in recursive,
            decision_reason="native_exact_hash_object_value",
        ))
    return ObjectMemoPlan(
        "modelx_graph.object_memo_native.v1_owned_pyobject_native_keys",
        tuple(rows),
    )


class ObjectMemoCythonEmitter:
    """Emit Cython wrappers for native-key / PyObject-value memo slots.

    The surrounding module must provide ``cdef unsigned int _epoch`` and
    ``cdef long long _memo_error``.  Each wrapped Cell must provide a miss function
    named ``objcalc_<uid>(Py_ssize_t policy_i, ...)`` returning ``object``.
    """

    def __init__(self, plan: ObjectMemoPlan, cells: Mapping[str, NativeFormulaCell]):
        self.plan = plan
        self.cells = dict(cells)
        missing = sorted(row.uid for row in plan.cells if row.uid not in self.cells)
        if missing:
            raise ObjectMemoNativeError("object memo cells missing from catalog: " + ",".join(missing))

    @staticmethod
    def _safe(uid: str) -> str:
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in uid)

    @staticmethod
    def _ctype(dtype: str) -> str:
        if dtype == "float64":
            return "double"
        if dtype == "int64":
            return "long long"
        if dtype == "bool":
            return "bint"
        raise ObjectMemoNativeError(f"unsupported object memo key dtype {dtype!r}")

    @staticmethod
    def _hash_key_ctype(dtype: str) -> str:
        if dtype == "float64":
            return "uint64_t"
        if dtype == "int64":
            return "int64_t"
        if dtype == "bool":
            return "unsigned char"
        raise ObjectMemoNativeError(dtype)

    def _values(self, uid: str) -> str:
        return f"_om_{self._safe(uid)}_values"

    def _state(self, uid: str) -> str:
        return f"_om_{self._safe(uid)}_state"

    def _gen(self, uid: str) -> str:
        return f"_om_{self._safe(uid)}_gen"

    def _busy(self, uid: str) -> str:
        return f"_om_{self._safe(uid)}_busy"

    def _key(self, uid: str, j: int) -> str:
        return f"_om_{self._safe(uid)}_key{j}"

    def _hits(self, uid: str) -> str:
        return f"_om_{self._safe(uid)}_hits"

    def _misses(self, uid: str) -> str:
        return f"_om_{self._safe(uid)}_misses"

    def import_source(self) -> str:
        has_hash = any(row.kind == "hash" for row in self.plan.cells)
        lines = [
            "from cpython.object cimport PyObject",
            "from cpython.ref cimport Py_INCREF, Py_DECREF",
        ]
        if has_hash:
            lines += [
                "from libc.stdint cimport uint64_t, int64_t",
                "from libc.string cimport memcpy",
            ]
        return "\n".join(lines) + "\n"

    def helper_source(self) -> str:
        if not any(row.kind == "hash" for row in self.plan.cells):
            return ""
        return "\n".join([
            "cdef inline uint64_t __objmemo_mix64(uint64_t x) noexcept nogil:",
            "    x ^= x >> 30",
            "    x *= <uint64_t>0xbf58476d1ce4e5b9ULL",
            "    x ^= x >> 27",
            "    x *= <uint64_t>0x94d049bb133111ebULL",
            "    x ^= x >> 31",
            "    return x",
            "",
            "cdef inline uint64_t __objmemo_f64_bits(double x) noexcept nogil:",
            "    cdef uint64_t bits",
            "    if x == 0.0:",
            "        return <uint64_t>0",
            "    memcpy(&bits, &x, sizeof(double))",
            "    return bits",
            "",
        ])

    def declaration_source(self) -> str:
        lines: list[str] = []
        for row in self.plan.cells:
            cell = self.cells[row.uid]
            slots = 1 if row.kind == "scalar" else (row.size if row.kind == "dense" else row.hash_capacity)
            lines += [
                f"cdef PyObject* {self._values(row.uid)}[{slots}]",
                f"cdef unsigned long long {self._hits(row.uid)} = 0",
                f"cdef unsigned long long {self._misses(row.uid)} = 0",
            ]
            if row.kind in {"scalar", "dense"}:
                lines.append(f"cdef unsigned int {self._state(row.uid)}[{slots}]")
            else:
                lines += [
                    f"cdef unsigned int {self._gen(row.uid)}[{slots}]",
                    f"cdef unsigned char {self._busy(row.uid)}[{slots}]",
                ]
                for j, dtype in enumerate(cell.parameter_dtypes):
                    lines.append(
                        f"cdef {self._hash_key_ctype(dtype)} {self._key(row.uid, j)}[{slots}]"
                    )
        return "\n".join(lines) + ("\n" if lines else "")

    def _signature(self, cell: NativeFormulaCell, prefix: str) -> str:
        args = ["Py_ssize_t policy_i"]
        args.extend(
            f"{self._ctype(dtype)} {name}"
            for name, dtype in zip(cell.parameters, cell.parameter_dtypes)
        )
        return f"cdef object {prefix}_{cell.uid}({', '.join(args)})"

    def _dense_wrapper(self, cell: NativeFormulaCell, row: ObjectMemoCell) -> list[str]:
        values = self._values(row.uid)
        state = self._state(row.uid)
        args = ", ".join(cell.parameters)
        suffix = "" if not args else ", " + args
        lines = [
            self._signature(cell, "r") + ":",
            "    cdef Py_ssize_t idx = 0",
            "    cdef object val",
            "    global _memo_error, " + self._hits(row.uid) + ", " + self._misses(row.uid),
        ]
        if row.kind == "dense":
            checks: list[str] = []
            terms: list[str] = []
            for name, (lo, hi), stride in zip(cell.parameters, row.bounds, row.strides):
                checks.extend((f"{name} < {lo}", f"{name} > {hi}"))
                term = f"({name} - ({lo}))"
                if stride != 1:
                    term = f"({term}) * {stride}"
                terms.append(term)
            lines += [
                "    if " + " or ".join(checks) + ":",
                "        _memo_error = 4",
                "        return None",
                "    idx = " + " + ".join(terms),
            ]
        lines += [
            f"    if ({state}[idx] & <unsigned int>0x7fffffff) == _epoch:",
            f"        if ({state}[idx] & <unsigned int>0x80000000) != 0:",
            "            _memo_error = 2",
            "            return None",
            f"        if {values}[idx] == NULL:",
            "            _memo_error = 6",
            "            return None",
            f"        {self._hits(row.uid)} += 1",
            f"        return <object>{values}[idx]",
            f"    {self._misses(row.uid)} += 1",
            f"    {state}[idx] = _epoch | <unsigned int>0x80000000",
            "    try:",
            f"        val = objcalc_{cell.uid}(policy_i{suffix})",
            "    except BaseException:",
            f"        {state}[idx] = 0",
            "        raise",
            "    if _memo_error != 0:",
            f"        {state}[idx] = 0",
            "        return val",
            f"    if {values}[idx] != NULL:",
            f"        Py_DECREF(<object>{values}[idx])",
            "    Py_INCREF(val)",
            f"    {values}[idx] = <PyObject*>val",
            f"    {state}[idx] = _epoch",
            "    return val",
            "",
        ]
        return lines

    def _hash_wrapper(self, cell: NativeFormulaCell, row: ObjectMemoCell) -> list[str]:
        cap = int(row.hash_capacity)
        if cap <= 0 or cap & (cap - 1):
            raise ObjectMemoNativeError(f"{row.uid}: hash capacity must be a power of two")
        mask = cap - 1
        values = self._values(row.uid)
        gen = self._gen(row.uid)
        busy = self._busy(row.uid)
        args = ", ".join(cell.parameters)
        suffix = "" if not args else ", " + args
        lines = [
            self._signature(cell, "r") + ":",
            f"    cdef uint64_t h = <uint64_t>{_native_hash_seed(row.uid)}ULL",
            "    cdef Py_ssize_t pos",
            "    cdef Py_ssize_t probe",
            "    cdef object val",
            "    cdef bint match",
            "    global _memo_error, " + self._hits(row.uid) + ", " + self._misses(row.uid),
        ]
        for j, dtype in enumerate(cell.parameter_dtypes):
            lines.append(f"    cdef {self._hash_key_ctype(dtype)} key{j}")
        for j, (name, dtype) in enumerate(zip(cell.parameters, cell.parameter_dtypes)):
            if dtype == "float64":
                lines += [
                    f"    if {name} != {name}:",
                    "        _memo_error = 5",
                    "        return None",
                    f"    key{j} = __objmemo_f64_bits({name})",
                    f"    h = __objmemo_mix64(h ^ __objmemo_mix64(key{j} + <uint64_t>{j + 1}ULL))",
                ]
            elif dtype == "int64":
                lines += [
                    f"    key{j} = <int64_t>{name}",
                    f"    h = __objmemo_mix64(h ^ __objmemo_mix64(<uint64_t>key{j} + <uint64_t>{j + 1}ULL))",
                ]
            elif dtype == "bool":
                lines += [
                    f"    key{j} = <unsigned char>{name}",
                    f"    h = __objmemo_mix64(h ^ __objmemo_mix64(<uint64_t>key{j} + <uint64_t>{j + 1}ULL))",
                ]
            else:
                raise ObjectMemoNativeError(dtype)
        lines += [
            f"    pos = <Py_ssize_t>(h & <uint64_t>{mask})",
            f"    for probe in range({cap}):",
            f"        if {gen}[pos] != _epoch:",
            f"            {gen}[pos] = _epoch",
            f"            {busy}[pos] = 1",
        ]
        for j in range(cell.arity):
            lines.append(f"            {self._key(row.uid, j)}[pos] = key{j}")
        lines += [
            f"            {self._misses(row.uid)} += 1",
            "            try:",
            f"                val = objcalc_{cell.uid}(policy_i{suffix})",
            "            except BaseException:",
            f"                {gen}[pos] = 0",
            f"                {busy}[pos] = 0",
            "                raise",
            "            if _memo_error != 0:",
            f"                {gen}[pos] = 0",
            f"                {busy}[pos] = 0",
            "                return val",
            f"            if {values}[pos] != NULL:",
            f"                Py_DECREF(<object>{values}[pos])",
            "            Py_INCREF(val)",
            f"            {values}[pos] = <PyObject*>val",
            f"            {busy}[pos] = 0",
            "            return val",
            "        match = True",
        ]
        for j in range(cell.arity):
            lines.append(f"        match = match and ({self._key(row.uid, j)}[pos] == key{j})")
        lines += [
            "        if match:",
            f"            if {busy}[pos]:",
            "                _memo_error = 2",
            "                return None",
            f"            if {values}[pos] == NULL:",
            "                _memo_error = 6",
            "                return None",
            f"            {self._hits(row.uid)} += 1",
            f"            return <object>{values}[pos]",
            f"        pos = (pos + 1) & {mask}",
            "    _memo_error = 1",
            "    return None",
            "",
        ]
        return lines

    def wrapper_source(self) -> str:
        lines: list[str] = []
        for row in self.plan.cells:
            cell = self.cells[row.uid]
            if row.kind in {"scalar", "dense"}:
                lines += self._dense_wrapper(cell, row)
            elif row.kind == "hash":
                lines += self._hash_wrapper(cell, row)
            else:
                raise ObjectMemoNativeError(f"unsupported object memo kind {row.kind!r}")
        return "\n".join(lines) + ("\n" if lines else "")

    def clear_source(self, function_name: str = "_clear_object_memos") -> str:
        lines = [f"cdef void {function_name}():", "    cdef Py_ssize_t i"]
        if not self.plan.cells:
            lines.append("    pass")
        for row in self.plan.cells:
            slots = 1 if row.kind == "scalar" else (row.size if row.kind == "dense" else row.hash_capacity)
            values = self._values(row.uid)
            lines += [
                f"    for i in range({slots}):",
                f"        if {values}[i] != NULL:",
                f"            Py_DECREF(<object>{values}[i])",
                f"            {values}[i] = NULL",
            ]
            if row.kind in {"scalar", "dense"}:
                lines.append(f"        {self._state(row.uid)}[i] = 0")
            else:
                lines += [
                    f"        {self._gen(row.uid)}[i] = 0",
                    f"        {self._busy(row.uid)}[i] = 0",
                ]
        lines.append("")
        return "\n".join(lines)

    def generation_clear_lines(self, indent: str = "    ") -> list[str]:
        """Reset object-cache generations without touching owned values.

        This is used only when the shared 31-bit epoch wraps inside ``nogil``.
        The value pointers remain owned and are decref'd when their slots are
        next replaced or when the ordinary GIL-held clear path runs.
        """
        lines: list[str] = []
        for row in self.plan.cells:
            slots = 1 if row.kind == "scalar" else (
                row.size if row.kind == "dense" else row.hash_capacity
            )
            metadata = self._state(row.uid) if row.kind in {"scalar", "dense"} else self._gen(row.uid)
            lines.append(
                f"{indent}memset(&{metadata}[0], 0, {slots} * sizeof(unsigned int))"
            )
            if row.kind == "hash":
                lines.append(
                    f"{indent}memset(&{self._busy(row.uid)}[0], 0, {slots} * sizeof(unsigned char))"
                )
        return lines

    def stats_source(self, function_name: str = "object_memo_stats") -> str:
        lines = [f"cpdef dict {function_name}():", "    return {"]
        for row in self.plan.cells:
            lines.append(
                f"        {row.uid!r}: {{'kind': {row.kind!r}, "
                f"'hits': int({self._hits(row.uid)}), 'misses': int({self._misses(row.uid)}), "
                "'value_abi': 'owned_pyobject'},"
            )
        lines += ["    }", ""]
        return "\n".join(lines)

    def reset_stats_source(self, function_name: str = "reset_object_memo_stats") -> str:
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
    "ObjectMemoNativeError",
    "ObjectMemoCell",
    "ObjectMemoPlan",
    "build_object_memo_plan",
    "ObjectMemoCythonEmitter",
]
