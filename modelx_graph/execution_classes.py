from __future__ import annotations

"""Architecture-B execution classification and unified recursive ABI contract.

Stage 2B keeps the Stage-2A N0/N1/P graph classification but removes the
separate logical bridge calling convention.  Every cacheable native-ABI Cell is
called through the existing ``r_<uid>(policy_index, ...)`` recursive ABI.

Execution classes:

* N0 - fully native formula candidate; all transitive formula dependencies are N0.
* N1 - native or ranked ABI, but at least one dependency requires N1/P execution.
* P  - the preserved Cell body needs Python semantics on a compiler-owned memo
  miss.

For a Python-backed P Cell, ``r_<uid>`` performs the native scalar/dense/exact-
hash memo lookup first. Only a miss enters the GIL and evaluates the detached
formula body. N1 formula source therefore calls ``r_<dep>`` uniformly for N0,
N1 and P dependencies.
"""

import ast
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .native_formula import NativeFormulaCell


N0 = "N0"
N1 = "N1"
P = "P"
NATIVE_SCALAR_VALUE_ABI = "native_scalar"
PYTHON_OBJECT_VALUE_ABI = "python_object"
RANKED_ARRAY_VALUE_ABI = "ranked_array"
UNSUPPORTED_VALUE_ABI = "unsupported"
_NATIVE_DTYPES = frozenset({"bool", "int64", "float64"})


class ExecutionClassError(RuntimeError):
    pass


def _cython_type(dtype: str) -> str:
    if dtype == "bool":
        return "bint"
    if dtype == "int64":
        # Match the existing recursive runtime ABI exactly. Stage 2B must not
        # create a second integer ABI merely because a Cell is Python-backed.
        return "long"
    if dtype == "float64":
        return "double"
    raise ExecutionClassError(f"unsupported native recursive ABI dtype {dtype!r}")


def _native_scalar_abi(cell: NativeFormulaCell) -> bool:
    return (
        cell.logical_result_ndim == 0
        and cell.dtype in _NATIVE_DTYPES
        and all(dtype in _NATIVE_DTYPES for dtype in cell.parameter_dtypes)
    )


def _value_abi(cell: NativeFormulaCell) -> str:
    """Return the recursive result ABI independently from the key ABI."""
    if cell.execution_role != "formula":
        return UNSUPPORTED_VALUE_ABI
    if not all(dtype in _NATIVE_DTYPES for dtype in cell.parameter_dtypes):
        return UNSUPPORTED_VALUE_ABI
    if cell.logical_result_ndim == 1 and cell.logical_result_shape is not None:
        return RANKED_ARRAY_VALUE_ABI
    if cell.logical_result_ndim > 1 and cell.logical_result_shape is not None:
        # Fixed higher-rank results stay as ordinary compiler-owned PyObjects.
        # Native callers cannot consume that shape directly, but detached Python
        # formulas (including the selected root) can retain NumPy/pandas semantics
        # while the object memo owns the value.
        return PYTHON_OBJECT_VALUE_ABI
    if cell.logical_result_ndim != 0:
        return UNSUPPORTED_VALUE_ABI
    if cell.dtype in _NATIVE_DTYPES:
        return NATIVE_SCALAR_VALUE_ABI
    if cell.dtype == "object":
        return PYTHON_OBJECT_VALUE_ABI
    return UNSUPPORTED_VALUE_ABI


def _tree_parts(cell: NativeFormulaCell) -> tuple[ast.AST, ...]:
    if cell.ranked_function is not None and cell.logical_result_ndim == 1:
        return (cell.ranked_function,)
    if cell.reduction is not None:
        r = cell.reduction
        return (r.init, r.body_expr, *r.range_args, *r.filters)
    if cell.function is not None:
        return (cell.function,)
    return ()


def _dense_size(bounds: Sequence[tuple[int, int]]) -> int:
    size = 1
    for lo, hi in bounds:
        width = int(hi) - int(lo) + 1
        if width <= 0:
            raise ExecutionClassError(f"invalid dense bounds {(lo, hi)!r}")
        size *= width
    return size


@dataclass(frozen=True)
class NativeMemoContract:
    """Memo-key contract for a Cell with a bridgeable recursive value ABI.

    ``physicalization_ready`` means the compiler memo runtime has enough
    authority to allocate the requested storage immediately.  A hash Cell can
    still be ``eligible`` while awaiting complete-run capacity evidence. Stage 2B
    physicalizes it once explicit capacity evidence is supplied; Stage 2C/census
    work should automate that evidence rather than falling back to Python dicts.
    """

    eligible: bool
    kind: str  # none | scalar | dense | hash
    physicalization_ready: bool
    reason: str
    value_abi: str = UNSUPPORTED_VALUE_ABI
    bounds: tuple[tuple[int, int], ...] = ()
    size: int = 0
    hash_capacity: int = 0

    def manifest(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "kind": self.kind,
            "physicalization_ready": self.physicalization_ready,
            "reason": self.reason,
            "value_abi": self.value_abi,
            "bounds": [list(x) for x in self.bounds],
            "size": self.size,
            "hash_capacity": self.hash_capacity,
            "python_dict_required": False,
        }


@dataclass(frozen=True)
class ExecutionClassCell:
    uid: str
    source_name: str
    source_fullname: str
    execution_class: str
    dependencies: tuple[str, ...]
    direct_blockers: tuple[tuple[str, str], ...]
    classification_reasons: tuple[str, ...]
    native_scalar_abi: bool
    value_abi: str
    bridgeable_python_fallback: bool
    parameter_dtypes: tuple[str, ...]
    result_dtype: str
    memo: NativeMemoContract
    execution_role: str = "formula"
    language_validation: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_name": self.source_name,
            "source_fullname": self.source_fullname,
            "execution_role": self.execution_role,
            "execution_class": self.execution_class,
            "dependencies": list(self.dependencies),
            "direct_blockers": [
                {"kind": kind, "detail": detail}
                for kind, detail in self.direct_blockers
            ],
            "classification_reasons": list(self.classification_reasons),
            "native_scalar_abi": self.native_scalar_abi,
            "value_abi": self.value_abi,
            "bridgeable_python_fallback": self.bridgeable_python_fallback,
            "parameter_dtypes": list(self.parameter_dtypes),
            "result_dtype": self.result_dtype,
            "memo": self.memo.manifest(),
            "language_validation": self.language_validation,
        }


@dataclass(frozen=True)
class FallbackPolicyContextContract:
    """Ownership/lifetime contract for Stage-2B fallback state."""

    owner: str = "batch_worker"
    policy_index_dtype: str = "Py_ssize_t"
    policy_key_source: str = "fast_plan.run_keys[policy_index]"
    creation: str = "lazy_on_first_python_backed_native_cache_miss_for_active_policy"
    lifetime: str = "one_active_policy_per_worker"
    reset: str = "discard_before_worker_advances_to_next_policy"
    backend: str = "ordinary_modelx_cython_or_python_cell_on_native_cache_miss"
    cache_scope: str = "policy_local_only"
    full_portfolio_cache_accumulation: bool = False
    thread_ownership: str = "one_context_per_worker_thread"

    def manifest(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "policy_index_dtype": self.policy_index_dtype,
            "policy_key_source": self.policy_key_source,
            "creation": self.creation,
            "lifetime": self.lifetime,
            "reset": self.reset,
            "backend": self.backend,
            "cache_scope": self.cache_scope,
            "full_portfolio_cache_accumulation": self.full_portfolio_cache_accumulation,
            "thread_ownership": self.thread_ownership,
        }


@dataclass(frozen=True)
class FallbackBridgeSpec:
    """Internal cache-miss contract for a Python-backed native-ABI Cell.

    The historical class name is retained for handoff/API compatibility, but
    Stage 2B no longer exposes a ``bridge_<uid>()`` logical ABI.  Native callers
    always call ``r_<uid>()``.  The native memo wrapper invokes ``calc_<uid>()``
    only on a cache miss, and for a P Cell that ``calc_<uid>()`` implementation
    acquires the GIL, evaluates the original Cell and converts the result back
    to the declared native scalar ABI.

    The generated miss implementation is ``noexcept nogil`` and captures a
    Python exception for deferred re-raise at the batch boundary.  This avoids
    Cython's ``except * nogil`` post-call GIL check on native-cache hits.
    """

    callee_uid: str
    callee_source_name: str
    callee_source_fullname: str
    caller_uids: tuple[str, ...]
    recursive_wrapper_name: str
    miss_implementation_name: str
    policy_context_parameter: str
    parameter_names: tuple[str, ...]
    parameter_dtypes: tuple[str, ...]
    result_dtype: str
    recursive_signature: str
    miss_signature: str
    call_mode: str = "positional_only"
    gil_mode: str = "miss_implementation_acquires_gil_internally"
    exception_mode: str = "capture_original_exception_and_reraise_after_nogil_batch"
    argument_boxing: str = "native_bool_int64_float64_to_python_scalars_on_miss"
    result_conversion: str = "checked_python_scalar_to_declared_native_dtype_on_miss"
    memo_ownership: str = "r_wrapper_native_scalar_dense_or_exact_hash"
    caller_convention: str = "uniform_r_cell_recursive_abi"

    @property
    def bridge_name(self) -> str:
        """Deprecated compatibility alias; not a callable ABI in Stage 2B."""
        return self.miss_implementation_name

    @property
    def cython_signature(self) -> str:
        """Deprecated compatibility alias for the internal miss signature."""
        return self.miss_signature

    def manifest(self) -> dict[str, Any]:
        return {
            "callee_uid": self.callee_uid,
            "callee_source_name": self.callee_source_name,
            "callee_source_fullname": self.callee_source_fullname,
            "caller_uids": list(self.caller_uids),
            "recursive_wrapper_name": self.recursive_wrapper_name,
            "miss_implementation_name": self.miss_implementation_name,
            "policy_context_parameter": self.policy_context_parameter,
            "parameter_names": list(self.parameter_names),
            "parameter_dtypes": list(self.parameter_dtypes),
            "result_dtype": self.result_dtype,
            "recursive_signature": self.recursive_signature,
            "miss_signature": self.miss_signature,
            "caller_convention": self.caller_convention,
            "call_mode": self.call_mode,
            "gil_mode": self.gil_mode,
            "exception_mode": self.exception_mode,
            "argument_boxing": self.argument_boxing,
            "result_conversion": self.result_conversion,
            "memo_ownership": self.memo_ownership,
            "separate_bridge_call_exposed_to_native_callers": False,
        }


@dataclass(frozen=True)
class ExecutionClassPlan:
    schema: str
    output_uid: str
    cells: Mapping[str, ExecutionClassCell]
    python_backed_misses: Mapping[str, FallbackBridgeSpec]
    fallback_context: FallbackPolicyContextContract

    @property
    def bridges(self) -> Mapping[str, FallbackBridgeSpec]:
        """Compatibility alias for dev127 readers. These are miss specs, not calls."""
        return self.python_backed_misses

    @property
    def output_class(self) -> str:
        row = self.cells.get(self.output_uid)
        return P if row is None else row.execution_class

    @property
    def mixed_candidate_ready(self) -> bool:
        return self.output_class in {N0, N1}

    def cell(self, uid: str) -> ExecutionClassCell:
        try:
            return self.cells[uid]
        except KeyError as exc:
            raise ExecutionClassError(f"unknown execution-class Cell {uid!r}") from exc

    @property
    def n0_uids(self) -> tuple[str, ...]:
        return tuple(sorted(uid for uid, row in self.cells.items() if row.execution_class == N0))

    @property
    def n1_uids(self) -> tuple[str, ...]:
        return tuple(sorted(uid for uid, row in self.cells.items() if row.execution_class == N1))

    @property
    def p_uids(self) -> tuple[str, ...]:
        return tuple(sorted(uid for uid, row in self.cells.items() if row.execution_class == P))

    @property
    def selected_closure_uids(self) -> tuple[str, ...]:
        if self.output_uid not in self.cells:
            return ()
        seen: set[str] = set()
        pending = [self.output_uid]
        while pending:
            uid = pending.pop()
            if uid in seen or uid not in self.cells:
                continue
            seen.add(uid)
            pending.extend(self.cells[uid].dependencies)
        return tuple(sorted(seen))

    @property
    def selected_closure_counts(self) -> dict[str, int]:
        out = {N0: 0, N1: 0, P: 0}
        for uid in self.selected_closure_uids:
            out[self.cells[uid].execution_class] += 1
        return out

    @property
    def recursive_abi_uids(self) -> tuple[str, ...]:
        """Cells reached through native ``r_<uid>`` calls before a P boundary.

        A bridgeable P Cell itself is included because it owns a native memo
        wrapper, but its Python-owned dependencies are not traversed.
        """
        if self.output_uid not in self.cells:
            return ()
        seen: set[str] = set()
        pending = [self.output_uid]
        while pending:
            uid = pending.pop()
            if uid in seen or uid not in self.cells:
                continue
            row = self.cells[uid]
            seen.add(uid)
            if row.execution_class == P:
                continue
            pending.extend(row.dependencies)
        return tuple(sorted(seen))

    @property
    def gil_cython_candidate_uids(self) -> tuple[str, ...]:
        """P Cells whose only direct blocker is explicit Python language semantics.

        These Cells retain a native scalar ABI and can therefore execute their
        translated Cython formula under the GIL while calling dependencies through
        the same ``r_<cell>()`` ABI. Provider/object-backed P Cells are excluded.
        """
        return tuple(sorted(
            uid for uid, row in self.cells.items()
            if row.execution_class == P
            and row.execution_role == "formula"
            and row.value_abi in {
                NATIVE_SCALAR_VALUE_ABI,
                PYTHON_OBJECT_VALUE_ABI,
                RANKED_ARRAY_VALUE_ABI,
            }
            and row.bridgeable_python_fallback
            and bool(row.direct_blockers)
            and all(kind == "python_semantics" for kind, _detail in row.direct_blockers)
        ))

    @property
    def mixed_recursive_abi_uids(self) -> tuple[str, ...]:
        """Recursive ABI closure for dev131 expression-level Python semantics.

        Unlike the historical closure, traversal continues through GIL-Cython P
        Cells because their generated formula bodies call ``r_<dep>()`` directly.
        Whole-Cell modelx fallback P Cells remain terminal boundaries.
        """
        if self.output_uid not in self.cells:
            return ()
        gil_backed = set(self.gil_cython_candidate_uids)
        seen: set[str] = set()
        pending = [self.output_uid]
        while pending:
            uid = pending.pop()
            if uid in seen or uid not in self.cells:
                continue
            row = self.cells[uid]
            seen.add(uid)
            if row.execution_class == P and uid not in gil_backed:
                continue
            pending.extend(row.dependencies)
        return tuple(sorted(seen))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "output_uid": self.output_uid,
            "output_class": self.output_class,
            "mixed_candidate_ready": self.mixed_candidate_ready,
            "counts": {
                N0: len(self.n0_uids),
                N1: len(self.n1_uids),
                P: len(self.p_uids),
            },
            "selected_closure_counts": self.selected_closure_counts,
            "selected_closure_uids": list(self.selected_closure_uids),
            "cells": {
                uid: row.manifest() for uid, row in sorted(self.cells.items())
            },
            "python_backed_cache_misses": {
                uid: row.manifest() for uid, row in sorted(self.python_backed_misses.items())
            },
            "fallback_context": self.fallback_context.manifest(),
            "n0_hot_path_requires_python": False,
            "n1_python_work_is_cell_granular": True,
            "recursive_call_abi": "uniform_r_<cell>",
            "recursive_abi_uids": list(self.recursive_abi_uids),
            "mixed_recursive_abi_uids": list(self.mixed_recursive_abi_uids),
            "gil_cython_candidate_uids": list(self.gil_cython_candidate_uids),
            "python_entry_condition": "python_backed_r_wrapper_native_cache_miss_only",
            "separate_bridge_calling_convention": False,
            "expression_level_python_islands": bool(self.gil_cython_candidate_uids),
            "python_semantic_p_strategy": "generated_Cython_formula_with_GIL_and_native_r_dependencies",
            "classification_authority": "direct_provider_type_abi_blockers_plus_cell_dependency_graph",
            "n0_language_authority": "successful_Cython_noexcept_nogil_compile",
            "n1_language_authority": "Stage2B_Cython_unified_recursive_abi_compile",
        }


def _memo_contract(
    cell: NativeFormulaCell,
    *,
    parameter_ranges: Mapping[str, Sequence[tuple[int, int] | None]],
    native_hash_capacity_by_uid: Mapping[str, int],
    max_dense_slots_per_cell: int,
) -> NativeMemoContract:
    if cell.execution_role != "formula":
        return NativeMemoContract(False, "none", True, "materialized_provider_not_recursive_formula")
    value_abi = _value_abi(cell)
    if value_abi == UNSUPPORTED_VALUE_ABI:
        return NativeMemoContract(False, "none", False, "native_key_and_value_abi_required")
    if not cell.parameters:
        return NativeMemoContract(
            True, "scalar", True,
            (
                "zero_arity_native_scalar"
                if value_abi == NATIVE_SCALAR_VALUE_ABI
                else "zero_arity_ranked_array"
                if value_abi == RANKED_ARRAY_VALUE_ABI
                else "zero_arity_python_object"
            ),
            value_abi=value_abi, size=1,
        )

    requested = tuple(parameter_ranges.get(cell.uid, ()))
    if len(requested) != len(cell.parameters):
        requested = tuple(None for _ in cell.parameters)
    dense_bounds: list[tuple[int, int]] = []
    dense_safe = True
    for dtype, item in zip(cell.parameter_dtypes, requested):
        if item is None and dtype == "bool":
            item = (0, 1)
        if dtype not in {"int64", "bool"} or item is None:
            dense_safe = False
            break
        lo, hi = int(item[0]), int(item[1])
        if lo > hi:
            raise ExecutionClassError(f"{cell.uid}: invalid parameter range {(lo, hi)!r}")
        dense_bounds.append((lo, hi))
    if dense_safe and len(dense_bounds) == len(cell.parameters):
        size = _dense_size(dense_bounds)
        if size <= int(max_dense_slots_per_cell):
            return NativeMemoContract(
                True, "dense", True, "native_cache_bounded_dense",
                value_abi, tuple(dense_bounds), size,
            )

    capacity = int(native_hash_capacity_by_uid.get(cell.uid, 0) or 0)
    return NativeMemoContract(
        True,
        "hash",
        capacity > 0,
        (
            "native_exact_hash_capacity_proven"
            if capacity > 0
            else "native_exact_hash_complete_run_capacity_evidence_required"
        ),
        value_abi,
        tuple(dense_bounds),
        0,
        capacity,
    )


def _python_backed_miss_spec(
    callee: NativeFormulaCell,
    callers: Sequence[str],
) -> FallbackBridgeSpec:
    if not _native_scalar_abi(callee):
        raise ExecutionClassError(f"{callee.uid}: cannot build Python-backed miss contract for non-native scalar ABI")
    args = ["Py_ssize_t __mx_policy_i"]
    for name, dtype in zip(callee.parameters, callee.parameter_dtypes):
        args.append(f"{_cython_type(dtype)} {name}")
    recursive_signature = (
        f"cdef inline {_cython_type(callee.dtype)} r_{callee.uid}("
        + ", ".join(args)
        + ") noexcept nogil"
    )
    miss_signature = (
        f"cdef inline {_cython_type(callee.dtype)} calc_{callee.uid}("
        + ", ".join(args)
        + ") noexcept nogil"
    )
    return FallbackBridgeSpec(
        callee_uid=callee.uid,
        callee_source_name=callee.source_name,
        callee_source_fullname=callee.source_fullname,
        caller_uids=tuple(sorted(set(callers))),
        recursive_wrapper_name=f"r_{callee.uid}",
        miss_implementation_name=f"calc_{callee.uid}",
        policy_context_parameter="__mx_policy_i",
        parameter_names=callee.parameters,
        parameter_dtypes=callee.parameter_dtypes,
        result_dtype=callee.dtype,
        recursive_signature=recursive_signature,
        miss_signature=miss_signature,
    )


def build_execution_class_plan(
    *,
    output_uid: str,
    cells: Mapping[str, NativeFormulaCell],
    direct_blockers: Mapping[str, Sequence[tuple[str, str]]],
    parameter_ranges: Mapping[str, Sequence[tuple[int, int] | None]] | None = None,
    native_hash_capacity_by_uid: Mapping[str, int] | None = None,
    max_dense_slots_per_cell: int = 1_000_000,
) -> ExecutionClassPlan:
    """Classify a source-transparent Cell graph without transitive P collapse.

    ``direct_blockers`` must contain only blockers intrinsic to the Cell before
    historical ``fallback_dependency`` propagation.  Known Cell calls are graph
    edges rather than provider blockers.
    """

    parameter_ranges = parameter_ranges or {}
    native_hash_capacity_by_uid = native_hash_capacity_by_uid or {}
    cells = dict(cells)
    all_uids = set(cells)
    deps: dict[str, tuple[str, ...]] = {}
    for uid, cell in cells.items():
        if cell.execution_role != "formula":
            deps[uid] = ()
            continue
        found = {
            node.func.id
            for tree in _tree_parts(cell)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in all_uids
        }
        deps[uid] = tuple(sorted(found))

    direct: dict[str, tuple[tuple[str, str], ...]] = {
        uid: tuple(sorted(set(tuple(x) for x in direct_blockers.get(uid, ()))))
        for uid in cells
    }
    classes: dict[str, str] = {}
    reasons: dict[str, list[str]] = {}
    bridgeable: dict[str, bool] = {}
    value_abis: dict[str, str] = {}

    for uid, cell in cells.items():
        abi = _native_scalar_abi(cell)
        value_abi = _value_abi(cell)
        value_abis[uid] = value_abi
        bridgeable[uid] = value_abi != UNSUPPORTED_VALUE_ABI
        if cell.execution_role != "formula":
            classes[uid] = N0
            reasons[uid] = ["materialized_provider"]
        elif direct[uid]:
            classes[uid] = P
            reasons[uid] = [
                f"direct_{kind}:{detail}" for kind, detail in direct[uid]
            ]
        elif value_abi == UNSUPPORTED_VALUE_ABI:
            classes[uid] = P
            reasons[uid] = ["direct_recursive_value_abi_unavailable"]
        elif value_abi == PYTHON_OBJECT_VALUE_ABI:
            classes[uid] = P
            reasons[uid] = ["direct_python_object_value_abi"]
        else:
            classes[uid] = N0
            reasons[uid] = ["direct_source_candidate_native_abi"]

    # Monotone graph classification. A native candidate becomes N1 when it must
    # call N1 or bridgeable P. It becomes P only when a direct dependency cannot
    # be represented by the Cell-level native bridge ABI. Once a Cell is P, its
    # own *interface* may still be bridgeable to an outer N1 caller.
    changed = True
    while changed:
        changed = False
        for uid in sorted(cells):
            if classes[uid] == P or cells[uid].execution_role != "formula":
                continue
            dep_rows = [(dep, classes[dep], bridgeable[dep]) for dep in deps[uid]]
            unbridgeable = [dep for dep, cls, can_bridge in dep_rows if cls == P and not can_bridge]
            if unbridgeable:
                classes[uid] = P
                reasons[uid] = [
                    "unbridgeable_python_dependency:" + dep
                    for dep in sorted(unbridgeable)
                ]
                changed = True
                continue
            object_deps = [
                dep for dep, cls, _can_bridge in dep_rows
                if cls == P and value_abis[dep] in {
                    PYTHON_OBJECT_VALUE_ABI,
                    RANKED_ARRAY_VALUE_ABI,
                }
            ]
            if object_deps and cells[uid].logical_result_ndim == 0:
                # A scalar body cannot call an object-returning wrapper from its
                # nogil native implementation.  Keep its scalar ABI, but move
                # the body to detached Python/GIL execution.
                classes[uid] = P
                reasons[uid] = [
                    "python_object_recursive_dependency:" + dep
                    for dep in sorted(object_deps)
                ]
                changed = True
                continue
            fallback_deps = [dep for dep, cls, _ in dep_rows if cls == P]
            n1_deps = [dep for dep, cls, _ in dep_rows if cls == N1]
            if classes[uid] == N0 and (fallback_deps or n1_deps):
                classes[uid] = N1
                reasons[uid] = (
                    ["python_backed_recursive_dependency:" + dep for dep in sorted(fallback_deps)]
                    + ["n1_dependency:" + dep for dep in sorted(n1_deps)]
                )
                changed = True
            elif classes[uid] == N1:
                # Refresh reasons when another member of a cycle transitions to N1.
                new_reasons = (
                    ["python_backed_recursive_dependency:" + dep for dep in sorted(fallback_deps)]
                    + ["n1_dependency:" + dep for dep in sorted(n1_deps)]
                )
                if new_reasons and new_reasons != reasons[uid]:
                    reasons[uid] = new_reasons

    rows: dict[str, ExecutionClassCell] = {}
    for uid, cell in cells.items():
        rows[uid] = ExecutionClassCell(
            uid=uid,
            source_name=cell.source_name,
            source_fullname=cell.source_fullname,
            execution_class=classes[uid],
            dependencies=deps[uid],
            direct_blockers=direct[uid],
            classification_reasons=tuple(reasons[uid]),
            native_scalar_abi=_native_scalar_abi(cell),
            value_abi=value_abis[uid],
            bridgeable_python_fallback=bridgeable[uid],
            parameter_dtypes=cell.parameter_dtypes,
            result_dtype=cell.dtype,
            memo=_memo_contract(
                cell,
                parameter_ranges=parameter_ranges,
                native_hash_capacity_by_uid=native_hash_capacity_by_uid,
                max_dense_slots_per_cell=max_dense_slots_per_cell,
            ),
            execution_role=cell.execution_role,
            language_validation=(
                "not_applicable_materialized_provider"
                if cell.execution_role != "formula"
                else "ordinary_modelx_cython_python_fallback"
                if classes[uid] == P
                else "Stage2B_Cython_unified_recursive_abi_compile_required"
                if classes[uid] == N1
                else "Cython_noexcept_nogil_compile_required"
            ),
        )

    python_miss_callers: dict[str, list[str]] = {}
    for uid, row in rows.items():
        if row.execution_class != N1:
            continue
        for dep in row.dependencies:
            target = rows[dep]
            if (
                target.execution_class == P
                and target.bridgeable_python_fallback
                and target.value_abi == NATIVE_SCALAR_VALUE_ABI
            ):
                python_miss_callers.setdefault(dep, []).append(uid)
    python_backed_misses = {
        uid: _python_backed_miss_spec(cells[uid], callers)
        for uid, callers in sorted(python_miss_callers.items())
    }

    return ExecutionClassPlan(
        schema="modelx_graph.execution_class_plan.v4_ranked_array_value_abi",
        output_uid=output_uid,
        cells=rows,
        python_backed_misses=python_backed_misses,
        fallback_context=FallbackPolicyContextContract(),
    )


__all__ = [
    "N0", "N1", "P", "ExecutionClassError", "NativeMemoContract",
    "NATIVE_SCALAR_VALUE_ABI", "PYTHON_OBJECT_VALUE_ABI", "RANKED_ARRAY_VALUE_ABI",
    "UNSUPPORTED_VALUE_ABI",
    "ExecutionClassCell", "FallbackPolicyContextContract", "FallbackBridgeSpec",
    "ExecutionClassPlan", "build_execution_class_plan",
]
