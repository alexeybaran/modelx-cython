from __future__ import annotations

"""Fast-recursive backend adapter driven by modelx-cython frontend metadata.

The production authority in this module is intentionally modelx-cython's existing
lexical/runtime/spec pipeline:

* ``ModuleVisitor`` supplies the complete exported Cell catalogue;
* ``MxCallTraceLogger`` supplies observed argument/return types;
* ``TransSpec`` supplies explicit overrides and cache bounds.

No target dependency graph, SCC analysis, Stage schedule or global static type/domain
fixed point is required for admission.  The ordinary modelx-cython backend remains the
fallback for Cells that are object-typed or whose provider expression is not yet worth
specializing.
"""

import ast
import copy
import gc
import hashlib
import json
import numbers
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from time import perf_counter

import numpy as np
import libcst as cst

from modelx_cython.builder import ClassInfo, ModuleInfo
from modelx_cython.cli import iter_module_files
from .latest_frontend_compat import run_sample
from .cell_alias_normalizer import normalize_cell_aliases
from modelx_cython.config import TransSpec
from modelx_cython.parser import ModuleVisitor
from modelx_cython.typedefs import str_to_type
from modelx_cython.usage import RESOLVED, build_resolver

from .authoritative_evidence import DemandEvidenceRequest, resolve_demand_driven_evidence
from .cython_expression import infer_local_types, expression_dtype as infer_expression_dtype
from .fast_formula_translation_service import (
    CanonicalModel, FastFormulaTranslationService, FrontendError,
    InputRegistry, InputSpec, ReductionSpec,
)
from .finite_vector_reduction import (
    FiniteVectorElementHelper,
    build_comprehension_vector_helpers,
    build_recursive_append_vector_helper,
    normalize_finite_vector_consumers,
    normalize_bounded_filtered_min_listcomps,
)
from .modelx_cython_formula_engine import normalize_formula_function
from .modelx_cython_category_abi import (
    BindPointCategorySubscript,
    ModelxCythonCategoryPlan,
    CategoryABIError,
    build_modelx_cython_category_plan,
    prove_consumer_category_result_domains,
)
from .native_formula import NativeFormulaCell, NativeFormulaProgram
from .execution_classes import ExecutionClassPlan, build_execution_class_plan
from .local_control_semantics import (
    bounded_add_range_specs,
    bounded_monotone_positive_evidence_requirements_by_while_id,
    range_call_ids_for_supported_bounded_add,
    range_call_ids_for_supported_bounded_filtered_min,
    range_call_ids_for_supported_bounded_filtered_add,
    supported_bounded_add_for_ids,
    supported_bounded_filtered_add_for_ids,
    continue_stmt_ids_for_supported_bounded_filtered_add,
    supported_bounded_filtered_min_for_ids,
    fixed_count_state_update_specs,
    range_call_ids_for_supported_first_hit,
    range_call_ids_for_supported_fixed_count,
    supported_first_hit_for_ids,
    supported_bounded_monotone_while_ids,
    supported_fixed_count_for_ids,
    update_stmt_ids_for_supported_bounded_monotone,
)
from .normalized_operator import NormalizedOperator
from .semantic_ir import SemanticFormulaIR, FormulaNormalizationResult
from .lookup_semantics import recognize_asof_lookups
from .semantic_passes import AstSemanticPass, SemanticPassPipeline
from .program_inputs import OptimizedInputSpec
from .program_semantics import GuardSpec, normalize_formula_guards
from .semantic_reuse import diagnose_function_semantics, unroll_literal_reductions
from .static_type_domain import StaticValueDomain, _local_domains, numeric_domain
from .int64_range_proof import (
    INT64_MAX, INT64_MIN, Int64Interval, derive_local_int64_intervals,
    interval_rows, intervals_from_rows, prove_int64_expression,
)
from .structured_provider import (
    RowMap, StructuredFramePlan, StructuredProviderError,
)


class ModelxCythonFastBackendError(RuntimeError):
    pass


_ALLOWED_DTYPES = {"float64", "int64", "bool"}
_RUNTIME_INPUT_CALLS = {
    "point_input", "ranked_input", "global_input", "array_input", "table_input"
}
_NATIVE_INTRINSICS = {
    "min", "max", "abs", "round", "int", "float", "bool",
    "__is_missing__", "__is_inf__", "__erf__", "__exact_axis_code__", "__exact_axis_code_valid__",
    "__step_lookup_1d__", "__interval_lookup_1d__", "__interp_lookup_1d__",
    "__sparse_table_lookup__", "__prepared_point_lookup__",
    "__guard_fail__",
    "ranked_where",
}
_MATH_CALLS = {"exp", "log", "sqrt", "sin", "cos", "floor", "ceil"}
_SAFE_IDENT = re.compile(r"[^0-9A-Za-z_]+")


def _safe_ident(value: str) -> str:
    out = _SAFE_IDENT.sub("_", value).strip("_") or "cell"
    if out[0].isdigit():
        out = "_" + out
    return out


def _translation_discovery_keys(sample_keys: Sequence[Any]) -> tuple[Any, ...]:
    """Select the bounded modelx graph-discovery witness set.

    The exported modelx-cython trace may use the caller's whole representative
    sample for type observation.  The second, much heavier modelx dependency
    trace is not proof authority, so one deterministic witness is sufficient for
    discovery.  Complete proofs remain tied to ``run_keys`` elsewhere.
    """

    keys = tuple(sample_keys)
    return keys[:1]


def _process_memory_kb() -> dict[str, int | None]:
    """Return cheap retained/peak process memory diagnostics on Linux.

    Planning can legitimately run in constrained build workers.  Reading
    ``/proc/self/status`` costs essentially nothing relative to provider evidence
    construction and, unlike Python-only allocators, includes NumPy/pandas/modelx
    retained memory.  On non-Linux platforms the fields remain ``None``.
    """

    out: dict[str, int | None] = {"rss_kb": None, "hwm_kb": None}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                out["rss_kb"] = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                out["hwm_kb"] = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return out


def _append_planning_diagnostic(row: Mapping[str, Any]) -> None:
    """Optionally persist phase progress so an OOM-killed planner is diagnosable.

    The sink is deliberately opt-in and diagnostic-only; production planning has
    no file-system side effect unless ``MODELX_GRAPH_PLAN_DIAGNOSTICS`` is set.
    """

    target = os.environ.get("MODELX_GRAPH_PLAN_DIAGNOSTICS")
    if not target:
        return
    try:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    except OSError:
        # Diagnostics must never change planner semantics.
        return


def _uid(module_name: str, class_name: str, cell_name: str) -> str:
    qname = f"{module_name}::{class_name}::{cell_name}"
    digest = hashlib.sha256(qname.encode("utf-8")).hexdigest()[:12]
    return f"mc_{digest}_{_safe_ident(cell_name)}"


def _fast_dtype(tp: type | None) -> str:
    if tp is None:
        return "object"
    try:
        if issubclass(tp, (bool, np.bool_)):
            return "bool"
        if issubclass(tp, numbers.Integral):
            return "int64"
        if issubclass(tp, numbers.Real):
            return "float64"
    except TypeError:
        pass
    return "object"


def _spec_type(value: Any, *, context: str) -> type:
    if not isinstance(value, str) or value not in str_to_type:
        raise ModelxCythonFastBackendError(f"{context}: invalid type override {value!r}")
    return str_to_type[value]


def _freeze_input(spec: InputSpec) -> OptimizedInputSpec:
    return OptimizedInputSpec(
        key=spec.key,
        dtype=spec.dtype,
        ndim=int(spec.ndim),
        scope=spec.scope,
        description=spec.description,
        expected_shape=spec.expected_shape,
        domain_token=spec.domain_token,
        normalized_table=spec.normalized_table,
        point_row_source=spec.point_row_source,
        point_field=spec.point_field,
        normalized_structured_column=spec.normalized_structured_column,
        enum_labels=spec.enum_labels,
        category_validity_key=spec.category_validity_key,
        normalized_lookup_1d=spec.normalized_lookup_1d,
        lookup_1d_role=spec.lookup_1d_role,
        prepared_exact_int_value=spec.prepared_exact_int_value,
        prepared_exact_int_authority=spec.prepared_exact_int_authority,
        prepared_exact_int_proof_domain_size=spec.prepared_exact_int_proof_domain_size,
        prepared_exact_int_source_name=spec.prepared_exact_int_source_name,
    )


def _prepared_exact_int_bounds(input_specs: Mapping[str, Any]) -> dict[str, int]:
    """Exact integer global-input bounds certified by preparation.

    The local-control layer consumes only this tiny key/value view.  All proof
    provenance and runtime validation remain owned by the existing input layer.
    """

    out: dict[str, int] = {}
    for key, spec in input_specs.items():
        value = getattr(spec, "prepared_exact_int_value", None)
        authority = getattr(spec, "prepared_exact_int_authority", None)
        if (
            type(value) is int
            and -(1 << 63) <= value <= (1 << 63) - 1
            and isinstance(authority, str)
            and authority
        ):
            out[str(key)] = int(value)
    return out


def _prepare_fixed_count_range_bound_inputs(
    evidence: FastFormulaTranslationService,
    fn: ast.FunctionDef,
) -> None:
    """Attach exact-int contracts to source refs used by structured bounded ranges.

    This does not rewrite the formula.  A source reference is first proven
    invariant over the complete frozen RunDomain.  A copy of the function then
    substitutes only those candidate range arguments with literals so the
    existing local-control recognizers can decide whether the surrounding loop
    shape is actually admissible. Only refs belonging to recognized loops are
    upgraded to validated exact-int scalar inputs. The historical function name
    is retained because existing I4a callers/tests import it directly.
    """

    def could_be_additive_range_shape(stmt: ast.For) -> bool:
        """Cheap syntax-only gate before any static-reference proof work.

        Nested loops are new in dev122.  Do not ask the evidence layer to prove
        every reference appearing in every nested range merely because one
        narrow nested additive semantic exists.  The full recognizer below is
        still the admission authority after exact values are substituted.
        """

        if stmt.orelse or len(stmt.body) != 1 or not isinstance(stmt.target, ast.Name):
            return False
        update = stmt.body[0]
        if (
            isinstance(update, ast.AugAssign)
            and isinstance(update.target, ast.Name)
            and isinstance(update.op, ast.Add)
        ):
            return True
        if not (
            isinstance(update, ast.Assign)
            and len(update.targets) == 1
            and isinstance(update.targets[0], ast.Name)
            and isinstance(update.value, ast.BinOp)
            and isinstance(update.value.op, ast.Add)
        ):
            return False
        acc = update.targets[0].id
        left_is_acc = isinstance(update.value.left, ast.Name) and update.value.left.id == acc
        right_is_acc = isinstance(update.value.right, ast.Name) and update.value.right.id == acc
        return left_is_acc != right_is_acc

    candidates: dict[str, int] = {}
    refs = evidence.refs
    top_level_for_ids = {id(stmt) for stmt in fn.body if isinstance(stmt, ast.For)}
    for stmt in ast.walk(fn):
        if not (
            isinstance(stmt, ast.For)
            and (id(stmt) in top_level_for_ids or could_be_additive_range_shape(stmt))
            and isinstance(stmt.iter, ast.Call)
            and isinstance(stmt.iter.func, ast.Name)
            and stmt.iter.func.id == "range"
            and not stmt.iter.keywords
            and 1 <= len(stmt.iter.args) <= 3
        ):
            continue
        for arg in stmt.iter.args:
            if not (
                isinstance(arg, ast.Name)
                and isinstance(arg.ctx, ast.Load)
                and arg.id in refs
            ):
                continue
            ok, value = evidence._proven_static_reference_value(arg.id)
            if (
                ok
                and type(value) is int
                and -(1 << 63) <= int(value) <= (1 << 63) - 1
            ):
                candidates[arg.id] = int(value)
    if not candidates:
        return

    probe = copy.deepcopy(fn)
    for stmt in ast.walk(probe):
        if not (
            isinstance(stmt, ast.For)
            and isinstance(stmt.iter, ast.Call)
            and isinstance(stmt.iter.func, ast.Name)
            and stmt.iter.func.id == "range"
        ):
            continue
        stmt.iter.args = [
            ast.copy_location(ast.Constant(candidates[arg.id]), arg)
            if isinstance(arg, ast.Name) and arg.id in candidates
            else arg
            for arg in stmt.iter.args
        ]
    probe = ast.fix_missing_locations(probe)
    # I4a admission is defined on the same scalar-local syntax that reaches
    # native emission.  Reuse the mature local-syntax lowering on this
    # throwaway recognition copy so harmless fixed-arity unpacking before the
    # loop (for example ``lo, hi = 0.0, cap``) cannot hide the fact that both
    # state locals are definitely initialized.  The production candidate is
    # not rewritten here; it follows the ordinary normalization pipeline.
    probe = evidence._normalize_simple_local_syntax(probe)
    admitted_lines = {spec.line for spec in fixed_count_state_update_specs(probe)}
    admitted_lines |= {spec.line for spec in bounded_add_range_specs(probe)}
    if not admitted_lines:
        return

    for stmt in ast.walk(fn):
        if not (
            isinstance(stmt, ast.For)
            and int(getattr(stmt, "lineno", 0) or 0) in admitted_lines
            and isinstance(stmt.iter, ast.Call)
        ):
            continue
        for arg in stmt.iter.args:
            if isinstance(arg, ast.Name) and arg.id in candidates:
                evidence.registry.validated_exact_int_scalar_ref(
                    arg.id,
                    refs[arg.id],
                    expected_value=candidates[arg.id],
                    proof_domain_size=len(evidence.run_keys),
                )


def _attach_local_integer_control_evidence(
    cell: NativeFormulaCell,
    *,
    callee_dtypes: Mapping[str, str],
    callee_value_domains: Mapping[str, StaticValueDomain] | None = None,
    input_specs: Mapping[str, Any],
    module_refs: Mapping[str, str] | None = None,
    parameter_int64_intervals: Mapping[str, Int64Interval] | None = None,
) -> NativeFormulaCell:
    """Attach authoritative local integer and finite-positive control facts.

    The inference itself is the existing Cython scalar-local type inference. Its
    Cell-call result types come from the same modelx-cython/demand evidence that
    already selected ``NativeFormulaCell.dtype``.  Unknown locals default away
    from int64, so this bridge is fail-closed for local-control purposes.

    Type facts reuse the existing Cython local inference plus a dependency-
    completeness gate.  I4b-1B2A additionally reuses the backend-neutral static
    value-domain engine: a local is marked positive only when its complete exact
    int64 domain is finite, proven, non-empty and every value is > 0.
    """

    if cell.function is None:
        return cell
    emitter = SimpleNamespace(
        v={uid: SimpleNamespace(dtype=dtype) for uid, dtype in callee_dtypes.items()},
        input_specs=input_specs,
    )
    parameter_types = dict(zip(cell.parameters, cell.parameter_dtypes))
    local_types = infer_local_types(emitter, cell.function, parameter_types)
    local_scalar_dtypes = tuple(sorted(
        (name, dtype)
        for name, dtype in local_types.items()
        if dtype in _ALLOWED_DTYPES
    ))
    callee_value_domains = dict(callee_value_domains or {})
    result_domains: dict[str, StaticValueDomain] = {}
    for uid, dtype in callee_dtypes.items():
        domain = callee_value_domains.get(uid)
        if not (
            domain is not None
            and domain.proven
            and domain.numeric_dtype == dtype
        ):
            domain = numeric_domain(dtype, f"native_result_type:{uid}")
        result_domains[uid] = domain

    def dependencies_known(node: ast.AST, known_names: set[str]) -> bool:
        """Require every value dependency behind a control-type fact to be typed.

        ``infer_local_types`` intentionally performs useful best-effort scalar
        inference for code generation.  Control admission needs stronger
        authority: an expression such as ``12 // unknown_call()`` must not
        become an int64 proof merely because its left operand is integral.
        This gate checks dependency completeness while leaving dtype inference
        itself in the existing shared implementation.
        """

        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return node.id in known_names
        if isinstance(node, ast.UnaryOp):
            return dependencies_known(node.operand, known_names)
        if isinstance(node, ast.BinOp):
            return dependencies_known(node.left, known_names) and dependencies_known(node.right, known_names)
        if isinstance(node, ast.BoolOp):
            return all(dependencies_known(x, known_names) for x in node.values)
        if isinstance(node, ast.Compare):
            return dependencies_known(node.left, known_names) and all(
                dependencies_known(x, known_names) for x in node.comparators
            )
        if isinstance(node, ast.IfExp):
            return (
                dependencies_known(node.test, known_names)
                and dependencies_known(node.body, known_names)
                and dependencies_known(node.orelse, known_names)
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in callee_dtypes:
                return all(dependencies_known(x, known_names) for x in node.args)
            if name in _RUNTIME_INPUT_CALLS:
                return bool(
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in input_specs
                    and all(dependencies_known(x, known_names) for x in node.args[1:])
                )
            if name in _NATIVE_INTRINSICS | _MATH_CALLS:
                return all(dependencies_known(x, known_names) for x in node.args)
            return False
        return False

    assignments: list[tuple[str, ast.AST]] = []
    for stmt in ast.walk(cell.function):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            assignments.append((stmt.targets[0].id, stmt.value))
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            assignments.append((stmt.target.id, stmt.value))

    known_names = set(parameter_types)
    trusted_integer: set[str] = set()
    # Iterate because local expressions can depend on previously inferred locals
    # (SPIA's ``t`` depends on ``step``).  Candidate dtypes still come entirely
    # from ``infer_local_types``; this loop only proves that no dependency is
    # untyped/opaque.
    for _ in range(max(1, len(assignments) + 1)):
        changed = False
        for name, expr in assignments:
            if name in trusted_integer or local_types.get(name) != "int64":
                continue
            if dependencies_known(expr, known_names):
                trusted_integer.add(name)
                known_names.add(name)
                changed = True
        if not changed:
            break
    integer_names = tuple(sorted(trusted_integer))
    parameter_domains = {
        name: numeric_domain(dtype, f"native_parameter_type:{cell.uid}:{name}")
        for name, dtype in parameter_types.items()
        if dtype in _ALLOWED_DTYPES
    }
    local_value_domains = _local_domains(
        cell.function,
        parameter_domains=parameter_domains,
        result_domains=result_domains,
        input_specs=input_specs,
        all_uids=set(result_domains),
        module_refs=dict(module_refs or {}),
    )
    assignment_counts: dict[str, int] = {}
    augmented_names: set[str] = set()
    for node in ast.walk(cell.function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignment_counts[node.targets[0].id] = assignment_counts.get(node.targets[0].id, 0) + 1
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            assignment_counts[node.target.id] = assignment_counts.get(node.target.id, 0) + 1
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            augmented_names.add(node.target.id)
    positive_domains: list[tuple[str, tuple[int, ...]]] = []
    for name in integer_names:
        # Exact finite-value domains are authoritative for stable source locals.
        # Loop-carried/multiply-assigned locals may accumulate a partial fixed-
        # point value set and therefore must not become positivity evidence.
        if assignment_counts.get(name, 0) != 1 or name in augmented_names:
            continue
        domain = local_value_domains.get(name)
        if not (
            domain is not None
            and domain.proven
            and domain.numeric_dtype == "int64"
            and domain.exact_values
            and all(type(value) is int and value > 0 for value in domain.exact_values)
        ):
            continue
        values = tuple(sorted({int(value) for value in domain.exact_values}))
        positive_domains.append((name, values))
    positive_names = tuple(name for name, _values in positive_domains)
    parameter_intervals = dict(parameter_int64_intervals or {})
    for name, dtype in parameter_types.items():
        if dtype == "int64" and name not in parameter_intervals:
            parameter_intervals[name] = Int64Interval.typed(
                f"native_int64_parameter_type:{cell.uid}:{name}"
            )
    referenced_call_names = {
        node.func.id
        for node in ast.walk(cell.function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    call_intervals: dict[str, Int64Interval] = {}
    for uid, dtype in callee_dtypes.items():
        if dtype != "int64" or uid not in referenced_call_names:
            continue
        domain = result_domains[uid]
        exact = tuple(
            int(value) for value in domain.exact_values
            if type(value) is int and INT64_MIN <= int(value) <= INT64_MAX
        )
        if exact and len(exact) == len(domain.exact_values):
            call_intervals[uid] = Int64Interval(
                min(exact), max(exact),
                (f"authoritative_exact_native_result_domain:{uid}",),
            )
        else:
            call_intervals[uid] = Int64Interval.typed(
                f"native_int64_result_type:{uid}"
            )
    referenced_input_keys = set(_input_keys(cell))
    input_intervals: dict[str, Int64Interval] = {}
    for key, spec in input_specs.items():
        if getattr(spec, "dtype", None) != "int64" or str(key) not in referenced_input_keys:
            continue
        exact = getattr(spec, "prepared_exact_int_value", None)
        authority = getattr(spec, "prepared_exact_int_authority", None)
        if type(exact) is int and INT64_MIN <= exact <= INT64_MAX and authority:
            input_intervals[str(key)] = Int64Interval.exact(
                int(exact), f"prepared_exact_int_input:{key}:{authority}"
            )
        else:
            input_intervals[str(key)] = Int64Interval.typed(
                f"native_int64_input_type:{key}"
            )
    local_intervals = derive_local_int64_intervals(
        cell.function,
        parameter_intervals=parameter_intervals,
        call_intervals=call_intervals,
        input_intervals=input_intervals,
    )
    # The mature Cython expression inference remains the type authority.  The
    # interval layer may refine only locals already proven int64 by that path.
    local_intervals = {
        name: domain for name, domain in local_intervals.items()
        if name in trusted_integer
    }
    evidence_source = "cython_expression.infer_local_types+modelx_cython_native_result_types"
    if positive_domains:
        evidence_source += "+static_type_domain.finite_exact_value_composition"
    return replace(
        cell,
        local_scalar_dtypes=local_scalar_dtypes,
        local_integer_names=integer_names,
        local_positive_int_names=positive_names,
        local_positive_int_domains=tuple(positive_domains),
        parameter_int64_intervals=interval_rows(parameter_intervals),
        local_int64_intervals=interval_rows(local_intervals),
        callee_int64_intervals=interval_rows(call_intervals),
        input_int64_intervals=interval_rows(input_intervals),
        local_control_evidence_source=evidence_source,
    )


def _tree_parts(cell: NativeFormulaCell) -> tuple[ast.AST, ...]:
    if cell.ranked_function is not None and cell.logical_result_ndim == 1:
        return (cell.ranked_function,)
    if cell.reduction is not None:
        r = cell.reduction
        return (r.init, r.body_expr, *r.range_args, *r.filters)
    if cell.function is not None:
        return (cell.function,)
    return ()


def _input_keys(cell: NativeFormulaCell) -> tuple[str, ...]:
    keys: set[str] = set()
    for tree in _tree_parts(cell):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _RUNTIME_INPUT_CALLS
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(str(node.args[0].value))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                keyed_intrinsics = {
                    "__sparse_table_lookup__": 2,
                    "__prepared_point_lookup__": 2,
                    "__step_lookup_1d__": 2,
                    "__interval_lookup_1d__": 3,
                    "__interp_lookup_1d__": 2,
                }
                count = keyed_intrinsics.get(node.func.id)
                if count is not None and len(node.args) >= count and all(
                    isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    for arg in node.args[:count]
                ):
                    keys.update(str(arg.value) for arg in node.args[:count])
    return tuple(sorted(keys))


def _exception_category(exc: Exception) -> str:
    text = str(exc).lower()
    if "space" in text or "itemspace" in text or "context" in text:
        return "context_space"
    if "table" in text or "dataframe" in text or "point field" in text or "series" in text:
        return "provider"
    if "dtype" in text or "type" in text:
        return "type"
    return "provider"


def _residual_blockers(
    cell: NativeFormulaCell,
    *,
    all_uids: set[str],
    module_refs: Mapping[str, str],
    prepared_exact_int_bounds: Mapping[str, int] | None = None,
) -> tuple[tuple[str, str], ...]:
    blockers: set[tuple[str, str]] = set()
    if cell.execution_role != "formula":
        return ()
    supported_for_ids: frozenset[int] = frozenset()
    supported_range_ids: frozenset[int] = frozenset()
    supported_filtered_min_for_ids: frozenset[int] = frozenset()
    supported_filtered_add_for_ids: frozenset[int] = frozenset()
    supported_filtered_add_continue_ids: frozenset[int] = frozenset()
    supported_while_ids: frozenset[int] = frozenset()
    supported_while_update_ids: frozenset[int] = frozenset()
    pending_while_ids: frozenset[int] = frozenset()
    pending_while_update_ids: frozenset[int] = frozenset()
    if cell.function is not None:
        prepared_exact_int_bounds = prepared_exact_int_bounds or {}
        integer_names = {
            name for name, dtype in zip(cell.parameters, cell.parameter_dtypes)
            if dtype == "int64"
        } | set(cell.local_integer_names)
        proof_name_intervals = intervals_from_rows(
            (*cell.parameter_int64_intervals, *cell.local_int64_intervals)
        )
        proof_call_intervals = intervals_from_rows(cell.callee_int64_intervals)
        proof_input_intervals = intervals_from_rows(cell.input_int64_intervals)
        scalar_dtypes = dict(zip(cell.parameters, cell.parameter_dtypes))
        scalar_dtypes.update(dict(cell.local_scalar_dtypes))
        positive_int_names = set(cell.local_positive_int_names)
        supported_filtered_min_for_ids = supported_bounded_filtered_min_for_ids(
            cell.function, integer_names=integer_names
        )
        supported_filtered_add_for_ids = supported_bounded_filtered_add_for_ids(
            cell.function,
            integer_names=integer_names,
            name_intervals=proof_name_intervals,
            call_intervals=proof_call_intervals,
            input_intervals=proof_input_intervals,
            scalar_dtypes=scalar_dtypes,
        )
        supported_add_for_ids = supported_bounded_add_for_ids(
            cell.function,
            prepared_exact_int_bounds=prepared_exact_int_bounds,
        )
        supported_filtered_add_continue_ids = continue_stmt_ids_for_supported_bounded_filtered_add(
            cell.function,
            integer_names=integer_names,
            name_intervals=proof_name_intervals,
            call_intervals=proof_call_intervals,
            input_intervals=proof_input_intervals,
            scalar_dtypes=scalar_dtypes,
        )
        prepared_mapping_for_ids = frozenset(
            id(node) for node in ast.walk(cell.function)
            if isinstance(node, ast.For)
            and bool(getattr(node, "_mx_prepared_mapping_range", False))
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and len(node.iter.args) == 2
        )
        prepared_mapping_range_ids = frozenset(
            id(node.iter) for node in ast.walk(cell.function)
            if isinstance(node, ast.For) and id(node) in prepared_mapping_for_ids
        )
        supported_for_ids = (
            prepared_mapping_for_ids
            | supported_first_hit_for_ids(cell.function)
            | supported_fixed_count_for_ids(
                cell.function,
                prepared_exact_int_bounds=prepared_exact_int_bounds,
            )
            | supported_filtered_min_for_ids
            | supported_filtered_add_for_ids
            | supported_add_for_ids
        )
        supported_range_ids = (
            prepared_mapping_range_ids
            | range_call_ids_for_supported_first_hit(cell.function)
            | range_call_ids_for_supported_fixed_count(
                cell.function,
                prepared_exact_int_bounds=prepared_exact_int_bounds,
            )
            | range_call_ids_for_supported_bounded_filtered_min(
                cell.function, integer_names=integer_names
            )
            | range_call_ids_for_supported_bounded_filtered_add(
                cell.function,
                integer_names=integer_names,
                name_intervals=proof_name_intervals,
                call_intervals=proof_call_intervals,
                input_intervals=proof_input_intervals,
                scalar_dtypes=scalar_dtypes,
            )
            | range_call_ids_for_supported_bounded_add(
                cell.function,
                prepared_exact_int_bounds=prepared_exact_int_bounds,
            )
        )
        supported_while_ids = supported_bounded_monotone_while_ids(
            cell.function,
            integer_names=integer_names,
            positive_int_names=positive_int_names,
        )
        supported_while_update_ids = update_stmt_ids_for_supported_bounded_monotone(
            cell.function,
            integer_names=integer_names,
            positive_int_names=positive_int_names,
        )
        pending = bounded_monotone_positive_evidence_requirements_by_while_id(
            cell.function, integer_names=integer_names
        )
        pending = {
            while_id: names
            for while_id, names in pending.items()
            if while_id not in supported_while_ids
        }
        pending_while_ids = frozenset(pending)
        pending_while_update_ids = frozenset(
            id(node.body[1])
            for node in cell.function.body
            if isinstance(node, ast.While) and id(node) in pending and len(node.body) == 2
        )
        for names in pending.values():
            for name in names:
                blockers.add(("control_evidence", f"positive_int_local:{name}"))
        allowed_stmt = (ast.FunctionDef, ast.Expr, ast.Assign, ast.AnnAssign, ast.Return, ast.If, ast.Pass)
        for node in ast.walk(cell.function):
            if isinstance(node, ast.For) and id(node) in supported_for_ids:
                continue
            if isinstance(node, ast.While) and id(node) in supported_while_ids:
                continue
            if isinstance(node, ast.While) and id(node) in pending_while_ids:
                continue
            if isinstance(node, (ast.AugAssign, ast.Assign)) and id(node) in supported_while_update_ids:
                continue
            if isinstance(node, (ast.AugAssign, ast.Assign)) and id(node) in pending_while_update_ids:
                continue
            if isinstance(node, ast.Continue) and id(node) in supported_filtered_add_continue_ids:
                continue
            if isinstance(node, ast.stmt) and not isinstance(node, allowed_stmt):
                blockers.add(("control_syntax", type(node).__name__))
    for tree in _tree_parts(cell):
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                blockers.add(("provider", "residual_subscript:" + ast.unparse(node)))
            elif isinstance(node, ast.Call):
                if id(node) in supported_range_ids:
                    continue
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                    if name in all_uids or name in _RUNTIME_INPUT_CALLS or name in _NATIVE_INTRINSICS:
                        continue
                    blockers.add(("provider", "residual_call:" + ast.unparse(node)))
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    receiver = node.func.value.id
                    module = module_refs.get(receiver)
                    if module in {"math", "numpy"} and node.func.attr in _MATH_CALLS:
                        continue
                    blockers.add(("provider", "residual_attribute_call:" + ast.unparse(node)))
                else:
                    blockers.add(("provider", "residual_dynamic_call:" + ast.unparse(node)))
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                blockers.add(("control_syntax", type(node).__name__))
    return tuple(sorted(blockers))


_ROUND_COMPAT_MIN_DIGITS = -22
_ROUND_COMPAT_MAX_DIGITS = 22


class _SourceTransparentDTypeView:
    """Minimal typed view used only to prove compatibility-intrinsic operands."""

    def __init__(
        self,
        *,
        cell_dtypes: Mapping[str, str],
        input_specs: Mapping[str, OptimizedInputSpec],
        module_refs: Mapping[str, str],
    ):
        self.v = {uid: type("_Cell", (), {"dtype": dtype})() for uid, dtype in cell_dtypes.items()}
        self.input_specs = input_specs
        self.module_refs = module_refs


def _source_round_compat_blockers(
    cell: NativeFormulaCell,
    node: ast.Call,
    *,
    cell_dtypes: Mapping[str, str],
    input_specs: Mapping[str, OptimizedInputSpec],
    module_refs: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    """Prove the narrow native float ``round(x, ndigits)`` compatibility domain.

    Cython owns the surrounding formula.  This proof exists only because Cython
    3.2 cannot lower Python's two-argument float round in ``nogil`` code even
    after both operands are typed.  Admission therefore requires a proven
    float64 value and signed-int64 precision.  When a nontrivial complete
    precision interval is available it must fit the exact helper's [-22, 22]
    domain; otherwise the emitted helper enforces that domain with a fail-closed
    runtime guard.  The guard is execution authority, never sampled evidence.
    """

    if node.keywords or len(node.args) not in {1, 2}:
        return (("python_semantics", "round_signature_requires_python"),)
    # Windows uses the compiled portable fixed-width helper; the same typed
    # domain proof and runtime guards apply on both compiler families.
    if len(node.args) == 1:
        # Existing one-argument behavior is unchanged in this stage.
        return ()

    env = dict(zip(cell.parameters, cell.parameter_dtypes))
    env.update(dict(cell.local_scalar_dtypes))
    dtype_view = _SourceTransparentDTypeView(
        cell_dtypes=cell_dtypes, input_specs=input_specs, module_refs=module_refs
    )
    value_dtype = infer_expression_dtype(dtype_view, node.args[0], env)
    if value_dtype != "float64":
        return (("python_semantics", f"round_value_requires_float64:{value_dtype}"),)

    ndigits_dtype = infer_expression_dtype(dtype_view, node.args[1], env)
    if ndigits_dtype != "int64":
        return (("python_semantics", f"round_ndigits_requires_int64:{ndigits_dtype}"),)

    name_intervals = intervals_from_rows(
        (*cell.parameter_int64_intervals, *cell.local_int64_intervals)
    )
    ndigits = prove_int64_expression(
        node.args[1],
        name_intervals=name_intervals,
        call_intervals=intervals_from_rows(cell.callee_int64_intervals),
        input_intervals=intervals_from_rows(cell.input_int64_intervals),
    )
    if ndigits is None:
        # Type authority is sufficient for code generation because the helper
        # carries its own exact [-22, 22] runtime guard.  Unknown precision
        # ranges therefore fail closed at execution rather than being guessed.
        return ()
    if ndigits.lower == INT64_MIN and ndigits.upper == INT64_MAX:
        # A typed-only interval is not a range proof.  Keep the compatibility
        # helper's runtime guard as the authority instead of demoting a Cell
        # whose complete value range is not otherwise needed by the compiler.
        return ()
    if (
        ndigits.lower < _ROUND_COMPAT_MIN_DIGITS
        or ndigits.upper > _ROUND_COMPAT_MAX_DIGITS
    ):
        return ((
            "python_semantics",
            f"round_ndigits_outside_native_compat:{ndigits.lower}:{ndigits.upper}",
        ),)
    return ()



def _literal_range_values(node: ast.AST, *, cap: int = 64) -> tuple[int, ...] | None:
    """Return an exact small source-literal ``range`` domain, or ``None``.

    This helper is intentionally source-authoritative.  It does not use observed
    trace extrema to invent a finite generator domain.
    """

    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and not node.keywords
        and 1 <= len(node.args) <= 3
    ):
        return None
    values: list[int] = []
    for arg in node.args:
        sign = 1
        value = arg
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
            sign = -1 if isinstance(value.op, ast.USub) else 1
            value = value.operand
        if not (isinstance(value, ast.Constant) and type(value.value) is int):
            return None
        values.append(sign * int(value.value))
    try:
        if len(values) == 1:
            domain = tuple(range(values[0]))
        elif len(values) == 2:
            domain = tuple(range(values[0], values[1]))
        else:
            domain = tuple(range(values[0], values[1], values[2]))
    except (ValueError, OverflowError):
        return None
    if not domain or len(domain) > cap:
        return None
    return domain


def _substitute_bound_name(node: ast.AST, name: str, value: int) -> ast.AST:
    return ast.fix_missing_locations(
        _NameSubstitute(name, ast.Constant(value)).visit(copy.deepcopy(node))
    )


def _prove_int_predicate_true(
    node: ast.AST,
    *,
    name_intervals: Mapping[str, Int64Interval],
    call_intervals: Mapping[str, Int64Interval],
    input_intervals: Mapping[str, Int64Interval],
) -> bool:
    """Prove a narrow integer predicate true on the complete native domain."""

    if isinstance(node, ast.Constant) and type(node.value) is bool:
        return bool(node.value)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(
                _prove_int_predicate_true(
                    value,
                    name_intervals=name_intervals,
                    call_intervals=call_intervals,
                    input_intervals=input_intervals,
                )
                for value in node.values
            )
        if isinstance(node.op, ast.Or):
            return any(
                _prove_int_predicate_true(
                    value,
                    name_intervals=name_intervals,
                    call_intervals=call_intervals,
                    input_intervals=input_intervals,
                )
                for value in node.values
            )
        return False
    if not (isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1):
        return False
    left = prove_int64_expression(
        node.left,
        name_intervals=name_intervals,
        call_intervals=call_intervals,
        input_intervals=input_intervals,
    )
    right = prove_int64_expression(
        node.comparators[0],
        name_intervals=name_intervals,
        call_intervals=call_intervals,
        input_intervals=input_intervals,
    )
    if left is None or right is None:
        return False
    op = node.ops[0]
    if isinstance(op, ast.GtE):
        return left.lower >= right.upper
    if isinstance(op, ast.Gt):
        return left.lower > right.upper
    if isinstance(op, ast.LtE):
        return left.upper <= right.lower
    if isinstance(op, ast.Lt):
        return left.upper < right.lower
    if isinstance(op, ast.Eq):
        return left.lower == left.upper == right.lower == right.upper
    if isinstance(op, ast.NotEq):
        return left.upper < right.lower or right.upper < left.lower
    return False


def _native_generator_reduction_lowering(cell: NativeFormulaCell) -> NativeFormulaCell:
    """Lower Cython-unsupported bounded generator ``any``/``max`` returns.

    Cython 3.2 can compile ordinary scalar ``min``/``max`` in ``nogil`` code but
    generator expressions allocate Python generator state and therefore require
    the GIL.  Architecture-B keeps Cython as language authority and fills only
    this demonstrated gap:

    * ``any(expr for i in range(...) if ...)`` becomes a short-circuit native
      ``for`` loop.  Empty-domain semantics are naturally ``False``.
    * ``max(expr for i in range(<small source-literal domain>) if ...)`` is
      finite-unrolled only when the *first* source-range element is proven to
      pass every filter on the complete int64 parameter domain.  That proves
      non-emptiness and preserves evaluation order and Python's first-value max
      tie/NaN behavior.  Otherwise the formula is left untouched and fails
      closed to detached P execution.

    Only direct ``return`` reductions are lowered.  This keeps control/evaluation
    order obvious and avoids manufacturing a general comprehension engine.
    """

    if cell.function is None or cell.execution_role != "formula":
        return cell
    name_intervals = intervals_from_rows(
        (*cell.parameter_int64_intervals, *cell.local_int64_intervals)
    )
    call_intervals = intervals_from_rows(cell.callee_int64_intervals)
    input_intervals = intervals_from_rows(cell.input_int64_intervals)
    used_names = {
        node.id for node in ast.walk(cell.function) if isinstance(node, ast.Name)
    } | set(cell.parameters)
    counter = [0]

    def fresh(prefix: str) -> str:
        while True:
            counter[0] += 1
            name = f"__mx_gen_{prefix}_{counter[0]}"
            if name not in used_names:
                used_names.add(name)
                return name

    def reduction(node: ast.AST):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"any", "max"}
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.GeneratorExp)
            and len(node.args[0].generators) == 1
        ):
            return None
        gen = node.args[0]
        comp = gen.generators[0]
        if comp.is_async or not isinstance(comp.target, ast.Name):
            return None
        if not (
            isinstance(comp.iter, ast.Call)
            and isinstance(comp.iter.func, ast.Name)
            and comp.iter.func.id == "range"
            and not comp.iter.keywords
            and 1 <= len(comp.iter.args) <= 3
        ):
            return None
        # ``any`` has correct empty-range semantics after direct loop lowering,
        # so Cython's final typed ``nogil`` compile remains the authority for
        # dynamic range bounds. ``max`` needs the stronger source/domain proof
        # below because empty generators raise in Python and cannot be silently
        # assigned a native sentinel.
        if node.func.id == "max" and any(
            prove_int64_expression(
                arg,
                name_intervals=name_intervals,
                call_intervals=call_intervals,
                input_intervals=input_intervals,
            ) is None
            for arg in comp.iter.args
        ):
            return None
        return node.func.id, gen, comp

    def lower_return(stmt: ast.Return) -> list[ast.stmt] | None:
        info = reduction(stmt.value)
        if info is None:
            return None
        kind, gen, comp = info
        binder = comp.target.id
        if kind == "any":
            truth = ast.If(
                test=copy.deepcopy(gen.elt),
                body=[ast.Return(ast.Constant(True))],
                orelse=[],
            )
            body: list[ast.stmt] = [truth]
            for condition in reversed(comp.ifs):
                body = [ast.If(test=copy.deepcopy(condition), body=body, orelse=[])]
            loop = ast.For(
                target=copy.deepcopy(comp.target),
                iter=copy.deepcopy(comp.iter),
                body=body,
                orelse=[],
                type_comment=None,
            )
            return [
                ast.copy_location(loop, stmt),
                ast.copy_location(ast.Return(ast.Constant(False)), stmt),
            ]

        domain = _literal_range_values(comp.iter)
        if domain is None:
            return None
        first = domain[0]
        first_filters = [
            _substitute_bound_name(condition, binder, first) for condition in comp.ifs
        ]
        if not all(
            _prove_int_predicate_true(
                condition,
                name_intervals=name_intervals,
                call_intervals=call_intervals,
                input_intervals=input_intervals,
            )
            for condition in first_filters
        ):
            return None
        acc = fresh("max")
        tmp = fresh("value")
        out: list[ast.stmt] = [
            ast.Assign(
                targets=[ast.Name(acc, ast.Store())],
                value=_substitute_bound_name(gen.elt, binder, first),
            )
        ]
        for value in domain[1:]:
            filters = [_substitute_bound_name(condition, binder, value) for condition in comp.ifs]
            test: ast.AST
            if not filters:
                test = ast.Constant(True)
            elif len(filters) == 1:
                test = filters[0]
            else:
                test = ast.BoolOp(ast.And(), filters)
            candidate = _substitute_bound_name(gen.elt, binder, value)
            update = [
                ast.Assign([ast.Name(tmp, ast.Store())], candidate),
                ast.If(
                    ast.Compare(ast.Name(tmp, ast.Load()), [ast.Gt()], [ast.Name(acc, ast.Load())]),
                    [ast.Assign([ast.Name(acc, ast.Store())], ast.Name(tmp, ast.Load()))],
                    [],
                ),
            ]
            out.append(ast.If(test=test, body=update, orelse=[]))
        out.append(ast.Return(ast.Name(acc, ast.Load())))
        return [ast.copy_location(row, stmt) for row in out]

    def lower_block(statements: Sequence[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in statements:
            if isinstance(stmt, ast.Return):
                lowered = lower_return(stmt)
                out.extend(lowered if lowered is not None else [copy.deepcopy(stmt)])
                continue
            row = copy.deepcopy(stmt)
            if isinstance(row, ast.If):
                row.body = lower_block(row.body)
                row.orelse = lower_block(row.orelse)
            elif isinstance(row, (ast.For, ast.While)):
                row.body = lower_block(row.body)
                row.orelse = lower_block(row.orelse)
            out.append(row)
        return out

    transformed = copy.deepcopy(cell.function)
    transformed.body = lower_block(transformed.body)
    transformed = ast.fix_missing_locations(transformed)
    if ast.dump(transformed, include_attributes=False) == ast.dump(cell.function, include_attributes=False):
        return cell
    return replace(cell, function=transformed)

def _source_transparent_residual_blockers(
    cell: NativeFormulaCell,
    *,
    all_uids: set[str],
    module_refs: Mapping[str, str],
    cell_dtypes: Mapping[str, str],
    input_specs: Mapping[str, OptimizedInputSpec],
) -> tuple[tuple[str, str], ...]:
    """Stage-1B residual screening that ignores ordinary Python syntax.

    The historical emitter needs a statement/control whitelist because it must
    reproduce those statements itself.  The source-preserving backend does not.
    Its pre-Cython screening therefore asks only whether compiler-owned/modelx
    leaves have been resolved to a native provider ABI.  ``range`` is an
    ordinary Cython/Python language builtin and is deliberately allowed without
    a bespoke loop recognizer.

    This is *candidate* screening, not the final N0 language validator.  Cython
    still has authority to reject comprehensions, object allocations, dynamic
    local constructs, or any other syntax that cannot compile ``nogil``.
    """

    blockers: set[tuple[str, str]] = set()
    if cell.execution_role != "formula":
        return ()
    ordinary_cython_builtins = {"range"}
    for tree in _tree_parts(cell):
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                # Native model/provider subscripts should have been normalized
                # into prepared-input intrinsics before this point.  A residual
                # subscript therefore still represents an unresolved provider
                # or Python-object dependency rather than mere control syntax.
                blockers.add(("provider", "residual_subscript:" + ast.unparse(node)))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                    # Python's two-argument float round is a demonstrated
                    # Cython 3.2 ``nogil`` gap.  Admit only the exact, explicitly
                    # proven compatibility domain; Cython still owns every
                    # surrounding expression and control-flow construct.
                    if name == "round":
                        round_blockers = _source_round_compat_blockers(
                            cell, node, cell_dtypes=cell_dtypes,
                            input_specs=input_specs, module_refs=module_refs,
                        )
                        if round_blockers:
                            blockers.update(round_blockers)
                            continue
                    if (
                        name in all_uids
                        or name in _RUNTIME_INPUT_CALLS
                        or name in _NATIVE_INTRINSICS
                        or name in ordinary_cython_builtins
                    ):
                        continue
                    blockers.add(("provider", "residual_call:" + ast.unparse(node)))
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    receiver = node.func.value.id
                    module = module_refs.get(receiver)
                    if module in {"math", "numpy"} and node.func.attr in _MATH_CALLS:
                        continue
                    blockers.add(("provider", "residual_attribute_call:" + ast.unparse(node)))
                else:
                    blockers.add(("provider", "residual_dynamic_call:" + ast.unparse(node)))
    return tuple(sorted(blockers))


def _ranked_source_transparent_residual_blockers(
    cell: NativeFormulaCell,
    *,
    all_uids: set[str],
    module_refs: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    """Screen a preserved rank-1 NumPy formula without scalarizing it.

    The ranked backend intentionally keeps NumPy array construction, slicing,
    transpose and matrix multiplication as Python/Cython object operations.
    Reusing the scalar ``nogil`` residual screen therefore rejects valid ranked
    formulas such as ``np.array(...).transpose() @ disc_factors()``.  Simply
    skipping residual screening is worse: pandas/table helpers then masquerade
    as native ranked formulas and fail much later in generated Cython.

    This validator admits only the small object surface owned by the ranked
    backend itself.  External Python names/methods stay blocked and can be
    handled by the existing prepared-provider / mixed-P machinery.
    """

    fn = cell.ranked_function
    if cell.execution_role != "formula" or fn is None:
        return ()

    blockers: set[tuple[str, str]] = set()
    local_names = {arg.arg for arg in fn.args.args}
    local_names.update(arg.arg for arg in fn.args.posonlyargs)
    local_names.update(arg.arg for arg in fn.args.kwonlyargs)
    if fn.args.vararg is not None:
        local_names.add(fn.args.vararg.arg)
    if fn.args.kwarg is not None:
        local_names.add(fn.args.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local_names.add(node.id)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for gen in node.generators:
                for target in ast.walk(gen.target):
                    if isinstance(target, ast.Name):
                        local_names.add(target.id)

    ranked_builtins = {
        "range", "list", "tuple", "min", "max", "abs", "round", "int",
        "float", "bool", "len", "sum", "ranked_series",
    }
    allowed_names = (
        set(all_uids)
        | set(_RUNTIME_INPUT_CALLS)
        | set(_NATIVE_INTRINSICS)
        | ranked_builtins
        | local_names
    )
    numpy_aliases = {
        name for name, module in module_refs.items() if module == "numpy"
    }
    allowed_names |= numpy_aliases

    numpy_calls = set(_MATH_CALLS) | {
        "array", "around", "round", "where", "maximum", "minimum",
        "nan_to_num", "isnan", "isinf", "exp", "log", "sqrt", "sin",
        "cos", "floor", "ceil", "zeros", "ones", "full",
    }

    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in allowed_names:
                blockers.add(("provider", "ranked_residual_name:" + node.id))
            continue

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name not in allowed_names:
                    blockers.add(("provider", "ranked_residual_call:" + ast.unparse(node)))
                    continue
                # Category-normalized recursive ABIs use integer codes.  A
                # preserved ranked body that still passes a string literal has
                # not gone through that lowering and must not be admitted as N0.
                if name in all_uids and any(
                    isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    for arg in node.args
                ):
                    blockers.add(("category", "ranked_string_cell_argument:" + ast.unparse(node)))
                continue

            if isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                receiver = node.func.value
                if isinstance(receiver, ast.Name):
                    if receiver.id in numpy_aliases and attr in numpy_calls:
                        continue
                    blockers.add(("provider", "ranked_residual_attribute_call:" + ast.unparse(node)))
                    continue
                # ``np.array(...).transpose()`` is the established ranked PV
                # idiom.  Other dynamic object methods (pandas reindex/set_axis,
                # list append, etc.) remain outside this backend.
                if attr == "transpose":
                    continue
                blockers.add(("provider", "ranked_residual_dynamic_call:" + ast.unparse(node)))
                continue

            blockers.add(("provider", "ranked_residual_dynamic_call:" + ast.unparse(node)))

        elif isinstance(node, ast.Attribute):
            # Calls are handled above.  A bare NumPy attribute used as a call
            # target is safe; arbitrary object attributes such as ``.index`` or
            # ``mort_table.columns`` are unresolved Python providers.
            if isinstance(node.ctx, ast.Load):
                if isinstance(node.value, ast.Name) and node.value.id in numpy_aliases:
                    if node.attr in numpy_calls:
                        continue
                # Dynamic-call nodes are also diagnosed above.  Recognize the
                # one admitted bare method attribute directly and otherwise
                # retain this additional provider diagnostic; duplicate detail
                # is preferable to silently admitting an object boundary.
                if node.attr != "transpose":
                    text = ast.unparse(node)
                    if not text.startswith("np."):
                        blockers.add(("provider", "ranked_residual_attribute:" + text))

        elif isinstance(node, ast.Subscript):
            base = node.value
            if isinstance(base, ast.Call) and isinstance(base.func, ast.Name):
                if base.func.id in all_uids | _RUNTIME_INPUT_CALLS | _NATIVE_INTRINSICS:
                    continue
            if isinstance(base, ast.Name) and base.id in local_names:
                continue
            blockers.add(("provider", "ranked_residual_subscript:" + ast.unparse(node)))

    return tuple(sorted(blockers))


@dataclass(frozen=True)
class FastParameterEvidence:
    name: str
    dtype: str
    type_source: str
    minimum: int | None = None
    maximum: int | None = None
    range_source: str = ""

    @property
    def dense_range(self) -> tuple[int, int] | None:
        if self.dtype not in {"int64", "bool"}:
            return None
        if self.minimum is None or self.maximum is None:
            return (0, 1) if self.dtype == "bool" else None
        return int(self.minimum), int(self.maximum)

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "type_source": self.type_source,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "range_source": self.range_source,
            "dense_range": None if self.dense_range is None else list(self.dense_range),
        }


@dataclass(frozen=True)
class FastScalarizedVectorHelper:
    """One modelx-cython array Cell scalarized into a memoized element function.

    The element dtype comes from modelx-cython runtime tracing.  This is a local
    performance rewrite, not a new type inference system: the original exported
    vector Cell remains catalogued as a materialized/object helper.
    """

    uid: str
    source_uid: str
    source_name: str
    parameter: str
    dtype: str
    dense_range: tuple[int, int] | None
    function: ast.FunctionDef = field(repr=False, compare=False)
    raw_element: ast.AST = field(repr=False, compare=False)

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_uid": self.source_uid,
            "source_name": self.source_name,
            "parameter": self.parameter,
            "dtype": self.dtype,
            "dense_range": None if self.dense_range is None else list(self.dense_range),
            "rewrite": "memoized_vector_element",
        }


@dataclass(frozen=True)
class FastCrossSpaceCall:
    """One metadata-resolved native call crossing a modelx Space boundary.

    Identity is taken from modelx-cython's ``CellsResolver``/``ModuleInfo``
    catalogue during preparation.  Generated code sees only ``callee_uid``.
    """

    caller_uid: str
    caller_class_fullname: str
    receiver_chain: tuple[str, ...]
    callee_uid: str
    callee_class_fullname: str
    callee_cell_fullname: str
    target_space_parameters: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "caller_uid": self.caller_uid,
            "caller_class_fullname": self.caller_class_fullname,
            "receiver_chain": list(self.receiver_chain),
            "callee_uid": self.callee_uid,
            "callee_class_fullname": self.callee_class_fullname,
            "callee_cell_fullname": self.callee_cell_fullname,
            "target_space_parameters": list(self.target_space_parameters),
            "identity_authority": "modelx_cython.CellsResolver+ModuleInfo",
            "runtime_call_form": "canonical_native_uid",
        }


@dataclass(frozen=True)
class FastRankedOutput:
    """Trace-authoritative logical output geometry for one selected Cell."""

    dtype: str
    ndim: int
    shape: tuple[int, ...]
    axis_keys: tuple[Any, ...]
    point_frame_ref: str | None
    authority: str = "modelx_cython.ReturnTypeInfo+RuntimeCellsInfo.array_shape"

    def manifest(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "ndim": self.ndim,
            "shape": list(self.shape),
            "axis_key_count": len(self.axis_keys),
            "point_frame_ref": self.point_frame_ref,
            "authority": self.authority,
            "layout": "rank1_contiguous_point_axis",
        }


@dataclass(frozen=True)
class FastFrontendCell:
    uid: str
    source_name: str
    source_fullname: str
    status: str  # native_formula | materialized_helper | object_fallback
    result_dtype: str
    result_type_source: str
    parameters: tuple[FastParameterEvidence, ...]
    blocker_items: tuple[tuple[str, str], ...] = ()
    provider_input_keys: tuple[str, ...] = ()
    result_category_domain: str | None = None
    parameter_category_domains: tuple[tuple[str, str | None], ...] = ()
    result_ndim: int = 0
    result_shape: tuple[int, ...] | None = None
    result_shape_source: str = "scalar"
    local_scalar_dtypes: tuple[tuple[str, str], ...] = ()
    local_integer_names: tuple[str, ...] = ()
    local_positive_int_names: tuple[str, ...] = ()
    local_positive_int_domains: tuple[tuple[str, tuple[int, ...]], ...] = ()
    parameter_int64_intervals: tuple[tuple[str, int, int, str], ...] = ()
    local_int64_intervals: tuple[tuple[str, int, int, str], ...] = ()
    callee_int64_intervals: tuple[tuple[str, int, int, str], ...] = ()
    input_int64_intervals: tuple[tuple[str, int, int, str], ...] = ()
    local_control_evidence_source: str = ""
    execution_class: str = ""
    execution_class_reasons: tuple[str, ...] = ()
    bridgeable_python_fallback: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_name": self.source_name,
            "source_fullname": self.source_fullname,
            "status": self.status,
            "result_dtype": self.result_dtype,
            "result_type_source": self.result_type_source,
            "parameters": [x.manifest() for x in self.parameters],
            "provider_input_keys": list(self.provider_input_keys),
            "result_category_domain": self.result_category_domain,
            "parameter_category_domains": {
                name: domain for name, domain in self.parameter_category_domains if domain is not None
            },
            "result_ndim": self.result_ndim,
            "result_shape": None if self.result_shape is None else list(self.result_shape),
            "result_shape_source": self.result_shape_source,
            "local_control_evidence": {
                "local_scalar_dtypes": dict(self.local_scalar_dtypes),
                "integer_names": list(self.local_integer_names),
                "positive_int_names": list(self.local_positive_int_names),
                "positive_int_domains": {
                    name: list(values) for name, values in self.local_positive_int_domains
                },
                "parameter_int64_intervals": {
                    name: {"lower": lower, "upper": upper, "authority": authority}
                    for name, lower, upper, authority in self.parameter_int64_intervals
                },
                "local_int64_intervals": {
                    name: {"lower": lower, "upper": upper, "authority": authority}
                    for name, lower, upper, authority in self.local_int64_intervals
                },
                "callee_int64_intervals": {
                    name: {"lower": lower, "upper": upper, "authority": authority}
                    for name, lower, upper, authority in self.callee_int64_intervals
                },
                "input_int64_intervals": {
                    name: {"lower": lower, "upper": upper, "authority": authority}
                    for name, lower, upper, authority in self.input_int64_intervals
                },
                "source": self.local_control_evidence_source,
            },
            "fallback_reasons": [
                {"kind": kind, "detail": detail} for kind, detail in self.blocker_items
            ],
            "execution_class": self.execution_class or None,
            "execution_class_reasons": list(self.execution_class_reasons),
            "bridgeable_python_fallback": self.bridgeable_python_fallback,
        }


@dataclass
class ModelxCythonFastPlan:
    schema: str
    module_name: str
    class_name: str
    output_name: str
    output_uid: str
    run_keys: tuple[Any, ...]
    cells: tuple[FastFrontendCell, ...]
    native_program: NativeFormulaProgram | None
    parameter_ranges: Mapping[str, tuple[tuple[int, int] | None, ...]]
    source_preserving_native_program: NativeFormulaProgram | None = None
    scalarized_vector_helpers: tuple[FastScalarizedVectorHelper, ...] = ()
    finite_vector_element_helpers: tuple[FiniteVectorElementHelper, ...] = ()
    runtime_inputs: Mapping[str, InputSpec] = field(default_factory=dict, repr=False)
    source_preserving_runtime_inputs: Mapping[str, InputSpec] = field(default_factory=dict, repr=False)
    source_preserving_candidate_blockers: Mapping[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    formula_lexical_sources: Mapping[str, str] = field(default_factory=dict, repr=False)
    # Dev136: normalized source functions before provider/category lowering.
    # These preserve ordinary Python formula syntax while binding modelx Cell
    # calls to canonical UIDs.  Mixed execution may give such a formula back to
    # Cython under the GIL while retaining the core r_<uid>() cache graph.
    formula_source_functions: Mapping[str, ast.FunctionDef] = field(default_factory=dict, repr=False)
    normalized_operators: tuple[NormalizedOperator, ...] = ()
    semantic_formula_ir: Mapping[str, SemanticFormulaIR] = field(default_factory=dict, repr=False)
    category_plan: ModelxCythonCategoryPlan | None = field(default=None, repr=False)
    preparation_timings: Mapping[str, float] = field(default_factory=dict)
    preparation_memory: Mapping[str, Mapping[str, int | None]] = field(default_factory=dict)
    preparation_diagnostics: Mapping[str, object] = field(default_factory=dict)
    module_info: ModuleInfo | None = field(default=None, repr=False)
    cross_space_calls: tuple[FastCrossSpaceCall, ...] = ()
    cross_space_cells: tuple[FastFrontendCell, ...] = ()
    ranked_output: FastRankedOutput | None = None
    semantic_oracle: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    authoritative_evidence: Mapping[str, object] = field(default_factory=dict)
    native_hash_capacity_evidence: Mapping[str, object] = field(default_factory=dict)
    native_hash_capacity_by_uid: Mapping[str, int] = field(default_factory=dict)
    execution_class_plan: ExecutionClassPlan | None = None
    source_transparent_cell_catalog: Mapping[str, NativeFormulaCell] = field(default_factory=dict, repr=False)
    source_transparent_direct_blockers: Mapping[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    source_transparent_module_refs: Mapping[str, str] = field(default_factory=dict, repr=False)

    def build_semantic_facade(self):
        """Return the behavior-preserving Phase-A semantic contract.

        The facade is deliberately derived from, and validated against, the
        existing plan.  Code generation does not consume it yet.
        """
        from .semantic_contract import build_semantic_facade

        return build_semantic_facade(self)

    @property
    def fast_admitted(self) -> bool:
        return self.native_program is not None and self.output_uid in self.native_program.cells

    @property
    def source_preserving_candidate_ready(self) -> bool:
        """Return whether pre-Cython native binding/ABI screening succeeded.

        Stage 1B deliberately does not call this final language admission: the
        generated ``noexcept nogil`` function must still compile in Cython.
        """
        return (
            self.source_preserving_native_program is not None
            and self.output_uid in self.source_preserving_native_program.cells
        )

    @property
    def source_preserving_admitted(self) -> bool:
        """Backward-compatible alias for the Stage-1B candidate-ready state."""
        return self.source_preserving_candidate_ready

    @property
    def mixed_execution_candidate_ready(self) -> bool:
        """Whether the selected output remains callable through the native recursive ABI.

        Stage 2B executes N1 through the same ``r_<cell>()`` ABI as N0. Any
        bridgeable P dependency enters Python only behind its own native memo
        wrapper on a cache miss.
        """
        return bool(
            self.execution_class_plan is not None
            and self.execution_class_plan.mixed_candidate_ready
        )

    @property
    def output_execution_class(self) -> str | None:
        if self.execution_class_plan is None:
            return None
        return self.execution_class_plan.output_class

    def execution_class_cell(self, uid: str):
        if self.execution_class_plan is None:
            raise ModelxCythonFastBackendError("fast plan has no N0/N1/P execution-class plan")
        return self.execution_class_plan.cell(uid)

    @property
    def fallback_cells(self) -> tuple[str, ...]:
        return tuple(x.source_name for x in self.cells if x.status == "object_fallback")

    @property
    def native_cells(self) -> tuple[str, ...]:
        return tuple(x.source_name for x in self.cells if x.status == "native_formula")

    def bind_inputs(self, keys: Sequence[Any] | None = None) -> dict[str, Any]:
        return self._bind_input_specs(self.runtime_inputs, keys)

    def bind_source_preserving_inputs(self, keys: Sequence[Any] | None = None) -> dict[str, Any]:
        """Bind the Stage-1B source-transparent candidate input set."""
        return self._bind_input_specs(self.source_preserving_runtime_inputs, keys)

    def _bind_input_specs(
        self,
        specs: Mapping[str, InputSpec],
        keys: Sequence[Any] | None,
    ) -> dict[str, Any]:
        holder = SimpleNamespace(
            run_keys=self.run_keys,
            inputs=dict(specs),
            validated_static_facts=(),
        )
        point_keys = (
            self.ranked_output.axis_keys
            if keys is None and self.ranked_output is not None
            else self.run_keys if keys is None else tuple(keys)
        )
        if self.ranked_output is None:
            raw = CanonicalModel.raw_inputs(holder, point_keys, None)
        else:
            # A ranked selected output has two independent domains: the modelx
            # invocation domain (normally the singleton ``()`` for vector
            # models) and the result/point axis.  Point inputs bind over the
            # latter, while scalar/global providers must continue to bind over
            # the former.
            raw: dict[str, Any] = {}
            batch_caches: dict[str, dict[Any, Any]] = {"point": {}, "global": {}}
            for name, spec in holder.inputs.items():
                selected = point_keys if spec.scope == "point" else self.run_keys
                if spec.fast_getter is not None:
                    raw[name] = spec.fast_getter(
                        selected, batch_caches.get(spec.scope, batch_caches["global"])
                    )
                else:
                    raw[name] = spec.getter(selected)
        return CanonicalModel.coerce_inputs(holder, raw)

    def build_recursive_program(
        self,
        *,
        max_dense_slots_per_cell: int = 1_000_000,
        native_hash_capacity_by_uid: Mapping[str, int] | None = None,
        native_hash_debug: bool = False,
    ):
        """Build the production recursive physicalization for this fast plan.

        Memo policy is intentionally not configurable here. Production is always
        native-cache + fail-closed. Historical cache-all/direct comparisons must
        use ``build_legacy_comparison_recursive_program(plan.native_program, ...)``
        explicitly so comparison machinery cannot leak back into normal planning.
        """
        if self.native_program is None:
            raise ModelxCythonFastBackendError("fast plan has no admitted native program")
        nan_uids = tuple(self.native_hash_capacity_evidence.get("nan_uids", ()))
        if nan_uids:
            raise ModelxCythonFastBackendError(
                "native exact hash does not admit NaN semantic memo keys: "
                + ",".join(nan_uids)
            )
        from .dense_recursive_native import build_dense_recursive_program
        return build_dense_recursive_program(
            self.native_program,
            parameter_ranges=self.parameter_ranges,
            max_dense_slots_per_cell=max_dense_slots_per_cell,
            native_hash_capacity_by_uid=(
                dict(self.native_hash_capacity_by_uid)
                if native_hash_capacity_by_uid is None else native_hash_capacity_by_uid
            ),
            native_hash_debug=native_hash_debug,
        )

    def build_source_preserving_program(
        self,
        *,
        max_dense_slots_per_cell: int = 1_000_000,
        native_hash_capacity_by_uid: Mapping[str, int] | None = None,
        native_hash_debug: bool = False,
    ):
        """Build the Architecture-B source-preserving formula backend.

        The physical runtime remains the production dense recursive backend.
        Only formula-body generation changes: ordinary Python syntax is left to
        Cython while modelx/provider leaves are rebound to the existing native
        ABI.  Complete prepared-input bounds used by stronger Cython semantics
        are derived from ``run_keys`` and revalidated by generated ``init_inputs``.

        Stage 1B uses the source-transparent candidate program when available,
        so ordinary control syntax does not have to pass the historical emitter
        whitelist before Cython sees it. The historical/current
        ``build_recursive_program`` path remains intact as a differential oracle.
        """
        from .source_preserving_native import build_source_preserving_program_from_fast_plan

        return build_source_preserving_program_from_fast_plan(
            self,
            max_dense_slots_per_cell=max_dense_slots_per_cell,
            native_hash_capacity_by_uid=native_hash_capacity_by_uid,
            native_hash_debug=native_hash_debug,
        )

    def build_mixed_recursive_program(
        self,
        *,
        max_dense_slots_per_cell: int = 1_000_000,
        native_hash_capacity_by_uid: Mapping[str, int] | None = None,
        native_hash_debug: bool = False,
    ):
        """Build mixed recursion with one ``r_<cell>()`` ABI for N0/N1/P Cells.

        Dev131 keeps Python-semantic scalar P formulas inside generated Cython:
        a native ``r_<cell>()`` cache miss acquires the GIL, executes the
        source-preserving formula, and resolves typed Cell dependencies back
        through native ``r_<dep>()`` calls. Provider/object-backed P Cells retain
        the compiler-owned modelx fallback as the conservative second tier.
        """
        from .mixed_recursive_native import build_mixed_recursive_program_from_fast_plan

        return build_mixed_recursive_program_from_fast_plan(
            self,
            max_dense_slots_per_cell=max_dense_slots_per_cell,
            native_hash_capacity_by_uid=native_hash_capacity_by_uid,
            native_hash_debug=native_hash_debug,
        )

    def build_fallback_target_plan(self):
        """Return owner-space resolver metadata for recursive P Cells."""
        from .mixed_recursive_native import build_fallback_target_plan

        return build_fallback_target_plan(self)

    def build_fallback_runtime(self, *, model: Any, space: Any):
        """Create runtime for detached P reference binding and any fallback hooks."""
        from .mixed_recursive_native import FallbackPolicyRuntime

        return FallbackPolicyRuntime.from_fast_plan(
            plan=self,
            model=model,
            space=space,
            target_plan=self.build_fallback_target_plan(),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "frontend_authority": ["ModuleVisitor", "MxCallTraceLogger", "TransSpec", "CombinedCellsInfo"],
            "upstream_method_reuse_policy": "reuse_modelx_cython_unless_documented_reason",
            "performance_goal": "orders_of_magnitude_faster_recursive_execution",
            "module_name": self.module_name,
            "class_name": self.class_name,
            "output_name": self.output_name,
            "output_uid": self.output_uid,
            "semantic_facade": self.build_semantic_facade().manifest(),
            "exported_cell_count": len(self.cells),
            "fast_native_formula_count": sum(x.status == "native_formula" for x in self.cells),
            "materialized_helper_count": sum(x.status == "materialized_helper" for x in self.cells),
            "object_fallback_count": sum(x.status == "object_fallback" for x in self.cells),
            "fast_admitted": self.fast_admitted,
            "source_preserving_admitted": self.source_preserving_admitted,
            "source_preserving_candidate_ready": self.source_preserving_candidate_ready,
            "source_preserving_language_authority": "Cython compile of generated noexcept nogil functions",
            "source_preserving_candidate_formula_count": (
                0 if self.source_preserving_native_program is None
                else sum(
                    cell.execution_role == "formula"
                    for cell in self.source_preserving_native_program.cells.values()
                )
            ),
            "source_preserving_candidate_blockers": {
                uid: [
                    {"kind": kind, "detail": detail}
                    for kind, detail in blockers
                ]
                for uid, blockers in sorted(self.source_preserving_candidate_blockers.items())
                if blockers
            },
            "source_transparent_direct_blockers": {
                uid: [
                    {"kind": kind, "detail": detail}
                    for kind, detail in blockers
                ]
                for uid, blockers in sorted(self.source_transparent_direct_blockers.items())
                if blockers
            },
            "mixed_execution_candidate_ready": self.mixed_execution_candidate_ready,
            "output_execution_class": self.output_execution_class,
            "execution_class_plan": (
                None if self.execution_class_plan is None
                else self.execution_class_plan.manifest()
            ),
            "source_transparent_cell_catalog_count": len(self.source_transparent_cell_catalog),
            "source_preserving_lexical_source_count": sum(
                bool(text) for text in self.formula_lexical_sources.values()
            ),
            "fallback_backend": "ordinary_modelx_cython",
            "graph_required": False,
            "static_type_domain_required": False,
            "global_static_type_fixed_point_required": False,
            "demand_driven_authoritative_evidence": dict(self.authoritative_evidence),
            "semantic_parameter_policy": None,
            "semantic_parameter_policy_role": "removed_from_production_plan; use explicit comparison utility",
            "production_memoization_policy": "native_cache",
            "production_overflow_policy": "fail_closed",
            # Retain Stage-1A manifest keys for readers of dev125 artifacts;
            # Stage 1B adds candidate admission rather than invalidating the
            # already-qualified alternate backend contract.
            "source_preserving_stage1a_available": True,
            "source_preserving_stage1a_role": "alternate_n0_formula_backend; current emitter retained as differential oracle",
            "source_preserving_stage1b_available": True,
            "source_preserving_stage1b_role": "syntax-transparent_n0_candidate_backend; current emitter retained as differential oracle",
            "architecture_b_stage2a_available": self.execution_class_plan is not None,
            "architecture_b_stage2b_recursive_abi": "uniform_r_<cell>",
            "architecture_b_stage2b_python_entry": "native_cache_miss_only",
            "architecture_b_stage2a_role": "N0_N1_P_cell_level_classification_retained",
            "architecture_b_stage2b_available": self.execution_class_plan is not None,
            "architecture_b_stage2b_role": "unified_recursive_r_cell_ABI_with_P_python_only_on_native_memo_miss",
            "architecture_b_stage2c_fallback_target_resolution": (
                self.build_fallback_target_plan().manifest()
            ),
            "native_hash_capacity_evidence": dict(self.native_hash_capacity_evidence),
            "native_hash_capacity_by_uid": dict(self.native_hash_capacity_by_uid),
            "scc_schedule_required": False,
            "stage_required": False,
            "run_key_count": len(self.run_keys),
            "inputs": {key: _freeze_input(spec).manifest() for key, spec in self.runtime_inputs.items()},
            "prepared_exact_int_range_bounds": {
                key: {
                    "value": int(spec.prepared_exact_int_value),
                    "authority": spec.prepared_exact_int_authority,
                    "proof_domain_size": spec.prepared_exact_int_proof_domain_size,
                    "source_name": spec.prepared_exact_int_source_name,
                }
                for key, spec in self.runtime_inputs.items()
                if spec.prepared_exact_int_value is not None
            },
            "normalized_operators": [x.manifest() for x in self.normalized_operators],
            "semantic_formula_ir": {
                uid: ir.manifest() for uid, ir in sorted(self.semantic_formula_ir.items())
            },
            "parameter_ranges": {
                uid: [None if x is None else list(x) for x in ranges]
                for uid, ranges in self.parameter_ranges.items()
            },
            "scalarized_vector_helpers": [x.manifest() for x in self.scalarized_vector_helpers],
            "finite_vector_element_helpers": [
                x.manifest() for x in self.finite_vector_element_helpers
            ],
            "cross_space_calls": [x.manifest() for x in self.cross_space_calls],
            "cross_space_cells": [x.manifest() for x in self.cross_space_cells],
            "ranked_output": (
                None if self.ranked_output is None else self.ranked_output.manifest()
            ),
            "semantic_oracle": {uid: dict(report) for uid, report in self.semantic_oracle.items()},
            "category_plan": None if self.category_plan is None else self.category_plan.manifest(),
            "preparation_timings": dict(self.preparation_timings),
            "preparation_memory": {
                phase: dict(values) for phase, values in self.preparation_memory.items()
            },
            "preparation_diagnostics": dict(self.preparation_diagnostics),
            "cells": [x.manifest() for x in self.cells],
        }


@dataclass(frozen=True)
class _FormulaSource:
    uid: str
    name: str
    fullname: str
    function: ast.FunctionDef
    parameters: tuple[str, ...]
    defaults: Mapping[str, ast.expr]
    guards: tuple[GuardSpec, ...]
    preparation_blockers: tuple[tuple[str, str], ...] = ()
    lexical_source: str = ""


class _BindAllCellCalls(ast.NodeTransformer):
    def __init__(
        self,
        uid_by_name: Mapping[str, str],
        params_by_name: Mapping[str, tuple[str, ...]],
        defaults_by_name: Mapping[str, Mapping[str, ast.expr]],
    ):
        self.uid_by_name = uid_by_name
        self.params_by_name = params_by_name
        self.defaults_by_name = defaults_by_name

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id not in self.uid_by_name:
            return node
        name = node.func.id
        return _bind_native_call(
            node,
            display_name=name,
            uid=self.uid_by_name[name],
            params=self.params_by_name[name],
            defaults=self.defaults_by_name[name],
        )


def _bind_native_call(
    node: ast.Call,
    *,
    display_name: str,
    uid: str,
    params: tuple[str, ...],
    defaults: Mapping[str, ast.expr],
) -> ast.Call:
    if len(node.args) > len(params) or any(k.arg is None for k in node.keywords):
        raise ModelxCythonFastBackendError(f"{display_name}: call is not positionally bindable")
    values: dict[str, ast.expr] = {}
    for p, value in zip(params, node.args):
        values[p] = value
    for kw in node.keywords:
        assert kw.arg is not None
        if kw.arg not in params or kw.arg in values:
            raise ModelxCythonFastBackendError(f"{display_name}: invalid keyword {kw.arg!r}")
        values[kw.arg] = kw.value
    args: list[ast.expr] = []
    for p in params:
        if p in values:
            args.append(values[p])
        elif p in defaults:
            args.append(copy.deepcopy(defaults[p]))
        else:
            raise ModelxCythonFastBackendError(
                f"{display_name}: missing required argument {p!r}"
            )
    return ast.copy_location(ast.Call(ast.Name(uid, ast.Load()), args, []), node)


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    """Return a normalized self-relative attribute chain, if statically named."""
    if not isinstance(node, ast.Attribute):
        return None
    attrs: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        attrs.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    attrs.append(cur.id)
    attrs.reverse()
    return tuple(attrs)


class _BindCrossSpaceCellCalls(ast.NodeTransformer):
    """Bind resolver-proven cross-Space calls to canonical native Cell UIDs."""

    def __init__(
        self,
        *,
        caller_uid: str,
        caller_class_fullname: str,
        resolver: Any,
        eligible_by_cell_fullname: Mapping[str, tuple[_FormulaSource, ClassInfo, tuple[str, ...]]],
        records: list[FastCrossSpaceCall],
    ) -> None:
        self.caller_uid = caller_uid
        self.caller_class_fullname = caller_class_fullname
        self.resolver = resolver
        self.eligible_by_cell_fullname = eligible_by_cell_fullname
        self.records = records

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        chain = _attribute_chain(node.func)
        if chain is None or len(chain) < 2:
            return node
        kind, cell_fullname = self.resolver.resolve_chain(
            self.caller_class_fullname, chain
        )
        if kind != RESOLVED:
            return node
        target = self.eligible_by_cell_fullname.get(cell_fullname)
        if target is None:
            return node
        source, class_info, space_params = target
        self.records.append(FastCrossSpaceCall(
            caller_uid=self.caller_uid,
            caller_class_fullname=self.caller_class_fullname,
            receiver_chain=chain,
            callee_uid=source.uid,
            callee_class_fullname=class_info.fqname,
            callee_cell_fullname=cell_fullname,
            target_space_parameters=space_params,
        ))
        return _bind_native_call(
            node,
            display_name=cell_fullname,
            uid=source.uid,
            params=source.parameters,
            defaults=source.defaults,
        )


def _formula_defs(source: str, class_name: str, class_info: ClassInfo) -> dict[str, ast.FunctionDef]:
    module = ast.parse(source)
    classes = [x for x in module.body if isinstance(x, ast.ClassDef) and x.name == class_name]
    if len(classes) != 1:
        raise ModelxCythonFastBackendError(f"class {class_name!r} not found uniquely")
    methods = {x.name: x for x in classes[0].body if isinstance(x, ast.FunctionDef)}
    result: dict[str, ast.FunctionDef] = {}
    for name, cell in class_info.cells.items():
        fname = "_f_" + name
        fn = methods.get(fname)
        if fn is None:
            fn = methods.get(name)
        if fn is None:
            raise ModelxCythonFastBackendError(f"exported Cell {name!r} has no formula method")
        result[name] = copy.deepcopy(fn)
    return result


def _signature(fn: ast.FunctionDef) -> tuple[tuple[str, ...], dict[str, ast.expr]]:
    args = list(fn.args.args)
    if not args or args[0].arg != "self":
        raise ModelxCythonFastBackendError(f"{fn.name}: exported formula has no self argument")
    args = args[1:]
    params = tuple(x.arg for x in args)
    defaults = list(fn.args.defaults)
    first = len(args) - len(defaults)
    by_name: dict[str, ast.expr] = {}
    for idx, default in enumerate(defaults, start=first):
        by_name[args[idx].arg] = copy.deepcopy(default)
    return params, by_name


def _build_formula_catalog(
    source: str,
    *,
    module_name: str,
    class_name: str,
    class_info: ClassInfo,
    next_guard: int = 1,
) -> tuple[dict[str, _FormulaSource], dict[str, str], int]:
    raw = _formula_defs(source, class_name, class_info)
    cst_module = cst.parse_module(source)
    lexical_sources: dict[str, str] = {}
    for statement in cst_module.body:
        if not isinstance(statement, cst.ClassDef) or statement.name.value != class_name:
            continue
        for member in statement.body.body:
            if not isinstance(member, cst.FunctionDef):
                continue
            method_name = member.name.value
            for cell_name in class_info.cells:
                expected = cell_name if cell_name.startswith("__") else "_f_" + cell_name
                if method_name == expected:
                    lexical_sources[cell_name] = cst_module.code_for_node(member)
                    break
        break
    params: dict[str, tuple[str, ...]] = {}
    defaults: dict[str, Mapping[str, ast.expr]] = {}
    uid_by_name = {name: _uid(module_name, class_name, name) for name in class_info.cells}
    for name, fn in raw.items():
        params[name], defaults[name] = _signature(fn)

    normalized: dict[str, ast.FunctionDef] = {}
    binding_blockers: dict[str, tuple[tuple[str, str], ...]] = {}
    for name, fn in raw.items():
        out = normalize_formula_function(fn, cell_name=name)
        out = normalize_cell_aliases(out, uid_by_name.keys()).function
        try:
            out = _BindAllCellCalls(uid_by_name, params, defaults).visit(out)
        except Exception as exc:
            binding_blockers[name] = ((
                "call_binding", f"{type(exc).__name__}:{exc}",
            ),)
        assert isinstance(out, ast.FunctionDef)
        out.name = uid_by_name[name]
        out.args.defaults = []
        out.args.kw_defaults = []
        normalized[name] = ast.fix_missing_locations(out)

    result: dict[str, _FormulaSource] = {}
    for name in sorted(normalized):
        blockers: tuple[tuple[str, str], ...] = binding_blockers.get(name, ())
        try:
            fn, guards, next_guard = normalize_formula_guards(
                normalized[name], uid=uid_by_name[name], next_code=next_guard
            )
        except Exception as exc:
            # A lexical Cell outside the selected output closure may contain a
            # guard that the native subset does not support. Keep the Cell in
            # the catalogue and make it fail closed locally instead of aborting
            # construction of the whole selected-output plan.
            fn = normalized[name]
            guards = ()
            blockers = tuple(sorted(set((*blockers, (
                "control_syntax", f"{type(exc).__name__}:{exc}"
            )))))
        cell = class_info.cells[name]
        result[uid_by_name[name]] = _FormulaSource(
            uid=uid_by_name[name],
            name=name,
            fullname=cell.fqname,
            function=fn,
            parameters=params[name],
            defaults=defaults[name],
            guards=guards,
            preparation_blockers=blockers,
            lexical_source=lexical_sources.get(name, ""),
        )
    return result, uid_by_name, next_guard


def _selector_expr_names(node: ast.AST) -> set[str]:
    """Return load-name dependencies of one parameterized-Space selector.

    This is deliberately syntax-only.  The caller decides which names are
    formula parameters and which belong to the active modelx ItemSpace.
    """

    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _eval_parameterized_selector_expr(
    node: ast.AST,
    *,
    context: Any,
    singleton_parameters: Mapping[str, Any],
) -> Any:
    """Evaluate the bounded selector subset against one active root ItemSpace.

    Parameterized native sidecars are prepared outside generated execution.  We
    therefore need the selected ItemSpace for each frozen RunDomain point, but
    we must not use ``eval`` or execute the owning formula.  This evaluator is a
    small, explicit subset covering literals, root references/parameters,
    provider calls, indexing and ordinary numeric selector expressions.
    """

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in singleton_parameters:
            return singleton_parameters[node.id]
        if hasattr(context, node.id):
            return getattr(context, node.id)
        raise ModelxCythonFastBackendError(
            f"parameterized Space selector name {node.id!r} is not resolvable"
        )
    if isinstance(node, ast.Attribute):
        base = _eval_parameterized_selector_expr(
            node.value, context=context, singleton_parameters=singleton_parameters
        )
        return getattr(base, node.attr)
    if isinstance(node, ast.Tuple):
        return tuple(
            _eval_parameterized_selector_expr(
                item, context=context, singleton_parameters=singleton_parameters
            )
            for item in node.elts
        )
    if isinstance(node, ast.List):
        return [
            _eval_parameterized_selector_expr(
                item, context=context, singleton_parameters=singleton_parameters
            )
            for item in node.elts
        ]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_parameterized_selector_expr(
            node.operand, context=context, singleton_parameters=singleton_parameters
        )
        return +value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    ):
        left = _eval_parameterized_selector_expr(
            node.left, context=context, singleton_parameters=singleton_parameters
        )
        right = _eval_parameterized_selector_expr(
            node.right, context=context, singleton_parameters=singleton_parameters
        )
        fn = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a ** b,
        }[type(node.op)]
        return fn(left, right)
    if isinstance(node, ast.Call):
        if node.keywords:
            raise ModelxCythonFastBackendError(
                "parameterized Space selector provider calls with keywords are not supported"
            )
        fn = _eval_parameterized_selector_expr(
            node.func, context=context, singleton_parameters=singleton_parameters
        )
        if not callable(fn):
            raise ModelxCythonFastBackendError(
                "parameterized Space selector resolved a non-callable provider"
            )
        args = [
            _eval_parameterized_selector_expr(
                arg, context=context, singleton_parameters=singleton_parameters
            )
            for arg in node.args
        ]
        return fn(*args)
    if isinstance(node, ast.Subscript):
        value = _eval_parameterized_selector_expr(
            node.value, context=context, singleton_parameters=singleton_parameters
        )
        key = _eval_parameterized_selector_expr(
            node.slice, context=context, singleton_parameters=singleton_parameters
        )
        return value[key]
    raise ModelxCythonFastBackendError(
        "unsupported parameterized Space selector expression: "
        + ast.dump(node, include_attributes=False)
    )


class _ParameterizedSpaceCallRewriter(ast.NodeTransformer):
    """Replace one local ``Space[selectors]`` alias with canonical Cell calls."""

    def __init__(
        self,
        *,
        alias_name: str,
        selection_node: ast.Subscript,
        target_by_name: Mapping[str, _FormulaSource],
    ) -> None:
        self.alias_name = alias_name
        self.selection_node = selection_node
        self.target_by_name = target_by_name
        self.used_uids: set[str] = set()
        self.escaped = False

    def visit_Assign(self, node: ast.Assign):  # noqa: N802
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == self.alias_name
            and ast.dump(node.value, include_attributes=False)
            == ast.dump(self.selection_node, include_attributes=False)
        ):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):  # noqa: N802
        # Visit arguments, but inspect the receiver before generic_visit turns a
        # nested alias use into something less obvious.
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == self.alias_name
        ):
            source = self.target_by_name.get(node.func.attr)
            if source is None:
                self.escaped = True
                return self.generic_visit(node)
            rewritten_args = [self.visit(copy.deepcopy(arg)) for arg in node.args]
            rewritten_keywords = [
                ast.keyword(arg=kw.arg, value=self.visit(copy.deepcopy(kw.value)))
                for kw in node.keywords
            ]
            rebound = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=self.alias_name, ctx=ast.Load()),
                    attr=node.func.attr,
                    ctx=ast.Load(),
                ),
                args=rewritten_args,
                keywords=rewritten_keywords,
            )
            rebound = _bind_native_call(
                rebound,
                display_name=source.fullname,
                uid=source.uid,
                params=source.parameters,
                defaults=source.defaults,
            )
            self.used_uids.add(source.uid)
            return ast.copy_location(rebound, node)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id == self.alias_name:
            self.escaped = True
        return node


class _ParameterizedSpaceTargetRewriter(ast.NodeTransformer):
    """Bind target-Space parameters and numeric array leaves to prepared ABI."""

    def __init__(
        self,
        *,
        parameter_inputs: Mapping[str, str],
        array_leaf_inputs: Mapping[str, tuple[str, str]],
    ) -> None:
        self.parameter_inputs = dict(parameter_inputs)
        self.array_leaf_inputs = dict(array_leaf_inputs)
        self.unsupported_array_calls: set[str] = set()

    @staticmethod
    def _point_input(key: str) -> ast.Call:
        return ast.Call(ast.Name("point_input", ast.Load()), [ast.Constant(key)], [])

    def visit_Name(self, node: ast.Name):  # noqa: N802
        if (
            isinstance(node.ctx, ast.Load)
            and node.id in self.parameter_inputs
        ):
            return ast.copy_location(self._point_input(self.parameter_inputs[node.id]), node)
        return node

    def visit_Subscript(self, node: ast.Subscript):  # noqa: N802
        # A numeric rank-1 zero-argument Cell leaf is frozen as a [point, axis]
        # table.  The scalar consumer remains a native Cell and therefore keeps
        # its ordinary native memo wrapper.
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in self.array_leaf_inputs
            and not node.value.args
            and not node.value.keywords
        ):
            table_key, ordinal_key = self.array_leaf_inputs[node.value.func.id]
            index = self.visit(copy.deepcopy(node.slice))
            return ast.copy_location(
                ast.Call(
                    ast.Name("table_input", ast.Load()),
                    [
                        ast.Constant(table_key),
                        self._point_input(ordinal_key),
                        index,
                    ],
                    [],
                ),
                node,
            )
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.array_leaf_inputs:
            self.unsupported_array_calls.add(node.func.id)
        return node


def _has_cross_space_candidate(
    formulas: Mapping[str, _FormulaSource], class_info: ClassInfo
) -> bool:
    roots = {
        name for name, ref in class_info.refs.items()
        if getattr(ref, "mx_class", "")
    } | set(class_info.spaces)
    if not roots:
        return False
    for source in formulas.values():
        for node in ast.walk(source.function):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if chain is not None and len(chain) >= 2 and chain[0] in roots:
                return True
    return False


def _module_catalogue(
    model_path: Path,
    *,
    root_module_name: str,
    root_module_info: ModuleInfo,
    root_source: str,
    root_path: Path,
    logger: Any,
    spec: TransSpec,
) -> tuple[dict[str, ModuleInfo], dict[str, str]]:
    """Parse the exported model module set with modelx-cython's own frontend."""
    infos: dict[str, ModuleInfo] = {root_module_name: root_module_info}
    sources: dict[str, str] = {root_module_name: root_source}
    root_resolved = root_path.resolve()
    for fqname, path in iter_module_files(model_path):
        path = Path(path)
        if path.resolve() == root_resolved:
            # Explicit module_name can differ in synthetic tests.  The current
            # root metadata remains authoritative for that case.
            if fqname != root_module_name:
                continue
            continue
        source = path.read_text(encoding="utf-8")
        visitor = ModuleVisitor(module=fqname, source=source)
        infos[fqname] = ModuleInfo(fqname, visitor, logger, spec)
        sources[fqname] = source
    return infos, sources


def _class_index(module_infos: Mapping[str, ModuleInfo]) -> dict[str, ClassInfo]:
    return {
        class_info.fqname: class_info
        for info in module_infos.values()
        for class_info in info.classes.values()
    }


def _cell_owner_index(
    class_by_fullname: Mapping[str, ClassInfo],
) -> dict[str, tuple[ClassInfo, Any]]:
    return {
        cell.fqname: (class_info, cell)
        for class_info in class_by_fullname.values()
        for cell in class_info.cells.values()
    }


def _resolved_cell_fullnames(
    source: _FormulaSource, *, caller_class: ClassInfo, resolver: Any
) -> tuple[str, ...]:
    result: set[str] = set()
    for node in ast.walk(source.function):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if chain is None or len(chain) < 2:
            continue
        kind, cell_fullname = resolver.resolve_chain(caller_class.fqname, chain)
        if kind == RESOLVED:
            result.add(cell_fullname)
    return tuple(sorted(result))


def _runtime_child_space(scope: Any, name: str) -> Any | None:
    spaces = getattr(scope, "spaces", None)
    try:
        if spaces is not None and name in spaces:
            return spaces[name]
    except Exception:
        pass
    try:
        return getattr(scope, name)
    except Exception:
        return None


def _live_space_map(
    *,
    root_class: ClassInfo,
    root_space: Any,
    class_by_fullname: Mapping[str, ClassInfo],
    resolver: Any,
) -> dict[str, Any]:
    """Map metadata classes to preparation-only live scopes, fail-closed.

    The association is driven exclusively by modelx-cython ref/child-space class
    metadata.  Runtime objects provide provider values only after that identity is
    established; they never decide which Cell a generated call targets.
    """
    result: dict[str, Any] = {root_class.fqname: root_space}
    queue = [root_class.fqname]
    while queue:
        owner = queue.pop(0)
        info = class_by_fullname.get(owner)
        scope = result.get(owner)
        if info is None or scope is None:
            continue
        refs = getattr(scope, "refs", {})
        for name, ref in info.refs.items():
            target = getattr(ref, "mx_class", "")
            if not target or target not in class_by_fullname or target in result:
                continue
            try:
                value = refs[name]
            except Exception:
                continue
            result[target] = value
            queue.append(target)
        # ``CellsResolver`` already derived this child-space map from exported
        # modelx-cython metadata.  Reuse it rather than recreating package rules.
        child_map = getattr(resolver, "_spaces", {}).get(owner, {})
        for name, target in child_map.items():
            if target not in class_by_fullname or target in result:
                continue
            value = _runtime_child_space(scope, name)
            if value is None:
                continue
            result[target] = value
            queue.append(target)
    return result


def _space_parameter_names(class_info: ClassInfo) -> tuple[str, ...]:
    params = getattr(class_info.logger, "param_info", {}).get(class_info.fqname, {})
    return tuple(params)


def _result_evidence(
    cell, *, allow_array: bool = False,
    fallback: tuple[str, str] | None = None,
) -> tuple[str, str]:
    spec = getattr(cell, "_spec", {}) or {}
    requested = spec.get(TransSpec.RET_T)
    if cell.has_typeinfo():
        series = getattr(cell._rt, "numeric_series_evidence", None)
        if series is not None:
            return (_fast_dtype(series[0]), "numeric_series_trace") if allow_array else ("object", "numeric_series_rank1")
        if cell._rt.ret_type.is_array:
            if not allow_array:
                return "object", f"runtime_array_ndim_{cell._rt.ret_type.ndim}"
            return (
                _fast_dtype(cell.norm_type),
                f"modelx_cython.CombinedCellsInfo.norm_type_array_ndim_{cell._rt.ret_type.ndim}",
            )
        dtype = _fast_dtype(cell.norm_type)
        if dtype not in _ALLOWED_DTYPES and cell.norm_type is object and fallback is not None:
            # Refine only modelx-cython's generic ``object`` surface type.  A
            # concrete unsupported runtime type such as ``str`` is conflicting
            # evidence and must not be replaced by static inference.
            return fallback[0], (
                "modelx_cython.CombinedCellsInfo.norm_type_object+" + fallback[1]
            )
        return dtype, "modelx_cython.CombinedCellsInfo.norm_type"
    if requested and requested != TransSpec.RET_MEMORYVIEW:
        return _fast_dtype(_spec_type(requested, context=f"{cell.fqname}.return_type")), "TransSpec.return_type_unobserved_extension"
    if fallback is not None:
        return fallback
    return "object", "unobserved_runtime_return"


def _result_geometry(cell) -> tuple[int, tuple[int, ...] | None, str]:
    """Read logical result geometry from modelx-cython trace metadata."""
    series = getattr(getattr(cell, "_rt", None), "numeric_series_evidence", None)
    if series is not None:
        return 1, series[1], "numeric_series_trace_with_labels"
    if not cell.has_typeinfo() or not cell._rt.ret_type.is_array:
        return 0, (), "modelx_cython.scalar_return"
    ndim = int(cell._rt.ret_type.ndim)
    shape = getattr(cell._rt, "array_shape", None)
    return (
        ndim,
        None if shape is None else tuple(int(value) for value in shape),
        "modelx_cython.ReturnTypeInfo+RuntimeCellsInfo.array_shape",
    )


def _ranked_output_evidence(cell, space: Any, dtype: str) -> FastRankedOutput | None:
    ndim, shape, _source = _result_geometry(cell)
    if ndim == 0:
        return None
    if ndim != 1 or shape is None or len(shape) != 1 or shape[0] <= 0:
        return None

    candidates: list[tuple[str, Any]] = []
    for name, value in getattr(space, "refs", {}).items():
        if (
            type(value).__name__ == "DataFrame"
            and type(value).__module__.startswith("pandas.")
            and len(value) == shape[0]
        ):
            candidates.append((str(name), value))
    point_frame_ref: str | None = None
    if candidates:
        first_name, first = candidates[0]
        first_keys = tuple(first.index.tolist())
        for _name, frame in candidates[1:]:
            if tuple(frame.index.tolist()) != first_keys:
                # Multiple incompatible point axes are ambiguous by contract.
                return None
        point_frame_ref = first_name
        axis_keys = first_keys
    else:
        axis_keys = tuple(range(shape[0]))
    if len(set(axis_keys)) != len(axis_keys):
        return None
    return FastRankedOutput(
        dtype=dtype, ndim=1, shape=shape, axis_keys=axis_keys,
        point_frame_ref=point_frame_ref,
    )


def _validate_ranked_series_labels(class_info, ranked_output) -> None:
    """Reject positional lowering when traced Series use another point axis."""
    if ranked_output is None:
        return
    for name, cell in class_info.cells.items():
        evidence = getattr(getattr(cell, "_rt", None), "numeric_series_evidence", None)
        if evidence is not None and evidence[1] == ranked_output.shape:
            if evidence[2] != ranked_output.axis_keys:
                raise ModelxCythonFastBackendError(
                    f"{name}: traced Series labels do not match the ranked point axis"
                )


def _parameter_evidence(
    cell, class_info: ClassInfo, *,
    fallbacks: Mapping[str, tuple[str, str]] | None = None,
) -> tuple[FastParameterEvidence, ...]:
    spec = getattr(cell, "_spec", {}) or {}
    type_overrides = spec.get(TransSpec.PARAM_T, {}) or {}
    range_overrides = spec.get(getattr(TransSpec, "PARAM_RANGE", "param_range"), {}) or {}
    if type_overrides and not isinstance(type_overrides, Mapping):
        raise ModelxCythonFastBackendError(f"{cell.fqname}.param_type must be a mapping")
    if range_overrides and not isinstance(range_overrides, Mapping):
        raise ModelxCythonFastBackendError(f"{cell.fqname}.param_range must be a mapping")
    rt = getattr(cell, "_rt", None)
    class_sizes = class_info.cells_arg_sizes.get(tuple(cell.params))
    out: list[FastParameterEvidence] = []
    for pos, name in enumerate(cell.params):
        if name in type_overrides:
            dtype = _fast_dtype(_spec_type(type_overrides[name], context=f"{cell.fqname}.param_type[{name!r}]"))
            source = "TransSpec.param_type"
        elif rt is not None and name in rt.arg_types:
            dtype = _fast_dtype(rt.arg_types[name])
            source = "MxCallTraceLogger.argument"
        elif fallbacks and name in fallbacks:
            dtype, source = fallbacks[name]
        else:
            dtype = "object"
            source = "unobserved_runtime_argument"

        lo: int | None = None
        hi: int | None = None
        range_source = ""
        if name in range_overrides:
            value = range_overrides[name]
            if not (isinstance(value, (tuple, list)) and len(value) == 2):
                raise ModelxCythonFastBackendError(f"{cell.fqname}.param_range[{name!r}] must be (min,max)")
            lo, hi = int(value[0]), int(value[1])
            if lo > hi:
                raise ModelxCythonFastBackendError(f"{cell.fqname}.param_range[{name!r}] is reversed")
            range_source = "TransSpec.param_range"
        elif rt is not None and name in getattr(rt, "min_args", {}) and name in rt.max_args:
            lo = int(rt.min_args[name])
            hi = int(rt.max_args[name])
            range_source = "MxCallTraceLogger.minmax"
            if class_sizes is not None and pos < len(class_sizes) and lo >= 0:
                hi = max(hi, int(class_sizes[pos]) - 1)
                range_source += "+TransSpec.cells_param_size"
        elif class_sizes is not None and pos < len(class_sizes) and dtype in {"int64", "bool"}:
            lo, hi = 0, int(class_sizes[pos]) - 1
            range_source = "TransSpec.cells_param_size"
        elif dtype == "bool":
            lo, hi = 0, 1
            range_source = "bool_domain"
        out.append(FastParameterEvidence(name, dtype, source, lo, hi, range_source))
    return tuple(out)


def _provider_parameter_domain(
    cell, class_info: ClassInfo, param: FastParameterEvidence, *,
    trace_covers_run_domain: bool = False,
) -> tuple[int, int, str] | None:
    """Return only provider-safe finite integer parameter authority.

    Pure representative-trace extrema are deliberately insufficient for provider
    materialization.  Explicit ``param_range`` is authoritative.  A modelx-cython
    min/max trace is also authoritative when the trace sample covers the complete
    frozen RunDomain selected for this compilation; in that case the extrema are
    complete-run call evidence rather than representative sampling.  For the
    ordinary modelx-cython Cells cache geometry, a non-negative traced parameter
    plus ``cells_param_size`` otherwise safely widens to the complete 0..size-1
    cache domain.  Negative-coordinate formulas do not take that branch and remain
    fail-closed unless an explicit range or complete-run trace is available.
    """
    if param.dtype != "int64" or param.minimum is None or param.maximum is None:
        return None
    if param.range_source == "TransSpec.param_range":
        return int(param.minimum), int(param.maximum), param.range_source
    if trace_covers_run_domain and param.range_source.startswith("MxCallTraceLogger.minmax"):
        return (
            int(param.minimum),
            int(param.maximum),
            param.range_source + "+complete_run_domain_trace",
        )
    if "TransSpec.cells_param_size" in param.range_source and int(param.minimum) >= 0:
        sizes = class_info.cells_arg_sizes.get(tuple(cell.params))
        if sizes is None:
            return None
        try:
            pos = tuple(cell.params).index(param.name)
        except ValueError:
            return None
        if pos >= len(sizes) or int(sizes[pos]) <= 0:
            return None
        return (
            0,
            int(sizes[pos]) - 1,
            param.range_source + "+complete_nonnegative_cells_cache_domain",
        )
    return None


def _control_parameter_intervals(
    evidence: FastFormulaTranslationService,
    uid: str,
) -> dict[str, Int64Interval]:
    """Adapt existing complete parameter-range authority for local control."""

    out: dict[str, Int64Interval] = {}
    for (owner_uid, name), row in getattr(
        evidence, "modelx_cython_parameter_domains", {}
    ).items():
        if owner_uid != uid:
            continue
        lower, upper, authority = row
        try:
            if type(lower) is not int or type(upper) is not int:
                continue
            out[str(name)] = Int64Interval(
                lower, upper,
                (f"complete_parameter_domain:{authority}",),
            )
        except (TypeError, ValueError):
            # Malformed/out-of-int64 evidence cannot authorize control.
            continue
    return out


def _module_refs(evidence: FastFormulaTranslationService) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, obj in evidence.refs.items():
        module_name = getattr(obj, "__name__", None)
        if module_name in {"math", "numpy"}:
            result[str(name)] = str(module_name)
    return result



class _NameSubstitute(ast.NodeTransformer):
    def __init__(self, name: str, replacement: ast.AST):
        self.name = name
        self.replacement = replacement

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id == self.name:
            return ast.copy_location(copy.deepcopy(self.replacement), node)
        return node


class _ExactSubtreeReplace(ast.NodeTransformer):
    def __init__(self, pattern: ast.AST, replacement: ast.AST):
        self.pattern = ast.dump(pattern, include_attributes=False)
        self.replacement = replacement
        self.count = 0

    def generic_visit(self, node):
        if isinstance(node, ast.AST) and ast.dump(node, include_attributes=False) == self.pattern:
            self.count += 1
            return ast.copy_location(copy.deepcopy(self.replacement), node)
        return super().generic_visit(node)


class _PreparedMetadataScalarCallNormalizer(ast.NodeTransformer):
    """Replace preparation-owned metadata Cell calls with typed global inputs."""

    def __init__(
        self,
        evidence: FastFormulaTranslationService,
        *,
        metadata_uids: set[str],
        source_name_by_uid: Mapping[str, str],
    ) -> None:
        self.evidence = evidence
        self.metadata_uids = metadata_uids
        self.source_name_by_uid = source_name_by_uid

    def visit_Call(self, node: ast.Call):  # noqa: N802
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in self.metadata_uids
            and not node.args
            and not node.keywords
        ):
            source_name = self.source_name_by_uid[node.func.id]
            key = self.evidence.registry.reference_metadata_scalar(source_name)
            return ast.copy_location(
                ast.Call(ast.Name("global_input", ast.Load()), [ast.Constant(key)], []),
                node,
            )
        return self.generic_visit(node)


def _single_return_expr(fn: ast.FunctionDef) -> ast.expr | None:
    body = [
        item for item in fn.body
        if not (
            isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
        )
    ]
    if len(body) == 1 and isinstance(body[0], ast.Return):
        return body[0].value
    return None


def _rank1_point_source_uids(
    formulas: Mapping[str, _FormulaSource], point_frame_ref: str | None
) -> set[str]:
    if point_frame_ref is None:
        return set()
    result: set[str] = set()
    for uid, source in formulas.items():
        value = _single_return_expr(source.function)
        if isinstance(value, ast.Name) and value.id == point_frame_ref:
            result.add(uid)
    return result


class _StructuredProviderAnalyzer:
    """Recover a small columnar/row-domain IR from DataFrame-producing Cells.

    This is intentionally not a pandas emulator.  It recognizes generic
    structured operations whose observable semantics are column lineage plus a
    deterministic row mapping: source frames, left joins, row-repeat expansion,
    and dtype-only casts.  Unsupported operations simply yield no provider plan.
    """

    def __init__(
        self,
        evidence: FastFormulaTranslationService,
        formulas: Mapping[str, _FormulaSource],
        *,
        expected_ranked_size: int | None,
    ) -> None:
        self.evidence = evidence
        self.formulas = formulas
        self.expected_ranked_size = expected_ranked_size
        self.cache: dict[str, StructuredFramePlan | None] = {}
        self.active: set[str] = set()
        self.source_name_by_uid = {uid: row.name for uid, row in formulas.items()}

    @staticmethod
    def _docless(body: Sequence[ast.stmt]) -> list[ast.stmt]:
        return [
            st for st in body
            if not (
                isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)
                and isinstance(st.value.value, str)
            )
        ]

    def _module_attr(self, node: ast.AST, module_name: str, attr: str) -> bool:
        return (
            isinstance(node, ast.Attribute) and node.attr == attr
            and isinstance(node.value, ast.Name)
            and getattr(self.evidence.refs.get(node.value.id), "__name__", None) == module_name
        )

    def _source_frame(self, name: str) -> StructuredFramePlan | None:
        value = self.evidence.refs.get(name)
        if value is None:
            return None
        try:
            return StructuredFramePlan.source(name, value)
        except StructuredProviderError:
            return None

    def _static_length(self, node: ast.AST) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, numbers.Integral):
            value = int(node.value)
            return value if value >= 0 else None
        if isinstance(node, ast.Name) and node.id in self.evidence.refs:
            value = self.evidence.refs[node.id]
            try:
                return int(len(value))
            except Exception:
                return None
        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "len" and len(node.args) == 1 and not node.keywords
        ):
            target = node.args[0]
            if isinstance(target, ast.Name) and target.id in self.evidence.refs:
                try:
                    return int(len(self.evidence.refs[target.id]))
                except Exception:
                    return None
            if (
                isinstance(target, ast.Call) and isinstance(target.func, ast.Name)
                and not target.args and not target.keywords and target.func.id in self.formulas
            ):
                source_name = self.formulas[target.func.id].name
                try:
                    # Cardinality is preparation metadata only.  The resulting
                    # structured columns never call this Cell at binding/runtime.
                    value = self.evidence.space.cells[source_name]()
                    return int(len(value))
                except Exception:
                    return None
        return None

    def _frame_expr(
        self, node: ast.AST, env: Mapping[str, StructuredFramePlan]
    ) -> StructuredFramePlan | None:
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            return self._source_frame(node.id)

        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and not node.args and not node.keywords and node.func.id in self.formulas
        ):
            return self.analyze(node.func.id)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "astype":
                # astype changes representation/dtype, not row/column lineage.
                return self._frame_expr(node.func.value, env)

            if node.func.attr == "join" and len(node.args) == 1:
                parent = self._frame_expr(node.func.value, env)
                if parent is None or not isinstance(node.args[0], ast.Name):
                    return None
                right_name = node.args[0].id
                right = self.evidence.refs.get(right_name)
                on = None
                for kw in node.keywords:
                    if kw.arg == "on" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        on = kw.value.value
                    else:
                        return None
                if on is None:
                    return None
                try:
                    return StructuredFramePlan.left_join(parent, right_name, right, on=on)
                except StructuredProviderError:
                    return None

            if self._module_attr(node.func, "pandas", "DataFrame") and node.args:
                values = node.args[0]
                if not (
                    isinstance(values, ast.Call)
                    and self._module_attr(values.func, "numpy", "repeat")
                    and len(values.args) >= 2
                ):
                    return None
                base_values = values.args[0]
                if not (
                    isinstance(base_values, ast.Attribute) and base_values.attr == "values"
                ):
                    return None
                parent = self._frame_expr(base_values.value, env)
                if parent is None:
                    return None
                axis = None
                if len(values.args) >= 3 and isinstance(values.args[2], ast.Constant):
                    axis = values.args[2].value
                for kw in values.keywords:
                    if kw.arg == "axis" and isinstance(kw.value, ast.Constant):
                        axis = kw.value.value
                    elif kw.arg != "axis":
                        return None
                if axis != 0:
                    return None
                repeat = self._static_length(values.args[1])
                if repeat is None and self.expected_ranked_size is not None and parent.output_size:
                    q, r = divmod(int(self.expected_ranked_size), int(parent.output_size))
                    if r == 0 and q > 0:
                        repeat = q
                if repeat is None or repeat <= 0:
                    return None
                # A DataFrame wrapping repeat(parent.values) preserves columns
                # only when columns=parent.columns.  Index construction is row
                # identity metadata and therefore intentionally irrelevant here.
                columns_ok = False
                for kw in node.keywords:
                    if kw.arg == "columns" and isinstance(kw.value, ast.Attribute) and kw.value.attr == "columns":
                        columns_parent = self._frame_expr(kw.value.value, env)
                        columns_ok = columns_parent == parent
                    elif kw.arg not in {"columns", "index"}:
                        return None
                if not columns_ok:
                    return None
                try:
                    return StructuredFramePlan.remap(
                        parent,
                        RowMap.repeat_each(parent.output_size, repeat),
                        note=f"repeat_each:{repeat}",
                    )
                except StructuredProviderError:
                    return None
        return None

    def analyze(self, uid: str) -> StructuredFramePlan | None:
        if uid in self.cache:
            return self.cache[uid]
        if uid in self.active or uid not in self.formulas:
            return None
        source = self.formulas[uid]
        if source.parameters:
            self.cache[uid] = None
            return None
        self.active.add(uid)
        try:
            env: dict[str, StructuredFramePlan] = {}
            result: StructuredFramePlan | None = None
            for st in self._docless(source.function.body):
                if (
                    isinstance(st, ast.Assign) and len(st.targets) == 1
                    and isinstance(st.targets[0], ast.Name)
                ):
                    frame = self._frame_expr(st.value, env)
                    if frame is not None:
                        env[st.targets[0].id] = frame
                    continue
                if isinstance(st, ast.Return) and st.value is not None:
                    result = self._frame_expr(st.value, env)
                    break
            self.cache[uid] = result
            return result
        finally:
            self.active.discard(uid)

    def discover(self) -> dict[str, StructuredFramePlan]:
        out: dict[str, StructuredFramePlan] = {}
        for uid in self.formulas:
            plan = self.analyze(uid)
            if plan is not None:
                out[uid] = plan
        return out


def _structured_field_selector(
    fn: ast.FunctionDef, structured_providers: Mapping[str, StructuredFramePlan]
) -> tuple[str, str] | None:
    value = _single_return_expr(fn)
    if not (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Call)
        and isinstance(value.value.func, ast.Name)
        and value.value.func.id in structured_providers
        and not value.value.args and not value.value.keywords
        and isinstance(value.slice, ast.Constant) and isinstance(value.slice.value, str)
    ):
        return None
    return value.value.func.id, value.slice.value


def _structured_category_evidence(
    formulas: Mapping[str, _FormulaSource],
    structured_providers: Mapping[str, StructuredFramePlan],
    refs: Mapping[str, Any],
) -> dict[str, tuple[str, tuple[Any, ...], bool]]:
    out: dict[str, tuple[str, tuple[Any, ...], bool]] = {}
    for uid, source in formulas.items():
        if source.parameters:
            continue
        selector = _structured_field_selector(source.function, structured_providers)
        if selector is None:
            continue
        provider_uid, field = selector
        try:
            type_evidence = structured_providers[provider_uid].column_type(field, refs)
        except StructuredProviderError:
            continue
        labels = type_evidence.category_labels
        nullable = type_evidence.nullable
        if labels and all(isinstance(x, str) for x in labels):
            out[uid] = (field, labels, nullable)
    return out


def _rank1_global_aggregate_uids(
    formulas: Mapping[str, _FormulaSource], point_array_uids: set[str]
) -> set[str]:
    result: set[str] = set()
    for uid, source in formulas.items():
        value = _single_return_expr(source.function)
        if not (
            not source.parameters
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in {"min", "max", "sum", "len"}
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Call)
            and isinstance(value.args[0].func, ast.Name)
            and value.args[0].func.id in point_array_uids
        ):
            continue
        result.add(uid)
    return result


def _rank1_matmul_reduction(fn: ast.FunctionDef) -> ast.FunctionDef:
    """Preserve a row-wise array dot product as one scalar reduction.

    The admitted form is deliberately exact: an array made from one generator,
    transposed, then multiplied by a rank-1 slice over the identical range.
    General loops and arbitrary matrix algebra remain outside M7D.
    """
    out = copy.deepcopy(fn)
    body = [
        item for item in out.body
        if not (
            isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
        )
    ]
    if len(body) != 2 or not (
        isinstance(body[0], ast.Assign)
        and len(body[0].targets) == 1
        and isinstance(body[0].targets[0], ast.Name)
        and isinstance(body[1], ast.Return)
        and isinstance(body[1].value, ast.BinOp)
        and isinstance(body[1].value.op, ast.MatMult)
    ):
        return out
    target = body[0].targets[0].id
    dot = body[1].value
    if not (isinstance(dot.left, ast.Name) and dot.left.id == target):
        return out
    produced = body[0].value
    if not (
        isinstance(produced, ast.Call)
        and isinstance(produced.func, ast.Attribute)
        and produced.func.attr == "transpose"
        and not produced.args and not produced.keywords
        and isinstance(produced.func.value, ast.Call)
        and isinstance(produced.func.value.func, ast.Attribute)
        and produced.func.value.func.attr == "array"
        and len(produced.func.value.args) == 1
    ):
        return out
    generated = produced.func.value.args[0]
    if (
        isinstance(generated, ast.Call)
        and isinstance(generated.func, ast.Name)
        and generated.func.id in {"list", "tuple"}
        and len(generated.args) == 1
    ):
        generated = generated.args[0]
    if not isinstance(generated, ast.GeneratorExp) or len(generated.generators) != 1:
        return out
    comp = generated.generators[0]
    if not (
        isinstance(comp.target, ast.Name)
        and isinstance(comp.iter, ast.Call)
        and isinstance(comp.iter.func, ast.Name)
        and comp.iter.func.id == "range"
        and not comp.ifs and not comp.is_async
        and isinstance(dot.right, ast.Subscript)
        and isinstance(dot.right.slice, ast.Slice)
        and dot.right.slice.lower is None
        and dot.right.slice.step is None
        and dot.right.slice.upper is not None
        and len(comp.iter.args) == 1
        and ast.dump(comp.iter.args[0], include_attributes=False)
        == ast.dump(dot.right.slice.upper, include_attributes=False)
    ):
        return out
    list_gen = ast.Call(
        func=ast.Name("list", ast.Load()),
        args=[copy.deepcopy(generated)],
        keywords=[],
    )
    summed = ast.Call(
        func=ast.Name("sum", ast.Load()),
        args=[ast.BinOp(list_gen, ast.Mult(), copy.deepcopy(dot.right))],
        keywords=[],
    )
    out.body = [ast.Return(value=summed)]
    return ast.fix_missing_locations(out)


class _Rank1ArrayFormulaNormalizer(ast.NodeTransformer):
    """Lower point-wise rank-1 formula syntax onto the existing scalar ABI."""

    def __init__(
        self,
        evidence: FastFormulaTranslationService,
        *,
        point_source_uids: set[str],
        point_frame_ref: str | None,
        global_aggregate_uids: set[str],
        source_name_by_uid: Mapping[str, str],
        scalarize: bool = True,
        structured_providers: Mapping[str, StructuredFramePlan] | None = None,
        current_uid: str | None = None,
        category_plan: ModelxCythonCategoryPlan | None = None,
    ) -> None:
        self.evidence = evidence
        self.point_source_uids = point_source_uids
        self.point_frame_ref = point_frame_ref
        self.global_aggregate_uids = global_aggregate_uids
        self.source_name_by_uid = source_name_by_uid
        self.scalarize = scalarize
        self.structured_providers = dict(structured_providers or {})
        self.current_uid = current_uid
        self.category_plan = category_plan

    def visit_Assign(self, node: ast.Assign):  # noqa: N802
        # Assigning a pandas index changes labels only; a scalarized element has
        # no index object and retains the same numeric value.
        if any(isinstance(target, ast.Attribute) and target.attr == "index" for target in node.targets):
            return None
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):  # noqa: N802
        source = node.value
        if (
            isinstance(source, ast.Call) and isinstance(source.func, ast.Name)
            and source.func.id in self.structured_providers
            and not source.args and not source.keywords
            and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
        ):
            labels = None
            if self.category_plan is not None and self.current_uid is not None:
                did = self.category_plan.encoding.result_domain_id(self.current_uid)
                if did is not None:
                    labels = self.category_plan.encoding.domain(did).labels
            provider = self.structured_providers[source.func.id]
            key = self.evidence.registry.ranked_structured_field(
                provider,
                node.slice.value,
                provider_id=source.func.id,
                category_labels=labels,
            )
            value = ast.Call(
                ast.Name("point_input" if self.scalarize else "ranked_input", ast.Load()),
                [ast.Constant(key)],
                [],
            )
            # Structured bool columns use int64 as their compact/native storage
            # ABI, but the formula result must retain Python/NumPy boolean
            # semantics.  This matters when a detached Python/pandas consumer
            # uses the value as a mask (Series.where, boolean indexing, etc.).
            try:
                logical_kind = provider.column_type(
                    node.slice.value, self.evidence.refs, labels=labels
                ).logical_kind
            except StructuredProviderError:
                logical_kind = None
            if logical_kind == "bool":
                value = ast.Compare(
                    left=value, ops=[ast.NotEq()], comparators=[ast.Constant(0)]
                )
            return ast.copy_location(value, node)
        is_point_frame = (
            isinstance(source, ast.Name)
            and source.id == self.point_frame_ref
        ) or (
            isinstance(source, ast.Call)
            and isinstance(source.func, ast.Name)
            and source.func.id in self.point_source_uids
            and not source.args and not source.keywords
        )
        if is_point_frame and isinstance(node.slice, ast.Constant):
            key = self.evidence.registry.ranked_point_field(node.slice.value)
            return ast.copy_location(
                ast.Call(
                    ast.Name("point_input" if self.scalarize else "ranked_input", ast.Load()),
                    [ast.Constant(key)],
                    [],
                ),
                node,
            )
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):  # noqa: N802
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in self.global_aggregate_uids
            and not node.args and not node.keywords
        ):
            name = self.source_name_by_uid[node.func.id]
            key = self.evidence.registry.reference_metadata_scalar(name)
            return ast.copy_location(
                ast.Call(ast.Name("global_input", ast.Load()), [ast.Constant(key)], []),
                node,
            )
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            module = self.evidence.refs.get(node.func.value.id)
            module_name = getattr(module, "__name__", None)
            if module_name == "pandas" and node.func.attr == "Series" and node.args:
                if not self.scalarize:
                    return ast.copy_location(
                        ast.Call(
                            ast.Name("ranked_series", ast.Load()),
                            [copy.deepcopy(node.args[0]), ast.Constant(len(self.evidence.point_keys))],
                            [],
                        ),
                        node,
                    )
                return ast.copy_location(copy.deepcopy(node.args[0]), node)
            if not self.scalarize:
                return node
            if module_name == "numpy" and node.func.attr in {"maximum", "minimum"} and len(node.args) == 2:
                name = "max" if node.func.attr == "maximum" else "min"
                return ast.copy_location(ast.Call(ast.Name(name, ast.Load()), node.args, []), node)
            if module_name == "numpy" and node.func.attr in {"around", "round"} and 1 <= len(node.args) <= 2:
                return ast.copy_location(ast.Call(ast.Name("round", ast.Load()), node.args, []), node)
            if module_name == "numpy" and node.func.attr == "where" and len(node.args) == 3:
                return ast.copy_location(ast.IfExp(node.args[0], node.args[1], node.args[2]), node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "where" and len(node.args) in {1, 2}:
            # pandas Series.where(cond, other=...) accepts ``other`` positionally
            # or by keyword.  The ranked path previously ignored the keyword
            # form and silently substituted NaN, corrupting otherwise-native
            # vector models such as BasicTerm_ME.  Admit only the exact subset
            # whose value semantics we can preserve.
            if any(kw.arg != "other" for kw in node.keywords):
                return node
            keyword_other = [kw.value for kw in node.keywords if kw.arg == "other"]
            if len(keyword_other) > 1 or (len(node.args) == 2 and keyword_other):
                return node
            if len(node.args) == 2:
                other = node.args[1]
            elif keyword_other:
                other = keyword_other[0]
            else:
                other = ast.Constant(np.nan)
            if not self.scalarize:
                return ast.copy_location(
                    ast.Call(
                        ast.Name("ranked_where", ast.Load()),
                        [node.args[0], node.func.value, other],
                        [],
                    ),
                    node,
                )
            return ast.copy_location(ast.IfExp(node.args[0], node.func.value, other), node)
        return node




class _RankedMultiIndexReindexNormalizer(ast.NodeTransformer):
    """Lower rank-1 pandas MultiIndex reindex/gather idioms to numeric inputs.

    The supported source shape is intentionally semantic rather than product
    specific::

        mi = pd.MultiIndex.from_arrays([row_labels, col_labels])
        values = provider.reindex(mi, fill_value=<numeric>).set_axis(...)

    ``provider`` may be either a global MultiIndex Series reference or a
    zero-argument local Cell returning a dense numeric two-level MultiIndex
    Series.  Pandas is preparation-only: generated ranked Cython receives one
    dense numeric 2-D input and performs NumPy advanced indexing.  Missing
    labels preserve ``Series.reindex`` semantics through an explicit validity
    mask and the requested fill value (NaN by default).
    """

    def __init__(
        self,
        evidence: FastFormulaTranslationService,
        *,
        source_name_by_uid: Mapping[str, str],
    ) -> None:
        self.evidence = evidence
        self.source_name_by_uid = dict(source_name_by_uid)
        self.index_arrays: dict[str, tuple[ast.expr, ast.expr]] = {}
        self.index_assignments: dict[str, ast.Assign] = {}

    def _is_pandas_multiindex_from_arrays(self, node: ast.AST) -> tuple[ast.expr, ast.expr] | None:
        if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
            return None
        func = node.func
        if not (
            isinstance(func, ast.Attribute) and func.attr == "from_arrays"
            and isinstance(func.value, ast.Attribute) and func.value.attr == "MultiIndex"
            and isinstance(func.value.value, ast.Name)
        ):
            return None
        alias = func.value.value.id
        module = self.evidence.refs.get(alias)
        if getattr(module, "__name__", None) != "pandas":
            return None
        arrays = node.args[0]
        if not isinstance(arrays, (ast.List, ast.Tuple)) or len(arrays.elts) != 2:
            return None
        return copy.deepcopy(arrays.elts[0]), copy.deepcopy(arrays.elts[1])

    def collect(self, fn: ast.FunctionDef) -> None:
        for st in fn.body:
            if not (
                isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
            ):
                continue
            arrays = self._is_pandas_multiindex_from_arrays(st.value)
            if arrays is None:
                continue
            name = st.targets[0].id
            self.index_arrays[name] = arrays
            self.index_assignments[name] = copy.deepcopy(st)

    @staticmethod
    def _numeric_fill(call: ast.Call) -> ast.expr | None:
        fill: ast.expr = ast.Attribute(ast.Name("np", ast.Load()), "nan", ast.Load())
        for kw in call.keywords:
            if kw.arg != "fill_value":
                return None
            if not isinstance(kw.value, ast.Constant) or not isinstance(
                kw.value.value, (bool, int, float, np.integer, np.floating)
            ):
                return None
            fill = copy.deepcopy(kw.value)
        return fill

    def _provider_table(self, node: ast.AST) -> tuple[str, tuple[int, int, int, int]] | None:
        # Global immutable/reference MultiIndex Series.
        if isinstance(node, ast.Name):
            value = self.evidence.refs.get(node.id)
            if value is None:
                return None
            try:
                key, _ = self.evidence.registry.global_multiindex_series(node.id, value)
            except FrontendError:
                return None
            return key, self.evidence.multiindex_axes[key]

        # Local zero-argument Cell returning a numeric dense MultiIndex Series.
        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and not node.args and not node.keywords
        ):
            source_name = self.source_name_by_uid.get(node.func.id)
            if source_name is None or source_name not in self.evidence.space.cells:
                return None
            try:
                cell = self.evidence.space.cells[source_name]
                key = self.evidence.registry.local_cell_multiindex_series(source_name, cell)
            except Exception:
                # Do not turn an arbitrary provider exception into admission.
                # Frontend residual screening will retain the original pandas
                # expression when this normalization cannot prove the table ABI.
                return None
            return key, self.evidence.multiindex_axes[key]
        return None

    @staticmethod
    def _axis_index(expr: ast.expr, start: int, step: int) -> ast.expr:
        out: ast.expr = copy.deepcopy(expr)
        if start:
            out = ast.BinOp(out, ast.Sub(), ast.Constant(int(start)))
        if step != 1:
            out = ast.BinOp(out, ast.FloorDiv(), ast.Constant(int(step)))
        return out

    @staticmethod
    def _valid_axis(expr: ast.expr, start: int, step: int, size: int) -> ast.expr:
        stop = start + step * (size - 1)
        lower = ast.Compare(copy.deepcopy(expr), [ast.GtE()], [ast.Constant(int(start))])
        upper = ast.Compare(copy.deepcopy(expr), [ast.LtE()], [ast.Constant(int(stop))])
        valid: ast.expr = ast.BinOp(lower, ast.BitAnd(), upper)
        if step != 1:
            mod = ast.BinOp(
                ast.BinOp(copy.deepcopy(expr), ast.Sub(), ast.Constant(int(start))),
                ast.Mod(), ast.Constant(int(step)),
            )
            aligned = ast.Compare(mod, [ast.Eq()], [ast.Constant(0)])
            valid = ast.BinOp(valid, ast.BitAnd(), aligned)
        return valid

    @staticmethod
    def _clip_index(expr: ast.expr, size: int) -> ast.expr:
        return ast.Call(
            ast.Attribute(ast.Name("np", ast.Load()), "maximum", ast.Load()),
            [
                ast.Constant(0),
                ast.Call(
                    ast.Attribute(ast.Name("np", ast.Load()), "minimum", ast.Load()),
                    [copy.deepcopy(expr), ast.Constant(int(size - 1))],
                    [],
                ),
            ],
            [],
        )

    def _lower_reindex(self, node: ast.AST) -> ast.expr | None:
        # set_axis only changes labels; after reindex the numeric order already
        # matches the rank-1 point axis, so it has no runtime numeric effect.
        if not (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_axis" and len(node.args) == 1
            and not node.keywords
        ):
            return None
        reindexed = node.func.value
        if not (
            isinstance(reindexed, ast.Call)
            and isinstance(reindexed.func, ast.Attribute)
            and reindexed.func.attr == "reindex"
            and len(reindexed.args) == 1
        ):
            return None
        index_arg = reindexed.args[0]
        if not isinstance(index_arg, ast.Name) or index_arg.id not in self.index_arrays:
            return None
        fill = self._numeric_fill(reindexed)
        if fill is None:
            return None
        resolved = self._provider_table(reindexed.func.value)
        if resolved is None:
            return None
        key, (row_start, row_step, col_start, col_step) = resolved
        spec = self.evidence.registry.specs[key]
        if spec.expected_shape is None or len(spec.expected_shape) != 2:
            return None
        nrow, ncol = (int(spec.expected_shape[0]), int(spec.expected_shape[1]))
        if nrow <= 0 or ncol <= 0:
            return None
        row_labels, col_labels = self.index_arrays[index_arg.id]
        row_index = self._axis_index(row_labels, row_start, row_step)
        col_index = self._axis_index(col_labels, col_start, col_step)
        valid = ast.BinOp(
            self._valid_axis(row_labels, row_start, row_step, nrow),
            ast.BitAnd(),
            self._valid_axis(col_labels, col_start, col_step, ncol),
        )
        gather = ast.Call(
            ast.Name("table_input", ast.Load()),
            [
                ast.Constant(key),
                self._clip_index(row_index, nrow),
                self._clip_index(col_index, ncol),
            ],
            [],
        )
        return ast.copy_location(
            ast.Call(
                ast.Attribute(ast.Name("np", ast.Load()), "where", ast.Load()),
                [valid, gather, fill],
                [],
            ),
            node,
        )

    def visit_Call(self, node: ast.Call):  # noqa: N802
        lowered = self._lower_reindex(node)
        if lowered is not None:
            return self.generic_visit(lowered)
        return self.generic_visit(node)

    def normalize(self, fn: ast.FunctionDef) -> ast.FunctionDef:
        self.collect(fn)
        out = self.visit(copy.deepcopy(fn))
        assert isinstance(out, ast.FunctionDef)
        remaining_loads = {
            node.id for node in ast.walk(out)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        out.body = [
            st for st in out.body
            if not (
                isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and st.targets[0].id in self.index_assignments
                and st.targets[0].id not in remaining_loads
            )
        ]
        return ast.fix_missing_locations(out)

def _normalize_ranked_formula(
    evidence: FastFormulaTranslationService,
    source: _FormulaSource,
    *,
    point_source_uids: set[str],
    point_frame_ref: str | None,
    global_aggregate_uids: set[str],
    source_name_by_uid: Mapping[str, str],
    reference_metadata_uids: set[str] | None = None,
    category_plan: ModelxCythonCategoryPlan | None = None,
    structured_providers: Mapping[str, StructuredFramePlan] | None = None,
) -> tuple[ast.FunctionDef | None, tuple[tuple[str, str], ...]]:
    """Normalize providers while preserving NumPy rank-1 formula semantics."""
    blockers: list[tuple[str, str]] = list(source.preparation_blockers)
    try:
        candidate = copy.deepcopy(source.function)
        if reference_metadata_uids:
            # Ranked formulas may consume preparation-owned scalar metadata too.
            # Those helpers have no recursive r_<uid> wrapper, so lower their
            # zero-argument calls to the same typed global inputs used by the
            # scalar source-preserving path before ranked normalization.
            candidate = _PreparedMetadataScalarCallNormalizer(
                evidence,
                metadata_uids=reference_metadata_uids,
                source_name_by_uid=source_name_by_uid,
            ).visit(candidate)
            assert isinstance(candidate, ast.FunctionDef)
            candidate = ast.fix_missing_locations(candidate)
        candidate = _Rank1ArrayFormulaNormalizer(
            evidence,
            point_source_uids=point_source_uids,
            point_frame_ref=point_frame_ref,
            global_aggregate_uids=global_aggregate_uids,
            source_name_by_uid=source_name_by_uid,
            scalarize=False,
            structured_providers=structured_providers,
            current_uid=source.uid,
            category_plan=category_plan,
        ).visit(candidate)
        assert isinstance(candidate, ast.FunctionDef)
        if category_plan is not None:
            # Direct structured-provider category selectors are already encoded by
            # ``ranked_structured_field`` using this Cell's result-domain labels.
            # Re-running the category AST rewriter would see only an opaque
            # ranked_input(...) call and (correctly, in the general case) demand
            # a separate dynamic-domain proof.  Here the binding itself is that
            # proof, so preserve the encoded integer vector and let downstream
            # consumers use the ordinary result-domain ABI.
            structured_selector = (
                _structured_field_selector(source.function, structured_providers or {})
                if structured_providers else None
            )
            if not (
                structured_selector is not None
                and category_plan.encoding.result_domain_id(source.uid) is not None
            ):
                candidate = category_plan.encoding.rewrite_function(source.uid, candidate)
        candidate = _RankedMultiIndexReindexNormalizer(
            evidence, source_name_by_uid=source_name_by_uid
        ).normalize(candidate)
        candidate = evidence._normalize_static_sequence_access(candidate)
        candidate = evidence._normalize_static_sequence_membership(candidate)
        candidate = evidence._normalize_external_tables(
            candidate, operator_uid=source.uid
        )
        lower = evidence._expr_lowerer(source.uid)
        lower.prepare_function(candidate)
        candidate.body = [lower.visit(copy.deepcopy(st)) for st in candidate.body]
        candidate = ast.fix_missing_locations(candidate)
        candidate = lower.cleanup_function(candidate)
        # Array expressions intentionally retain NumPy ufuncs, matrix products,
        # slicing, and comprehensions.  Only unresolved model/runtime attributes
        # block this path.
        for node in ast.walk(candidate):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "self":
                    blockers.append(("provider", "residual_self_attribute:" + ast.unparse(node)))
        return candidate, tuple(sorted(set(blockers)))
    except Exception as exc:
        kind = "category" if isinstance(exc, CategoryABIError) else _exception_category(exc)
        blockers.append((kind, f"{type(exc).__name__}:{exc}"))
        return None, tuple(sorted(set(blockers)))


def _vector_dense_range(class_info: ClassInfo) -> tuple[int, int] | None:
    # modelx-cython's cells_param_size is the existing cache-size contract.  A
    # vector helper indexed by the same time coordinate may reuse the largest
    # positive declared/observed dimension; no source dependency graph is needed.
    sizes = [
        int(v)
        for shape in class_info.cells_arg_sizes.values()
        for v in shape
        if isinstance(v, numbers.Integral) and int(v) > 0
    ]
    return (0, max(sizes) - 1) if sizes else None



def _selected_finite_integer_parameter_domains(
    evidence: FastFormulaTranslationService,
    formulas: Mapping[str, _FormulaSource],
    output_uid: str,
    *,
    max_values: int = 32,
    max_iterations: int = 32,
) -> tuple[dict[tuple[str, str], tuple[int, ...]], dict[tuple[str, str], str]]:
    """Prove complete finite integer parameter values inside the selected closure.

    Representative call extrema are deliberately not enough for symbolic field
    dispatch.  This proof starts at the selected output, expands only source-proven
    frozen integer sequence reductions, and propagates literal/finite integer call
    arguments through canonical native Cell calls.  Recursive forwarding is handled
    by a bounded monotone fixed point over tiny finite value sets.  Any unresolved
    inbound callsite makes that parameter unproven rather than guessed.
    """
    if output_uid not in formulas:
        return {}, {}

    # Demand-drive the proof.  Most actuarial formulas never construct a field
    # name from an integer parameter, so they should not pay for a second source
    # closure analysis merely because VUL does.
    wanted: set[tuple[str, str]] = set()
    for uid, source in formulas.items():
        params = set(source.parameters)
        if not params:
            continue
        for sub in (node for node in ast.walk(source.function) if isinstance(node, ast.Subscript)):
            for call in (node for node in ast.walk(sub.slice) if isinstance(node, ast.Call)):
                if (
                    isinstance(call.func, ast.Name)
                    and call.func.id == "str"
                    and len(call.args) == 1
                    and not call.keywords
                    and isinstance(call.args[0], ast.Name)
                    and call.args[0].id in params
                ):
                    wanted.add((uid, call.args[0].id))


        # Step-table row-prefix normalization also needs exact finite integer
        # parameter domains.  Keep this narrowly demand-driven: recognize the
        # same local row-view/list-comprehension shape as the StepLookup lowerer
        # and request proof only for parameters in that row prefix.  This avoids
        # turning every unrelated ``.loc[(..., param)]`` into a second closure
        # analysis while preserving complete source-call authority for the cases
        # that actually need finite dispatch.
        body = list(source.function.body)
        for index, statement in enumerate(body[:-1]):
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Subscript)
                and isinstance(statement.value.value, ast.Attribute)
                and statement.value.value.attr == "loc"
                and isinstance(statement.value.slice, ast.Tuple)
            ):
                continue
            row_name = statement.targets[0].id
            next_statement = body[index + 1]
            if not (
                isinstance(next_statement, ast.Assign)
                and isinstance(next_statement.value, ast.ListComp)
                and len(next_statement.value.generators) == 1
                and isinstance(next_statement.value.generators[0].iter, ast.Attribute)
                and next_statement.value.generators[0].iter.attr == "index"
                and isinstance(next_statement.value.generators[0].iter.value, ast.Name)
                and next_statement.value.generators[0].iter.value.id == row_name
            ):
                continue
            for name in (
                node for node in ast.walk(statement.value.slice)
                if isinstance(node, ast.Name) and node.id in params
            ):
                wanted.add((uid, name.id))
    if not wanted:
        return {}, {}

    all_uids = set(formulas)

    def callees(fn: ast.FunctionDef) -> set[str]:
        return {
            node.func.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in all_uids
        }

    # Raw reachability is structural and cheap.  Only selected Cells then pay for
    # complete-RunDomain frozen-sequence proof.
    reachable: set[str] = {output_uid}
    queue = [output_uid]
    while queue:
        caller = queue.pop()
        for callee in callees(formulas[caller].function):
            if callee not in reachable:
                reachable.add(callee)
                queue.append(callee)

    # Follow parameter value flow rather than whole-function ancestry. A lookup
    # parameter can sit under a very large recursive closure while only a small
    # chain of forwarded parameters can feed that argument. For each requested
    # parameter, inspect every reachable callsite to its Cell, expand frozen
    # integer sequence reductions only in those callers, and recursively request
    # proof for caller parameters referenced by the argument expression.  This is
    # complete for the selected output while avoiding unrelated recursive logic.
    callers_by_callee: dict[str, set[str]] = {}
    for caller in sorted(reachable):
        for callee in callees(formulas[caller].function):
            if callee in reachable:
                callers_by_callee.setdefault(callee, set()).add(caller)

    proof_functions: dict[str, ast.FunctionDef] = {}

    def proof_function(uid: str) -> ast.FunctionDef:
        cached = proof_functions.get(uid)
        if cached is not None:
            return cached
        fn = copy.deepcopy(formulas[uid].function)
        try:
            fn = evidence._normalize_static_integer_sequence_reductions(fn)
            fn = unroll_literal_reductions(fn)
        except Exception:
            # Optional evidence must fail closed. Raw canonical calls can still
            # prove literal/parameter forwarding even when a sequence provider is
            # outside this narrow bridge.
            fn = copy.deepcopy(formulas[uid].function)
        fn = ast.fix_missing_locations(fn)
        proof_functions[uid] = fn
        return fn

    def parameter_dependencies(expr: ast.AST, caller_uid: str):
        params = set(formulas[caller_uid].parameters)
        if isinstance(expr, ast.Constant) and type(expr.value) is int:
            return set()
        if isinstance(expr, ast.Name):
            if expr.id not in params:
                return None
            return {(caller_uid, expr.id)}
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            return parameter_dependencies(expr.operand, caller_uid)
        if isinstance(expr, ast.IfExp):
            left = parameter_dependencies(expr.body, caller_uid)
            right = parameter_dependencies(expr.orelse, caller_uid)
            if left is None or right is None:
                return None
            return left | right
        if isinstance(expr, ast.BinOp) and isinstance(
            expr.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
        ):
            left = parameter_dependencies(expr.left, caller_uid)
            right = parameter_dependencies(expr.right, caller_uid)
            if left is None or right is None:
                return None
            return left | right
        return None

    inbound: dict[tuple[str, str], list[tuple[str, ast.AST]]] = {}
    pending = [key for key in sorted(wanted) if key[0] in reachable]
    requested: set[tuple[str, str]] = set()
    while pending:
        key = pending.pop()
        if key in requested:
            continue
        requested.add(key)
        callee_uid, param_name = key
        callee = formulas.get(callee_uid)
        if callee is None or param_name not in callee.parameters:
            continue
        position = callee.parameters.index(param_name)
        for caller_uid in sorted(callers_by_callee.get(callee_uid, ())):
            fn = proof_function(caller_uid)
            sites = [
                node for node in ast.walk(fn)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == callee_uid
            ]
            if not sites:
                inbound.setdefault(key, []).append((caller_uid, ast.Constant(None)))
                continue
            for node in sites:
                if len(node.args) != len(callee.parameters) or node.keywords:
                    inbound.setdefault(key, []).append((caller_uid, ast.Constant(None)))
                    continue
                arg = copy.deepcopy(node.args[position])
                inbound.setdefault(key, []).append((caller_uid, arg))
                deps = parameter_dependencies(arg, caller_uid)
                if deps is None:
                    continue
                for dep in sorted(deps):
                    if dep[0] in reachable and dep not in requested:
                        pending.append(dep)

    domains: dict[tuple[str, str], set[int]] = {}

    def eval_values(expr: ast.AST, caller_uid: str) -> set[int] | None:
        if isinstance(expr, ast.Constant) and type(expr.value) is int:
            return {int(expr.value)}
        if isinstance(expr, ast.Name):
            vals = domains.get((caller_uid, expr.id))
            return None if not vals else set(vals)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            vals = eval_values(expr.operand, caller_uid)
            if vals is None:
                return None
            return {v if isinstance(expr.op, ast.UAdd) else -v for v in vals}
        if isinstance(expr, ast.IfExp):
            left = eval_values(expr.body, caller_uid)
            right = eval_values(expr.orelse, caller_uid)
            if left is None or right is None:
                return None
            out = left | right
            return out if len(out) <= max_values else None
        if isinstance(expr, ast.BinOp) and isinstance(
            expr.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
        ):
            left = eval_values(expr.left, caller_uid)
            right = eval_values(expr.right, caller_uid)
            if left is None or right is None or len(left) * len(right) > max_values * max_values:
                return None
            out: set[int] = set()
            for a in left:
                for b in right:
                    try:
                        if isinstance(expr.op, ast.Add):
                            value = a + b
                        elif isinstance(expr.op, ast.Sub):
                            value = a - b
                        elif isinstance(expr.op, ast.Mult):
                            value = a * b
                        elif isinstance(expr.op, ast.FloorDiv):
                            if b == 0:
                                return None
                            value = a // b
                        else:
                            if b == 0:
                                return None
                            value = a % b
                    except Exception:
                        return None
                    if not (-(1 << 63) <= value <= (1 << 63) - 1):
                        return None
                    out.add(int(value))
                    if len(out) > max_values:
                        return None
            return out
        return None

    def expr_dependencies(expr: ast.AST, caller_uid: str):
        """Return referenced caller parameter keys for the supported integer subset."""
        if isinstance(expr, ast.Constant) and type(expr.value) is int:
            return set()
        if isinstance(expr, ast.Name):
            if expr.id not in set(formulas[caller_uid].parameters):
                return None
            return {(caller_uid, expr.id)}
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            return expr_dependencies(expr.operand, caller_uid)
        if isinstance(expr, ast.IfExp):
            left = expr_dependencies(expr.body, caller_uid)
            right = expr_dependencies(expr.orelse, caller_uid)
            if left is None or right is None:
                return None
            return left | right
        if isinstance(expr, ast.BinOp) and isinstance(
            expr.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
        ):
            left = expr_dependencies(expr.left, caller_uid)
            right = expr_dependencies(expr.right, caller_uid)
            if left is None or right is None:
                return None
            return left | right
        return None

    # Add whatever is currently provable, then let recursive/name forwarding grow
    # the finite sets to convergence.  Sets never shrink, so the bound is explicit.
    for _ in range(max_iterations):
        changed = False
        for key, sites in inbound.items():
            target = domains.setdefault(key, set())
            before = len(target)
            for caller_uid, expr in sites:
                values = eval_values(expr, caller_uid)
                if values is None:
                    continue
                target.update(values)
                if len(target) > max_values:
                    target.clear()
                    break
            if len(target) != before:
                changed = True
        if not changed:
            break

    # Completeness is separate from value propagation.  A parameter that received
    # one known call plus one unresolved call must not lend its partial value set to
    # a downstream symbolic field.  Build a parameter-dependency graph and solve it
    # by SCC so recursive forwarding cycles are accepted only when every external
    # dependency is itself complete.
    nodes = set(inbound)
    deps_by_key: dict[tuple[str, str], set[tuple[str, str]]] = {}
    blocked_keys: set[tuple[str, str]] = set()
    for key, sites in inbound.items():
        deps: set[tuple[str, str]] = set()
        for caller_uid, expr in sites:
            found = expr_dependencies(expr, caller_uid)
            if found is None:
                blocked_keys.add(key)
                continue
            deps.update(found)
        if any(dep not in nodes for dep in deps):
            blocked_keys.add(key)
        deps_by_key[key] = {dep for dep in deps if dep in nodes}

    index = 0
    stack: list[tuple[str, str]] = []
    on_stack: set[tuple[str, str]] = set()
    indices: dict[tuple[str, str], int] = {}
    lowlinks: dict[tuple[str, str], int] = {}
    components: list[set[tuple[str, str]]] = []

    def strongconnect(node: tuple[str, str]) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for dep in deps_by_key.get(node, ()):
            if dep not in indices:
                strongconnect(dep)
                lowlinks[node] = min(lowlinks[node], lowlinks[dep])
            elif dep in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[dep])
        if lowlinks[node] == indices[node]:
            component: set[tuple[str, str]] = set()
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == node:
                    break
            components.append(component)

    for node in sorted(nodes):
        if node not in indices:
            strongconnect(node)

    complete_keys: set[tuple[str, str]] = set()
    pending = list(components)
    for _ in range(len(components) + 1):
        changed = False
        rest: list[set[tuple[str, str]]] = []
        for component in pending:
            if any(member in blocked_keys for member in component):
                rest.append(component)
                continue
            outside = {
                dep
                for member in component
                for dep in deps_by_key.get(member, ())
                if dep not in component
            }
            if outside <= complete_keys:
                complete_keys.update(component)
                changed = True
            else:
                rest.append(component)
        pending = rest
        if not changed:
            break

    proven: dict[tuple[str, str], tuple[int, ...]] = {}
    sources: dict[tuple[str, str], str] = {}
    for key, sites in inbound.items():
        if key not in complete_keys:
            continue
        values = domains.get(key, set())
        if not values or len(values) > max_values:
            continue
        complete = True
        union: set[int] = set()
        for caller_uid, expr in sites:
            vals = eval_values(expr, caller_uid)
            if vals is None:
                complete = False
                break
            union.update(vals)
            if len(union) > max_values:
                complete = False
                break
        if not complete or union != values:
            continue
        if key not in wanted:
            continue
        proven[key] = tuple(sorted(values))
        sources[key] = (
            "selected_closure_source_calls+complete_RunDomain_frozen_integer_sequences"
        )
    return proven, sources


def _selected_guarded_parameter_lower_bounds(
    formulas: Mapping[str, _FormulaSource],
    output_uid: str,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], str]]:
    """Prove demand-driven parameter lower bounds from complete source call paths.

    This evidence exists for provider-domain admission, not general type inference.
    A parameter is requested only when it appears in the query of the explicit
    last-anchor-<=x StepLookup source idiom with a bare ``max(keys)`` result.  Every
    reachable callsite to that parameter must then have a sound lower bound.  Simple
    post-return guards are carried to later calls, so patterns such as::

        if t < 0: return ...
        if t == 0: return ...
        value(t)

    prove ``value.t >= 1`` for integer ``t`` without observing representative runtime
    extrema.  Unknown expressions, call binding, or cyclic/unproved forwarding fail
    closed.
    """

    def step_query_parameters(source: _FormulaSource) -> set[str]:
        fn = source.function
        params = set(source.parameters)
        body = list(fn.body)
        wanted: set[str] = set()
        for index, statement in enumerate(body[:-1]):
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                continue
            row_name = statement.targets[0].id
            next_statement = body[index + 1]
            if not (
                isinstance(next_statement, ast.Assign)
                and len(next_statement.targets) == 1
                and isinstance(next_statement.targets[0], ast.Name)
                and isinstance(next_statement.value, ast.ListComp)
                and len(next_statement.value.generators) == 1
            ):
                continue
            comp = next_statement.value.generators[0]
            if not (
                isinstance(comp.iter, ast.Attribute)
                and comp.iter.attr == "index"
                and isinstance(comp.iter.value, ast.Name)
                and comp.iter.value.id == row_name
                and len(comp.ifs) == 1
                and isinstance(comp.ifs[0], ast.Compare)
                and len(comp.ifs[0].ops) == 1
                and isinstance(comp.ifs[0].ops[0], ast.LtE)
                and len(comp.ifs[0].comparators) == 1
            ):
                continue
            keys_name = next_statement.targets[0].id
            bare_max = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "max"
                and len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == keys_name
                for later in body[index + 2:]
                for node in ast.walk(later)
            )
            if not bare_max:
                continue
            query = comp.ifs[0].comparators[0]
            wanted.update(
                node.id for node in ast.walk(query)
                if isinstance(node, ast.Name) and node.id in params
            )
        return wanted

    wanted = {
        (uid, parameter)
        for uid, source in formulas.items()
        for parameter in step_query_parameters(source)
    }
    if not wanted or output_uid not in formulas:
        return {}, {}

    def callees(fn: ast.FunctionDef) -> set[str]:
        return {
            node.func.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in formulas
        }

    reachable: set[str] = set()
    queue = [output_uid]
    while queue:
        uid = queue.pop()
        if uid in reachable or uid not in formulas:
            continue
        reachable.add(uid)
        queue.extend(sorted(callees(formulas[uid].function) - reachable))
    wanted = {key for key in wanted if key[0] in reachable}
    if not wanted:
        return {}, {}

    def copy_expr(node: ast.AST) -> ast.AST:
        return copy.deepcopy(node)

    def negate(node: ast.AST) -> ast.AST:
        return ast.UnaryOp(op=ast.Not(), operand=copy_expr(node))

    def block_always_terminates(statements: Sequence[ast.stmt]) -> bool:
        for statement in statements:
            if isinstance(statement, (ast.Return, ast.Raise)):
                return True
            if (
                isinstance(statement, ast.If)
                and statement.orelse
                and block_always_terminates(statement.body)
                and block_always_terminates(statement.orelse)
            ):
                return True
        return False

    # callee uid -> [(caller uid, call, path guards, local lower bounds)]
    #
    # Phase 2 carries one narrow but useful piece of loop semantics into provider
    # domain proofs: targets iterating a Python ``range`` (or a local alias built
    # from ``list(range(...))``) inherit the range's lower bound.  This is source
    # semantics, not representative trace evidence, and applies generically to
    # finite projection/table loops.
    callsites: dict[
        str, list[tuple[str, ast.Call, tuple[ast.AST, ...], dict[str, float]]]
    ] = {}

    def syntactic_range_lower_bound(
        node: ast.AST,
        local_bounds: Mapping[str, float],
        sequence_bounds: Mapping[str, float],
    ) -> float | None:
        if isinstance(node, ast.Name):
            return sequence_bounds.get(node.id)
        expr = node
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id in {"list", "tuple"}
            and len(expr.args) == 1
            and not expr.keywords
        ):
            expr = expr.args[0]
        if not (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "range"
            and expr.args
            and len(expr.args) <= 3
            and not expr.keywords
        ):
            return None
        start_expr = ast.Constant(0) if len(expr.args) == 1 else expr.args[0]
        if isinstance(start_expr, ast.Constant) and isinstance(start_expr.value, (int, float)) and not isinstance(start_expr.value, bool):
            return float(start_expr.value)
        if isinstance(start_expr, ast.Name):
            return local_bounds.get(start_expr.id)
        return None

    def walk_expr(
        node: ast.AST | None,
        caller_uid: str,
        guards: tuple[ast.AST, ...],
        local_bounds: Mapping[str, float],
        sequence_bounds: Mapping[str, float],
    ) -> None:
        if node is None:
            return
        if isinstance(node, ast.BoolOp):
            prior: list[ast.AST] = []
            for value in node.values:
                if isinstance(node.op, ast.And):
                    extra = tuple(copy_expr(x) for x in prior)
                else:
                    extra = tuple(negate(x) for x in prior)
                walk_expr(value, caller_uid, guards + extra, local_bounds, sequence_bounds)
                prior.append(value)
            return
        if isinstance(node, ast.IfExp):
            walk_expr(node.test, caller_uid, guards, local_bounds, sequence_bounds)
            walk_expr(node.body, caller_uid, guards + (copy_expr(node.test),), local_bounds, sequence_bounds)
            walk_expr(node.orelse, caller_uid, guards + (negate(node.test),), local_bounds, sequence_bounds)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            nested_bounds = dict(local_bounds)
            for comp in node.generators:
                walk_expr(comp.iter, caller_uid, guards, nested_bounds, sequence_bounds)
                if isinstance(comp.target, ast.Name):
                    lower = syntactic_range_lower_bound(comp.iter, nested_bounds, sequence_bounds)
                    if lower is not None:
                        nested_bounds[comp.target.id] = float(lower)
                for filt in comp.ifs:
                    walk_expr(filt, caller_uid, guards, nested_bounds, sequence_bounds)
            if isinstance(node, ast.DictComp):
                walk_expr(node.key, caller_uid, guards, nested_bounds, sequence_bounds)
                walk_expr(node.value, caller_uid, guards, nested_bounds, sequence_bounds)
            else:
                walk_expr(node.elt, caller_uid, guards, nested_bounds, sequence_bounds)
            return
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in reachable:
                callsites.setdefault(node.func.id, []).append(
                    (caller_uid, node, tuple(copy_expr(x) for x in guards), dict(local_bounds))
                )
            for arg in node.args:
                walk_expr(arg, caller_uid, guards, local_bounds, sequence_bounds)
            for keyword in node.keywords:
                walk_expr(keyword.value, caller_uid, guards, local_bounds, sequence_bounds)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                walk_expr(child, caller_uid, guards, local_bounds, sequence_bounds)
            elif isinstance(child, ast.comprehension):
                walk_expr(child.iter, caller_uid, guards, local_bounds, sequence_bounds)
                for filt in child.ifs:
                    walk_expr(filt, caller_uid, guards, local_bounds, sequence_bounds)

    def walk_statements(
        statements: Sequence[ast.stmt],
        caller_uid: str,
        guards: tuple[ast.AST, ...],
        local_bounds: Mapping[str, float] | None = None,
        sequence_bounds: Mapping[str, float] | None = None,
    ) -> None:
        active = tuple(copy_expr(x) for x in guards)
        locals_now = dict(local_bounds or {})
        sequences_now = dict(sequence_bounds or {})
        for statement in statements:
            if isinstance(statement, ast.If):
                walk_expr(statement.test, caller_uid, active, locals_now, sequences_now)
                walk_statements(statement.body, caller_uid, active + (copy_expr(statement.test),), locals_now, sequences_now)
                if statement.orelse:
                    walk_statements(statement.orelse, caller_uid, active + (negate(statement.test),), locals_now, sequences_now)
                body_ends = block_always_terminates(statement.body)
                else_ends = bool(statement.orelse) and block_always_terminates(statement.orelse)
                if body_ends and else_ends:
                    return
                if body_ends and (not statement.orelse or not else_ends):
                    active = active + (negate(statement.test),)
                elif else_ends and not body_ends:
                    active = active + (copy_expr(statement.test),)
                continue
            if isinstance(statement, ast.Return):
                walk_expr(statement.value, caller_uid, active, locals_now, sequences_now)
                return
            if isinstance(statement, ast.Raise):
                walk_expr(statement.exc, caller_uid, active, locals_now, sequences_now)
                return
            if isinstance(statement, ast.Assign):
                walk_expr(statement.value, caller_uid, active, locals_now, sequences_now)
                if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                    lower = syntactic_range_lower_bound(statement.value, locals_now, sequences_now)
                    if lower is not None:
                        sequences_now[statement.targets[0].id] = float(lower)
                continue
            if isinstance(statement, ast.AnnAssign):
                walk_expr(statement.value, caller_uid, active, locals_now, sequences_now)
                continue
            if isinstance(statement, ast.AugAssign):
                walk_expr(statement.value, caller_uid, active, locals_now, sequences_now)
                continue
            if isinstance(statement, ast.Expr):
                walk_expr(statement.value, caller_uid, active, locals_now, sequences_now)
                continue
            if isinstance(statement, ast.For):
                walk_expr(statement.iter, caller_uid, active, locals_now, sequences_now)
                child_bounds = dict(locals_now)
                if isinstance(statement.target, ast.Name):
                    lower = syntactic_range_lower_bound(statement.iter, child_bounds, sequences_now)
                    if lower is not None:
                        child_bounds[statement.target.id] = float(lower)
                walk_statements(statement.body, caller_uid, active, child_bounds, sequences_now)
                walk_statements(statement.orelse, caller_uid, active, locals_now, sequences_now)
                continue
            if isinstance(statement, ast.While):
                walk_expr(statement.test, caller_uid, active, locals_now, sequences_now)
                walk_statements(statement.body, caller_uid, active, locals_now, sequences_now)
                walk_statements(statement.orelse, caller_uid, active, locals_now, sequences_now)
                continue
            for child in ast.iter_child_nodes(statement):
                if isinstance(child, ast.expr):
                    walk_expr(child, caller_uid, active, locals_now, sequences_now)

    for caller_uid in sorted(reachable):
        walk_statements(formulas[caller_uid].function.body, caller_uid, ())

    def required_guard_atoms(node: ast.AST, truth: bool = True) -> tuple[tuple[ast.AST, bool], ...]:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return required_guard_atoms(node.operand, not truth)
        if isinstance(node, ast.BoolOp):
            if (truth and isinstance(node.op, ast.And)) or (
                (not truth) and isinstance(node.op, ast.Or)
            ):
                rows: list[tuple[ast.AST, bool]] = []
                for child in node.values:
                    rows.extend(required_guard_atoms(child, truth))
                return tuple(rows)
        return ((node, truth),)

    def guard_integer_lower_bound(
        guards: tuple[ast.AST, ...], parameter: str
    ) -> float | None:
        lower: int | None = None
        excluded: set[int] = set()
        for guard in guards:
            for candidate, truth in required_guard_atoms(guard):
                if not (
                    isinstance(candidate, ast.Compare)
                    and len(candidate.ops) == 1
                    and len(candidate.comparators) == 1
                ):
                    continue
                left, right, op = candidate.left, candidate.comparators[0], candidate.ops[0]
                if (
                    isinstance(left, ast.Name)
                    and left.id == parameter
                    and isinstance(right, ast.Constant)
                    and type(right.value) is int
                ):
                    value = int(right.value)
                elif (
                    isinstance(right, ast.Name)
                    and right.id == parameter
                    and isinstance(left, ast.Constant)
                    and type(left.value) is int
                ):
                    value = int(left.value)
                    swaps = {
                        ast.Lt: ast.Gt, ast.LtE: ast.GtE,
                        ast.Gt: ast.Lt, ast.GtE: ast.LtE,
                    }
                    op = swaps.get(type(op), type(op))()
                else:
                    continue

                if not truth:
                    inverses = {
                        ast.Lt: ast.GtE, ast.LtE: ast.Gt,
                        ast.Gt: ast.LtE, ast.GtE: ast.Lt,
                        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
                    }
                    inverse = inverses.get(type(op))
                    if inverse is None:
                        continue
                    op = inverse()

                bound: int | None = None
                if isinstance(op, ast.GtE):
                    bound = value
                elif isinstance(op, ast.Gt):
                    bound = value + 1
                elif isinstance(op, ast.Eq):
                    bound = value
                elif isinstance(op, ast.NotEq):
                    excluded.add(value)
                if bound is not None:
                    lower = bound if lower is None else max(lower, bound)

        if lower is None:
            return None
        while lower in excluded:
            lower += 1
        return float(lower)

    inbound: dict[
        tuple[str, str], list[tuple[str, ast.AST, tuple[ast.AST, ...], dict[str, float]]]
    ] = {}
    blocked: set[tuple[str, str]] = set()
    for callee_uid, sites in callsites.items():
        callee = formulas[callee_uid]
        for caller_uid, call, guards, local_bounds in sites:
            if len(call.args) != len(callee.parameters) or call.keywords:
                blocked.update((callee_uid, parameter) for parameter in callee.parameters)
                continue
            for parameter, arg in zip(callee.parameters, call.args):
                inbound.setdefault((callee_uid, parameter), []).append(
                    (caller_uid, copy_expr(arg), guards, dict(local_bounds))
                )

    known: dict[tuple[str, str], float] = {}

    def expression_lower_bound(
        node: ast.AST,
        caller_uid: str,
        guards: tuple[ast.AST, ...],
        local_bounds: Mapping[str, float],
    ) -> float | None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            return None
        if isinstance(node, ast.Name):
            if node.id in local_bounds:
                return float(local_bounds[node.id])
            if node.id not in formulas[caller_uid].parameters:
                return None
            guarded = guard_integer_lower_bound(guards, node.id)
            inherited = known.get((caller_uid, node.id))
            if guarded is None:
                return inherited
            if inherited is None:
                return guarded
            return max(float(guarded), float(inherited))
        if isinstance(node, ast.UnaryOp):
            value = expression_lower_bound(node.operand, caller_uid, guards, local_bounds)
            if isinstance(node.op, ast.UAdd):
                return value
            return None
        if isinstance(node, ast.BinOp):
            left = expression_lower_bound(node.left, caller_uid, guards, local_bounds)
            right = expression_lower_bound(node.right, caller_uid, guards, local_bounds)
            if isinstance(node.op, ast.Add) and left is not None and right is not None:
                return left + right
            if isinstance(node.op, ast.Sub) and left is not None and isinstance(
                node.right, ast.Constant
            ) and isinstance(node.right.value, (int, float)) and not isinstance(node.right.value, bool):
                return left - float(node.right.value)
            if isinstance(node.op, ast.Mult) and left is not None and right is not None:
                # Multiplication is monotone only when both proven intervals are
                # non-negative. Lower-bound-only evidence cannot safely do more.
                if left >= 0.0 and right >= 0.0:
                    return left * right
            return None
        if isinstance(node, ast.IfExp):
            body = expression_lower_bound(node.body, caller_uid, guards + (copy_expr(node.test),), local_bounds)
            other = expression_lower_bound(node.orelse, caller_uid, guards + (negate(node.test),), local_bounds)
            if body is None or other is None:
                return None
            return min(body, other)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"int", "float"} and len(node.args) == 1 and not node.keywords:
                return expression_lower_bound(node.args[0], caller_uid, guards, local_bounds)
            if node.func.id == "max" and node.args and not node.keywords:
                rows = [expression_lower_bound(arg, caller_uid, guards, local_bounds) for arg in node.args]
                known_rows = [row for row in rows if row is not None]
                return max(known_rows) if known_rows else None
            if node.func.id == "min" and node.args and not node.keywords:
                rows = [expression_lower_bound(arg, caller_uid, guards, local_bounds) for arg in node.args]
                if any(row is None for row in rows):
                    return None
                return min(float(row) for row in rows if row is not None)
        return None

    # Monotone fixed point: a key becomes proven only when every reachable callsite
    # has a lower bound. Cycles with no independently bounded edge remain unknown.
    all_keys = {
        (uid, parameter)
        for uid in reachable
        for parameter in formulas[uid].parameters
    }
    for _ in range(len(all_keys) + 1):
        changed = False
        for key in sorted(all_keys):
            if key in blocked:
                continue
            sites = inbound.get(key)
            if not sites:
                continue
            rows: list[float] = []
            complete = True
            for caller_uid, expr, guards, local_bounds in sites:
                value = expression_lower_bound(expr, caller_uid, guards, local_bounds)
                if value is None:
                    complete = False
                    break
                rows.append(float(value))
            if not complete or not rows:
                continue
            value = min(rows)
            if key not in known or value > known[key]:
                known[key] = value
                changed = True
        if not changed:
            break

    proven = {key: known[key] for key in wanted if key in known}
    sources = {
        key: "selected_closure_source_calls+dominating_early_return_guards"
        for key in proven
    }
    return proven, sources

def _build_scalarized_vector_helpers(
    evidence: FastFormulaTranslationService,
    formulas: Mapping[str, _FormulaSource],
    class_info: ClassInfo,
) -> dict[str, FastScalarizedVectorHelper]:
    result: dict[str, FastScalarizedVectorHelper] = {}
    default_range = _vector_dense_range(class_info)
    for source_uid, (parameter, _range_call, element) in evidence.vector_elements.items():
        source = formulas.get(source_uid)
        if source is None:
            continue
        cell_info = class_info.cells[source.name]
        rt = getattr(cell_info, "_rt", None)
        if rt is None or not rt.ret_type.is_array:
            continue
        dtype = _fast_dtype(rt.ret_type.value_type)
        if dtype not in _ALLOWED_DTYPES:
            continue
        digest = hashlib.sha256((source_uid + "::vector_element").encode("utf-8")).hexdigest()[:12]
        uid = f"vh_{digest}_{_safe_ident(source.name)}_element"
        try:
            raw_element = evidence._inline_vector_subscripts(copy.deepcopy(element))
            fn = ast.FunctionDef(
                name=uid,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=parameter)],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=[ast.Return(value=copy.deepcopy(raw_element))],
                decorator_list=[],
            )
            fn = ast.fix_missing_locations(fn)
            fn = evidence._normalize_external_tables(fn, operator_uid=source_uid)
            lower = evidence._expr_lowerer(source_uid)
            lower.prepare_function(fn)
            fn.body = [lower.visit(copy.deepcopy(st)) for st in fn.body]
            fn = ast.fix_missing_locations(fn)
            fn = lower.cleanup_function(fn)
            fn = evidence._normalize_scalar_intrinsics(fn)
            fn = evidence._normalize_simple_local_syntax(fn)
        except Exception:
            # Scalarization is an optimization only.  If its element cannot use
            # the existing mature provider normalizer, normal vector inlining
            # remains the compatibility behavior.
            continue
        result[source_uid] = FastScalarizedVectorHelper(
            uid=uid,
            source_uid=source_uid,
            source_name=source.name,
            parameter=parameter,
            dtype=dtype,
            dense_range=default_range,
            function=fn,
            raw_element=ast.fix_missing_locations(raw_element),
        )
    return result


def _build_finite_vector_element_helpers(
    evidence: FastFormulaTranslationService,
    formulas: Mapping[str, _FormulaSource],
    class_info: ClassInfo,
    *,
    max_slots: int = 4096,
) -> dict[tuple[str, int | None], FiniteVectorElementHelper]:
    """Recognize and validate bounded list/tuple vector producers.

    This is intentionally separate from modelx-cython's NumPy-array helper path.
    The target producer returns Python list/tuple objects in ordinary modelx, but
    the scalarized native representation never does.  Result shape is proved by
    the accepted source constructor and bounded over the complete RunDomain; live
    model evaluation is used only to validate numeric element dtype/max length,
    never to materialize recursive vector values into production inputs.
    """

    raw: dict[tuple[str, int | None], FiniteVectorElementHelper] = {}
    for uid, source in sorted(formulas.items()):
        for helper in build_comprehension_vector_helpers(
            source_uid=uid,
            source_name=source.name,
            parameters=source.parameters,
            function=source.function,
        ):
            raw[helper.key] = helper

    # Recursive builders may depend on already recognized tuple/vector helpers.
    # Iterate to a fixed point over formula count, not over runtime state.
    for _ in range(max(1, len(formulas))):
        changed = False
        for uid, source in sorted(formulas.items()):
            if (uid, None) in raw:
                continue
            helper = build_recursive_append_vector_helper(
                source_uid=uid,
                source_name=source.name,
                parameters=source.parameters,
                function=source.function,
                known_helpers=raw,
            )
            if helper is not None:
                raw[helper.key] = helper
                changed = True
        if not changed:
            break

    validated: dict[tuple[str, int | None], FiniteVectorElementHelper] = {}
    # Use the frozen RunDomain ordinal rather than the key object itself so the
    # preparation cache also works for legitimate unhashable invocation keys.
    value_cache: dict[tuple[str, int, tuple[Any, ...]], Any] = {}
    for key, helper in sorted(raw.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1])):
        source = formulas[helper.source_uid]
        cell_info = class_info.cells[source.name]
        params = _parameter_evidence(cell_info, class_info)
        if len(params) != len(source.parameters):
            continue
        # Synthetic vector elements currently need a dense native memo range.
        # Do not create a synthetic non-dense key whose complete-domain hash
        # cardinality cannot be observed in the exported source trace.
        if any(p.dtype not in {"int64", "bool"} or p.dense_range is None for p in params):
            if params:
                continue
        if any(
            isinstance(node, ast.Name) and node.id in set(source.parameters)
            for node in ast.walk(helper.length_expr)
        ):
            continue
        probe_args = tuple(int(p.dense_range[0]) for p in params)  # type: ignore[index]
        observed_kind = "bool"
        max_length = 0
        valid = True
        for run_ordinal, run_key in enumerate(evidence.run_keys):
            cache_key = (helper.source_uid, run_ordinal, probe_args)
            try:
                if cache_key not in value_cache:
                    value_cache[cache_key] = getattr(
                        evidence._space_instance(run_key), source.name
                    )(*probe_args)
                value = value_cache[cache_key]
                if helper.component is not None:
                    if not isinstance(value, (tuple, list)) or helper.component >= len(value):
                        valid = False
                        break
                    value = value[helper.component]
                if not isinstance(value, (tuple, list)) or not value or len(value) > max_slots:
                    valid = False
                    break
                max_length = max(max_length, len(value))
                for item in value:
                    if isinstance(item, (bool, np.bool_)):
                        continue
                    if isinstance(item, (int, np.integer)):
                        if observed_kind == "bool":
                            observed_kind = "int64"
                        continue
                    if isinstance(item, (float, np.floating)):
                        if not np.isfinite(float(item)):
                            valid = False
                            break
                        observed_kind = "float64"
                        continue
                    valid = False
                    break
                if not valid:
                    break
            except Exception:
                valid = False
                break
        if not valid or max_length <= 0:
            continue
        from dataclasses import replace
        validated[key] = replace(
            helper,
            dtype=observed_kind,
            max_length=max_length,
            parameter_dtypes=tuple(p.dtype for p in params) + ("int64",),
            parameter_ranges=tuple(p.dense_range for p in params) + ((0, max_length - 1),),
        )
    return validated


def _normalize_finite_vector_element_helpers(
    evidence: FastFormulaTranslationService,
    helpers: Mapping[tuple[str, int | None], FiniteVectorElementHelper],
    *,
    category_plan: ModelxCythonCategoryPlan,
) -> dict[tuple[str, int | None], FiniteVectorElementHelper]:
    """Run synthetic element formulas through the mature scalar provider pipeline."""
    from dataclasses import replace

    result: dict[tuple[str, int | None], FiniteVectorElementHelper] = {}
    for key, helper in helpers.items():
        try:
            candidate = copy.deepcopy(helper.function)
            candidate = category_plan.encoding.rewrite_function(helper.source_uid, candidate)
            candidate = evidence._inline_vector_subscripts(candidate)
            candidate = evidence._normalize_external_tables(candidate, operator_uid=helper.source_uid)
            lower = evidence._expr_lowerer(helper.source_uid)
            lower.prepare_function(candidate)
            candidate.body = [lower.visit(copy.deepcopy(st)) for st in candidate.body]
            candidate = ast.fix_missing_locations(candidate)
            candidate = lower.cleanup_function(candidate)
            candidate = evidence._normalize_scalar_intrinsics(candidate)
            candidate = evidence._normalize_simple_local_syntax(candidate)
        except Exception:
            continue
        result[key] = replace(helper, function=ast.fix_missing_locations(candidate))
    return result


def _scalarize_reduction_vector_elements(
    reduction: ReductionSpec,
    helpers: Mapping[str, FastScalarizedVectorHelper],
) -> ReductionSpec:
    body = copy.deepcopy(reduction.body_expr)
    # Replace the largest helper element first.  This naturally prefers a full
    # vector helper such as disc_factors over a nested helper such as
    # disc_rate_mth, without product/cell-name rules.
    ordered = sorted(
        helpers.values(),
        key=lambda h: -sum(1 for _ in ast.walk(h.raw_element)),
    )
    for helper in ordered:
        pattern = _NameSubstitute(
            helper.parameter, ast.Name(id=reduction.loop_var, ctx=ast.Load())
        ).visit(copy.deepcopy(helper.raw_element))
        assert isinstance(pattern, ast.AST)
        replacement = ast.Call(
            func=ast.Name(id=helper.uid, ctx=ast.Load()),
            args=[ast.Name(id=reduction.loop_var, ctx=ast.Load())],
            keywords=[],
        )
        rewrite = _ExactSubtreeReplace(ast.fix_missing_locations(pattern), replacement)
        body = rewrite.visit(body)
        body = ast.fix_missing_locations(body)
    return ReductionSpec(
        copy.deepcopy(reduction.init),
        reduction.target,
        reduction.loop_var,
        tuple(copy.deepcopy(x) for x in reduction.range_args),
        body,
        tuple(copy.deepcopy(x) for x in reduction.filters),
    )


def _normalize_formula(
    evidence: FastFormulaTranslationService,
    source: _FormulaSource,
    *,
    all_uids: set[str],
    module_refs: Mapping[str, str],
    vector_helpers: Mapping[str, FastScalarizedVectorHelper] | None = None,
    finite_vector_helpers: Mapping[tuple[str, int | None], FiniteVectorElementHelper] | None = None,
    category_plan: ModelxCythonCategoryPlan | None = None,
    point_uid: str | None = None,
    rank1_pointwise: bool = False,
    rank1_point_source_uids: set[str] | None = None,
    rank1_point_frame_ref: str | None = None,
    rank1_global_aggregate_uids: set[str] | None = None,
    source_name_by_uid: Mapping[str, str] | None = None,
    reference_metadata_uids: set[str] | None = None,
    structured_providers: Mapping[str, StructuredFramePlan] | None = None,
) -> FormulaNormalizationResult:
    blockers: list[tuple[str, str]] = list(source.preparation_blockers)
    fn: ast.FunctionDef | None = None
    reduction: ReductionSpec | None = None
    semantic: SemanticFormulaIR | None = None
    normalized_guards: tuple[GuardSpec, ...] = tuple(source.guards)
    try:
        candidate = unroll_literal_reductions(source.function)
        if reference_metadata_uids:
            candidate = _PreparedMetadataScalarCallNormalizer(
                evidence,
                metadata_uids=reference_metadata_uids,
                source_name_by_uid=source_name_by_uid or {},
            ).visit(candidate)
            assert isinstance(candidate, ast.FunctionDef)
            candidate = ast.fix_missing_locations(candidate)
        static_ref_names = set(evidence.refs)

        def predicate_has_static_candidate(test: ast.AST) -> bool:
            """Whether one branch predicate may consume authoritative static facts.

            Triggering the mature specialization pass is intentionally broader than
            the old ``is``/``is not`` special case, but admission remains narrow:
            ``_proven_static_reference_value`` / ``_proven_static_cell_value`` still
            require complete-RunDomain proof before any predicate is replaced.
            """
            for part in ast.walk(test):
                if (
                    isinstance(part, ast.Name)
                    and isinstance(part.ctx, ast.Load)
                    and part.id in static_ref_names
                ):
                    return True
                if (
                    isinstance(part, ast.Call)
                    and isinstance(part.func, ast.Name)
                    and not part.args and not part.keywords
                ):
                    source_name = (source_name_by_uid or {}).get(part.func.id, part.func.id)
                    if source_name in evidence.base_funcs:
                        return True
            return False

        needs_predicate_static = any(
            predicate_has_static_candidate(node.test)
            for node in ast.walk(candidate)
            if isinstance(node, (ast.If, ast.IfExp))
        )
        needs_value_materialization = any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in static_ref_names
            and isinstance(evidence.refs.get(node.value.id), (tuple, str))
            for node in ast.walk(candidate)
        )
        needs_static_reference_materialization = any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in static_ref_names
            and (isinstance(evidence.refs.get(node.id), str) or evidence.refs.get(node.id) is None)
            for node in ast.walk(candidate)
        )
        if needs_predicate_static or needs_value_materialization or needs_static_reference_materialization:
            candidate = evidence._specialize_static_facts_and_domains(candidate)
        if needs_value_materialization:
            candidate = evidence._materialize_static_value_operators(source.uid, candidate)
        if rank1_pointwise:
            candidate = _rank1_matmul_reduction(candidate)
            candidate = _Rank1ArrayFormulaNormalizer(
                evidence,
                point_source_uids=rank1_point_source_uids or set(),
                point_frame_ref=rank1_point_frame_ref,
                global_aggregate_uids=rank1_global_aggregate_uids or set(),
                source_name_by_uid=source_name_by_uid or {},
                structured_providers=structured_providers,
                current_uid=source.uid,
                category_plan=category_plan,
            ).visit(candidate)
            assert isinstance(candidate, ast.FunctionDef)
            candidate = ast.fix_missing_locations(candidate)
        # Phase 1 semantic boundary: source-shape normalization is now an
        # explicit, ordered pass pipeline.  The payload remains AST for now so
        # behavior is unchanged; future lookup/table IR nodes enter here rather
        # than teaching physical Cython lowering more source syntax.
        semantic = SemanticFormulaIR.from_function(source.uid, source.name, candidate)
        semantic_passes: list[AstSemanticPass] = []
        if finite_vector_helpers:
            semantic_passes.append(AstSemanticPass(
                "finite_vector_consumers",
                lambda fn: normalize_finite_vector_consumers(fn, finite_vector_helpers),
                ("recursive finite vectors remain runtime-computed",),
            ))
        semantic_passes.extend((
            AstSemanticPass(
                "bounded_filtered_min_sequences",
                normalize_bounded_filtered_min_listcomps,
                ("bounded Python list/filter/min syntax becomes scalar reduction semantics",),
            ),
            AstSemanticPass("static_sequence_access", evidence._normalize_static_sequence_access),
            AstSemanticPass("static_sequence_membership", evidence._normalize_static_sequence_membership),
            AstSemanticPass(
                "static_integer_sequence_reductions",
                evidence._normalize_static_integer_sequence_reductions,
            ),
        ))
        if category_plan is not None:
            semantic_passes.append(AstSemanticPass(
                "static_category_sequence_reductions",
                lambda fn: evidence._normalize_static_category_sequence_reductions(
                    fn, category_plan.encoding
                ),
            ))
        semantic_passes.append(AstSemanticPass("literal_reduction_unroll", unroll_literal_reductions))
        semantic = SemanticPassPipeline(
            semantic_passes, name="source_shape_normalization"
        ).run(semantic)
        candidate = copy.deepcopy(semantic.function)
        if category_plan is not None:
            field = category_plan.point_fields.get(source.uid)
            parameterized = category_plan.parameterized_point_bindings.get(source.uid)
            if field is not None or parameterized is not None:
                if point_uid is None or evidence.model_point_row is None:
                    raise ModelxCythonFastBackendError(f"{source.uid}: category point binding lacks point row")
                did = category_plan.encoding.result_domain_id(source.uid)
                if did is None:
                    raise ModelxCythonFastBackendError(f"{source.uid}: category result domain is missing")
                domain = category_plan.encoding.domain(did)
                if any(not isinstance(label, str) for label in domain.labels):
                    raise ModelxCythonFastBackendError(
                        f"{source.uid}: point category provider currently requires string-only labels"
                    )
                if field is not None:
                    key = evidence.registry.point_field(
                        evidence.model_point_row, field, enum_labels=tuple(domain.labels),
                        enum_allow_missing=category_plan.encoding.result_nullable(source.uid),
                    )
                    candidate = BindPointCategorySubscript(point_uid, field, key).visit(candidate)
                    candidate = ast.fix_missing_locations(candidate)
                if parameterized is not None:
                    required_by_field: dict[str, list[Any]] = {}
                    for case in parameterized.cases:
                        bucket = required_by_field.setdefault(case.field, [])
                        for key_value in case.required_run_keys:
                            if key_value not in bucket:
                                bucket.append(key_value)
                    for selected_field, required_keys in required_by_field.items():
                        key = evidence.registry.point_field(
                            evidence.model_point_row, selected_field,
                            enum_labels=tuple(domain.labels),
                            enum_required_keys=tuple(required_keys),
                            enum_allow_missing=parameterized.nullable,
                        )
                        candidate = BindPointCategorySubscript(
                            point_uid, selected_field, key
                        ).visit(candidate)
                        candidate = ast.fix_missing_locations(candidate)
            structured_selector = (
                _structured_field_selector(source.function, structured_providers or {})
                if rank1_pointwise and structured_providers else None
            )
            if not (
                structured_selector is not None
                and category_plan.encoding.result_domain_id(source.uid) is not None
            ):
                candidate = category_plan.encoding.rewrite_function(source.uid, candidate)
        # Record category binding/encoding as one semantic transition.  The
        # existing category implementation remains authoritative in Phase 1.
        if semantic is None:
            semantic = SemanticFormulaIR.from_function(source.uid, source.name, candidate)
        else:
            semantic = semantic.with_function(
                candidate,
                pass_name="category_binding_and_encoding",
                notes=("finite category ABI remains unchanged",),
            )

        operator_uids_before = set(evidence._normalized_operators)
        semantic = SemanticPassPipeline((
            AstSemanticPass("vector_subscript_inlining", evidence._inline_vector_subscripts),
            AstSemanticPass(
                "local_mapping_iteration",
                lambda fn: evidence._normalize_local_mapping_iteration(
                    fn, operator_uid=source.uid
                ),
                ("finite Mapping traversal lowers before expression lowering",),
            ),
        ), name="provider_pre_semantics").run(semantic)

        # Phase 2: ordered lookup syntax is recognized once at the semantic
        # boundary.  Physical provider lowering consumes the resulting IR rather
        # than independently rediscovering the source idiom.
        asof_ops = recognize_asof_lookups(semantic.function)
        semantic = semantic.with_semantic_ops((*semantic.semantic_ops, *asof_ops))
        semantic = semantic.with_function(
            semantic.function,
            pass_name="asof_lookup_recognition",
            notes=((f"recognized {len(asof_ops)} ordered lookup(s)",) if asof_ops else ("no ordered lookup",)),
        )
        semantic = SemanticPassPipeline((
            AstSemanticPass(
                "structured_provider_normalization",
                lambda fn: evidence._normalize_external_tables(
                    fn, operator_uid=source.uid, semantic_ops=semantic.semantic_ops
                ),
                ("semantic lookup/table operations lower before physical expression lowering",),
            ),
        ), name="provider_semantics").run(semantic)
        operator_uids_after = set(evidence._normalized_operators)
        semantic = semantic.with_operator_uids(tuple(operator_uids_after - operator_uids_before))
        candidate = copy.deepcopy(semantic.function)

        # Semantic operations may introduce fail-closed Python raises *after* the
        # source catalogue's ordinary guard pass.  Normalize those raises through
        # the same backend-neutral GuardSpec mechanism.  A deterministic high code
        # range keeps late semantic guards disjoint from source guards without
        # introducing a plan-order dependency; 1024 codes are reserved per UID.
        if any(isinstance(node, ast.Raise) for node in ast.walk(candidate)):
            digest = int(hashlib.sha256(source.uid.encode("utf-8")).hexdigest()[:12], 16)
            semantic_guard_base = 1_000_000_000 + digest * 1024
            candidate, late_guards, _next_semantic_guard = normalize_formula_guards(
                candidate, uid=source.uid, next_code=semantic_guard_base
            )
            normalized_guards = tuple((*source.guards, *late_guards))
            semantic = semantic.with_function(
                candidate,
                pass_name="semantic_runtime_guard_normalization",
                notes=(f"normalized {len(late_guards)} semantic runtime guard(s)",),
            )
        _prepare_fixed_count_range_bound_inputs(evidence, candidate)
        lower = evidence._expr_lowerer(source.uid)
        lower.prepare_function(candidate)
        reduction = evidence._normalize_reduction(candidate)
        if reduction is not None:
            if vector_helpers:
                reduction = _scalarize_reduction_vector_elements(reduction, vector_helpers)
            reduction = ReductionSpec(
                lower.visit(copy.deepcopy(reduction.init)),
                reduction.target,
                reduction.loop_var,
                tuple(lower.visit(copy.deepcopy(x)) for x in reduction.range_args),
                lower.visit(copy.deepcopy(reduction.body_expr)),
                tuple(lower.visit(copy.deepcopy(x)) for x in reduction.filters),
            )
            ast.fix_missing_locations(reduction.body_expr)
        else:
            candidate.body = [lower.visit(copy.deepcopy(st)) for st in candidate.body]
            candidate = ast.fix_missing_locations(candidate)
            candidate = lower.cleanup_function(candidate)
            candidate = evidence._normalize_scalar_intrinsics(candidate)
            candidate = evidence._normalize_simple_local_syntax(candidate)
            fn = candidate
    except Exception as exc:
        kind = "category" if isinstance(exc, CategoryABIError) else _exception_category(exc)
        blockers.append((kind, f"{type(exc).__name__}:{exc}"))

    # Residual/control admission is intentionally deferred until the final
    # NativeFormulaCell has authoritative result/parameter/local type evidence.
    # Earlier versions screened here with placeholder int64 parameter types,
    # which made it impossible to carry one evidence representation into both
    # admission and codegen for I4b local control.
    return FormulaNormalizationResult(
        semantic=semantic,
        function=fn,
        reduction=reduction,
        blockers=tuple(sorted(set(blockers))),
        guards=normalized_guards,
    )


def _callee_uids(cell: NativeFormulaCell, all_uids: set[str]) -> set[str]:
    return {
        node.func.id
        for tree in _tree_parts(cell)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in all_uids
    }


def build_modelx_cython_fast_plan(
    classes_path: str | Path,
    *,
    model_path: str | Path,
    sample_path: str | Path,
    spec_data: Mapping[str, Any] | None,
    model: Any,
    space: Any,
    output_name: str,
    sample_keys: Sequence[Any],
    run_keys: Sequence[Any],
    class_name: str = "_c_Projection",
    module_name: str | None = None,
) -> ModelxCythonFastPlan:
    """Build a fast specialization from modelx-cython frontend metadata.

    Every lexical Cell is catalogued. Numeric/provider-compatible Cells enter the
    sidecar native program. Object/unobserved/unsupported Cells remain explicit
    ``object_fallback`` entries owned by the ordinary modelx-cython backend.
    """

    total_started = perf_counter()
    timings: dict[str, float] = {}
    phase_memory: dict[str, dict[str, int | None]] = {}

    def finish_phase(name: str, started: float) -> None:
        timings[name] = perf_counter() - started
        memory = _process_memory_kb()
        phase_memory[name] = memory
        _append_planning_diagnostic(
            {
                "phase": name,
                "elapsed_s": timings[name],
                **memory,
            }
        )

    phase_started = total_started
    classes_path = Path(classes_path)
    model_path = Path(model_path)
    from .invocation import complete_space_invocation_keys
    run_keys = complete_space_invocation_keys(space, run_keys)
    sample_keys = complete_space_invocation_keys(space, sample_keys)
    module_name = module_name or f"{model_path.name}._mx_classes"
    source_text = classes_path.read_text(encoding="utf-8")
    finish_phase("source_load", phase_started)

    # ``run_sample`` is normally only a representative type-discovery trace and
    # therefore its min/max argument observations are not provider proof.  When
    # the caller deliberately traced every frozen RunDomain key, however, those
    # observations are complete call evidence for the selected output.  Preserve
    # that distinction explicitly rather than smuggling representative extrema
    # into provider normalization.
    trace_covers_run_domain = (
        len(sample_keys) == len(run_keys)
        and set(sample_keys) == set(run_keys)
    )

    phase_started = perf_counter()
    logger = run_sample(model_path, str(sample_path))
    finish_phase("modelx_cython_trace", phase_started)

    phase_started = perf_counter()
    visitor = ModuleVisitor(module=module_name, source=source_text)
    spec = TransSpec(dict(spec_data or {}))
    module_info = ModuleInfo(module_name, visitor, logger, spec)
    if class_name not in module_info.classes:
        raise ModelxCythonFastBackendError(f"modelx-cython class {class_name!r} is absent")
    class_info = module_info.classes[class_name]
    formulas, uid_by_name, next_guard = _build_formula_catalog(
        source_text,
        module_name=module_name,
        class_name=class_name,
        class_info=class_info,
    )
    if output_name not in uid_by_name:
        raise ModelxCythonFastBackendError(f"output Cell {output_name!r} is absent")
    output_uid = uid_by_name[output_name]
    output_result_dtype, _output_result_source = _result_evidence(
        class_info.cells[output_name], allow_array=True
    )
    ranked_output = _ranked_output_evidence(
        class_info.cells[output_name], space, output_result_dtype
    )
    _validate_ranked_series_labels(class_info, ranked_output)

    # M7C: discover cross-Space identity only when the selected class actually
    # contains a call rooted in a metadata-typed Space ref/child.  The common
    # same-Space path therefore does not pay for a whole-export module scan.
    cross_formulas: dict[str, _FormulaSource] = {}
    cross_class_by_uid: dict[str, ClassInfo] = {}
    cross_cell_by_uid: dict[str, Any] = {}
    cross_live_scope_by_uid: dict[str, Any] = {}
    cross_records: list[FastCrossSpaceCall] = []
    if _has_cross_space_candidate(formulas, class_info):
        scan_started = perf_counter()
        module_infos, module_sources = _module_catalogue(
            model_path, root_module_name=module_name, root_module_info=module_info,
            root_source=source_text, root_path=classes_path, logger=logger, spec=spec,
        )
        resolver = build_resolver(module_infos)
        class_by_fullname = _class_index(module_infos)
        cell_owner = _cell_owner_index(class_by_fullname)
        live_by_class = _live_space_map(
            root_class=class_info, root_space=space,
            class_by_fullname=class_by_fullname, resolver=resolver,
        )
        root_space_params = tuple(space.parameters or ())

        # Build formula catalogues lazily for only classes reached by a resolved
        # call from the selected class/its cross-Space callees.  Parsing the
        # exported module set is cheap; translating hundreds of unrelated formulas
        # merely to discover one call would be a very human sort of inefficiency.
        catalog_by_class: dict[str, dict[str, _FormulaSource]] = {}
        eligible_by_cell_fullname: dict[
            str, tuple[_FormulaSource, ClassInfo, tuple[str, ...]]
        ] = {}
        eligible_by_uid: dict[str, tuple[_FormulaSource, ClassInfo]] = {}

        def ensure_class_catalog(target_class: ClassInfo) -> None:
            nonlocal next_guard
            if target_class.fqname in catalog_by_class:
                return
            live_scope = live_by_class.get(target_class.fqname)
            target_space_params = _space_parameter_names(target_class)
            if (
                live_scope is None
                or any(name not in root_space_params for name in target_space_params)
            ):
                catalog_by_class[target_class.fqname] = {}
                return
            source_text_for_class = module_sources.get(target_class.module.fqname)
            if source_text_for_class is None or not target_class.cells:
                catalog_by_class[target_class.fqname] = {}
                return
            try:
                catalog, _target_uid_by_name, candidate_next_guard = _build_formula_catalog(
                    source_text_for_class,
                    module_name=target_class.module.fqname,
                    class_name=target_class.name,
                    class_info=target_class,
                    next_guard=next_guard,
                )
            except Exception:
                catalog_by_class[target_class.fqname] = {}
                return
            next_guard = candidate_next_guard
            catalog_by_class[target_class.fqname] = catalog
            for uid, row in catalog.items():
                cell = target_class.cells[row.name]
                result_dtype, _result_source = _result_evidence(cell)
                params = _parameter_evidence(cell, target_class)
                if result_dtype not in _ALLOWED_DTYPES:
                    continue
                if any(p.dtype not in {"int64", "bool"} for p in params):
                    continue
                eligible_by_cell_fullname[cell.fqname] = (
                    row, target_class, target_space_params
                )
                eligible_by_uid[uid] = (row, target_class)
                cross_class_by_uid[uid] = target_class
                cross_cell_by_uid[uid] = cell
                cross_live_scope_by_uid[uid] = live_scope

        def ensure_cell(cell_fullname: str) -> None:
            owner = cell_owner.get(cell_fullname)
            if owner is None:
                return
            target_class, cell = owner
            if target_class.fqname == class_info.fqname:
                return
            # Reject object/unobserved targets before parsing their entire class.
            result_dtype, _source = _result_evidence(cell)
            params = _parameter_evidence(cell, target_class)
            if result_dtype not in _ALLOWED_DTYPES:
                return
            if any(p.dtype not in {"int64", "bool"} for p in params):
                return
            ensure_class_catalog(target_class)

        def bind_cross(source: _FormulaSource, caller_class: ClassInfo) -> _FormulaSource:
            for target_fullname in _resolved_cell_fullnames(
                source, caller_class=caller_class, resolver=resolver
            ):
                ensure_cell(target_fullname)
            binder = _BindCrossSpaceCellCalls(
                caller_uid=source.uid, caller_class_fullname=caller_class.fqname,
                resolver=resolver, eligible_by_cell_fullname=eligible_by_cell_fullname,
                records=cross_records,
            )
            function = binder.visit(copy.deepcopy(source.function))
            assert isinstance(function, ast.FunctionDef)
            return _FormulaSource(
                uid=source.uid,
                name=source.name,
                fullname=source.fullname,
                function=ast.fix_missing_locations(function),
                parameters=source.parameters,
                defaults=source.defaults,
                guards=source.guards,
                preparation_blockers=source.preparation_blockers,
                lexical_source=source.lexical_source,
            )

        formulas = {
            uid: bind_cross(row, class_info) for uid, row in formulas.items()
        }
        queue: list[str] = []
        for row in formulas.values():
            for node in ast.walk(row.function):
                if (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in eligible_by_uid and node.func.id not in queue
                ):
                    queue.append(node.func.id)
        seen_cross: set[str] = set()
        while queue:
            uid = queue.pop(0)
            if uid in seen_cross or uid not in eligible_by_uid:
                continue
            seen_cross.add(uid)
            source, caller_class = eligible_by_uid[uid]
            bound = bind_cross(source, caller_class)
            cross_formulas[uid] = bound
            # Same-Space calls were already UID-bound by _build_formula_catalog;
            # cross-Space calls have just joined them.  Queue only eligible native
            # targets, not an execution graph or whole-model closure structure.
            for node in ast.walk(bound.function):
                if (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in eligible_by_uid
                    and node.func.id not in seen_cross
                    and node.func.id not in queue
                ):
                    queue.append(node.func.id)
        finish_phase("cross_space_metadata", scan_started)

    # De-duplicate repeated call sites while keeping deterministic evidence.
    cross_records = list({
        (r.caller_uid, r.caller_class_fullname, r.receiver_chain, r.callee_uid): r
        for r in cross_records
    }.values())
    cross_records.sort(key=lambda r: (
        r.caller_uid, r.receiver_chain, r.callee_uid, r.callee_class_fullname
    ))
    finish_phase("modelx_cython_lexical", phase_started)

    phase_started = perf_counter()
    # The modelx dependency trace inside the preparation service is discovery
    # evidence, not completeness authority.  Feeding the entire representative
    # trace set into that graph capture multiplies retained modelx/networkx state
    # on highly recursive products (IUL is the canonical example) without making
    # any proof stronger.  Keep one deterministic discovery witness here; type
    # evidence still comes from the modelx-cython trace above and all static/
    # provider/range/hash proofs continue to use the complete ``run_keys`` domain.
    translation_trace_keys = _translation_discovery_keys(sample_keys)
    evidence = FastFormulaTranslationService(
        model,
        space=space,
        output=output_name,
        sample_keys=translation_trace_keys,
        run_keys=run_keys,
        point_keys=None if ranked_output is None else ranked_output.axis_keys,
        point_frame=(
            None if ranked_output is None or ranked_output.point_frame_ref is None
            else space.refs[ranked_output.point_frame_ref]
        ),
    )
    evidence.registry = InputRegistry(evidence)
    evidence.multiindex_axes = {}
    evidence._normalized_operators = {}

    # Dev146: a parameterized Space selected inside a scalar formula may expose
    # ordinary numeric Cells behind a local alias, e.g.::
    #
    #     comf = comm_table[sex, rate, table]
    #     return comf.Axn(x, n) + comf.AnnDuenx(x, n)
    #
    # Treating ``comf`` as an opaque Python object leaves the selected Space's
    # recursive modelx cache graph on the hot path.  Physicalize one proven
    # selection context per target class instead: target scalar Cells join the
    # canonical native Cell catalogue; target Space parameters become prepared
    # point inputs; zero-argument numeric rank-1 leaf Cells become prepared
    # [point, axis] tables.  The recursive scalar Cells themselves therefore use
    # the same C memo wrappers as every other N0/N1 Cell.
    class_by_fqname = {info.fqname: info for info in module_info.classes.values()}
    parameterized_contexts: dict[str, dict[str, Any]] = {}

    def root_context(run_key: Any) -> Any:
        if isinstance(run_key, tuple):
            return space[run_key]
        return space[run_key]

    def selector_tuple(node: ast.Subscript) -> tuple[ast.AST, ...]:
        if isinstance(node.slice, ast.Tuple):
            return tuple(copy.deepcopy(item) for item in node.slice.elts)
        return (copy.deepcopy(node.slice),)

    def formula_singletons(source: _FormulaSource) -> dict[str, Any]:
        cell = class_info.cells[source.name]
        result: dict[str, Any] = {}
        for param in _parameter_evidence(cell, class_info):
            if param.minimum is None or param.maximum is None or param.minimum != param.maximum:
                continue
            value: Any = param.minimum
            if param.dtype in {"int64", "bool"}:
                value = int(value)
            elif param.dtype == "float64":
                value = float(value)
            result[param.name] = value
        return result

    def selected_item_factory(
        ref_name: str,
        selectors: tuple[ast.AST, ...],
        singletons: Mapping[str, Any],
    ):
        def selected(run_key: Any):
            context = root_context(run_key)
            values = tuple(
                _eval_parameterized_selector_expr(
                    node, context=context, singleton_parameters=singletons
                )
                for node in selectors
            )
            target = getattr(context, ref_name)
            key: Any = values[0] if len(values) == 1 else values
            return target[key]

        return selected

    def register_parameterized_target(
        *,
        target_class: ClassInfo,
        selected_item: Any,
        selection_fingerprint: tuple[Any, ...],
    ) -> dict[str, Any] | None:
        nonlocal next_guard
        existing = parameterized_contexts.get(target_class.fqname)
        if existing is not None:
            if existing["fingerprint"] != selection_fingerprint:
                return None
            return existing

        try:
            target_catalog, _target_uid_by_name, candidate_next_guard = _build_formula_catalog(
                source_text,
                module_name=module_name,
                class_name=target_class.name,
                class_info=target_class,
                next_guard=next_guard,
            )
        except Exception:
            return None
        next_guard = candidate_next_guard
        target_by_name = {row.name: row for row in target_catalog.values()}

        scalar_uids: set[str] = set()
        array_leaf_uids: set[str] = set()
        for target_uid, target_source in target_catalog.items():
            if target_source.name == "__call__":
                continue
            target_cell = target_class.cells[target_source.name]
            ndim, _shape, _shape_source = _result_geometry(target_cell)
            if ndim == 0:
                dtype, dtype_source = _result_evidence(target_cell)
                params = _parameter_evidence(target_cell, target_class)
                # An unobserved branch in the modelx-cython trace is not
                # evidence that the Cell is non-numeric.  Keep such scalar
                # formulas in the specialization catalogue and let the later
                # demand-driven type pass prove them from native callers and
                # already-proven callees.  Concrete object/string evidence is
                # still rejected here.  TradLife CommTable.Exn is the canonical
                # case: the representative TERM policy never executes the ENDW
                # branch, but gross_prem_rate still contains a statically
                # reachable Exn call that must not fall back to a prepared
                # modelx method input.
                result_admissible = (
                    dtype in _ALLOWED_DTYPES
                    or (dtype == "object" and dtype_source == "unobserved_runtime_return")
                )
                params_admissible = all(
                    param.dtype in _ALLOWED_DTYPES
                    or (
                        param.dtype == "object"
                        and param.type_source == "unobserved_runtime_argument"
                    )
                    for param in params
                )
                if result_admissible and params_admissible:
                    scalar_uids.add(target_uid)
            elif ndim == 1 and not target_source.parameters:
                dtype, _dtype_source = _result_evidence(target_cell, allow_array=True)
                if dtype in _ALLOWED_DTYPES:
                    array_leaf_uids.add(target_uid)

        if not scalar_uids:
            return None

        target_space_params = _space_parameter_names(target_class)
        parameter_evidence = {
            p.name: p
            for p in _parameter_evidence(target_class.cells["__call__"], target_class)
        } if "__call__" in target_class.cells else {}
        used_space_params = {
            node.id
            for target_uid in scalar_uids
            for node in ast.walk(target_catalog[target_uid].function)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in target_space_params
        }
        parameter_inputs: dict[str, str] = {}
        for name in sorted(used_space_params):
            param = parameter_evidence.get(name)
            if param is None or param.dtype not in _ALLOWED_DTYPES:
                return None

            def getter(keys, name=name, dtype=param.dtype, selected_item=selected_item):
                values = [getattr(selected_item(key), name) for key in keys]
                np_dtype = (
                    np.float64 if dtype == "float64" else
                    np.bool_ if dtype == "bool" else np.int64
                )
                return np.asarray(values, dtype=np_dtype)

            key = evidence.registry.frozen_point_scalar(
                (
                    "parameterized_space_parameter",
                    target_class.fqname,
                    name,
                    selection_fingerprint,
                ),
                param.dtype,
                getter,
                f"parameterized Space {target_class.fqname} parameter {name}",
            )
            parameter_inputs[name] = key

        ordinal_key = evidence.registry.point_ordinal()
        array_leaf_inputs: dict[str, tuple[str, str]] = {}
        for leaf_uid in sorted(array_leaf_uids):
            source = target_catalog[leaf_uid]
            cell = target_class.cells[source.name]
            dtype, _dtype_source = _result_evidence(cell, allow_array=True)

            def materialize(keys, source=source, dtype=dtype, selected_item=selected_item):
                rows: list[np.ndarray] = []
                width: int | None = None
                for run_key in keys:
                    item = selected_item(run_key)
                    value = getattr(item, source.name)()
                    index = getattr(value, "index", None)
                    if index is not None:
                        labels = tuple(index)
                        if labels != tuple(range(len(labels))):
                            raise FrontendError(
                                f"parameterized Space array leaf {source.fullname} requires "
                                "a zero-based contiguous integer axis"
                            )
                    np_dtype = (
                        np.float64 if dtype == "float64" else
                        np.bool_ if dtype == "bool" else np.int64
                    )
                    row = np.asarray(value, dtype=np_dtype)
                    if row.ndim != 1:
                        raise FrontendError(
                            f"parameterized Space array leaf {source.fullname} is not rank-1"
                        )
                    if width is None:
                        width = int(row.shape[0])
                    elif int(row.shape[0]) != width:
                        raise FrontendError(
                            f"parameterized Space array leaf {source.fullname} changed width"
                        )
                    rows.append(row)
                if not rows:
                    return np.empty((0, 0), dtype=np.float64)
                return np.ascontiguousarray(np.stack(rows, axis=0))

            sample = materialize(run_keys)
            key = evidence.registry._key("psarr")
            evidence.registry.specs[key] = InputSpec(
                key=key,
                dtype=dtype,
                ndim=2,
                scope="global",
                getter=materialize,
                description=f"parameterized Space numeric array leaf {source.fullname}",
                expected_shape=tuple(int(x) for x in sample.shape),
                domain_token=repr((
                    "parameterized_space_array_leaf",
                    target_class.fqname,
                    source.name,
                    selection_fingerprint,
                )),
            )
            array_leaf_inputs[leaf_uid] = (key, ordinal_key)

        rewritten_catalog: dict[str, _FormulaSource] = {}
        for target_uid in sorted(scalar_uids):
            source = target_catalog[target_uid]
            transformer = _ParameterizedSpaceTargetRewriter(
                parameter_inputs=parameter_inputs,
                array_leaf_inputs=array_leaf_inputs,
            )
            function = transformer.visit(copy.deepcopy(source.function))
            assert isinstance(function, ast.FunctionDef)
            if transformer.unsupported_array_calls:
                return None
            ast.fix_missing_locations(function)
            rewritten_catalog[target_uid] = replace(
                source,
                function=function,
                lexical_source=ast.unparse(function),
            )

        live_scope = selected_item(run_keys[0]) if run_keys else None
        for target_uid, target_source in rewritten_catalog.items():
            cross_formulas[target_uid] = target_source
            cross_class_by_uid[target_uid] = target_class
            cross_cell_by_uid[target_uid] = target_class.cells[target_source.name]
            cross_live_scope_by_uid[target_uid] = live_scope

        context = {
            "fingerprint": selection_fingerprint,
            "target_by_name": {
                source.name: source for source in rewritten_catalog.values()
            },
            "target_class": target_class,
        }
        parameterized_contexts[target_class.fqname] = context
        return context

    rewritten_formulas = dict(formulas)
    for root_uid, source in list(formulas.items()):
        singletons = formula_singletons(source)
        candidate_function = copy.deepcopy(rewritten_formulas[root_uid].function)
        changed = False
        for statement in list(candidate_function.body):
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Subscript)
                and isinstance(statement.value.value, ast.Name)
            ):
                continue
            alias_name = statement.targets[0].id
            ref_name = statement.value.value.id
            ref_info = class_info.refs.get(ref_name)
            target_fqname = getattr(ref_info, "mx_class", "") if ref_info is not None else ""
            target_class = class_by_fqname.get(target_fqname)
            if target_class is None:
                continue
            target_space_params = _space_parameter_names(target_class)
            selectors = selector_tuple(statement.value)
            if not target_space_params or len(selectors) != len(target_space_params):
                continue
            formula_param_names = set(source.parameters)
            selector_formula_names = set().union(*(
                _selector_expr_names(node) & formula_param_names for node in selectors
            ))
            if any(name not in singletons for name in selector_formula_names):
                continue
            selected_item = selected_item_factory(ref_name, selectors, singletons)
            try:
                selected_values = []
                for run_key in run_keys:
                    context = root_context(run_key)
                    values = tuple(
                        _eval_parameterized_selector_expr(
                            node, context=context, singleton_parameters=singletons
                        )
                        for node in selectors
                    )
                    selected_values.append(tuple(repr(value) for value in values))
                fingerprint = tuple(selected_values)
                context_info = register_parameterized_target(
                    target_class=target_class,
                    selected_item=selected_item,
                    selection_fingerprint=fingerprint,
                )
            except Exception:
                context_info = None
            if context_info is None:
                continue
            rewriter = _ParameterizedSpaceCallRewriter(
                alias_name=alias_name,
                selection_node=statement.value,
                target_by_name=context_info["target_by_name"],
            )
            transformed = rewriter.visit(copy.deepcopy(candidate_function))
            assert isinstance(transformed, ast.FunctionDef)
            if rewriter.escaped or not rewriter.used_uids:
                continue
            ast.fix_missing_locations(transformed)
            candidate_function = transformed
            changed = True
            for callee_uid in sorted(rewriter.used_uids):
                callee = cross_formulas[callee_uid]
                cross_records.append(FastCrossSpaceCall(
                    caller_uid=root_uid,
                    caller_class_fullname=class_info.fqname,
                    receiver_chain=(ref_name, callee.name),
                    callee_uid=callee_uid,
                    callee_class_fullname=target_class.fqname,
                    callee_cell_fullname=callee.fullname,
                    target_space_parameters=target_space_params,
                ))
        if changed:
            rewritten_formulas[root_uid] = replace(
                source,
                function=candidate_function,
                lexical_source=ast.unparse(candidate_function),
            )
    formulas = rewritten_formulas

    evidence.base_funcs = {uid: copy.deepcopy(row.function) for uid, row in formulas.items()}
    evidence.configure_formula_catalog({uid: row.name for uid, row in formulas.items()})
    # Preserve modelx-cython's own finite integer parameter evidence for provider
    # normalization.  Some object-returning helper paths are absent from the older
    # GraphModelCompiler VariantTrace catalogue even though CombinedCellsInfo has
    # authoritative argument type/min/max facts.  Provider lowerers may consume
    # this map, but it grants no execution scheduling authority.
    evidence.modelx_cython_parameter_domains = {}
    for uid, row in formulas.items():
        cell_info = class_info.cells[row.name]
        for param in _parameter_evidence(cell_info, class_info):
            domain = _provider_parameter_domain(
                cell_info, class_info, param,
                trace_covers_run_domain=trace_covers_run_domain,
            )
            if domain is not None:
                evidence.modelx_cython_parameter_domains[(uid, param.name)] = domain
    for uid, row in cross_formulas.items():
        owner = cross_class_by_uid[uid]
        cell_info = cross_cell_by_uid[uid]
        for param in _parameter_evidence(cell_info, owner):
            domain = _provider_parameter_domain(
                cell_info, owner, param,
                trace_covers_run_domain=trace_covers_run_domain,
            )
            if domain is not None:
                evidence.modelx_cython_parameter_domains[(uid, param.name)] = domain
    evidence.vector_elements = evidence._discover_vector_helpers()
    if evidence.model_point_row is not None:
        point_uid = uid_by_name.get(evidence.model_point_row.cell_name)
        if point_uid is not None:
            # ``replace`` is imported at module scope.  A conditional local import here
            # shadows it for the whole planner function and breaks fallback propagation
            # on paths that do not execute this branch (including cross-Space/ranked roots).
            evidence.model_point_row = replace(evidence.model_point_row, cell_name=point_uid)

    root_uids = set(formulas)
    all_uids = root_uids | set(cross_formulas)
    (
        evidence.finite_symbolic_parameter_domains,
        evidence.finite_symbolic_parameter_domain_sources,
    ) = _selected_finite_integer_parameter_domains(
        evidence, formulas, output_uid
    )
    (
        evidence.source_guarded_parameter_lower_bounds,
        evidence.source_guarded_parameter_lower_bound_sources,
    ) = _selected_guarded_parameter_lower_bounds(formulas, output_uid)
    vector_uids = {uid for uid in evidence.vector_elements if uid in root_uids}
    point_array_uids = {
        uid for uid, row in formulas.items()
        if ranked_output is not None
        and _result_geometry(class_info.cells[row.name])[:2]
        == (1, ranked_output.shape)
    }
    point_source_uids = _rank1_point_source_uids(
        formulas, None if ranked_output is None else ranked_output.point_frame_ref
    )
    global_aggregate_uids = _rank1_global_aggregate_uids(
        formulas, point_array_uids
    )
    reference_metadata_uids = {
        uid
        for uid, row in formulas.items()
        if uid != output_uid
        and not row.parameters
        and evidence._reference_metadata_scalar_ok(uid)
    }
    point_uid = None
    if evidence.model_point_row is not None and evidence.model_point_row.cell_name in all_uids:
        point_uid = evidence.model_point_row.cell_name
    materialized_uids = point_source_uids | global_aggregate_uids | reference_metadata_uids
    if ranked_output is None:
        materialized_uids |= set(vector_uids)
    if point_uid is not None:
        materialized_uids.add(point_uid)
    module_refs = _module_refs(evidence)
    scalar_vector_helpers = _build_scalarized_vector_helpers(evidence, formulas, class_info)
    scalar_vector_uids = {helper.uid for helper in scalar_vector_helpers.values()}
    finite_vector_helpers = _build_finite_vector_element_helpers(
        evidence, formulas, class_info
    )
    finite_vector_uids = {helper.uid for helper in finite_vector_helpers.values()}
    all_native_uids = all_uids | scalar_vector_uids | finite_vector_uids
    finish_phase("provider_evidence_setup", phase_started)

    phase_started = perf_counter()
    structured_providers: dict[str, StructuredFramePlan] = {}
    structured_point_categories: dict[str, tuple[str, tuple[Any, ...], bool]] = {}
    if ranked_output is not None:
        structured_providers = _StructuredProviderAnalyzer(
            evidence, formulas, expected_ranked_size=int(ranked_output.shape[0])
        ).discover()
        structured_point_categories = _structured_category_evidence(
            formulas, structured_providers, evidence.refs
        )
    category_plan = build_modelx_cython_category_plan(
        formulas, class_info=class_info, evidence=evidence, point_uid=point_uid,
        output_uid=output_uid,
        structured_point_categories=structured_point_categories,
    )
    if not category_plan.supported:
        raise ModelxCythonFastBackendError(
            "modelx-cython category plan is blocked: " + "; ".join(category_plan.encoding.blockers)
        )

    # A structured-provider selector has stronger physical type evidence than
    # modelx-cython's generic ndarray[object] trace.  The column lineage is
    # already proven and binding freezes it to a primitive numeric/category
    # vector, so carry that exact physical dtype through ordinary native
    # admission.  This is generic structured-column evidence, not a model-point
    # or CashValue exception.
    structured_result_dtypes: dict[str, tuple[str, str]] = {}
    for uid, source in formulas.items():
        selector = _structured_field_selector(source.function, structured_providers)
        if selector is None:
            continue
        provider_uid, field = selector
        did = category_plan.encoding.result_domain_id(uid)
        labels = None
        if did is not None:
            labels = tuple(category_plan.encoding.domain(did).labels)
        try:
            type_evidence = structured_providers[provider_uid].column_type(
                field, evidence.refs, labels=labels
            )
            dtype = (
                "bool" if type_evidence.logical_kind == "bool"
                else type_evidence.physical_dtype
            )
        except StructuredProviderError:
            continue
        if dtype in _ALLOWED_DTYPES:
            structured_result_dtypes[uid] = (
                dtype, f"structured_provider_column:{provider_uid}:{field}"
            )

    # Reuse the mature table normalizer's finite-selector bridge. The evidence
    # comes from modelx-cython/provider/local literals, never StaticTypeDomainProof.
    evidence.static_category_result_domains = {
        uid: tuple(category_plan.encoding.domain(did).labels)
        for uid, did in category_plan.encoding.result_domains.items()
        if all(isinstance(label, str) for label in category_plan.encoding.domain(did).labels)
    }
    evidence.static_category_parameter_domains = {
        (uid, parameter): tuple(category_plan.encoding.domain(did).labels)
        for (uid, parameter), did in category_plan.encoding.parameter_domains.items()
        if all(isinstance(label, str) for label in category_plan.encoding.domain(did).labels)
    }
    evidence.static_category_nullable_parameters = set(category_plan.encoding.nullable_parameters)
    evidence.static_category_reachable_result_domains = prove_consumer_category_result_domains(
        formulas, evidence=evidence, output_uid=output_uid, encoding=category_plan.encoding,
    )
    finite_vector_helpers = _normalize_finite_vector_element_helpers(
        evidence, finite_vector_helpers, category_plan=category_plan
    )
    finite_vector_uids = {helper.uid for helper in finite_vector_helpers.values()}
    all_native_uids = all_uids | scalar_vector_uids | finite_vector_uids
    finish_phase("category_evidence", phase_started)

    phase_started = perf_counter()
    normalized: dict[str, FormulaNormalizationResult] = {}
    ranked_normalized: dict[
        str, tuple[ast.FunctionDef | None, tuple[tuple[str, str], ...]]
    ] = {}
    for uid, row in formulas.items():
        if uid in materialized_uids:
            continue
        normalized[uid] = _normalize_formula(
            evidence, row, all_uids=all_native_uids, module_refs=module_refs,
            vector_helpers=scalar_vector_helpers, category_plan=category_plan,
            finite_vector_helpers=finite_vector_helpers,
            point_uid=point_uid,
            rank1_pointwise=uid in point_array_uids,
            rank1_point_source_uids=point_source_uids,
            rank1_point_frame_ref=(
                None if ranked_output is None else ranked_output.point_frame_ref
            ),
            rank1_global_aggregate_uids=global_aggregate_uids,
            source_name_by_uid={key: value.name for key, value in formulas.items()},
            reference_metadata_uids=reference_metadata_uids,
            structured_providers=structured_providers,
        )
        if ranked_output is not None and _result_geometry(
            class_info.cells[row.name]
        )[0] == 1:
            ranked_normalized[uid] = _normalize_ranked_formula(
                evidence,
                row,
                point_source_uids=point_source_uids,
                point_frame_ref=ranked_output.point_frame_ref,
                global_aggregate_uids=global_aggregate_uids,
                source_name_by_uid={key: value.name for key, value in formulas.items()},
                reference_metadata_uids=reference_metadata_uids,
                category_plan=category_plan,
                structured_providers=structured_providers,
            )
    cross_normalized: dict[str, FormulaNormalizationResult] = {}
    for uid, row in cross_formulas.items():
        scope = cross_live_scope_by_uid[uid]
        refs = getattr(scope, "refs", {})
        with evidence.translation_refs(refs):
            cross_normalized[uid] = _normalize_formula(
                evidence, row, all_uids=all_native_uids, module_refs=module_refs,
                vector_helpers=None, category_plan=None, point_uid=None,
            )
    finish_phase("formula_normalization", phase_started)

    # Everything below this point consumes compact proof summaries, normalized
    # formula ASTs and prepared InputSpecs.  The preparation context's realized
    # modelx trace and source-analysis caches are discovery/translation working
    # state only; retaining them through recursive admission/hash profiling is a
    # pure memory cost on large recursive products.
    prepared_input_specs = dict(evidence.registry.specs)
    prepared_normalized_operators = tuple(
        sorted(evidence._normalized_operators.values(), key=lambda x: x.uid)
    )
    prepared_proof_diagnostics = evidence.proof_diagnostics()
    prepared_control_parameter_intervals = {
        uid: _control_parameter_intervals(evidence, uid)
        for uid in all_native_uids
    }
    phase_started = perf_counter()
    del evidence
    gc.collect()
    finish_phase("translation_working_state_release", phase_started)

    phase_started = perf_counter()
    # Phase 1 semantic-boundary invariant: authoritative demand analysis starts
    # from the semantic-normalized formula when one exists.  Raw source remains
    # provenance/fallback only; downstream analysis no longer gets a second
    # independent interpretation of already-normalized source syntax.
    combined_sources = {**formulas, **cross_formulas}
    combined_normalized = {**normalized, **cross_normalized}
    demand_requests: dict[str, DemandEvidenceRequest] = {}
    for uid, row in combined_sources.items():
        normalized_result = combined_normalized.get(uid)
        semantic_function = (
            None
            if normalized_result is None or normalized_result.semantic is None
            else normalized_result.semantic.function
        )
        demand_requests[uid] = DemandEvidenceRequest(
            uid=uid,
            source_name=row.name,
            parameters=row.parameters,
            function=copy.deepcopy(semantic_function or row.function),
        )
    demand_overrides: dict[str, ast.FunctionDef] = {}
    for uid, normalized_result in {**normalized, **cross_normalized}.items():
        if normalized_result.function is not None and normalized_result.reduction is None:
            demand_overrides[uid] = copy.deepcopy(normalized_result.function)

    demand_result_anchors: dict[str, tuple[str, str]] = {}
    demand_parameter_anchors: dict[str, dict[str, tuple[str, str]]] = {}
    for uid, row in formulas.items():
        cell_info = class_info.cells[row.name]
        dtype, source = _result_evidence(
            cell_info, allow_array=ranked_output is not None and uid == output_uid
        )
        if uid in structured_result_dtypes:
            dtype, source = structured_result_dtypes[uid]
        if dtype in _ALLOWED_DTYPES:
            demand_result_anchors[uid] = (dtype, source)
        pf = {
            p.name: (p.dtype, p.type_source)
            for p in _parameter_evidence(cell_info, class_info)
            if p.dtype in _ALLOWED_DTYPES
        }
        if pf:
            demand_parameter_anchors[uid] = pf
    for uid, row in cross_formulas.items():
        cell_info = cross_cell_by_uid[uid]; owner = cross_class_by_uid[uid]
        dtype, source = _result_evidence(cell_info)
        if dtype in _ALLOWED_DTYPES:
            demand_result_anchors[uid] = (dtype, source)
        pf = {
            p.name: (p.dtype, p.type_source)
            for p in _parameter_evidence(cell_info, owner)
            if p.dtype in _ALLOWED_DTYPES
        }
        if pf:
            demand_parameter_anchors[uid] = pf
    for uid, row in formulas.items():
        did = category_plan.encoding.result_domain_id(uid)
        if did is not None:
            demand_result_anchors[uid] = ("int64", "modelx_cython_category_abi:" + did)
        for name in row.parameters:
            pdid = category_plan.encoding.parameter_domain_id(uid, name)
            if pdid is not None:
                demand_parameter_anchors.setdefault(uid, {})[name] = (
                    "int64", "modelx_cython_category_abi:" + pdid
                )
    # Reuse modelx-cython's sampled ref types as authoritative scalar evidence
    # for preserved Python/unobserved branches.  This breaks an otherwise
    # artificial cycle where native normalization waits for a Cell type while
    # demand typing waits for normalization to materialize a numeric ref.
    demand_reference_anchors: dict[str, dict[str, tuple[str, str]]] = {}

    def typed_ref_anchors(owner: ClassInfo, live_scope: Any | None) -> dict[str, tuple[str, str]]:
        result: dict[str, tuple[str, str]] = {}
        for name, ref in getattr(owner, "refs", {}).items():
            sampled_type = getattr(ref, "type_", None)
            dtype = _fast_dtype(sampled_type)
            source = f"modelx_cython.CombinedRefInfo:{name}"
            # An unobserved conditional branch may never touch an otherwise
            # ordinary scalar ref, so modelx-cython legitimately has no trace
            # sample for it.  The owner ref itself is static model state, not a
            # Cell formula call; reading its already-bound value during planning
            # preserves modelx-cython's ref-copy semantics without creating a
            # live-modelx execution path.
            if dtype not in _ALLOWED_DTYPES and live_scope is not None:
                try:
                    value = getattr(live_scope, name)
                except Exception:
                    value = None
                candidate = _fast_dtype(type(value)) if value is not None else "object"
                if candidate in _ALLOWED_DTYPES:
                    dtype = candidate
                    source = f"modelx_owner_ref_value:{name}"
            if dtype in _ALLOWED_DTYPES:
                result[name] = (dtype, source)
        return result

    root_typed_refs = typed_ref_anchors(class_info, space)
    if root_typed_refs:
        for uid in formulas:
            demand_reference_anchors[uid] = root_typed_refs
    for uid, owner in cross_class_by_uid.items():
        typed = typed_ref_anchors(owner, cross_live_scope_by_uid.get(uid))
        if typed:
            demand_reference_anchors[uid] = typed

    demand_evidence = resolve_demand_driven_evidence(
        demand_requests, output_uid=output_uid, formula_overrides=demand_overrides,
        input_specs=prepared_input_specs, module_refs=module_refs,
        result_anchors=demand_result_anchors, parameter_anchors=demand_parameter_anchors,
        reference_anchors=demand_reference_anchors,
    )
    demand_result_fallbacks = {
        uid: fallback for uid in demand_evidence.selected_uids
        if (fallback := demand_evidence.result_fallback(uid)) is not None
    }
    demand_parameter_fallbacks = {
        uid: {
            name: fallback for name in demand_requests[uid].parameters
            if (fallback := demand_evidence.parameter_fallback(uid, name)) is not None
        }
        for uid in demand_evidence.selected_uids
    }
    finish_phase("authoritative_demand_evidence", phase_started)

    phase_started = perf_counter()
    # Build one result-dtype catalogue from the same modelx-cython/demand/category
    # evidence used for final NativeFormulaCell dtypes. Local-control type
    # inference consumes only this catalogue; it does not run a parallel type
    # system or inspect live modelx values.
    control_callee_dtypes: dict[str, str] = {}
    for uid, row in formulas.items():
        dtype, _source = _result_evidence(
            class_info.cells[row.name],
            allow_array=ranked_output is not None,
            fallback=demand_result_fallbacks.get(uid),
        )
        if uid in structured_result_dtypes:
            dtype, _source = structured_result_dtypes[uid]
        if category_plan.encoding.result_domain_id(uid) is not None:
            dtype = "int64"
        if dtype in _ALLOWED_DTYPES:
            control_callee_dtypes[uid] = dtype
    for uid, row in cross_formulas.items():
        dtype, _source = _result_evidence(
            cross_cell_by_uid[uid], fallback=demand_result_fallbacks.get(uid)
        )
        if dtype in _ALLOWED_DTYPES:
            control_callee_dtypes[uid] = dtype
    control_callee_dtypes.update({helper.uid: helper.dtype for helper in scalar_vector_helpers.values()})
    control_callee_dtypes.update({helper.uid: helper.dtype for helper in finite_vector_helpers.values()})
    control_callee_value_domains: dict[str, StaticValueDomain] = {}
    for uid, dtype in control_callee_dtypes.items():
        domain = demand_evidence.result_domains.get(uid)
        if (
            domain is not None
            and domain.proven
            and domain.numeric_dtype == dtype
        ):
            control_callee_value_domains[uid] = domain
        else:
            control_callee_value_domains[uid] = numeric_domain(
                dtype, f"native_result_type:{uid}"
            )
    prepared_exact_int_bounds = _prepared_exact_int_bounds(prepared_input_specs)

    frontend_rows: dict[str, FastFrontendCell] = {}
    native_cells: dict[str, NativeFormulaCell] = {}
    cross_native_cells: dict[str, NativeFormulaCell] = {}
    source_native_cells: dict[str, NativeFormulaCell] = {}
    source_cross_native_cells: dict[str, NativeFormulaCell] = {}
    cross_frontend_rows: dict[str, FastFrontendCell] = {}
    synthetic_native: dict[str, NativeFormulaCell] = {}
    source_synthetic_native: dict[str, NativeFormulaCell] = {}
    source_candidate_blockers: dict[str, tuple[tuple[str, str], ...]] = {}
    source_transparent_cell_catalog: dict[str, NativeFormulaCell] = {}
    parameter_ranges: dict[str, tuple[tuple[int, int] | None, ...]] = {}
    for helper in scalar_vector_helpers.values():
        candidate = NativeFormulaCell(
            uid=helper.uid,
            source_name=helper.source_name + "[]",
            source_fullname=formulas[helper.source_uid].fullname + "[]",
            dtype=helper.dtype,
            parameters=(helper.parameter,),
            parameter_dtypes=("int64",),
            function=copy.deepcopy(helper.function),
            reduction=None,
            guards=(),
        )
        candidate = _attach_local_integer_control_evidence(
            candidate, callee_dtypes=control_callee_dtypes,
            callee_value_domains=control_callee_value_domains,
            input_specs=prepared_input_specs, module_refs=module_refs,
        )
        if not _residual_blockers(
            candidate,
            all_uids=all_native_uids,
            module_refs=module_refs,
            prepared_exact_int_bounds=prepared_exact_int_bounds,
        ):
            synthetic_native[helper.uid] = candidate
            parameter_ranges[helper.uid] = (helper.dense_range,)
        source_blockers = _source_transparent_residual_blockers(
            candidate, all_uids=all_native_uids, module_refs=module_refs,
            cell_dtypes=control_callee_dtypes, input_specs=prepared_input_specs,
        )
        source_transparent_cell_catalog[helper.uid] = candidate
        source_candidate_blockers[helper.uid] = source_blockers
        if not source_blockers:
            source_synthetic_native[helper.uid] = candidate
            parameter_ranges[helper.uid] = (helper.dense_range,)

    for helper in finite_vector_helpers.values():
        component = "" if helper.component is None else f"[{helper.component}]"
        candidate = NativeFormulaCell(
            uid=helper.uid,
            source_name=f"{helper.source_name}{component}[]",
            source_fullname=f"{formulas[helper.source_uid].fullname}{component}[]",
            dtype=helper.dtype,
            parameters=helper.parameters,
            parameter_dtypes=helper.parameter_dtypes,
            function=copy.deepcopy(helper.function),
            reduction=None,
            guards=(),
        )
        candidate = _attach_local_integer_control_evidence(
            candidate, callee_dtypes=control_callee_dtypes,
            callee_value_domains=control_callee_value_domains,
            input_specs=prepared_input_specs, module_refs=module_refs,
        )
        if not _residual_blockers(
            candidate,
            all_uids=all_native_uids,
            module_refs=module_refs,
            prepared_exact_int_bounds=prepared_exact_int_bounds,
        ):
            synthetic_native[helper.uid] = candidate
            parameter_ranges[helper.uid] = helper.parameter_ranges
        source_blockers = _source_transparent_residual_blockers(
            candidate, all_uids=all_native_uids, module_refs=module_refs,
            cell_dtypes=control_callee_dtypes, input_specs=prepared_input_specs,
        )
        source_transparent_cell_catalog[helper.uid] = candidate
        source_candidate_blockers[helper.uid] = source_blockers
        if not source_blockers:
            source_synthetic_native[helper.uid] = candidate
            parameter_ranges[helper.uid] = helper.parameter_ranges

    for name in sorted(class_info.cells):
        cell_info = class_info.cells[name]
        uid = uid_by_name[name]
        source = formulas[uid]
        result_dtype, result_source = _result_evidence(
            cell_info, allow_array=ranked_output is not None,
            fallback=demand_result_fallbacks.get(uid),
        )
        if uid in structured_result_dtypes:
            result_dtype, result_source = structured_result_dtypes[uid]
        result_ndim, result_shape, result_shape_source = _result_geometry(cell_info)
        result_category_domain = category_plan.encoding.result_domain_id(uid)
        if result_category_domain is not None:
            result_dtype = "int64"
            result_source = "modelx_cython_category_abi:" + result_category_domain

        base_params = _parameter_evidence(
            cell_info, class_info, fallbacks=demand_parameter_fallbacks.get(uid)
        )
        parameterized_category = category_plan.parameterized_point_bindings.get(uid)
        if parameterized_category is not None:
            category_ranges = {
                name: (lo, hi) for name, lo, hi in parameterized_category.parameter_ranges
            }
            base_params = tuple(
                FastParameterEvidence(
                    p.name, p.dtype, p.type_source,
                    category_ranges.get(p.name, (p.minimum, p.maximum))[0],
                    category_ranges.get(p.name, (p.minimum, p.maximum))[1],
                    (p.range_source + "+parameterized_category_provider_dispatch"
                     if p.name in category_ranges and p.range_source
                     else "parameterized_category_provider_dispatch"
                     if p.name in category_ranges else p.range_source),
                )
                for p in base_params
            )
        params_list: list[FastParameterEvidence] = []
        parameter_category_domains: list[tuple[str, str | None]] = []
        for p in base_params:
            did = category_plan.encoding.parameter_domain_id(uid, p.name)
            parameter_category_domains.append((p.name, did))
            if did is None:
                params_list.append(p)
            else:
                domain = category_plan.encoding.domain(did)
                params_list.append(FastParameterEvidence(
                    p.name, "int64", "modelx_cython_category_abi:" + did,
                    0, len(domain.labels) - 1, "category_code_domain",
                ))
        params = tuple(params_list)
        parameter_ranges[uid] = tuple(p.dense_range for p in params)

        if uid in materialized_uids:
            helper_kind = "model_point_row" if uid == point_uid else "vector_provider"
            materialized = NativeFormulaCell(
                uid=uid,
                source_name=name,
                source_fullname=source.fullname,
                dtype=result_dtype,
                parameters=source.parameters,
                parameter_dtypes=tuple(p.dtype for p in params),
                function=source.function,
                reduction=None,
                guards=source.guards,
                execution_role="materialized_helper",
                logical_result_ndim=result_ndim,
                logical_result_shape=result_shape,
            )
            native_cells[uid] = materialized
            source_native_cells[uid] = materialized
            source_transparent_cell_catalog[uid] = materialized
            source_candidate_blockers[uid] = ()
            frontend_rows[uid] = FastFrontendCell(
                uid, name, source.fullname, "materialized_helper",
                result_dtype, result_source + "+" + helper_kind, params,
                result_ndim=result_ndim, result_shape=result_shape,
                result_shape_source=result_shape_source,
            )
            continue

        normalized_result = normalized.get(uid)
        fn = None if normalized_result is None else normalized_result.function
        reduction = None if normalized_result is None else normalized_result.reduction
        provider_blockers = () if normalized_result is None else normalized_result.blockers
        normalized_guards = source.guards if normalized_result is None else normalized_result.guards
        ranked_fn, ranked_blockers = ranked_normalized.get(uid, (None, ()))
        blockers: list[tuple[str, str]] = list(
            ranked_blockers if ranked_output is not None and result_ndim == 1
            else provider_blockers
        )
        if result_ndim > 1 or (
            result_ndim == 1 and ranked_output is None and uid not in point_array_uids
        ):
            blockers.append((
                "rank_geometry",
                f"array_result_ndim_{result_ndim}_shape_{result_shape!r}_does_not_match_selected_rank1_axis",
            ))
        if uid == output_uid and result_ndim > 0 and ranked_output is None:
            blockers.append((
                "rank_geometry",
                "selected_array_output_has_ambiguous_or_unsupported_geometry",
            ))
        if result_dtype not in _ALLOWED_DTYPES:
            blockers.append(("type", f"result:{result_dtype}:{result_source}"))
        for p in params:
            if p.dtype not in {"int64", "bool", "float64"}:
                blockers.append(("type", f"parameter:{p.name}:{p.dtype}:{p.type_source}"))
        if fn is None and reduction is None and ranked_fn is None:
            blockers.append(("provider", "formula_not_fast_normalized"))

        native = NativeFormulaCell(
            uid=uid,
            source_name=name,
            source_fullname=source.fullname,
            dtype=result_dtype,
            parameters=source.parameters,
            parameter_dtypes=tuple(p.dtype for p in params),
            function=fn,
            reduction=reduction,
            guards=normalized_guards,
            result_category_domain=result_category_domain,
            parameter_category_domains=tuple(did for _name, did in parameter_category_domains),
            logical_result_ndim=result_ndim,
            logical_result_shape=result_shape,
            ranked_function=ranked_fn,
        )
        native = _attach_local_integer_control_evidence(
            native, callee_dtypes=control_callee_dtypes,
            callee_value_domains=control_callee_value_domains,
            input_specs=prepared_input_specs, module_refs=module_refs,
            parameter_int64_intervals=prepared_control_parameter_intervals.get(uid, {}),
        )
        native = _native_generator_reduction_lowering(native)
        source_blockers = [
            item for item in blockers if item[0] != "control_syntax"
        ]
        # Rank-1 formulas deliberately preserve a small NumPy/Cython object
        # surface.  Validate that surface with its own residual screen rather
        # than either applying scalar ``nogil`` rules or admitting arbitrary
        # Python-object methods.
        if native.logical_result_ndim == 1 and native.ranked_function is not None:
            source_blockers.extend(_ranked_source_transparent_residual_blockers(
                native, all_uids=all_native_uids, module_refs=module_refs,
            ))
        else:
            source_blockers.extend(_source_transparent_residual_blockers(
                native, all_uids=all_native_uids, module_refs=module_refs,
                cell_dtypes=control_callee_dtypes, input_specs=prepared_input_specs,
            ))
        source_blockers = list(sorted(set(source_blockers)))
        source_transparent_cell_catalog[uid] = native
        source_candidate_blockers[uid] = tuple(source_blockers)
        if not source_blockers:
            source_native_cells[uid] = native

        blockers.extend(_residual_blockers(
            native, all_uids=all_native_uids, module_refs=module_refs,
            prepared_exact_int_bounds=prepared_exact_int_bounds,
        ))
        if blockers:
            frontend_rows[uid] = FastFrontendCell(
                uid, name, source.fullname, "object_fallback",
                result_dtype, result_source, params,
                tuple(sorted(set(blockers))),
                result_category_domain=result_category_domain,
                parameter_category_domains=tuple(parameter_category_domains),
                result_ndim=result_ndim, result_shape=result_shape,
                result_shape_source=result_shape_source,
                local_scalar_dtypes=native.local_scalar_dtypes,
                local_integer_names=native.local_integer_names,
                local_positive_int_names=native.local_positive_int_names,
                local_positive_int_domains=native.local_positive_int_domains,
                parameter_int64_intervals=native.parameter_int64_intervals,
                local_int64_intervals=native.local_int64_intervals,
                callee_int64_intervals=native.callee_int64_intervals,
                input_int64_intervals=native.input_int64_intervals,
                local_control_evidence_source=native.local_control_evidence_source,
            )
            continue
        native_cells[uid] = native
        frontend_rows[uid] = FastFrontendCell(
            uid, name, source.fullname, "native_formula",
            result_dtype, result_source, params,
            provider_input_keys=_input_keys(native),
            result_category_domain=result_category_domain,
            parameter_category_domains=tuple(parameter_category_domains),
            result_ndim=result_ndim, result_shape=result_shape,
            result_shape_source=result_shape_source,
            local_scalar_dtypes=native.local_scalar_dtypes,
            local_integer_names=native.local_integer_names,
            local_positive_int_names=native.local_positive_int_names,
            local_positive_int_domains=native.local_positive_int_domains,
            parameter_int64_intervals=native.parameter_int64_intervals,
            local_int64_intervals=native.local_int64_intervals,
            callee_int64_intervals=native.callee_int64_intervals,
            input_int64_intervals=native.input_int64_intervals,
            local_control_evidence_source=native.local_control_evidence_source,
        )

    # Cross-Space callees are sidecar-only implementation cells.  They do not
    # enter ``plan.cells`` or alter the root exported-cell status census.
    for uid, source in cross_formulas.items():
        cell_info = cross_cell_by_uid[uid]
        owner = cross_class_by_uid[uid]
        result_dtype, result_source = _result_evidence(
            cell_info, fallback=demand_result_fallbacks.get(uid)
        )
        result_ndim, result_shape, result_shape_source = _result_geometry(cell_info)
        params = _parameter_evidence(
            cell_info, owner, fallbacks=demand_parameter_fallbacks.get(uid)
        )
        parameter_ranges[uid] = tuple(p.dense_range for p in params)
        normalized_result = cross_normalized.get(uid)
        fn = None if normalized_result is None else normalized_result.function
        reduction = None if normalized_result is None else normalized_result.reduction
        provider_blockers = () if normalized_result is None else normalized_result.blockers
        normalized_guards = source.guards if normalized_result is None else normalized_result.guards
        blockers: list[tuple[str, str]] = list(provider_blockers)
        if result_dtype not in _ALLOWED_DTYPES:
            blockers.append(("type", f"result:{result_dtype}:{result_source}"))
        for p in params:
            if p.dtype not in {"int64", "bool", "float64"}:
                blockers.append(("type", f"parameter:{p.name}:{p.dtype}:{p.type_source}"))
        if fn is None and reduction is None:
            blockers.append(("provider", "formula_not_fast_normalized"))
        native = NativeFormulaCell(
            uid=uid, source_name=source.name, source_fullname=source.fullname,
            dtype=result_dtype,
            parameters=source.parameters,
            parameter_dtypes=tuple(p.dtype for p in params),
            function=fn, reduction=reduction, guards=normalized_guards,
            logical_result_ndim=result_ndim, logical_result_shape=result_shape,
        )
        native = _attach_local_integer_control_evidence(
            native, callee_dtypes=control_callee_dtypes,
            callee_value_domains=control_callee_value_domains,
            input_specs=prepared_input_specs, module_refs=module_refs,
            parameter_int64_intervals=prepared_control_parameter_intervals.get(uid, {}),
        )
        native = _native_generator_reduction_lowering(native)
        source_blockers = [
            item for item in blockers if item[0] != "control_syntax"
        ]
        if native.logical_result_ndim == 1 and native.ranked_function is not None:
            source_blockers.extend(_ranked_source_transparent_residual_blockers(
                native, all_uids=all_native_uids, module_refs=module_refs,
            ))
        else:
            source_blockers.extend(_source_transparent_residual_blockers(
                native, all_uids=all_native_uids, module_refs=module_refs,
                cell_dtypes=control_callee_dtypes, input_specs=prepared_input_specs,
            ))
        source_blockers = list(sorted(set(source_blockers)))
        source_transparent_cell_catalog[uid] = native
        source_candidate_blockers[uid] = tuple(source_blockers)
        if not source_blockers:
            source_cross_native_cells[uid] = native

        blockers.extend(_residual_blockers(
            native, all_uids=all_native_uids, module_refs=module_refs,
            prepared_exact_int_bounds=prepared_exact_int_bounds,
        ))
        if blockers:
            cross_frontend_rows[uid] = FastFrontendCell(
                uid, source.name, source.fullname, "object_fallback",
                result_dtype, result_source, params, tuple(sorted(set(blockers))),
                result_ndim=result_ndim, result_shape=result_shape,
                result_shape_source=result_shape_source,
                local_scalar_dtypes=native.local_scalar_dtypes,
                local_integer_names=native.local_integer_names,
                local_positive_int_names=native.local_positive_int_names,
                local_positive_int_domains=native.local_positive_int_domains,
                parameter_int64_intervals=native.parameter_int64_intervals,
                local_int64_intervals=native.local_int64_intervals,
                callee_int64_intervals=native.callee_int64_intervals,
                input_int64_intervals=native.input_int64_intervals,
                local_control_evidence_source=native.local_control_evidence_source,
            )
            continue
        cross_native_cells[uid] = native
        cross_frontend_rows[uid] = FastFrontendCell(
            uid, source.name, source.fullname, "native_formula",
            result_dtype, result_source, params, provider_input_keys=_input_keys(native),
            result_ndim=result_ndim, result_shape=result_shape,
            result_shape_source=result_shape_source,
            local_scalar_dtypes=native.local_scalar_dtypes,
            local_integer_names=native.local_integer_names,
            local_positive_int_names=native.local_positive_int_names,
            local_positive_int_domains=native.local_positive_int_domains,
            parameter_int64_intervals=native.parameter_int64_intervals,
            local_int64_intervals=native.local_int64_intervals,
            callee_int64_intervals=native.callee_int64_intervals,
            input_int64_intervals=native.input_int64_intervals,
            local_control_evidence_source=native.local_control_evidence_source,
        )

    # Cache/no-cache prediction is no longer a production planning input.
    # Historical semantic-parameter classification remains available only via
    # the explicit semantic_parameter_policy comparison utility.

    def available_synthetic_uids(
        formula_active: set[str],
        synthetic_pool: Mapping[str, NativeFormulaCell],
    ) -> set[str]:
        """Synthetic helpers whose native dependencies are all available.

        This is a tiny local inclusion fixed point, not an execution graph or
        scheduler. Synthetic vector-element helpers may call other synthetic
        helpers, so a one-level membership test would incorrectly discard a
        valid scalarized chain.
        """
        available: set[str] = set()
        changed_local = True
        while changed_local:
            changed_local = False
            allowed = formula_active | available
            for uid, cell in synthetic_pool.items():
                if uid in available:
                    continue
                deps = _callee_uids(cell, all_native_uids)
                # Ordinary recursive Cells are admissible with their own native
                # memo busy/cycle semantics. Synthetic element Cells must obey
                # the same rule: a self-call is not a missing dependency.
                if deps <= (allowed | {uid}):
                    available.add(uid)
                    changed_local = True
        return available

    def terminal_unavailable_dependencies(
        uid: str,
        *,
        formula_active: set[str],
        available_synthetic: set[str],
        synthetic_pool: Mapping[str, NativeFormulaCell],
        seen: set[str] | None = None,
    ) -> set[str]:
        """Flatten one unavailable synthetic helper to real missing callees."""
        if uid in formula_active or uid in available_synthetic:
            return set()
        if uid not in synthetic_pool:
            return {uid}
        seen = set() if seen is None else set(seen)
        if uid in seen:
            return set()
        seen.add(uid)
        out: set[str] = set()
        for dep in _callee_uids(synthetic_pool[uid], all_native_uids):
            if dep == uid:
                continue
            out.update(terminal_unavailable_dependencies(
                dep,
                formula_active=formula_active,
                available_synthetic=available_synthetic,
                synthetic_pool=synthetic_pool,
                seen=seen,
            ))
        return out

    # Local fallback propagation only. This is not target-closure graph analysis:
    # all root Cells remain catalogued and this merely guarantees that no emitted
    # native formula calls a Cell delegated to ordinary modelx-cython.
    changed = True
    while changed:
        changed = False
        lexical_active = {uid for uid, row in frontend_rows.items() if row.status == "native_formula"}
        cross_active = set(cross_native_cells)
        formula_active = lexical_active | cross_active
        available_synthetic = available_synthetic_uids(formula_active, synthetic_native)
        active = formula_active | available_synthetic

        for uid in tuple(sorted(cross_active)):
            native = cross_native_cells[uid]
            dangling = _callee_uids(native, all_native_uids) - active
            if dangling:
                cross_native_cells.pop(uid, None)
                row = cross_frontend_rows[uid]
                names = tuple(sorted(
                    cross_formulas[x].fullname if x in cross_formulas
                    else formulas[x].name if x in formulas else x
                    for x in dangling
                ))
                cross_frontend_rows[uid] = replace(
                    row,
                    status="object_fallback",
                    blocker_items=row.blocker_items + (("fallback_dependency", ",".join(names)),),
                )
                changed = True

        # Recompute after cross targets were removed so root callers fail closed in
        # the same iteration rather than retaining a dangling UID.
        cross_active = set(cross_native_cells)
        formula_active = lexical_active | cross_active
        available_synthetic = available_synthetic_uids(formula_active, synthetic_native)
        active = formula_active | available_synthetic
        for uid in tuple(sorted(lexical_active)):
            native = native_cells[uid]
            dangling = _callee_uids(native, all_native_uids) - active
            if dangling:
                terminal: set[str] = set()
                for dep in dangling:
                    terminal.update(terminal_unavailable_dependencies(
                        dep,
                        formula_active=formula_active,
                        available_synthetic=available_synthetic,
                        synthetic_pool=synthetic_native,
                    ))
                if terminal:
                    dangling = terminal
                def display(dep: str) -> str:
                    if dep in formulas:
                        return formulas[dep].name
                    if dep in cross_formulas:
                        return cross_formulas[dep].fullname
                    helper = synthetic_native.get(dep)
                    return helper.source_name if helper is not None else dep
                names = tuple(sorted(display(x) for x in dangling))
                native_cells.pop(uid, None)
                row = frontend_rows[uid]
                frontend_rows[uid] = replace(
                    row,
                    status="object_fallback",
                    blocker_items=row.blocker_items + (("fallback_dependency", ",".join(names)),),
                )
                changed = True

    # Preserve the direct, Cell-intrinsic Stage-1B blocker view before the
    # historical N0-only projection adds transitive ``fallback_dependency``
    # diagnostics. Stage 2A consumes this direct view to classify N0/N1/P
    # without collapsing bridgeable callers to P.
    source_transparent_direct_blockers = dict(source_candidate_blockers)

    # Stage 1B repeats only dependency availability for the source-transparent
    # candidate set.  Unlike the historical path above, ordinary statement
    # syntax has already been ignored; a Cell leaves this set only because a
    # model/provider/type dependency is unavailable.  Language-level legality
    # is intentionally deferred to Cython's actual ``noexcept nogil`` compile.
    source_changed = True
    while source_changed:
        source_changed = False
        lexical_active = {
            uid for uid, cell in source_native_cells.items()
            if cell.execution_role == "formula"
        }
        cross_active = set(source_cross_native_cells)
        formula_active = lexical_active | cross_active
        available_synthetic = available_synthetic_uids(
            formula_active, source_synthetic_native
        )
        active = formula_active | available_synthetic

        for uid in tuple(sorted(cross_active)):
            native = source_cross_native_cells[uid]
            dangling = _callee_uids(native, all_native_uids) - active
            if not dangling:
                continue
            source_cross_native_cells.pop(uid, None)
            names = tuple(sorted(
                cross_formulas[x].fullname if x in cross_formulas
                else formulas[x].name if x in formulas else x
                for x in dangling
            ))
            prior = source_candidate_blockers.get(uid, ())
            source_candidate_blockers[uid] = tuple(sorted(set(
                prior + (("fallback_dependency", ",".join(names)),)
            )))
            source_changed = True

        lexical_active = {
            uid for uid, cell in source_native_cells.items()
            if cell.execution_role == "formula"
        }
        cross_active = set(source_cross_native_cells)
        formula_active = lexical_active | cross_active
        available_synthetic = available_synthetic_uids(
            formula_active, source_synthetic_native
        )
        active = formula_active | available_synthetic
        for uid in tuple(sorted(lexical_active)):
            native = source_native_cells[uid]
            dangling = _callee_uids(native, all_native_uids) - active
            if not dangling:
                continue
            terminal: set[str] = set()
            for dep in dangling:
                terminal.update(terminal_unavailable_dependencies(
                    dep,
                    formula_active=formula_active,
                    available_synthetic=available_synthetic,
                    synthetic_pool=source_synthetic_native,
                ))
            if terminal:
                dangling = terminal
            def source_display(dep: str) -> str:
                if dep in formulas:
                    return formulas[dep].name
                if dep in cross_formulas:
                    return cross_formulas[dep].fullname
                helper = source_synthetic_native.get(dep)
                return helper.source_name if helper is not None else dep
            names = tuple(sorted(source_display(x) for x in dangling))
            source_native_cells.pop(uid, None)
            prior = source_candidate_blockers.get(uid, ())
            source_candidate_blockers[uid] = tuple(sorted(set(
                prior + (("fallback_dependency", ",".join(names)),)
            )))
            source_changed = True

    native_program: NativeFormulaProgram | None = None
    source_preserving_native_program: NativeFormulaProgram | None = None
    native_hash_capacity_evidence: dict[str, object] = {}
    native_hash_capacity_by_uid: dict[str, int] = {}
    selected_inputs: dict[str, InputSpec] = {}
    source_selected_inputs: dict[str, InputSpec] = {}
    referenced_helpers: tuple[FastScalarizedVectorHelper, ...] = ()
    referenced_finite_helpers: tuple[FiniteVectorElementHelper, ...] = ()
    if output_uid in native_cells:
        # Keep only the sidecar-specialized formulas plus semantic helpers. The
        # ordinary modelx-cython package owns every fallback Cell.
        sidecar_cells = dict(native_cells)
        sidecar_cells.update(cross_native_cells)
        formula_cells = list(native_cells.values()) + list(cross_native_cells.values())
        synthetic_uid_set = set(synthetic_native)
        referenced_synthetic = set().union(*(
            _callee_uids(cell, synthetic_uid_set) for cell in formula_cells
        )) if formula_cells else set()
        pending = list(referenced_synthetic)
        while pending:
            current = pending.pop()
            cell = synthetic_native.get(current)
            if cell is None:
                continue
            for dep in _callee_uids(cell, synthetic_uid_set):
                if dep not in referenced_synthetic:
                    referenced_synthetic.add(dep)
                    pending.append(dep)
        for uid in sorted(referenced_synthetic):
            if uid not in synthetic_native:
                raise ModelxCythonFastBackendError(
                    f"native formula retained unavailable scalarized vector helper {uid}"
                )
            sidecar_cells[uid] = synthetic_native[uid]
        by_synth_uid = {h.uid: h for h in scalar_vector_helpers.values()}
        referenced_helpers = tuple(
            by_synth_uid[uid] for uid in sorted(referenced_synthetic)
            if uid in by_synth_uid
        )
        by_finite_uid = {h.uid: h for h in finite_vector_helpers.values()}
        referenced_finite_helpers = tuple(
            by_finite_uid[uid] for uid in sorted(referenced_synthetic)
            if uid in by_finite_uid
        )
        for uid in materialized_uids:
            if uid in native_cells:
                continue
            helper = frontend_rows[uid]
            # A helper is needed in correspondence metadata only; it is not emitted.
            source = formulas[uid]
            sidecar_cells[uid] = NativeFormulaCell(
                uid=uid,
                source_name=source.name,
                source_fullname=source.fullname,
                dtype=helper.result_dtype,
                parameters=source.parameters,
                parameter_dtypes=tuple(p.dtype for p in helper.parameters),
                function=source.function,
                reduction=None,
                guards=source.guards,
                execution_role="materialized_helper",
                logical_result_ndim=helper.result_ndim,
                logical_result_shape=helper.result_shape,
            )
        used_input_keys: set[str] = set()
        for cell in sidecar_cells.values():
            if cell.execution_role == "formula":
                used_input_keys.update(_input_keys(cell))
        for key in tuple(used_input_keys):
            spec = prepared_input_specs.get(key)
            if spec is not None and spec.category_validity_key is not None:
                used_input_keys.add(spec.category_validity_key)
        selected_inputs = {
            key: spec for key, spec in prepared_input_specs.items()
            if key in used_input_keys
        }
        frozen_inputs = tuple(_freeze_input(spec) for spec in selected_inputs.values())
        identity = [module_name, class_name, output_uid, *selected_inputs]
        identity.extend(
            f"{uid}:{sidecar_cells[uid].execution_role}:{sidecar_cells[uid].dtype}:{sidecar_cells[uid].source()}"
            for uid in sorted(sidecar_cells)
        )
        puid = "mxf_" + hashlib.sha256("\0".join(identity).encode("utf-8", "backslashreplace")).hexdigest()[:16]
        native_program = NativeFormulaProgram(
            schema="modelx_graph.native_formula.modelx_cython_fast.v2",
            uid=puid,
            output_uid=output_uid,
            cells=sidecar_cells,
            input_order=tuple(selected_inputs),
            inputs=frozen_inputs,
            normalized_operators=prepared_normalized_operators,
            module_refs=tuple(sorted(module_refs.items())),
            category_domains=tuple(
                (did, tuple(domain.labels))
                for did, domain in sorted(category_plan.encoding.domains.items())
            ),
            provenance="modelx_cython_frontend_fast_recursive_v2_category_provider",
        )
        # Production native hash tables use fixed capacities certified over the
        # complete frozen RunDomain.  Pay this trace cost only when the native
        # cache planner actually identifies a non-dense/hash Cell; dense/scalar
        # products keep the ordinary modelx-cython sample-only planning cost.
        from .dense_recursive_native import build_dense_memo_plan
        memo_probe = build_dense_memo_plan(
            native_program,
            parameter_ranges=parameter_ranges,
            memoization_policy="native_cache",
            native_hash_capacity=16,
        )
        hash_uids = tuple(row.uid for row in memo_probe.cells if row.kind == "hash")
        if hash_uids:
            hash_started = perf_counter()
            from .native_hash_capacity_evidence import profile_native_hash_capacity_evidence
            capacity = profile_native_hash_capacity_evidence(
                model_path=model_path,
                space=space,
                output_name=output_name,
                run_keys=run_keys,
                output_cell=native_program.cells[output_uid],
                hash_cells={uid: native_program.cells[uid] for uid in hash_uids},
                range_cells=native_program.cells,
                target_load_factor=0.50,
                min_capacity=16,
                max_capacity=1 << 20,
            )

            # The ordinary modelx-cython trace is deliberately sample-sized, so
            # its integral min/max is type evidence, not a production range
            # authority.  When hash memoization already requires a complete
            # frozen-RunDomain pass, reuse that pass to strengthen every observed
            # integral parameter bound before physical memo planning.  Unioning
            # with explicit/upstream bounds preserves stronger static domains and
            # can only widen a dense allocation; it never narrows one.
            strengthened_ranges: dict[str, list[dict[str, object]]] = {}
            for uid, observed in capacity.complete_integral_ranges_by_uid.items():
                current = tuple(parameter_ranges.get(uid, ()))
                if len(current) != len(observed):
                    continue
                merged: list[tuple[int, int] | None] = []
                changes: list[dict[str, object]] = []
                for pos, (old_range, complete_range) in enumerate(zip(current, observed)):
                    if complete_range is None:
                        merged.append(old_range)
                        continue
                    if old_range is None:
                        new_range = (int(complete_range[0]), int(complete_range[1]))
                    else:
                        new_range = (
                            min(int(old_range[0]), int(complete_range[0])),
                            max(int(old_range[1]), int(complete_range[1])),
                        )
                    merged.append(new_range)
                    if old_range != new_range:
                        changes.append({
                            "parameter_position": pos,
                            "before": None if old_range is None else list(old_range),
                            "complete_run_observed": list(complete_range),
                            "after": list(new_range),
                        })
                if changes:
                    parameter_ranges[uid] = tuple(merged)
                    strengthened_ranges[uid] = changes

            # Re-plan after strengthening.  A previously unbounded integral
            # parameter may make an existing hash Cell dense (RILA's bucket Cell
            # is one example), while widening an existing dense box can push a Cell
            # over the dense-slot ceiling.  The first complete trace sized only the
            # Cells that were hash candidates before strengthening, so fail closed
            # if the final plan contains a new hash candidate rather than inventing
            # a capacity.
            final_probe = build_dense_memo_plan(
                native_program,
                parameter_ranges=parameter_ranges,
                memoization_policy="native_cache",
                native_hash_capacity=16,
            )
            final_hash_uids = tuple(row.uid for row in final_probe.cells if row.kind == "hash")
            missing_hash_evidence = sorted(set(final_hash_uids) - set(hash_uids))
            if missing_hash_evidence:
                raise ModelxCythonFastBackendError(
                    "complete-RunDomain range widening created new native hash candidates "
                    "without key-capacity evidence: " + ",".join(missing_hash_evidence)
                )

            native_hash_capacity_evidence = capacity.manifest()
            native_hash_capacity_evidence["integral_parameter_range_authority"] = (
                "complete_frozen_run_domain_exported_formula_trace_union_static_bounds"
            )
            native_hash_capacity_evidence["strengthened_parameter_ranges"] = strengthened_ranges
            native_hash_capacity_by_uid = {
                uid: int(capacity.capacities_by_uid[uid]) for uid in final_hash_uids
            }
            finish_phase("native_hash_complete_run_capacity_trace", hash_started)

    if output_uid in source_native_cells:
        # Stage 1B builds a second sidecar candidate from the same normalized
        # provider/ABI evidence but without the historical ordinary-syntax
        # whitelist.  This program is not fed to the old emitter.
        source_sidecar_cells = dict(source_native_cells)
        source_sidecar_cells.update(source_cross_native_cells)
        source_formula_cells = [
            cell for cell in source_sidecar_cells.values()
            if cell.execution_role == "formula"
        ]
        source_synth_uids = set(source_synthetic_native)
        source_referenced_synthetic = set().union(*(
            _callee_uids(cell, source_synth_uids) for cell in source_formula_cells
        )) if source_formula_cells else set()
        pending = list(source_referenced_synthetic)
        while pending:
            current = pending.pop()
            helper = source_synthetic_native.get(current)
            if helper is None:
                continue
            for dep in _callee_uids(helper, source_synth_uids):
                if dep not in source_referenced_synthetic:
                    source_referenced_synthetic.add(dep)
                    pending.append(dep)
        for uid in sorted(source_referenced_synthetic):
            helper = source_synthetic_native.get(uid)
            if helper is None:
                raise ModelxCythonFastBackendError(
                    f"source-transparent formula retained unavailable helper {uid}"
                )
            source_sidecar_cells[uid] = helper

        source_used_input_keys: set[str] = set()
        for cell in source_sidecar_cells.values():
            if cell.execution_role == "formula":
                source_used_input_keys.update(_input_keys(cell))
        for key in tuple(source_used_input_keys):
            spec = prepared_input_specs.get(key)
            if spec is not None and spec.category_validity_key is not None:
                source_used_input_keys.add(spec.category_validity_key)
        source_selected_inputs = {
            key: spec for key, spec in prepared_input_specs.items()
            if key in source_used_input_keys
        }
        source_frozen_inputs = tuple(
            _freeze_input(spec) for spec in source_selected_inputs.values()
        )
        source_identity = [
            module_name, class_name, output_uid, "source_transparent_stage1b",
            *source_selected_inputs,
        ]
        source_identity.extend(
            f"{uid}:{source_sidecar_cells[uid].execution_role}:"
            f"{source_sidecar_cells[uid].dtype}:{source_sidecar_cells[uid].source()}"
            for uid in sorted(source_sidecar_cells)
        )
        source_puid = "mxs_" + hashlib.sha256(
            "\0".join(source_identity).encode("utf-8", "backslashreplace")
        ).hexdigest()[:16]
        source_preserving_native_program = NativeFormulaProgram(
            schema="modelx_graph.native_formula.modelx_cython_source_transparent.v1",
            uid=source_puid,
            output_uid=output_uid,
            cells=source_sidecar_cells,
            input_order=tuple(source_selected_inputs),
            inputs=source_frozen_inputs,
            normalized_operators=tuple(sorted(
                prepared_normalized_operators, key=lambda x: x.uid
            )),
            module_refs=tuple(sorted(module_refs.items())),
            category_domains=tuple(
                (did, tuple(domain.labels))
                for did, domain in sorted(category_plan.encoding.domains.items())
            ),
            provenance=(
                "modelx_cython_frontend_fast_recursive_v3_"
                "source_transparent_cython_language_authority"
            ),
        )

    # Stage 2A classifies the *pre-propagation* source-transparent catalogue.
    # The Stage-1B ``source_preserving_native_program`` above remains an N0-only
    # projection that can be compiled today. The execution-class plan preserves
    # bridgeable callers as N1 instead of removing them merely because a callee
    # is owned by ordinary modelx-cython/Python fallback.
    execution_class_plan = build_execution_class_plan(
        output_uid=output_uid,
        cells=source_transparent_cell_catalog,
        direct_blockers=source_transparent_direct_blockers,
        parameter_ranges=parameter_ranges,
        native_hash_capacity_by_uid=native_hash_capacity_by_uid,
    )

    # The detached-P builder can traverse provider/Python-semantic boundaries,
    # including a P selected root. The older mixed_recursive_abi_uids property
    # stops at some of those boundaries and mixed_candidate_ready rejects P
    # roots entirely. Collect build-time evidence over the selected dependency
    # closure, a conservative superset, so detached native wrappers are sized.
    # This does not change runtime admission or the runtime recursive closure.
    # Production still uses fixed native tables with no Python overflow cache.
    mixed_missing_hash_uids = tuple(
        uid
        for uid in execution_class_plan.selected_closure_uids
        if execution_class_plan.cell(uid).memo.kind == "hash"
        and not execution_class_plan.cell(uid).memo.physicalization_ready
    )
    if mixed_missing_hash_uids:
        mixed_hash_started = perf_counter()
        from .native_hash_capacity_evidence import profile_native_hash_capacity_evidence

        missing_catalog = sorted(set(mixed_missing_hash_uids) - set(source_transparent_cell_catalog))
        if missing_catalog:
            raise ModelxCythonFastBackendError(
                "mixed recursive native hash capacity evidence cannot resolve Cells: "
                + ",".join(missing_catalog)
            )
        # The same complete exported-model pass that proves exact-hash
        # cardinality also carries complete integral call extrema.  Reuse it
        # across the selected dependency closure so representative trace
        # extrema never become unsafe dense memo bounds at runtime.
        mixed_range_cells = {
            uid: source_transparent_cell_catalog[uid]
            for uid in execution_class_plan.selected_closure_uids
            if uid in source_transparent_cell_catalog
            and source_transparent_cell_catalog[uid].execution_role == "formula"
            and source_transparent_cell_catalog[uid].parameters
            and "[]" not in source_transparent_cell_catalog[uid].source_fullname
        }
        mixed_hash_uids = tuple(
            uid
            for uid in execution_class_plan.selected_closure_uids
            if execution_class_plan.cell(uid).memo.kind == "hash"
        )
        mixed_capacity = profile_native_hash_capacity_evidence(
            model_path=model_path,
            space=space,
            output_name=output_name,
            run_keys=run_keys,
            output_cell=source_transparent_cell_catalog[output_uid],
            hash_cells={
                uid: source_transparent_cell_catalog[uid]
                for uid in mixed_hash_uids
            },
            range_cells=mixed_range_cells,
            target_load_factor=0.50,
            min_capacity=16,
            max_capacity=1 << 20,
        )

        strengthened_mixed_ranges: dict[str, list[dict[str, object]]] = {}
        for uid, observed in mixed_capacity.complete_integral_ranges_by_uid.items():
            current = tuple(parameter_ranges.get(uid, ()))
            if len(current) != len(observed):
                continue
            merged: list[tuple[int, int] | None] = []
            changes: list[dict[str, object]] = []
            for pos, (old_range, complete_range) in enumerate(zip(current, observed)):
                if complete_range is None:
                    merged.append(old_range)
                    continue
                if old_range is None:
                    new_range = (int(complete_range[0]), int(complete_range[1]))
                else:
                    new_range = (
                        min(int(old_range[0]), int(complete_range[0])),
                        max(int(old_range[1]), int(complete_range[1])),
                    )
                merged.append(new_range)
                if old_range != new_range:
                    changes.append({
                        "parameter_position": pos,
                        "before": None if old_range is None else list(old_range),
                        "complete_run_observed": list(complete_range),
                        "after": list(new_range),
                    })
            if changes:
                parameter_ranges[uid] = tuple(merged)
                strengthened_mixed_ranges[uid] = changes

        native_hash_capacity_by_uid = dict(native_hash_capacity_by_uid)
        native_hash_capacity_by_uid.update(
            {uid: int(mixed_capacity.capacities_by_uid[uid]) for uid in mixed_hash_uids}
        )
        prior_capacity_evidence = dict(native_hash_capacity_evidence)
        mixed_manifest = mixed_capacity.manifest()
        if prior_capacity_evidence:
            prior_capacity_evidence["mixed_recursive_completion"] = mixed_manifest
            prior_capacity_evidence["nan_uids"] = sorted(
                set(prior_capacity_evidence.get("nan_uids", ()))
                | set(mixed_capacity.nan_uids)
            )
            native_hash_capacity_evidence = prior_capacity_evidence
        else:
            native_hash_capacity_evidence = dict(mixed_manifest)
            native_hash_capacity_evidence["scope"] = "selected_dependency_closure_capacity_completion"

        execution_class_plan = build_execution_class_plan(
            output_uid=output_uid,
            cells=source_transparent_cell_catalog,
            direct_blockers=source_transparent_direct_blockers,
            parameter_ranges=parameter_ranges,
            native_hash_capacity_by_uid=native_hash_capacity_by_uid,
        )
        final_hash_uids = tuple(
            uid
            for uid in execution_class_plan.selected_closure_uids
            if execution_class_plan.cell(uid).memo.kind == "hash"
        )
        missing_hash_evidence = tuple(
            uid for uid in final_hash_uids
            if uid not in mixed_capacity.capacities_by_uid
            and int(native_hash_capacity_by_uid.get(uid, 0) or 0) <= 0
        )
        if missing_hash_evidence:
            raise ModelxCythonFastBackendError(
                "complete-RunDomain mixed range widening created new native hash candidates "
                "without key-capacity evidence: " + ",".join(missing_hash_evidence)
            )
        nan_final_hash = tuple(
            uid for uid in final_hash_uids if uid in set(mixed_capacity.nan_uids)
        )
        if nan_final_hash:
            raise ModelxCythonFastBackendError(
                "native exact hash does not admit NaN semantic memo keys: "
                + ",".join(nan_final_hash)
            )
        unresolved_after_profile = tuple(
            uid
            for uid in final_hash_uids
            if not execution_class_plan.cell(uid).memo.physicalization_ready
        )
        if unresolved_after_profile:
            raise ModelxCythonFastBackendError(
                "mixed recursive exact-hash capacity evidence incomplete: "
                + ",".join(unresolved_after_profile)
            )
        if native_hash_capacity_evidence:
            native_hash_capacity_evidence["mixed_recursive_strengthened_parameter_ranges"] = (
                strengthened_mixed_ranges
            )
            native_hash_capacity_evidence["mixed_recursive_integral_parameter_range_authority"] = (
                "complete_frozen_run_domain_exported_formula_trace_union_static_bounds"
            )
        finish_phase("mixed_recursive_hash_complete_run_capacity_trace", mixed_hash_started)
    for uid, row in tuple(frontend_rows.items()):
        classification = execution_class_plan.cells.get(uid)
        if classification is None:
            continue
        frontend_rows[uid] = replace(
            row,
            execution_class=classification.execution_class,
            execution_class_reasons=classification.classification_reasons,
            bridgeable_python_fallback=classification.bridgeable_python_fallback,
        )
    for uid, row in tuple(cross_frontend_rows.items()):
        classification = execution_class_plan.cells.get(uid)
        if classification is None:
            continue
        cross_frontend_rows[uid] = replace(
            row,
            execution_class=classification.execution_class,
            execution_class_reasons=classification.classification_reasons,
            bridgeable_python_fallback=classification.bridgeable_python_fallback,
        )

    finish_phase("admission_and_sidecar", phase_started)
    finish_phase("total", total_started)

    oracle_sources = dict(formulas)
    oracle_sources.update(cross_formulas)
    semantic_oracle = {
        uid: diagnose_function_semantics(source.function).manifest()
        for uid, source in sorted(oracle_sources.items())
    }

    plan = ModelxCythonFastPlan(
        schema="modelx_graph.modelx_cython_fast_plan.v7",
        module_name=module_name,
        class_name=class_name,
        output_name=output_name,
        output_uid=output_uid,
        run_keys=run_keys,
        cells=tuple(frontend_rows[uid] for uid in sorted(frontend_rows)),
        native_program=native_program,
        source_preserving_native_program=source_preserving_native_program,
        parameter_ranges=parameter_ranges,
        scalarized_vector_helpers=referenced_helpers,
        # Keep all validated finite-vector element evidence in the plan manifest,
        # even when the selected root is later blocked by an unrelated downstream
        # dependency. Only ``referenced_finite_helpers`` enter an admitted native
        # sidecar; this field is diagnostic/preparation evidence, not runtime state.
        finite_vector_element_helpers=tuple(
            helper for _key, helper in sorted(
                finite_vector_helpers.items(),
                key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1]),
            )
        ),
        runtime_inputs=selected_inputs if native_program is not None else dict(prepared_input_specs),
        source_preserving_runtime_inputs=(
            source_selected_inputs
            if source_preserving_native_program is not None
            else dict(prepared_input_specs)
        ),
        source_preserving_candidate_blockers=dict(source_candidate_blockers),
        formula_lexical_sources={
            uid: source.lexical_source
            for uid, source in {**formulas, **cross_formulas}.items()
            if source.lexical_source
        },
        formula_source_functions={
            uid: copy.deepcopy(source.function)
            for uid, source in {**formulas, **cross_formulas}.items()
        },
        normalized_operators=prepared_normalized_operators,
        semantic_formula_ir={
            uid: result.semantic
            for uid, result in {**normalized, **cross_normalized}.items()
            if result.semantic is not None
        },
        category_plan=category_plan,
        preparation_timings=dict(timings),
        preparation_memory={key: dict(value) for key, value in phase_memory.items()},
        preparation_diagnostics={
            "semantic_proof": prepared_proof_diagnostics,
            "translation_trace_keys": list(translation_trace_keys),
            "translation_trace_policy": (
                "single_deterministic_discovery_witness_full_run_domain_remains_proof_authority"
            ),
        },
        module_info=module_info,
        cross_space_calls=tuple(cross_records),
        cross_space_cells=tuple(cross_frontend_rows[uid] for uid in sorted(cross_frontend_rows)),
        ranked_output=ranked_output,
        semantic_oracle=semantic_oracle,
        authoritative_evidence=demand_evidence.manifest(),
        native_hash_capacity_evidence=native_hash_capacity_evidence,
        native_hash_capacity_by_uid=native_hash_capacity_by_uid,
        execution_class_plan=execution_class_plan,
        source_transparent_cell_catalog=dict(source_transparent_cell_catalog),
        source_transparent_direct_blockers=dict(source_transparent_direct_blockers),
        source_transparent_module_refs=dict(module_refs),
    )
    # Phase A is intentionally assertion-only: derive the new semantic facade
    # before returning the plan so any disagreement with the legacy semantic/ABI
    # representations fails closed, while all existing lowering/codegen paths
    # continue to consume their historical fields unchanged.
    plan.build_semantic_facade()
    return plan


__all__ = [
    "ModelxCythonFastBackendError",
    "FastParameterEvidence",
    "FastScalarizedVectorHelper",
    "FastCrossSpaceCall",
    "FastRankedOutput",
    "FastFrontendCell",
    "ModelxCythonFastPlan",
    "build_modelx_cython_fast_plan",
]


