from __future__ import annotations

"""Finite-category evidence adapter for the modelx-cython fast backend.

This module deliberately consumes only modelx-cython frontend metadata, local lexical
literal evidence, and complete provider columns.  It does not invoke static graph/type
analysis.  The stable category code ABI itself is reused from ``static_category_abi``
for backwards compatibility with the dev80 diagnostic path.
"""

import ast
import copy
import itertools
import numbers
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from modelx_cython.config import TransSpec

from .category_abi import (
    CategoryABIError,
    CategoryDomain,
    CategoryEncodingPlan,
    attach_parameter_positions,
)
from .finite_domain_category import (
    FiniteValue,
    TruthValue,
    assign_finite_value,
    environment_for_label,
    evaluate_predicate,
    finite_value,
)


class ModelxCythonCategoryABIError(RuntimeError):
    pass


# Optional consumer-scoped reachability is an optimization, never an admission
# prerequisite. Keep exact modelx-space walks tightly bounded; larger products
# simply retain the global category domain and normal table validation.
CONSUMER_CATEGORY_REACHABILITY_PROOF_WORK_LIMIT = 8_192
SELECTED_CATEGORY_COMPLETION_MAX_LABELS = 16
SELECTED_CATEGORY_COMPLETION_MAX_ITERATIONS = 32


def _literal(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _provider_missing(value: Any) -> bool:
    """Provider missingness is physical state, not a semantic category label."""
    if value is None:
        return True
    try:
        probe = np.asarray(value)
        if probe.ndim == 0 and probe.dtype.kind in "fc":
            return bool(np.isnan(probe.item()))
    except Exception:
        pass
    try:
        result = value != value
        return bool(result) if isinstance(result, (bool, np.bool_)) else False
    except Exception:
        return type(value).__name__ in {"NAType", "NaTType"}


def _ordered(values: Sequence[Any]) -> tuple[Any, ...]:
    if not values or not all(_literal(x) for x in values):
        return ()
    return tuple(sorted(set(values), key=lambda value: (0, "") if value is None else (1, value)))


def _spec_values(value: Any, *, context: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, (tuple, list, set)):
        raise ModelxCythonCategoryABIError(f"{context} must be a finite sequence of string/None labels")
    out = _ordered(tuple(value))
    if not out or len(out) > 16 or len(out) != len(set(value)):
        raise ModelxCythonCategoryABIError(f"{context} must contain 1..16 unique string/None labels")
    return out


@dataclass(frozen=True)
class FastCategoryEvidence:
    target: str
    domain_id: str
    labels: tuple[Any, ...]
    source: str
    detail: str = ""
    nullable: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "domain_id": self.domain_id,
            "labels": list(self.labels),
            "source": self.source,
            "detail": self.detail,
            "nullable": self.nullable,
        }


@dataclass(frozen=True)
class ParameterizedPointCategoryCase:
    field: str
    predicate: ast.expr = field(repr=False, compare=False)
    predicate_source: str = ""
    required_run_keys: tuple[Any, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "predicate": self.predicate_source or ast.unparse(self.predicate),
            "required_run_keys": list(self.required_run_keys),
        }


@dataclass(frozen=True)
class ParameterizedPointCategoryBinding:
    uid: str
    parameters: tuple[str, ...]
    parameter_ranges: tuple[tuple[str, int, int], ...]
    cases: tuple[ParameterizedPointCategoryCase, ...]
    labels: tuple[Any, ...]
    nullable: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "parameters": list(self.parameters),
            "parameter_ranges": [
                {"name": name, "min": lo, "max": hi}
                for name, lo, hi in self.parameter_ranges
            ],
            "labels": list(self.labels),
            "cases": [case.manifest() for case in self.cases],
            "nullable": self.nullable,
            "provider_validity": "explicit_code_plus_validity_input",
        }


@dataclass(frozen=True)
class ModelxCythonCategoryPlan:
    encoding: CategoryEncodingPlan
    evidence: tuple[FastCategoryEvidence, ...]
    point_fields: Mapping[str, str]
    parameterized_point_bindings: Mapping[str, ParameterizedPointCategoryBinding] = field(default_factory=dict)
    result_domain_proofs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    parameter_domain_completion_proofs: Mapping[tuple[str, str], Mapping[str, Any]] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return self.encoding.supported

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "modelx_graph.modelx_cython_category_plan.v1",
            "frontend": "modelx_cython_trace_spec_provider_local_literals",
            "static_type_domain_required": False,
            "encoding": self.encoding.manifest(),
            "evidence": [row.manifest() for row in self.evidence],
            "point_fields": dict(sorted(self.point_fields.items())),
            "parameterized_point_bindings": {
                uid: binding.manifest()
                for uid, binding in sorted(self.parameterized_point_bindings.items())
            },
            "result_domain_proofs": {
                uid: dict(proof) for uid, proof in sorted(self.result_domain_proofs.items())
            },
            "parameter_domain_completion_proofs": [
                {
                    "uid": uid,
                    "parameter": parameter,
                    **dict(proof),
                }
                for (uid, parameter), proof in sorted(
                    self.parameter_domain_completion_proofs.items()
                )
            ],
            "nullable_result_uids": sorted(self.encoding.nullable_results),
        }


def _literal_return_values(fn: ast.FunctionDef) -> tuple[Any, ...]:
    values: list[Any] = []

    def collect(expr: ast.AST | None) -> bool:
        if expr is None:
            return False
        if isinstance(expr, ast.Constant) and _literal(expr.value):
            values.append(expr.value)
            return True
        if isinstance(expr, ast.IfExp):
            left = collect(expr.body)
            right = collect(expr.orelse)
            return left and right
        return False

    returns = [node for node in ast.walk(fn) if isinstance(node, ast.Return)]
    if not returns:
        return ()
    accepted = False
    for node in returns:
        if collect(node.value):
            accepted = True
        elif node.value is not None:
            # A formula that mixes category literals and non-category returns is not
            # a finite-category result merely because one branch contains a string.
            return ()
    return _ordered(values) if accepted else ()


def _literal_sequence_return_values(fn: ast.FunctionDef) -> tuple[Any, ...]:
    """Return the exact finite element domain of literal category sequences.

    Every value-returning ``return`` in the formula must return one tuple/list/set
    whose elements are category literals.  Raises/guard-fail branches are ignored.
    This is deliberately lexical and execution-independent; it describes only the
    element domain, not sequence length/order at runtime.
    """
    values: list[Any] = []
    returns = [node for node in ast.walk(fn) if isinstance(node, ast.Return)]
    if not returns:
        return ()
    accepted = False
    for node in returns:
        expr = node.value
        if expr is None:
            continue
        if not isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
            return ()
        row: list[Any] = []
        for elt in expr.elts:
            if not isinstance(elt, ast.Constant) or not _literal(elt.value):
                return ()
            row.append(elt.value)
        if not row:
            return ()
        values.extend(row)
        accepted = True
    return _ordered(values) if accepted else ()


def _terminal_block(body: Sequence[ast.stmt]) -> bool:
    if not body:
        return False
    last = body[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    if isinstance(last, ast.Expr) and isinstance(last.value, ast.Call):
        return isinstance(last.value.func, ast.Name) and last.value.func.id == "__guard_fail__"
    if isinstance(last, ast.If):
        return bool(last.orelse) and _terminal_block(last.body) and _terminal_block(last.orelse)
    return False


def _comparison_literal_for_parameter(test: ast.AST, parameter: str) -> Any | None:
    if not (
        isinstance(test, ast.Compare) and len(test.ops) == 1
        and isinstance(test.ops[0], (ast.Eq, ast.Is)) and len(test.comparators) == 1
    ):
        return None
    left, right = test.left, test.comparators[0]
    for maybe_name, maybe_value in ((left, right), (right, left)):
        if (
            isinstance(maybe_name, ast.Name) and maybe_name.id == parameter
            and isinstance(maybe_value, ast.Constant) and _literal(maybe_value.value)
        ):
            return maybe_value.value
    return None


def _exhaustive_name_dispatch_literals(fn: ast.FunctionDef, name: str) -> tuple[Any, ...]:
    """Prove one exact category parameter domain from a terminal if/elif chain.

    This admits only a chain where every successful branch terminates and the final
    else terminates (usually by raise).  It therefore differs materially from the
    weaker ``_comparison_parameter_literals`` census: these labels are the complete
    accepted semantic dispatch domain, not merely values mentioned somewhere.
    """
    candidates: list[tuple[Any, ...]] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        labels: list[Any] = []
        current: ast.If | None = node
        ok = True
        while current is not None:
            label = _comparison_literal_for_parameter(current.test, name)
            if label is None or not _terminal_block(current.body):
                ok = False
                break
            labels.append(label)
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                continue
            if not _terminal_block(current.orelse):
                ok = False
            current = None
        if ok and labels:
            ordered = _ordered(labels)
            if ordered:
                candidates.append(ordered)
    if not candidates:
        return ()
    # Prefer the largest exact terminal dispatch.  Nested elif nodes are also
    # visited by ast.walk and yield suffixes of the same chain.
    return max(candidates, key=len)


def _exhaustive_parameter_dispatch_literals(source: Any, parameter: str) -> tuple[Any, ...]:
    return _exhaustive_name_dispatch_literals(source.function, parameter)


def _parameter_none_test(test: ast.AST, parameter: str) -> bool | None:
    """Return the predicate truth value when ``parameter is None``.

    Only the simple optional-parameter shape is admitted.  This is source proof,
    not a general boolean evaluator.
    """
    if not (
        isinstance(test, ast.Compare) and len(test.ops) == 1
        and len(test.comparators) == 1
    ):
        return None
    left, right = test.left, test.comparators[0]
    pair = ((left, right), (right, left))
    matched = any(
        isinstance(maybe_name, ast.Name) and maybe_name.id == parameter
        and isinstance(maybe_none, ast.Constant) and maybe_none.value is None
        for maybe_name, maybe_none in pair
    )
    if not matched:
        return None
    op = test.ops[0]
    if isinstance(op, (ast.Eq, ast.Is)):
        return True
    if isinstance(op, (ast.NotEq, ast.IsNot)):
        return False
    return None


def _optional_static_default_dispatch_literals(
    source: Any, parameter: str, evidence: Any,
) -> tuple[Any, ...]:
    """Exact domain for ``static_default if p is None else p`` dispatch.

    This is the category-ABI companion to I1B static-reference materialization.
    The static default must be proven over the complete RunDomain, and the local
    alias must feed an exhaustive terminal dispatch.  Therefore widening is tied
    to real source dataflow rather than the overly broad global-dispatch rule that
    Block 2B-2 intentionally rejected.
    """
    fn = source.function
    for node in ast.walk(fn):
        if not (
            isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.IfExp)
        ):
            continue
        local = node.targets[0].id
        expr = node.value
        none_truth = _parameter_none_test(expr.test, parameter)
        if none_truth is None:
            continue
        static_expr = expr.body if none_truth else expr.orelse
        param_expr = expr.orelse if none_truth else expr.body
        if not (isinstance(param_expr, ast.Name) and param_expr.id == parameter):
            continue
        if isinstance(static_expr, ast.Constant) and _literal(static_expr.value):
            default = static_expr.value
        elif isinstance(static_expr, ast.Name):
            ok, default = evidence._proven_static_reference_value(static_expr.id)
            if not ok or not _literal(default):
                continue
        else:
            continue
        dispatch = _exhaustive_name_dispatch_literals(fn, local)
        if not dispatch or default not in dispatch:
            continue
        return _ordered((None, *dispatch))
    return ()


def _direct_call_literal_args(formulas: Mapping[str, Any]) -> dict[tuple[str, int], tuple[Any, ...]]:
    """Collect exact finite category arguments from reachable lexical callsites.

    This is the first legacy category recognizer migrated onto the Phase-B finite
    domain evaluator.  The walk tracks simple assignments, conditionals and bounded
    literal ``for`` loops.  Opaque control explores both paths; nested function/
    comprehension scopes remain isolated.  Evidence is still lexical/callsite
    driven and never invents labels from a callee's source dispatch.
    """

    found: dict[tuple[str, int], list[Any]] = {}

    def iterable_values(node: ast.AST, env: Mapping[str, Any]) -> tuple[Any, ...]:
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            values: list[Any] = []
            for elt in node.elts:
                value = finite_value(elt, env)
                if value is None:
                    return ()
                values.extend(value.values)
            return _ordered(tuple(values))
        return ()

    for source in formulas.values():
        def collect_calls(node: ast.AST | None, env: Mapping[str, Any]) -> None:
            if node is None:
                return

            class Calls(ast.NodeVisitor):
                def visit_Call(self, call: ast.Call):  # noqa: N802
                    if isinstance(call.func, ast.Name) and call.func.id in formulas:
                        target = formulas[call.func.id]
                        args = list(enumerate(call.args))
                        args += [
                            (target.parameters.index(keyword.arg), keyword.value)
                            for keyword in call.keywords
                            if keyword.arg in target.parameters
                        ]
                        for pos, arg in args:
                            value = finite_value(arg, env)
                            if value is not None:
                                found.setdefault((call.func.id, pos), []).extend(value.values)
                    self.generic_visit(call)

                # Do not leak the current finite environment into nested lexical
                # scopes whose targets may shadow the same names.
                def visit_GeneratorExp(self, node):  # noqa: N802
                    return None
                visit_ListComp = visit_GeneratorExp
                visit_SetComp = visit_GeneratorExp
                visit_DictComp = visit_GeneratorExp
                visit_Lambda = visit_GeneratorExp
                def visit_FunctionDef(self, node):  # noqa: N802
                    return None
                visit_AsyncFunctionDef = visit_FunctionDef
                visit_ClassDef = visit_FunctionDef

            Calls().visit(node)

        def walk_block(
            stmts: Sequence[ast.stmt], states: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            live = [dict(state) for state in states]
            for stmt in stmts:
                if not live:
                    break
                next_live: list[dict[str, Any]] = []
                for env in live:
                    if isinstance(stmt, ast.If):
                        collect_calls(stmt.test, env)
                        proof = evaluate_predicate(stmt.test, env)
                        if proof.unsupported_category:
                            # This collector is evidence-only.  Unsupported category
                            # control cannot prove reachability, so explore both
                            # rather than using it as positive narrowing evidence.
                            branches = [stmt.body, stmt.orelse]
                        elif proof.truth == TruthValue.TRUE:
                            branches = [stmt.body]
                        elif proof.truth == TruthValue.FALSE:
                            branches = [stmt.orelse]
                        else:
                            branches = [stmt.body, stmt.orelse]
                        for branch in branches:
                            if branch:
                                next_live.extend(walk_block(branch, [dict(env)]))
                            else:
                                next_live.append(dict(env))
                        continue

                    if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                        collect_calls(stmt.iter, env)
                        labels = iterable_values(stmt.iter, env)
                        if labels:
                            for label in labels:
                                loop_env = dict(env)
                                loop_env[stmt.target.id] = FiniteValue.one(label)
                                walk_block(stmt.body, [loop_env])
                            post = dict(env)
                            post[stmt.target.id] = FiniteValue(frozenset(labels))
                            if stmt.orelse:
                                next_live.extend(walk_block(stmt.orelse, [post]))
                            else:
                                next_live.append(post)
                            continue
                        # Unknown iteration domain: scan calls without claiming a
                        # category value for the target.
                        collect_calls(stmt, env)
                        next_live.append(dict(env))
                        continue

                    if isinstance(stmt, ast.Return):
                        collect_calls(stmt.value, env)
                        continue
                    if isinstance(stmt, ast.Raise):
                        collect_calls(stmt.exc, env)
                        continue
                    if (
                        isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                    ):
                        collect_calls(stmt.value, env)
                        assign_finite_value(env, stmt.targets[0].id, stmt.value)
                        next_live.append(env)
                        continue
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        collect_calls(stmt.value, env)
                        assign_finite_value(env, stmt.target.id, stmt.value)
                        next_live.append(env)
                        continue
                    collect_calls(stmt, env)
                    next_live.append(env)
                live = next_live
            return live

        walk_block(source.function.body, [{}])

    return {key: _ordered(labels) for key, labels in found.items() if _ordered(labels)}


def _bound_table_parameter_labels(formulas, evidence):
    """Read finite string selector domains from complete bound table indexes.

    Admit only a model reference (or its zero-argument provider) and direct
    loc/at row selectors. Never evaluate a formula expression or infer from a
    trace sample. Values stay snapshot-bound through the existing input binder.
    """
    import pandas as pd
    if not hasattr(evidence, "_context_root_for_key"):
        return {}
    found = {}
    def provider(expr, key):
        call = isinstance(expr, ast.Call)
        if call:
            if expr.args or expr.keywords:
                raise ValueError("parameterized provider")
            expr = expr.func
        attrs = []
        while isinstance(expr, ast.Attribute):
            attrs.append(expr.attr)
            expr = expr.value
        if not isinstance(expr, ast.Name) or expr.id not in evidence.refs:
            raise ValueError("not a bound table reference")
        value = evidence._context_root_for_key(expr.id, key)
        for name in reversed(attrs):
            value = getattr(value, name)
        return value() if call else value
    for uid, source in formulas.items():
        for node in ast.walk(source.function):
            if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr in {"loc", "at"}):
                continue
            candidates = {}
            try:
                for key in evidence.run_keys:
                    table = provider(node.value.value, key)
                    if not isinstance(table, (pd.DataFrame, pd.Series)):
                        raise ValueError("not a pandas table")
                    row = node.slice
                    if isinstance(table, pd.DataFrame) and isinstance(row, ast.Tuple):
                        if len(row.elts) != 2 or not isinstance(row.elts[1], ast.Constant) or row.elts[1].value not in table.columns:
                            raise ValueError("ambiguous row/column selector")
                        row = row.elts[0]
                    axes = row.elts if isinstance(row, ast.Tuple) else [row]
                    if len(axes) > table.index.nlevels:
                        raise ValueError("too many row axes")
                    for level, selector in enumerate(axes):
                        if not isinstance(selector, ast.Name) or selector.id not in source.parameters:
                            continue
                        values = tuple(table.index.get_level_values(level).unique())
                        if not values or not all(isinstance(v, str) for v in values):
                            continue
                        labels = candidates.setdefault(selector.id, set())
                        labels.update(values)
                        if len(labels) > 256:
                            raise ValueError("table category domain exceeds 256 labels")
            except Exception:
                continue
            for parameter, labels in candidates.items():
                found.setdefault((uid, parameter), set()).update(labels)
    return {key: _ordered(tuple(labels)) for key, labels in found.items() if len(labels) <= 256}


def _comparison_parameter_literals(source: Any) -> dict[str, tuple[Any, ...]]:
    by_name: dict[str, list[Any]] = {name: [] for name in source.parameters}
    params = set(source.parameters)
    for node in ast.walk(source.function):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
            continue
        left, right = node.left, node.comparators[0]
        if isinstance(left, ast.Name) and left.id in params:
            if isinstance(right, ast.Constant) and _literal(right.value):
                by_name[left.id].append(right.value)
            elif isinstance(right, (ast.Tuple, ast.List, ast.Set)):
                vals = [elt.value for elt in right.elts if isinstance(elt, ast.Constant) and _literal(elt.value)]
                if len(vals) == len(right.elts):
                    by_name[left.id].extend(vals)
        if isinstance(right, ast.Name) and right.id in params and isinstance(left, ast.Constant) and _literal(left.value):
            by_name[right.id].append(left.value)
    return {name: _ordered(values) for name, values in by_name.items() if _ordered(values)}


def _point_field_source(fn: ast.FunctionDef, point_uid: str) -> str | None:
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]

    def field_from_expr(expr: ast.AST) -> str | None:
        if not (
            isinstance(expr, ast.Subscript)
            and isinstance(expr.value, ast.Call)
            and isinstance(expr.value.func, ast.Name)
            and expr.value.func.id == point_uid
            and not expr.value.args and not expr.value.keywords
            and isinstance(expr.slice, ast.Constant)
            and isinstance(expr.slice.value, str)
        ):
            return None
        return str(expr.slice.value)

    if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
        return field_from_expr(body[0].value)
    if (
        len(body) >= 2 and isinstance(body[0], ast.Assign)
        and len(body[0].targets) == 1 and isinstance(body[0].targets[0], ast.Name)
    ):
        field = field_from_expr(body[0].value)
        local = body[0].targets[0].id
        if field is not None and isinstance(body[-1], ast.Return) and isinstance(body[-1].value, ast.Name) and body[-1].value.id == local:
            return field
    return None


def _point_values(evidence: Any, field: str) -> tuple[tuple[Any, ...], bool]:
    source = evidence.model_point_row
    if source is None:
        return (), False
    frame = source.provider()
    if field not in frame.columns:
        return (), False
    nparams = len(evidence.space_params)
    pos = int(source.semantic.key_position)
    rows = [key[pos] if nparams > 1 and isinstance(key, tuple) else key for key in evidence.run_keys]
    try:
        values = frame.loc[rows, field].tolist()
    except Exception as exc:
        raise ModelxCythonCategoryABIError(f"point field {field!r} cannot be read over run domain: {exc}") from exc
    nullable = any(_provider_missing(value) for value in values)
    concrete = tuple(value for value in values if not _provider_missing(value))
    return _ordered(concrete), nullable



def _point_field_expr(expr: ast.AST, point_uid: str) -> str | None:
    if not (
        isinstance(expr, ast.Subscript)
        and isinstance(expr.value, ast.Call)
        and isinstance(expr.value.func, ast.Name)
        and expr.value.func.id == point_uid
        and not expr.value.args and not expr.value.keywords
        and isinstance(expr.slice, ast.Constant)
        and isinstance(expr.slice.value, str)
    ):
        return None
    return str(expr.slice.value)


def _and_expr(parts: Sequence[ast.expr]) -> ast.expr:
    if not parts:
        return ast.Constant(True)
    if len(parts) == 1:
        return copy.deepcopy(parts[0])
    return ast.BoolOp(ast.And(), [copy.deepcopy(x) for x in parts])


def _not_expr(expr: ast.expr) -> ast.expr:
    return ast.UnaryOp(ast.Not(), copy.deepcopy(expr))


def _extract_parameterized_point_cases(
    fn: ast.FunctionDef, point_uid: str
) -> tuple[tuple[str, ast.expr], ...] | None:
    """Recognize bounded if/elif trees returning direct model-point fields.

    The returned predicates include prior-branch fallthrough conditions, so provider
    validity can be checked only where a selected field is actually reachable.
    """

    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    cases: list[tuple[str, ast.expr]] = []

    def walk(stmts: Sequence[ast.stmt], paths: list[list[ast.expr]]) -> list[list[ast.expr]] | None:
        active = paths
        for stmt in stmts:
            if not active:
                break
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    return None
                field_name = _point_field_expr(stmt.value, point_uid)
                if field_name is None:
                    return None
                for path in active:
                    cases.append((field_name, ast.fix_missing_locations(_and_expr(path))))
                active = []
                continue
            if (
                isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "__guard_fail__"
            ):
                active = []
                continue
            if isinstance(stmt, ast.If):
                next_active: list[list[ast.expr]] = []
                for path in active:
                    yes = walk(stmt.body, [path + [copy.deepcopy(stmt.test)]])
                    if yes is None:
                        return None
                    if stmt.orelse:
                        no = walk(stmt.orelse, [path + [_not_expr(stmt.test)]])
                        if no is None:
                            return None
                    else:
                        no = [path + [_not_expr(stmt.test)]]
                    next_active.extend(yes)
                    next_active.extend(no)
                active = next_active
                continue
            # Keep this deliberately narrow. Assignments, loops, dynamic dispatch,
            # and arbitrary successful expressions stay with ordinary modelx-cython.
            return None
        return active

    remaining = walk(body, [[]])
    if remaining is None or not cases:
        return None
    # Every successful path must be one of the direct-field returns; any remaining
    # fallthrough path would imply an implicit None/dynamic result.
    if remaining:
        return None
    return tuple(cases)



def _direct_call_integer_args(formulas: Mapping[str, Any]) -> dict[tuple[str, int], tuple[int, ...]]:
    found: dict[tuple[str, int], set[int]] = {}
    known = set(formulas)
    for source in formulas.values():
        for node in ast.walk(source.function):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in known):
                continue
            for pos, arg in enumerate(node.args):
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int) and not isinstance(arg.value, bool):
                    found.setdefault((node.func.id, pos), set()).add(int(arg.value))
    return {key: tuple(sorted(values)) for key, values in found.items()}


def _case_integer_values(
    cases: Sequence[tuple[str, ast.expr]], parameters: Sequence[str],
) -> Mapping[str, tuple[int, ...]] | None:
    params = tuple(parameters)
    values: dict[str, set[int]] = {name: set() for name in params}

    def positive_equalities(node: ast.AST, pinned: dict[str, set[int]], *, negated: bool = False) -> None:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            positive_equalities(node.operand, pinned, negated=not negated); return
        if isinstance(node, ast.BoolOp):
            for child in node.values: positive_equalities(child, pinned, negated=negated)
            return
        if negated or not (isinstance(node, ast.Compare) and len(node.ops)==1 and isinstance(node.ops[0],(ast.Eq,ast.Is)) and len(node.comparators)==1):
            return
        left,right=node.left,node.comparators[0]
        for maybe_name,maybe_value in ((left,right),(right,left)):
            if (isinstance(maybe_name,ast.Name) and maybe_name.id in pinned and isinstance(maybe_value,ast.Constant) and isinstance(maybe_value.value,int) and not isinstance(maybe_value.value,bool)):
                pinned[maybe_name.id].add(int(maybe_value.value))

    for _field,predicate in cases:
        pinned={name:set() for name in params}; positive_equalities(predicate,pinned)
        if any(len(pinned[name])!=1 for name in params): return None
        for name in params: values[name].update(pinned[name])
    if any(not values[name] for name in params): return None
    return {name:tuple(sorted(values[name])) for name in params}


def _integer_parameter_ranges(
    cell: Any, parameters: Sequence[str], *, class_info: Any | None = None,
    direct_integer_args: Mapping[tuple[str, int], tuple[int, ...]] | None = None,
    uid: str | None = None,
    structural_integer_values: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[tuple[str, int, int], ...] | None:
    spec = getattr(cell, "_spec", {}) or {}
    type_overrides = spec.get(getattr(TransSpec, "PARAM_T", "param_type"), {}) or {}
    range_overrides = spec.get(getattr(TransSpec, "PARAM_RANGE", "param_range"), {}) or {}
    rt = getattr(cell, "_rt", None)
    class_sizes = None
    if class_info is not None:
        try:
            class_sizes = class_info.cells_arg_sizes.get(tuple(cell.params))
        except Exception:
            class_sizes = None
    out: list[tuple[str, int, int]] = []
    for pos, name in enumerate(parameters):
        tp = None
        if isinstance(type_overrides, Mapping) and name in type_overrides:
            value = type_overrides[name]
            if value not in {"int", "int64", "long", "Integral", "bool"}:
                return None
            tp = int
        elif rt is not None and name in getattr(rt, "arg_types", {}):
            tp = rt.arg_types[name]
        try:
            if tp is None or not issubclass(tp, numbers.Integral):
                return None
        except TypeError:
            return None
        if isinstance(range_overrides, Mapping) and name in range_overrides:
            value = range_overrides[name]
            if not (isinstance(value, (tuple, list)) and len(value) == 2):
                return None
            lo, hi = int(value[0]), int(value[1])
        elif rt is not None and name in getattr(rt, "min_args", {}) and name in getattr(rt, "max_args", {}):
            lo, hi = int(rt.min_args[name]), int(rt.max_args[name])
            if class_sizes is not None and pos < len(class_sizes) and lo >= 0:
                hi = max(hi, int(class_sizes[pos]) - 1)
        elif class_sizes is not None and pos < len(class_sizes):
            lo, hi = 0, int(class_sizes[pos]) - 1
        else:
            return None
        direct = () if uid is None else tuple((direct_integer_args or {}).get((uid, pos), ()))
        if direct:
            lo = min(lo, min(direct)); hi = max(hi, max(direct))
        structural = tuple((structural_integer_values or {}).get(name, ()))
        if structural:
            lo = min(lo, min(structural)); hi = max(hi, max(structural))
        if lo > hi or hi - lo > 32:
            return None
        out.append((name, lo, hi))
    return tuple(out)


def _predicate_value(
    node: ast.AST, *, params: Mapping[str, Any], run_key: Any, formulas: Mapping[str, Any], evidence: Any
) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in params:
            return params[node.id]
        raise ModelxCythonCategoryABIError(f"unsupported predicate name {node.id!r}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(_predicate_value(node.operand, params=params, run_key=run_key, formulas=formulas, evidence=evidence))
    if isinstance(node, ast.BoolOp):
        vals = [_predicate_value(x, params=params, run_key=run_key, formulas=formulas, evidence=evidence) for x in node.values]
        return all(bool(x) for x in vals) if isinstance(node.op, ast.And) else any(bool(x) for x in vals)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        left = _predicate_value(node.left, params=params, run_key=run_key, formulas=formulas, evidence=evidence)
        right = _predicate_value(node.comparators[0], params=params, run_key=run_key, formulas=formulas, evidence=evidence)
        op = node.ops[0]
        if isinstance(op, (ast.Eq, ast.Is)):
            return left == right
        if isinstance(op, (ast.NotEq, ast.IsNot)):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        raise ModelxCythonCategoryABIError("unsupported predicate comparison")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in formulas:
        source = formulas[node.func.id]
        cell = getattr(evidence.space[run_key], source.name)
        args = [
            _predicate_value(x, params=params, run_key=run_key, formulas=formulas, evidence=evidence)
            for x in node.args
        ]
        if node.keywords:
            raise ModelxCythonCategoryABIError("predicate Cell calls must be positional")
        value = cell(*args)
        if not isinstance(value, (bool, numbers.Integral)):
            raise ModelxCythonCategoryABIError(
                f"predicate Cell {source.name!r} returned unsupported {type(value).__name__}"
            )
        return value
    raise ModelxCythonCategoryABIError(
        f"unsupported parameterized point-provider predicate {ast.unparse(node)!r}"
    )



def prove_consumer_category_result_domains(
    formulas: Mapping[str, Any], *, evidence: Any, output_uid: str,
    encoding: CategoryEncodingPlan,
) -> dict[tuple[str, str], tuple[Any, ...]]:
    """Prove consumer-scoped categorical result domains over the complete RunDomain.

    This is a deliberately small control-flow proof, not a dependency graph. For
    each formula it records direct Cell calls under source ``if`` paths. A predicate
    is used to narrow run keys only when it can be evaluated without formula
    parameters; otherwise both branches remain reachable. Reachability then flows
    from the selected output through those local call masks. The resulting subset
    never changes the category ABI or its integer codes; table normalization may use
    it only to prove that a missing semantic table label cannot reach that consumer.
    """
    run_keys = tuple(evidence.run_keys)
    # This proof is an optional table-normalization optimization. It must never
    # turn complete-domain evidence into an O(policies * formulas) preparation
    # tax for large models. If there is no zero-argument categorical provider, or
    # the exact proof would exceed the bounded work budget, decline the proof and
    # let normal table validation fail closed where narrowing is actually needed.
    categorical_providers = tuple(
        uid for uid in encoding.result_domains
        if uid in formulas and not formulas[uid].parameters
    )
    if not categorical_providers:
        return {}
    proof_work = len(run_keys) * max(1, len(formulas))
    if proof_work > CONSUMER_CATEGORY_REACHABILITY_PROOF_WORK_LIMIT:
        return {}

    all_keys = set(run_keys)
    formula_ids = frozenset(formulas)

    def calls_in(node: ast.AST | None) -> set[str]:
        if node is None:
            return set()
        return {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            and child.func.id in formula_ids
        }

    local_masks: dict[str, dict[str, set[Any]]] = {}
    for uid, source in formulas.items():
        found: dict[str, set[Any]] = {}

        def record(node: ast.AST | None, mask: set[Any]) -> None:
            if not mask or node is None:
                return
            for callee in calls_in(node):
                found.setdefault(callee, set()).update(mask)

        def split(test: ast.AST, mask: set[Any]) -> tuple[set[Any], set[Any]]:
            yes: set[Any] = set()
            no: set[Any] = set()
            for key in mask:
                try:
                    value = bool(_predicate_value(
                        test, params={}, run_key=key, formulas=formulas, evidence=evidence
                    ))
                except Exception:
                    # Parameter-dependent or otherwise unsupported predicates carry
                    # no run-key proof. Conservatively keep both paths.
                    yes.add(key)
                    no.add(key)
                else:
                    (yes if value else no).add(key)
            return yes, no

        def walk_block(stmts: Sequence[ast.stmt], incoming: set[Any]) -> set[Any]:
            live = set(incoming)
            for stmt in stmts:
                if not live:
                    break
                if isinstance(stmt, ast.If):
                    record(stmt.test, live)
                    yes, no = split(stmt.test, live)
                    yes_fall = walk_block(stmt.body, yes)
                    no_fall = walk_block(stmt.orelse, no) if stmt.orelse else no
                    live = yes_fall | no_fall
                    continue
                if isinstance(stmt, ast.Return):
                    record(stmt.value, live)
                    live.clear()
                    break
                if isinstance(stmt, ast.Assign):
                    record(stmt.value, live)
                    continue
                if isinstance(stmt, ast.AnnAssign):
                    record(stmt.value, live)
                    continue
                if isinstance(stmt, ast.AugAssign):
                    record(stmt.value, live)
                    continue
                if isinstance(stmt, ast.Expr):
                    record(stmt.value, live)
                    continue
                # Unsupported local control is not used to narrow anything. Its
                # calls are still conservatively reachable from the current mask.
                record(stmt, live)
            return live

        walk_block(source.function.body, set(all_keys))
        local_masks[uid] = found

    reachable: dict[str, set[Any]] = {uid: set() for uid in formulas}
    if output_uid not in reachable:
        return {}
    reachable[output_uid] = set(all_keys)
    changed = True
    while changed:
        changed = False
        for caller, caller_mask in reachable.items():
            if not caller_mask:
                continue
            for callee, local_mask in local_masks.get(caller, {}).items():
                contribution = caller_mask & local_mask
                before = len(reachable[callee])
                reachable[callee].update(contribution)
                if len(reachable[callee]) != before:
                    changed = True

    out: dict[tuple[str, str], tuple[Any, ...]] = {}
    for consumer, consumer_mask in reachable.items():
        if not consumer_mask:
            continue
        for provider, local_mask in local_masks.get(consumer, {}).items():
            did = encoding.result_domain_id(provider)
            if did is None or formulas[provider].parameters:
                continue
            call_mask = consumer_mask & local_mask
            if not call_mask:
                continue
            domain = encoding.domain(did)
            observed: set[Any] = set()
            try:
                for key in call_mask:
                    value = getattr(evidence.space[key], formulas[provider].name)()
                    if value not in domain.labels:
                        raise ModelxCythonCategoryABIError(
                            f"category provider {formulas[provider].name!r} returned "
                            f"{value!r} outside {did}"
                        )
                    observed.add(value)
            except Exception:
                # Failure to validate a complete-RunDomain provider subset simply
                # yields no narrowing proof; global category semantics remain intact.
                continue
            labels = tuple(label for label in domain.labels if label in observed)
            if labels and len(labels) < len(domain.labels):
                out[(consumer, provider)] = labels
    return out


def _build_parameterized_point_binding(
    uid: str, source: Any, *, cell: Any, class_info: Any,
    formulas: Mapping[str, Any], evidence: Any, point_uid: str,
    direct_integer_args: Mapping[tuple[str, int], tuple[int, ...]] | None = None,
) -> ParameterizedPointCategoryBinding | None:
    if not source.parameters:
        return None
    raw_cases = _extract_parameterized_point_cases(source.function, point_uid)
    if raw_cases is None:
        return None
    structural_integer_values = _case_integer_values(raw_cases, source.parameters)
    ranges = _integer_parameter_ranges(
        cell, source.parameters, class_info=class_info,
        direct_integer_args=direct_integer_args, uid=uid,
        structural_integer_values=structural_integer_values,
    )
    if ranges is None:
        return None
    size = 1
    for _name, lo, hi in ranges:
        size *= hi - lo + 1
    if size > 64:
        return None

    frame = evidence.model_point_row.provider()
    semantic = evidence.model_point_row.semantic
    nparams = len(evidence.space_params)
    pos = int(semantic.key_position)
    labels: list[Any] = []
    nullable = False
    cases: list[ParameterizedPointCategoryCase] = []
    names = tuple(name for name, _lo, _hi in ranges)
    domains = [range(lo, hi + 1) for _name, lo, hi in ranges]
    for field_name, predicate in raw_cases:
        if field_name not in frame.columns:
            return None
        required: list[Any] = []
        for run_key in evidence.run_keys:
            reachable = False
            for values in itertools.product(*domains):
                params = dict(zip(names, values))
                if bool(_predicate_value(
                    predicate, params=params, run_key=run_key, formulas=formulas, evidence=evidence
                )):
                    reachable = True
                    break
            if reachable:
                required.append(run_key)
                row = run_key[pos] if nparams > 1 and isinstance(run_key, tuple) else run_key
                value = frame.loc[row, field_name]
                if _provider_missing(value):
                    nullable = True
                    continue
                if not isinstance(value, str):
                    raise ModelxCythonCategoryABIError(
                        f"{uid}: reachable point field {field_name!r} has unsupported non-string value {value!r} on {run_key!r}"
                    )
                labels.append(value)
        cases.append(ParameterizedPointCategoryCase(
            field=field_name, predicate=copy.deepcopy(predicate),
            predicate_source=ast.unparse(predicate), required_run_keys=tuple(required),
        ))
    ordered = _ordered(labels)
    if not ordered:
        return None
    return ParameterizedPointCategoryBinding(
        uid=uid, parameters=tuple(source.parameters), parameter_ranges=ranges,
        cases=tuple(cases), labels=ordered, nullable=nullable,
    )


@dataclass(frozen=True)
class _CategoryValueSet:
    labels: tuple[Any, ...]
    nullable: bool = False

    @staticmethod
    def merge(values: Sequence["_CategoryValueSet"]) -> "_CategoryValueSet" | None:
        if not values:
            return None
        labels: list[Any] = []
        nullable = False
        for value in values:
            labels.extend(value.labels)
            nullable = nullable or value.nullable
        ordered = _ordered(labels)
        return _CategoryValueSet(ordered, nullable) if ordered else None


def _selected_formula_closure(formulas: Mapping[str, Any], output_uid: str | None) -> tuple[str, ...]:
    if output_uid is None or output_uid not in formulas:
        return ()
    known = set(formulas)
    selected: set[str] = set()
    queue = [output_uid]
    while queue:
        uid = queue.pop()
        if uid in selected or uid not in formulas:
            continue
        selected.add(uid)
        for node in ast.walk(formulas[uid].function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in known:
                queue.append(node.func.id)
    return tuple(sorted(selected))


def _category_alias_test_value(
    test: ast.AST, aliases: set[str], label: Any,
    other_labels: Mapping[str, Any] | None = None,
) -> tuple[bool | None, bool, bool]:
    """Evaluate one category guard through the generic finite-domain analyzer.

    Returns ``(decision, related, unsupported_category)``.  ``decision=None`` is
    ordinary three-valued UNKNOWN and is safe for conservative branch exploration
    when ``unsupported_category`` is false.
    """
    env = environment_for_label(tuple(sorted(aliases)), label, other_labels=other_labels)
    result = evaluate_predicate(test, env)
    return result.exact_bool, result.category_related, result.unsupported_category


def _simple_category_aliases(fn: ast.FunctionDef, parameter: str) -> set[str]:
    """Return locals that are unambiguously direct aliases of one parameter."""
    aliases = {parameter}
    assignments: dict[str, list[ast.AST]] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assignments.setdefault(node.targets[0].id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            assignments.setdefault(node.target.id, []).append(node.value)
    changed = True
    while changed:
        changed = False
        for name, values in assignments.items():
            if name in aliases or not values:
                continue
            if all(isinstance(value, ast.Name) and value.id in aliases for value in values):
                aliases.add(name)
                changed = True
    return aliases


def _selected_category_forward_demands(
    formulas: Mapping[str, Any], *, output_uid: str | None,
    domains: Mapping[str, CategoryDomain],
    parameter_domains: Mapping[tuple[str, str], str],
    result_domains: Mapping[str, str],
) -> dict[tuple[str, str], dict[Any, set[tuple[str, str]]]]:
    """Collect exact dynamic category demands from selected source callsites.

    Each caller category parameter is enumerated over its finite ABI domain.  Source
    guards involving that parameter are evaluated exactly, so labels handled by an
    earlier terminal branch do not leak into a later call.  Unrelated control is
    explored conservatively on both branches.
    """
    selected = set(_selected_formula_closure(formulas, output_uid))
    if not selected:
        return {}
    out: dict[tuple[str, str], dict[Any, set[tuple[str, str]]]] = {}

    # A category-valued Cell result passed directly into another category
    # parameter is an exact selected-closure demand even when the caller itself
    # has no category parameter.  This is common in Python-semantic bridge
    # formulas such as ``mort_rate_at_age(sex(life), ...)``.  Keep this separate
    # from parameter-forwarding analysis so demand completion remains tied to
    # actual callsites rather than globally widening every source dispatch.
    for caller_uid in sorted(selected):
        caller = formulas[caller_uid]
        for call in (
            node for node in ast.walk(caller.function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in selected
        ):
            callee_uid = call.func.id
            callee = formulas[callee_uid]

            def add_result_demand(arg: ast.AST, target: tuple[str, str]) -> None:
                if not (
                    isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Name)
                    and arg.func.id in result_domains
                ):
                    return
                did = result_domains[arg.func.id]
                domain = domains.get(did)
                if domain is None:
                    return
                for label in domain.labels:
                    out.setdefault(target, {}).setdefault(label, set()).add(
                        (caller_uid, f"result:{arg.func.id}")
                    )

            for pos, arg in enumerate(call.args):
                if pos >= len(callee.parameters):
                    continue
                target = (callee_uid, callee.parameters[pos])
                if target in parameter_domains:
                    add_result_demand(arg, target)
            for keyword in call.keywords:
                if keyword.arg is None or keyword.arg not in callee.parameters:
                    continue
                target = (callee_uid, keyword.arg)
                if target in parameter_domains:
                    add_result_demand(keyword.value, target)

    def scan_calls(
        node: ast.AST | None, *, caller_uid: str, caller_parameter: str,
        aliases: set[str], label: Any,
    ) -> None:
        if node is None:
            return
        for call in (
            child for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            and child.func.id in selected
        ):
            callee_uid = call.func.id
            callee = formulas[callee_uid]
            for pos, arg in enumerate(call.args):
                if pos >= len(callee.parameters):
                    continue
                target = (callee_uid, callee.parameters[pos])
                if target not in parameter_domains:
                    continue
                if isinstance(arg, ast.Name) and arg.id in aliases:
                    out.setdefault(target, {}).setdefault(label, set()).add(
                        (caller_uid, caller_parameter)
                    )
            for keyword in call.keywords:
                if keyword.arg is None or keyword.arg not in callee.parameters:
                    continue
                target = (callee_uid, keyword.arg)
                if target not in parameter_domains:
                    continue
                if isinstance(keyword.value, ast.Name) and keyword.value.id in aliases:
                    out.setdefault(target, {}).setdefault(label, set()).add(
                        (caller_uid, caller_parameter)
                    )

    def walk_block(
        stmts: Sequence[ast.stmt], *, caller_uid: str, caller_parameter: str,
        label: Any, initial_aliases: set[str],
    ) -> list[set[str]]:
        states = [set(initial_aliases)]
        for stmt in stmts:
            next_states: list[set[str]] = []
            for aliases in states:
                if isinstance(stmt, ast.If):
                    scan_calls(
                        stmt.test, caller_uid=caller_uid, caller_parameter=caller_parameter,
                        aliases=aliases, label=label,
                    )
                    decision, _related, unsupported = _category_alias_test_value(stmt.test, aliases, label)
                    branches: list[Sequence[ast.stmt]]
                    if decision is True:
                        branches = [stmt.body]
                    elif decision is False:
                        branches = [stmt.orelse]
                    else:
                        branches = [stmt.body, stmt.orelse]
                    for branch in branches:
                        if branch:
                            next_states.extend(walk_block(
                                branch, caller_uid=caller_uid,
                                caller_parameter=caller_parameter, label=label,
                                initial_aliases=set(aliases),
                            ))
                        else:
                            next_states.append(set(aliases))
                    continue
                if isinstance(stmt, ast.Return):
                    scan_calls(
                        stmt.value, caller_uid=caller_uid, caller_parameter=caller_parameter,
                        aliases=aliases, label=label,
                    )
                    continue
                if isinstance(stmt, ast.Raise):
                    scan_calls(
                        stmt.exc, caller_uid=caller_uid, caller_parameter=caller_parameter,
                        aliases=aliases, label=label,
                    )
                    continue
                if (
                    isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id == "__guard_fail__"
                ):
                    continue
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                ):
                    scan_calls(
                        stmt.value, caller_uid=caller_uid, caller_parameter=caller_parameter,
                        aliases=aliases, label=label,
                    )
                    target = stmt.targets[0].id
                    if isinstance(stmt.value, ast.Name) and stmt.value.id in aliases:
                        aliases.add(target)
                    else:
                        aliases.discard(target)
                    next_states.append(aliases)
                    continue
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    scan_calls(
                        stmt.value, caller_uid=caller_uid, caller_parameter=caller_parameter,
                        aliases=aliases, label=label,
                    )
                    if isinstance(stmt.value, ast.Name) and stmt.value.id in aliases:
                        aliases.add(stmt.target.id)
                    else:
                        aliases.discard(stmt.target.id)
                    next_states.append(aliases)
                    continue
                scan_calls(
                    stmt, caller_uid=caller_uid, caller_parameter=caller_parameter,
                    aliases=aliases, label=label,
                )
                next_states.append(aliases)
            states = next_states
            if not states:
                break
        return states

    for caller_uid in sorted(selected):
        source = formulas[caller_uid]
        for parameter in source.parameters:
            did = parameter_domains.get((caller_uid, parameter))
            if did is None:
                continue
            labels = tuple(domains[did].labels)
            if len(labels) > SELECTED_CATEGORY_COMPLETION_MAX_LABELS:
                continue
            for label in labels:
                walk_block(
                    source.function.body,
                    caller_uid=caller_uid,
                    caller_parameter=parameter,
                    label=label,
                    initial_aliases={parameter},
                )
    return out


def _parameter_label_terminal_outcomes(
    fn: ast.FunctionDef, parameter: str, label: Any, other_labels: Mapping[str, Any] | None = None,
) -> set[str]:
    """Classify terminal paths with the generic finite-domain abstract interpreter.

    The tracked parameter and any other proven category parameters start as exact
    singleton finite values.  Simple assignments/conditional expressions preserve
    finite-domain information.  Opaque non-category predicates explore both paths;
    only unsupported operations that consume a tracked category fail closed.
    """

    initial_env = environment_for_label((parameter,), label, other_labels=other_labels)

    def walk(
        stmts: Sequence[ast.stmt], states: list[dict[str, Any]],
    ) -> tuple[set[str], list[dict[str, Any]]]:
        outcomes: set[str] = set()
        live = [dict(x) for x in states]
        for stmt in stmts:
            if not live:
                break
            next_live: list[dict[str, Any]] = []
            for env in live:
                if isinstance(stmt, ast.If):
                    proof = evaluate_predicate(stmt.test, env)
                    if proof.unsupported_category:
                        outcomes.add("unknown_category_guard")
                        continue
                    if proof.truth == TruthValue.TRUE:
                        branches = [stmt.body]
                    elif proof.truth == TruthValue.FALSE:
                        branches = [stmt.orelse]
                    else:
                        branches = [stmt.body, stmt.orelse]
                    for branch in branches:
                        if not branch:
                            next_live.append(dict(env))
                            continue
                        branch_outcomes, branch_live = walk(branch, [dict(env)])
                        outcomes.update(branch_outcomes)
                        next_live.extend(branch_live)
                    continue
                if isinstance(stmt, ast.Return):
                    outcomes.add("return")
                    continue
                if isinstance(stmt, ast.Raise):
                    outcomes.add("raise")
                    continue
                if (
                    isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id == "__guard_fail__"
                ):
                    outcomes.add("raise")
                    continue
                if (
                    isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                ):
                    assign_finite_value(env, stmt.targets[0].id, stmt.value)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    assign_finite_value(env, stmt.target.id, stmt.value)
                next_live.append(env)
            live = next_live
        return outcomes, live

    outcomes, live = walk(fn.body, [initial_env])
    if live:
        outcomes.add("fallthrough")
    return outcomes


def _prove_parameter_accepts_category_label(
    formulas: Mapping[str, Any], *, selected: set[str], uid: str, parameter: str,
    label: Any, domains: Mapping[str, CategoryDomain],
    parameter_domains: Mapping[tuple[str, str], str],
    result_domains: Mapping[str, str],
    exhaustive_dispatch: Mapping[str, Mapping[str, tuple[Any, ...]]],
    cache: dict[tuple[str, str, Any], tuple[bool, str]],
    active: set[tuple[str, str, Any]],
    python_semantic_parameters: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[bool, str]:
    """Fail-closed source proof that one new label is valid for one parameter.

    The use proof is path-sensitive for finite category guards.  This matters for
    source such as ``if sex == 'mix': return ...`` followed by a table lookup that
    is valid only for the remaining labels.  A global AST scan would incorrectly
    reject ``'mix'`` because it sees the unreachable lookup after the terminal
    branch.  Unrelated control is still explored conservatively on both branches.
    """
    key = (uid, parameter, label)
    if key in cache:
        return cache[key]
    did = parameter_domains.get((uid, parameter))
    if did is not None and label in domains[did].labels:
        return True, "existing_category_domain"
    if uid not in selected or uid not in formulas:
        return False, "target_outside_selected_closure"
    exact = exhaustive_dispatch.get(uid, {}).get(parameter, ())
    if exact:
        result = (
            label in exact,
            "terminal_dispatch_accepts" if label in exact else "terminal_dispatch_rejects",
        )
        cache[key] = result
        return result
    other_domains = {
        name: domains[domain_id].labels
        for (owner, name), domain_id in parameter_domains.items()
        if owner == uid and name != parameter
    }
    combinations = 1
    for labels in other_domains.values():
        combinations *= len(labels)
    if combinations > 256:
        return False, "category_terminal_proof_work_bound"
    terminal_outcomes = set()
    for values in itertools.product(*other_domains.values()):
        terminal_outcomes.update(_parameter_label_terminal_outcomes(
            formulas[uid].function, parameter, label, dict(zip(other_domains, values))
        ))
    if "unknown_category_guard" in terminal_outcomes:
        cache[key] = (False, "unsupported_category_guard")
        return cache[key]
    if "raise" in terminal_outcomes:
        cache[key] = (False, "source_path_rejects_label")
        return cache[key]
    if key in active:
        return False, "unanchored_category_forwarding_cycle"
    active.add(key)
    try:
        fn = formulas[uid].function
        forwarded = False

        def check_fragment(node: ast.AST | None, aliases: set[str]) -> tuple[bool, str]:
            """Validate reachable uses of the tracked category in one expression.

            The fragment must not include nested statement bodies.  Calls into
            selected typed Cells are recursively proven; all other opaque uses
            fail closed.
            """
            nonlocal forwarded
            if node is None:
                return True, "no_category_use"
            parent: dict[int, ast.AST] = {}
            for item in ast.walk(node):
                for child in ast.iter_child_nodes(item):
                    parent[id(child)] = item
            for item in ast.walk(node):
                if not isinstance(item, ast.Name) or item.id not in aliases:
                    continue
                par = parent.get(id(item))
                if par is None:
                    # The fragment itself is a Name (for example ``return sex``).
                    return False, "unsupported_bare_category_use"
                if isinstance(par, ast.Compare):
                    _value, related, unsupported = _category_alias_test_value(par, aliases, label)
                    if related and not unsupported:
                        continue
                    return False, "unsupported_category_comparison"
                call: ast.Call | None = None
                if isinstance(par, ast.Call):
                    call = par
                elif isinstance(par, ast.keyword):
                    grand = parent.get(id(par))
                    if isinstance(grand, ast.Call):
                        call = grand
                if call is not None:
                    if not isinstance(call.func, ast.Name) or call.func.id not in selected:
                        return False, "category_forwarded_to_unknown_call"
                    callee_uid = call.func.id
                    callee = formulas[callee_uid]
                    target_parameter: str | None = None
                    for pos, arg in enumerate(call.args):
                        if arg is item and pos < len(callee.parameters):
                            target_parameter = callee.parameters[pos]
                            break
                    if target_parameter is None:
                        for keyword in call.keywords:
                            if keyword.value is item and keyword.arg in callee.parameters:
                                target_parameter = keyword.arg
                                break
                    if target_parameter is None:
                        return False, "category_forwarded_to_unbound_parameter"
                    target = (callee_uid, target_parameter)
                    forwarded = True
                    if target not in parameter_domains:
                        if target in python_semantic_parameters:
                            # The semantic label remains a Python object at this
                            # boundary.  Category ABI completion ends here; the
                            # detached-helper planner proves executability later.
                            continue
                        return False, "category_forwarded_to_untyped_parameter"
                    accepted, reason = _prove_parameter_accepts_category_label(
                        formulas, selected=selected, uid=callee_uid,
                        parameter=target_parameter, label=label, domains=domains,
                        parameter_domains=parameter_domains, result_domains=result_domains,
                        exhaustive_dispatch=exhaustive_dispatch, cache=cache, active=active,
                        python_semantic_parameters=python_semantic_parameters,
                    )
                    if not accepted:
                        return False, f"downstream_rejects:{callee_uid}:{target_parameter}:{reason}"
                    continue
                if isinstance(par, ast.Return):
                    result_did = result_domains.get(uid)
                    if result_did is not None and label in domains[result_did].labels:
                        continue
                    return False, "category_return_domain_does_not_accept_label"
                if isinstance(par, ast.Assign) and par.value is item:
                    continue
                if isinstance(par, ast.AnnAssign) and par.value is item:
                    continue
                return False, f"unsupported_category_use:{type(par).__name__}"
            return True, "reachable_category_uses_safe"

        def walk_block(stmts: Sequence[ast.stmt], states: list[set[str]]) -> tuple[bool, str, list[set[str]]]:
            """Check only source statements reachable for the concrete label."""
            live = [set(state) for state in states]
            for stmt in stmts:
                if not live:
                    break
                next_live: list[set[str]] = []
                for aliases in live:
                    if isinstance(stmt, ast.If):
                        decision, related, unsupported = _category_alias_test_value(stmt.test, aliases, label)
                        if unsupported:
                            return False, "unsupported_category_guard", []
                        if not related:
                            ok, reason = check_fragment(stmt.test, aliases)
                            if not ok:
                                return False, reason, []
                        branches: list[Sequence[ast.stmt]]
                        if decision is True:
                            branches = [stmt.body]
                        elif decision is False:
                            branches = [stmt.orelse]
                        else:
                            branches = [stmt.body, stmt.orelse]
                        for branch in branches:
                            if not branch:
                                next_live.append(set(aliases))
                                continue
                            ok, reason, branch_live = walk_block(branch, [set(aliases)])
                            if not ok:
                                return False, reason, []
                            next_live.extend(branch_live)
                        continue
                    if isinstance(stmt, ast.Return):
                        # Preserve Return as the parent so a direct category return
                        # can be checked against the result ABI.
                        ok, reason = check_fragment(stmt, aliases)
                        if not ok:
                            return False, reason, []
                        continue
                    if isinstance(stmt, ast.Raise):
                        ok, reason = check_fragment(stmt.exc, aliases)
                        if not ok:
                            return False, reason, []
                        continue
                    if (
                        isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Name)
                        and stmt.value.func.id == "__guard_fail__"
                    ):
                        continue
                    if (
                        isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                    ):
                        ok, reason = check_fragment(stmt.value, aliases)
                        if not ok and not (
                            isinstance(stmt.value, ast.Name) and stmt.value.id in aliases
                        ):
                            return False, reason, []
                        target = stmt.targets[0].id
                        if isinstance(stmt.value, ast.Name) and stmt.value.id in aliases:
                            aliases.add(target)
                        else:
                            aliases.discard(target)
                        next_live.append(aliases)
                        continue
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        ok, reason = check_fragment(stmt.value, aliases)
                        if not ok and not (
                            isinstance(stmt.value, ast.Name) and stmt.value.id in aliases
                        ):
                            return False, reason, []
                        if isinstance(stmt.value, ast.Name) and stmt.value.id in aliases:
                            aliases.add(stmt.target.id)
                        else:
                            aliases.discard(stmt.target.id)
                        next_live.append(aliases)
                        continue
                    # Unsupported compound control that touches the tracked value
                    # is rejected.  If it does not touch it, it cannot create a
                    # category forwarding edge and is harmless to this proof.
                    if isinstance(stmt, (ast.For, ast.While, ast.Try, ast.With, ast.Match)):
                        if any(
                            isinstance(item, ast.Name) and item.id in aliases
                            for item in ast.walk(stmt)
                        ):
                            return False, f"unsupported_category_control:{type(stmt).__name__}", []
                        next_live.append(aliases)
                        continue
                    ok, reason = check_fragment(stmt, aliases)
                    if not ok:
                        return False, reason, []
                    next_live.append(aliases)
                live = next_live
            return True, "reachable_paths_safe", live

        ok, reason, _live = walk_block(fn.body, [{parameter}])
        if not ok:
            cache[key] = (False, reason)
            return cache[key]
        # Sequential dispatches may interleave numerical work and early returns.
        # They need no downstream category call: concrete branch evaluation above
        # proved every reachable path returns, and the path-sensitive use scan
        # excluded opaque operations on the category itself.
        if not forwarded and terminal_outcomes == {"return"}:
            cache[key] = (True, "selected_source_terminal_paths_accept")
            return cache[key]
        if not forwarded:
            cache[key] = (False, "no_source_proven_category_forwarding")
            return cache[key]
        cache[key] = (True, "selected_source_forwarding_accepts")
        return cache[key]
    finally:
        active.discard(key)


def _python_semantic_parameter_targets(
    formulas: Mapping[str, Any], class_info: Any,
) -> frozenset[tuple[str, str]]:
    """Return selected Cell parameters whose traced ABI keeps Python objects.

    A category label forwarded into such a parameter crosses *out* of the native
    category ABI: no integer category code is required at that callee boundary.
    The later detached-P/Python-key planner remains responsible for proving that
    the Python formula itself can execute without a live modelx Cell.  Keeping
    these concerns separate prevents category completion from inventing a native
    domain for ordinary Python-semantic helper parameters.
    """
    out: set[tuple[str, str]] = set()
    for uid, source in formulas.items():
        cell = class_info.cells.get(source.name)
        rt = None if cell is None else getattr(cell, "_rt", None)
        arg_types = {} if rt is None else dict(getattr(rt, "arg_types", {}) or {})
        for name in source.parameters:
            typ = arg_types.get(name)
            if typ is None:
                continue
            try:
                numeric = issubclass(typ, numbers.Real)
            except TypeError:
                numeric = False
            if not numeric:
                out.add((uid, name))
    return frozenset(out)


def _complete_selected_category_parameter_domains(
    formulas: Mapping[str, Any], *, output_uid: str | None,
    domains: dict[str, CategoryDomain],
    parameter_domains: dict[tuple[str, str], str],
    result_domains: Mapping[str, str],
    exhaustive_dispatch: Mapping[str, Mapping[str, tuple[Any, ...]]],
    register_extension: Any,
    python_semantic_parameters: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], bool]:
    """Demand-complete observed category parameter domains inside one closure."""
    selected = set(_selected_formula_closure(formulas, output_uid))
    if not selected:
        return {}, True
    proofs: dict[tuple[str, str], Mapping[str, Any]] = {}
    converged = False
    for iteration in range(1, SELECTED_CATEGORY_COMPLETION_MAX_ITERATIONS + 1):
        demands = _selected_category_forward_demands(
            formulas, output_uid=output_uid, domains=domains,
            parameter_domains=parameter_domains, result_domains=result_domains,
        )
        changed = False
        acceptance_cache: dict[tuple[str, str, Any], tuple[bool, str]] = {}
        for target in sorted(demands):
            prior_did = parameter_domains.get(target)
            if prior_did is None:
                continue
            prior_domain = domains[prior_did]
            demanded = tuple(
                label for label in _ordered(tuple(demands[target]))
                if label not in prior_domain.labels
            )
            if not demanded:
                continue
            accepted: list[Any] = []
            rejected: list[dict[str, Any]] = []
            for label in demanded:
                ok, reason = _prove_parameter_accepts_category_label(
                    formulas, selected=selected, uid=target[0], parameter=target[1],
                    label=label, domains=domains, parameter_domains=parameter_domains,
                    result_domains=result_domains, exhaustive_dispatch=exhaustive_dispatch,
                    cache=acceptance_cache, active=set(),
                    python_semantic_parameters=python_semantic_parameters,
                )
                if ok:
                    accepted.append(label)
                else:
                    rejected.append({"label": label, "reason": reason})
            if len(prior_domain.labels) + len(accepted) > SELECTED_CATEGORY_COMPLETION_MAX_LABELS:
                rejected.extend(
                    {"label": label, "reason": "category_completion_label_bound"}
                    for label in accepted
                )
                accepted = []
            source_rows = sorted({
                f"{caller}:{parameter}"
                for label in demanded
                for caller, parameter in demands[target].get(label, set())
            })
            proof = {
                "schema": "modelx_graph.selected_category_parameter_completion.v1",
                "selected_output": output_uid,
                "observed_domain_id": prior_did,
                "observed_labels": list(prior_domain.labels),
                "demanded_labels": list(demanded),
                "accepted_labels": list(accepted),
                "rejected": rejected,
                "source_calls": source_rows,
                "iteration": iteration,
                "global_fixed_point": False,
                "execution_graph_authority": False,
                "existing_codes_preserved": True,
            }
            proofs[target] = proof
            if accepted:
                new_did = register_extension(
                    prior_did, accepted,
                    target=f"parameter:{target[0]}:{target[1]}",
                    detail=(
                        f"selected_output={output_uid};source_calls={','.join(source_rows)}"
                    ),
                )
                parameter_domains[target] = new_did
                changed = changed or new_did != prior_did
        if not changed:
            converged = True
            break
    return proofs, converged


def _propagate_selected_nullable_parameters(
    formulas: Mapping[str, Any],
    *,
    output_uid: str | None,
    nullable_results: set[str],
    nullable_parameters: set[tuple[str, str]],
    parameter_domains: Mapping[tuple[str, str], str],
    sequence_result_domains: Mapping[str, str],
    nullable_sequence_results: set[str],
) -> set[tuple[str, str]]:
    """Carry nullable state through selected dynamic category call arguments.

    Labels/domains remain owned by ``CategoryEncodingPlan``.  This pass tracks only
    physical validity state and is intentionally selected-closure/local: no global
    category graph or execution schedule is constructed.
    """

    selected = set(_selected_formula_closure(formulas, output_uid))
    out = set(nullable_parameters)

    class Flow(ast.NodeVisitor):
        def __init__(self, caller_uid: str):
            self.caller_uid = caller_uid
            self.locals: set[str] = set()

        def nullable(self, node: ast.AST) -> bool:
            if isinstance(node, ast.Name):
                return node.id in self.locals or (self.caller_uid, node.id) in out
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                return node.func.id in nullable_results
            if isinstance(node, ast.IfExp):
                return self.nullable(node.body) or self.nullable(node.orelse)
            return False

        def visit_Assign(self, node: ast.Assign):  # noqa: N802
            nullable = self.nullable(node.value)
            self.visit(node.value)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if nullable:
                    self.locals.add(node.targets[0].id)
                else:
                    self.locals.discard(node.targets[0].id)

        def visit_AnnAssign(self, node: ast.AnnAssign):  # noqa: N802
            if node.value is None or not isinstance(node.target, ast.Name):
                return self.generic_visit(node)
            nullable = self.nullable(node.value)
            self.visit(node.value)
            if nullable:
                self.locals.add(node.target.id)
            else:
                self.locals.discard(node.target.id)

        def visit_GeneratorExp(self, node: ast.GeneratorExp):  # noqa: N802
            if len(node.generators) != 1 or not isinstance(node.generators[0].target, ast.Name):
                return self.generic_visit(node)
            comp = node.generators[0]
            target = comp.target.id
            prior = target in self.locals
            sequence_nullable = (
                isinstance(comp.iter, ast.Call) and isinstance(comp.iter.func, ast.Name)
                and comp.iter.func.id in sequence_result_domains
                and comp.iter.func.id in nullable_sequence_results
            )
            self.visit(comp.iter)
            if sequence_nullable:
                self.locals.add(target)
            else:
                self.locals.discard(target)
            for cond in comp.ifs:
                self.visit(cond)
            self.visit(node.elt)
            if prior:
                self.locals.add(target)
            else:
                self.locals.discard(target)

        def visit_Call(self, node: ast.Call):  # noqa: N802
            if isinstance(node.func, ast.Name) and node.func.id in formulas:
                callee = formulas[node.func.id]
                for pos, arg in enumerate(node.args):
                    if pos >= len(callee.parameters):
                        continue
                    name = callee.parameters[pos]
                    if (node.func.id, name) not in parameter_domains:
                        continue
                    if self.nullable(arg):
                        out.add((node.func.id, name))
            self.generic_visit(node)

    # A nullable parameter discovered in one callee may make a deeper call nullable.
    # Bound iterations by selected Cell count and converge on the finite pair set.
    for _ in range(max(1, len(selected) + 1)):
        before = set(out)
        for uid in sorted(selected):
            Flow(uid).visit(formulas[uid].function)
        if out == before:
            break
    return out


def _prove_selected_category_results(
    formulas: Mapping[str, Any], *, output_uid: str | None,
    domains: Mapping[str, CategoryDomain], result_domains: Mapping[str, str],
    parameter_domains: Mapping[tuple[str, str], str],
    nullable_results: frozenset[str] | set[str],
    strong_result_uids: frozenset[str] | set[str],
) -> dict[str, tuple[_CategoryValueSet, Mapping[str, Any]]]:
    """Prove exact finite result domains inside the selected lexical closure only.

    The proof follows local assignments, already-proven category-returning Cell calls,
    and simple equality/membership branch constraints. Cycles without an authoritative
    anchor fail closed. This is an admission/evidence service, not an execution graph.
    """
    selected = set(_selected_formula_closure(formulas, output_uid))
    if not selected:
        return {}
    strong = set(strong_result_uids)
    nullable = set(nullable_results)
    cache: dict[str, _CategoryValueSet | None] = {}
    proofs: dict[str, Mapping[str, Any]] = {}
    resolving: set[str] = set()

    def domain_value(uid: str) -> _CategoryValueSet | None:
        did = result_domains.get(uid)
        if did is None:
            return None
        domain = domains[did]
        return _CategoryValueSet(tuple(domain.labels), uid in nullable)

    def parameter_env(uid: str) -> dict[str, _CategoryValueSet]:
        out: dict[str, _CategoryValueSet] = {}
        for name in formulas[uid].parameters:
            did = parameter_domains.get((uid, name))
            if did is not None:
                out[name] = _CategoryValueSet(tuple(domains[did].labels), False)
        return out

    def narrow(
        test: ast.AST, env: Mapping[str, _CategoryValueSet]
    ) -> tuple[dict[str, _CategoryValueSet], dict[str, _CategoryValueSet], bool]:
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            yes, no, ok = narrow(test.operand, env)
            return no, yes, ok
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1):
            return dict(env), dict(env), False
        left, right, op = test.left, test.comparators[0], test.ops[0]
        name = None
        literals: tuple[Any, ...] | None = None
        if isinstance(left, ast.Name):
            name = left.id
            if isinstance(right, ast.Constant) and _literal(right.value):
                literals = (right.value,)
            elif isinstance(right, (ast.Tuple, ast.List, ast.Set)):
                vals = [elt.value for elt in right.elts if isinstance(elt, ast.Constant) and _literal(elt.value)]
                if len(vals) == len(right.elts):
                    literals = tuple(vals)
        elif isinstance(right, ast.Name) and isinstance(left, ast.Constant) and _literal(left.value):
            name = right.id
            literals = (left.value,)
        if name is None or literals is None:
            return dict(env), dict(env), False
        allowed = set(literals)
        truth_is_inside = isinstance(op, (ast.Eq, ast.Is, ast.In))
        false_is_inside = isinstance(op, (ast.NotEq, ast.IsNot, ast.NotIn))
        if not (truth_is_inside or false_is_inside):
            return dict(env), dict(env), False
        yes, no = dict(env), dict(env)
        values = env.get(name)
        if values is None:
            # A guard may itself establish a finite category domain for a local
            # whose producer is opaque to this lexical proof.  Only the branch
            # constrained *inside* explicit string/None literals becomes typed;
            # the complementary branch remains unknown.  If that unknown branch
            # terminates (typically by raise), the surviving path can later prove
            # an exact category result.
            finite = _CategoryValueSet(tuple(_ordered(literals)), False)
            if truth_is_inside:
                yes[name] = finite
            else:
                no[name] = finite
            return yes, no, True
        inside = tuple(label for label in values.labels if label in allowed)
        outside = tuple(label for label in values.labels if label not in allowed)
        yes_labels, no_labels = (inside, outside) if truth_is_inside else (outside, inside)
        if yes_labels:
            yes[name] = _CategoryValueSet(yes_labels, values.nullable and not truth_is_inside)
        else:
            yes.pop(name, None)
        if no_labels:
            no[name] = _CategoryValueSet(no_labels, values.nullable and truth_is_inside)
        else:
            no.pop(name, None)
        return yes, no, True

    def expr(expr_node: ast.AST | None, env: Mapping[str, _CategoryValueSet]) -> _CategoryValueSet | None:
        if expr_node is None:
            return None
        if isinstance(expr_node, ast.Constant) and _literal(expr_node.value):
            return _CategoryValueSet((expr_node.value,), False)
        if isinstance(expr_node, ast.Name):
            return env.get(expr_node.id)
        if isinstance(expr_node, ast.Call) and isinstance(expr_node.func, ast.Name) and expr_node.func.id in selected:
            return resolve(expr_node.func.id)
        if isinstance(expr_node, ast.IfExp):
            yes, no, ok = narrow(expr_node.test, env)
            if not ok:
                return None
            left = expr(expr_node.body, yes)
            right = expr(expr_node.orelse, no)
            if left is None or right is None:
                return None
            return _CategoryValueSet.merge((left, right))
        return None

    def merge_env(a: Mapping[str, _CategoryValueSet], b: Mapping[str, _CategoryValueSet]) -> dict[str, _CategoryValueSet]:
        out: dict[str, _CategoryValueSet] = {}
        for name in set(a) & set(b):
            merged = _CategoryValueSet.merge((a[name], b[name]))
            if merged is not None:
                out[name] = merged
        return out

    def block(stmts: Sequence[ast.stmt], incoming: Mapping[str, _CategoryValueSet]):
        env = dict(incoming)
        returns: list[_CategoryValueSet] = []
        live = True
        for st in stmts:
            if not live:
                break
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                continue
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                value = expr(st.value, env)
                if value is None:
                    env.pop(st.targets[0].id, None)
                else:
                    env[st.targets[0].id] = value
                continue
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
                value = expr(st.value, env)
                if value is None:
                    env.pop(st.target.id, None)
                else:
                    env[st.target.id] = value
                continue
            if isinstance(st, ast.If):
                yes_env, no_env, ok = narrow(st.test, env)
                if not ok:
                    # Unrelated numeric/boolean control does not invalidate a
                    # category-result proof. Explore both branches from the same
                    # incoming category environment and merge the surviving
                    # category facts. This is conservative: it may retain labels
                    # from a runtime-unreachable branch, but it never invents a
                    # label absent from source-proven category expressions.
                    yes_env, no_env = dict(env), dict(env)
                yes = block(st.body, yes_env)
                no = block(st.orelse, no_env) if st.orelse else ([], no_env, True)
                if yes is None or no is None:
                    return None
                yret, yenv, ylive = yes
                nret, nenv, nlive = no
                returns.extend(yret); returns.extend(nret)
                if ylive and nlive:
                    env = merge_env(yenv, nenv)
                    live = True
                elif ylive:
                    env, live = yenv, True
                elif nlive:
                    env, live = nenv, True
                else:
                    live = False
                continue
            if isinstance(st, ast.Return):
                value = expr(st.value, env)
                if value is None:
                    return None
                returns.append(value)
                live = False
                continue
            if isinstance(st, ast.Raise):
                live = False
                continue
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Name) and st.value.func.id == "__guard_fail__":
                live = False
                continue
            # Loops, dynamic mutation and other local control are intentionally
            # deferred. A proof that cannot stay local/exact simply declines.
            return None
        return returns, env, live

    def resolve(uid: str) -> _CategoryValueSet | None:
        if uid in cache:
            return cache[uid]
        existing = domain_value(uid)
        if uid in strong:
            cache[uid] = existing
            return existing
        if uid in resolving:
            return existing if uid in strong else None
        resolving.add(uid)
        try:
            result = block(formulas[uid].function.body, parameter_env(uid))
            if result is None:
                cache[uid] = existing
                return existing
            returned, _env, live = result
            if live or not returned:
                cache[uid] = existing
                return existing
            inferred = _CategoryValueSet.merge(returned)
            if inferred is None:
                cache[uid] = existing
                return existing
            if existing is not None and any(label not in inferred.labels for label in existing.labels):
                cache[uid] = existing
                return existing
            cache[uid] = inferred
            proofs[uid] = {
                "schema": "modelx_graph.selected_category_result_proof.v1",
                "source": "selected_local_category_result_proof",
                "labels": list(inferred.labels),
                "nullable": inferred.nullable,
                "global_fixed_point": False,
                "execution_graph_authority": False,
            }
            return inferred
        finally:
            resolving.discard(uid)

    for uid in sorted(selected):
        resolve(uid)
    return {uid: (value, proofs[uid]) for uid, value in cache.items() if uid in proofs and value is not None}


def build_modelx_cython_category_plan(
    formulas: Mapping[str, Any],
    *,
    class_info: Any,
    evidence: Any,
    point_uid: str | None,
    output_uid: str | None = None,
    structured_point_categories: Mapping[str, tuple[str, tuple[Any, ...], bool]] | None = None,
) -> ModelxCythonCategoryPlan:
    """Build category ABI from upstream/provider facts plus selected local proofs."""

    domains: dict[str, CategoryDomain] = {}
    result_domains: dict[str, str] = {}
    sequence_result_domains: dict[str, str] = {}
    parameter_domains: dict[tuple[str, str], str] = {}
    evidence_rows: list[FastCategoryEvidence] = []
    point_fields: dict[str, str] = {}
    parameterized_point_bindings: dict[str, ParameterizedPointCategoryBinding] = {}
    nullable_results: set[str] = set()
    nullable_sequence_results: set[str] = set()
    nullable_parameters: set[tuple[str, str]] = set()
    blockers: list[str] = []
    result_sources: dict[str, str] = {}
    strong_result_uids: set[str] = set()
    result_domain_proofs: dict[str, Mapping[str, Any]] = {}
    parameter_domain_completion_proofs: dict[tuple[str, str], Mapping[str, Any]] = {}

    def register(
        labels: Sequence[Any], *, target: str, source: str, detail: str = "",
        nullable: bool = False,
    ) -> str:
        domain = CategoryDomain.from_labels(tuple(labels), source, target, detail)
        prior = domains.get(domain.domain_id)
        if prior is not None and prior.labels != domain.labels:
            blockers.append(f"domain_hash_collision:{domain.domain_id}")
        else:
            domains.setdefault(domain.domain_id, domain)
        evidence_rows.append(FastCategoryEvidence(
            target, domain.domain_id, domain.labels, source, detail, nullable
        ))
        return domain.domain_id

    def register_extension(
        prior_domain_id: str, labels: Sequence[Any], *, target: str, detail: str = "",
    ) -> str:
        prior = domains[prior_domain_id]
        domain = prior.extended(
            tuple(labels), "selected_closure_category_demand_completion", target, detail
        )
        existing = domains.get(domain.domain_id)
        if existing is not None and existing.labels != domain.labels:
            blockers.append(f"domain_hash_collision:{domain.domain_id}")
        else:
            domains.setdefault(domain.domain_id, domain)
        evidence_rows.append(FastCategoryEvidence(
            target=target,
            domain_id=domain.domain_id,
            labels=domain.labels,
            source="selected_closure_category_demand_completion",
            detail=detail,
            nullable=False,
        ))
        return domain.domain_id

    direct_integer_args = _direct_call_integer_args(formulas)
    if point_uid is not None and evidence.model_point_row is not None:
        for uid, source in formulas.items():
            cell = class_info.cells[source.name]
            try:
                binding = _build_parameterized_point_binding(
                    uid, source, cell=cell, class_info=class_info,
                    formulas=formulas, evidence=evidence, point_uid=point_uid,
                    direct_integer_args=direct_integer_args,
                )
            except ModelxCythonCategoryABIError as exc:
                blockers.append(f"parameterized_point_provider:{uid}:{exc}")
                binding = None
            if binding is not None:
                parameterized_point_bindings[uid] = binding
                if binding.nullable:
                    nullable_results.add(uid)

    direct_args = _direct_call_literal_args(formulas)
    bound_table_args = _bound_table_parameter_labels(formulas, evidence)
    comparisons = {uid: _comparison_parameter_literals(source) for uid, source in formulas.items()}
    exhaustive_dispatch = {
        uid: {
            name: labels
            for name in source.parameters
            if (labels := _exhaustive_parameter_dispatch_literals(source, name))
        }
        for uid, source in formulas.items()
    }

    for uid, source in formulas.items():
        cell = class_info.cells[source.name]
        spec = getattr(cell, "_spec", {}) or {}
        rt = getattr(cell, "_rt", None)

        result_labels = _spec_values(
            spec.get(getattr(TransSpec, "RET_VALUES", "return_values")),
            context=f"{cell.fqname}.return_values",
        )
        result_source = "TransSpec.return_values" if result_labels else ""

        field = _point_field_source(source.function, point_uid) if point_uid is not None and not source.parameters else None
        parameterized = parameterized_point_bindings.get(uid)
        if parameterized is not None and result_labels:
            outside = tuple(label for label in parameterized.labels if label not in set(result_labels))
            if outside:
                blockers.append(
                    f"parameterized_point_provider:{uid}:reachable provider labels {outside!r} "
                    f"outside explicit return_values {result_labels!r}"
                )
        if not result_labels and parameterized is not None and rt is not None:
            result_labels = parameterized.labels
            result_source = (
                "provider_guard_reachable_full_run_domain_nullable"
                if parameterized.nullable else "provider_guard_reachable_full_run_domain"
            )

        structured = (structured_point_categories or {}).get(uid)
        if not result_labels and structured is not None and rt is not None:
            structured_field, structured_labels, structured_nullable = structured
            if structured_labels and all(isinstance(x, str) for x in structured_labels):
                result_labels = tuple(structured_labels)
                result_source = (
                    "structured_provider_full_ranked_domain_nullable"
                    if structured_nullable else "structured_provider_full_ranked_domain"
                )
                point_fields[uid] = structured_field
                if structured_nullable:
                    nullable_results.add(uid)

        if not result_labels and field is not None and rt is not None:
            provider_labels, provider_nullable = _point_values(evidence, field)
            if provider_labels and all(isinstance(x, str) for x in provider_labels):
                result_labels = provider_labels
                result_source = (
                    "provider_full_run_domain_nullable"
                    if provider_nullable else "provider_full_run_domain"
                )
                point_fields[uid] = field
                if provider_nullable:
                    nullable_results.add(uid)

        if not result_labels and rt is not None:
            traced = tuple(getattr(rt, "literal_returns", ()) or ())
            if traced:
                result_labels = _ordered(traced)
                result_source = "MxCallTraceLogger.literal_returns"

        if not result_labels:
            lexical = _literal_return_values(source.function)
            if lexical:
                result_labels = lexical
                result_source = "local_literal_returns"

        if result_labels:
            result_domains[uid] = register(
                result_labels, target=f"result:{uid}", source=result_source,
                detail=(
                    f"point_field={field}"
                    if field is not None and (
                        result_source.startswith("provider_full_run_domain")
                        or result_source.startswith("structured_provider_full_ranked_domain")
                    )
                    else ""
                ),
                nullable=uid in nullable_results,
            )
            result_sources[uid] = result_source
            if (
                result_source == "TransSpec.return_values"
                or result_source.startswith("provider_")
                or result_source == "local_literal_returns"
            ):
                strong_result_uids.add(uid)

        sequence_labels = _literal_sequence_return_values(source.function)
        if sequence_labels:
            sequence_result_domains[uid] = register(
                sequence_labels,
                target=f"sequence_result:{uid}",
                source="local_literal_sequence_returns",
            )

        param_spec = spec.get(getattr(TransSpec, "PARAM_VALUES", "param_values"), {}) or {}
        if param_spec and not isinstance(param_spec, Mapping):
            raise ModelxCythonCategoryABIError(f"{cell.fqname}.param_values must be a mapping")
        for pos, name in enumerate(source.parameters):
            labels = (
                _spec_values(param_spec.get(name), context=f"{cell.fqname}.param_values[{name!r}]")
                if name in param_spec else ()
            )
            source_name = "TransSpec.param_values" if labels else ""
            if not labels and (uid, name) in bound_table_args:
                labels = bound_table_args[(uid, name)]
                source_name = "bound_table_index_full_run_domain"
            local_direct = direct_args.get((uid, pos), ())
            optional_dispatch = _optional_static_default_dispatch_literals(
                source, name, evidence
            )
            if not labels and optional_dispatch:
                labels = optional_dispatch
                source_name = "local_optional_static_default_terminal_dispatch"
            if not labels and rt is not None:
                traced = tuple((getattr(rt, "literal_args", {}) or {}).get(name, ()) or ())
                if traced:
                    labels = _ordered(tuple(traced) + tuple(local_direct))
                    source_name = (
                        "MxCallTraceLogger.literal_args+local_direct_call_literals"
                        if local_direct else "MxCallTraceLogger.literal_args"
                    )
            if not labels and local_direct:
                labels = local_direct
                source_name = "local_direct_call_literals"
            if not labels:
                local = comparisons.get(uid, {}).get(name, ())
                if local:
                    labels = local
                    source_name = "local_parameter_comparisons"
            if labels:
                parameter_domains[(uid, name)] = register(
                    labels, target=f"parameter:{uid}:{name}", source=source_name
                )

    # Block 2B-2: widen only the category parameter positions that are actually
    # fed by a proven finite category-sequence generator.  The callee's terminal
    # dispatch is a compatibility check, not a global source of parameter values.
    # This distinction preserves established kernels such as ULSG claim_pp(kind),
    # whose GRACE branch exists but is never passed to claim_pp by any caller.
    for caller_uid, caller in formulas.items():
        for gen in (
            node for node in ast.walk(caller.function)
            if isinstance(node, ast.GeneratorExp) and len(node.generators) == 1
        ):
            comp = gen.generators[0]
            if not (
                isinstance(comp.target, ast.Name)
                and isinstance(comp.iter, ast.Call)
                and isinstance(comp.iter.func, ast.Name)
                and comp.iter.func.id in sequence_result_domains
            ):
                continue
            source_did = sequence_result_domains[comp.iter.func.id]
            source_labels = tuple(domains[source_did].labels)
            target_name = comp.target.id
            for call in (
                node for node in ast.walk(gen.elt)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in formulas
            ):
                callee_uid = call.func.id
                callee = formulas[callee_uid]
                for pos, arg in enumerate(call.args):
                    if not (isinstance(arg, ast.Name) and arg.id == target_name):
                        continue
                    if pos >= len(callee.parameters):
                        continue
                    parameter = callee.parameters[pos]
                    exact = exhaustive_dispatch.get(callee_uid, {}).get(parameter, ())
                    if exact and any(label not in exact for label in source_labels):
                        blockers.append(
                            f"dynamic_category_argument:{caller_uid}->{callee_uid}:{parameter}:"
                            f"sequence labels {source_labels!r} outside terminal dispatch {exact!r}"
                        )
                        continue
                    prior_did = parameter_domains.get((callee_uid, parameter))
                    prior_labels = () if prior_did is None else tuple(domains[prior_did].labels)
                    combined = _ordered((*prior_labels, *source_labels))
                    if exact and any(label not in exact for label in combined):
                        blockers.append(
                            f"dynamic_category_argument:{caller_uid}->{callee_uid}:{parameter}:"
                            f"combined labels {combined!r} outside terminal dispatch {exact!r}"
                        )
                        continue
                    if not exact and prior_labels and any(label not in prior_labels for label in source_labels):
                        blockers.append(
                            f"dynamic_category_argument:{caller_uid}->{callee_uid}:{parameter}:"
                            f"sequence labels {source_labels!r} exceed proven parameter domain {prior_labels!r}"
                        )
                        continue
                    if combined and combined != prior_labels:
                        parameter_domains[(callee_uid, parameter)] = register(
                            combined,
                            target=f"parameter:{callee_uid}:{parameter}",
                            source="dynamic_sequence_argument",
                            detail=(
                                f"caller={caller_uid};iterator={comp.iter.func.id};"
                                f"terminal_dispatch={'yes' if exact else 'no'}"
                            ),
                        )

    # Stage 3B: representative modelx-cython observations are evidence, not the
    # sole category-domain authority.  Complete only parameter positions that are
    # dynamically demanded by the selected source closure, and only when the
    # Establish any exact category result that can already be proved from local
    # selected-source control flow before parameter-demand completion.  This lets
    # a real callsite such as ``callee(category_provider(...))`` contribute its
    # complete result domain to the callee parameter demand.  A second proof pass
    # below still runs after parameter completion and can discover additional
    # results enabled by newly completed parameter domains.
    precompletion_results = _prove_selected_category_results(
        formulas, output_uid=output_uid, domains=domains,
        result_domains=result_domains, parameter_domains=parameter_domains,
        nullable_results=nullable_results, strong_result_uids=strong_result_uids,
    )
    for uid, (value, proof) in precompletion_results.items():
        if uid in result_domains:
            continue
        if value.nullable:
            nullable_results.add(uid)
        result_domains[uid] = register(
            value.labels,
            target=f"result:{uid}",
            source="selected_local_category_result_proof",
            detail=f"selected_output={output_uid};phase=pre_parameter_completion",
            nullable=value.nullable,
        )
        result_sources[uid] = "selected_local_category_result_proof"
        result_domain_proofs[uid] = proof

    # callee's source semantics prove that each additional label is valid.  Domain
    # extension preserves all existing physical codes and appends new labels.
    python_semantic_parameters = _python_semantic_parameter_targets(formulas, class_info)
    parameter_domain_completion_proofs, completion_converged = (
        _complete_selected_category_parameter_domains(
            formulas,
            output_uid=output_uid,
            domains=domains,
            parameter_domains=parameter_domains,
            result_domains=result_domains,
            exhaustive_dispatch=exhaustive_dispatch,
            register_extension=register_extension,
            python_semantic_parameters=python_semantic_parameters,
        )
    )
    if not completion_converged:
        blockers.append(
            "selected_category_parameter_completion:bounded convergence not reached"
        )

    # FC1 Block 2B-1: fill sampled-result holes only inside the selected semantic
    # closure. Provider/spec/local-literal domains are immutable anchors; the proof
    # follows category-valued callees, locals and simple branch constraints.
    propagated = _prove_selected_category_results(
        formulas, output_uid=output_uid,
        domains=domains, result_domains=result_domains,
        parameter_domains=parameter_domains,
        nullable_results=nullable_results,
        strong_result_uids=strong_result_uids,
    )
    for uid, (value, proof) in propagated.items():
        existing_id = result_domains.get(uid)
        existing_labels = () if existing_id is None else tuple(domains[existing_id].labels)
        if existing_labels and any(label not in value.labels for label in existing_labels):
            blockers.append(
                f"selected_category_result_proof:{uid}:observed labels {existing_labels!r} "
                f"outside inferred exact domain {value.labels!r}"
            )
            continue
        if value.nullable:
            nullable_results.add(uid)
        if existing_labels != value.labels or existing_id is None:
            result_domains[uid] = register(
                value.labels,
                target=f"result:{uid}",
                source="selected_local_category_result_proof",
                detail=f"selected_output={output_uid}",
                nullable=value.nullable,
            )
            result_sources[uid] = "selected_local_category_result_proof"
        result_domain_proofs[uid] = proof

    nullable_parameters = _propagate_selected_nullable_parameters(
        formulas,
        output_uid=output_uid,
        nullable_results=nullable_results,
        nullable_parameters=nullable_parameters,
        parameter_domains=parameter_domains,
        sequence_result_domains=sequence_result_domains,
        nullable_sequence_results=nullable_sequence_results,
    )

    plan = CategoryEncodingPlan(
        schema="modelx_graph.modelx_cython_category_encoding.v2",
        domains=domains,
        result_domains=result_domains,
        parameter_domains=parameter_domains,
        sequence_result_domains=sequence_result_domains,
        nullable_results=frozenset(nullable_results),
        nullable_parameters=frozenset(nullable_parameters),
        nullable_sequence_results=frozenset(nullable_sequence_results),
        blockers=tuple(sorted(set(blockers))),
    )
    plan = attach_parameter_positions(
        plan, {uid: source.parameters for uid, source in formulas.items()}
    )
    return ModelxCythonCategoryPlan(
        encoding=plan,
        evidence=tuple(evidence_rows),
        point_fields=point_fields,
        parameterized_point_bindings=parameterized_point_bindings,
        result_domain_proofs=result_domain_proofs,
        parameter_domain_completion_proofs=parameter_domain_completion_proofs,
    )


class BindPointCategorySubscript(ast.NodeTransformer):
    def __init__(self, point_uid: str, field: str, input_key: str):
        self.point_uid = point_uid
        self.field = field
        self.input_key = input_key

    def visit_Subscript(self, node: ast.Subscript):  # noqa: N802
        node = self.generic_visit(node)
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == self.point_uid
            and not node.value.args and not node.value.keywords
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == self.field
        ):
            return ast.copy_location(
                ast.Call(ast.Name("point_input", ast.Load()), [ast.Constant(self.input_key)], []), node
            )
        return node


__all__ = [
    "ModelxCythonCategoryABIError",
    "FastCategoryEvidence",
    "ParameterizedPointCategoryCase",
    "ParameterizedPointCategoryBinding",
    "ModelxCythonCategoryPlan",
    "BindPointCategorySubscript",
    "build_modelx_cython_category_plan",
    "prove_consumer_category_result_domains",
    "CategoryABIError",
]
