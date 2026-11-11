from __future__ import annotations

"""Direct physical lowering from canonical Stage execution/storage evidence.

This module is intentionally independent of ExecutableGraph/OptimizedProgram and
legacy coord_phase.  It freezes the physical facts needed by the direct Stage
Python/Cython backend: iteration geometry, dependency-derived scan blocks,
recurrence boundary seeds, full-stage fence admission, and storage-free
PureMap ABI.  Stage placement and barrier discovery remain upstream semantic facts.
"""

import ast
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical_semantic_graph import CanonicalSemanticGraph
from .stage_execution_plan import StageExecutionPlan
from .stage_storage_plan import StageStoragePlan


class StagePhysicalLoweringError(RuntimeError):
    pass


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "backslashreplace"))
        h.update(b"\0")
    return f"{prefix}_{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class StageIterationDomain:
    uid: str
    stage_index: int
    parameter_name: str
    range_arg_sources: tuple[str, ...]
    range_arg_asts: tuple[str, ...]
    coordinate_step: int
    scan_direction: str
    materialization_extension_min: int = 0
    materialization_extension_max: int = 0
    evidence_uids: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "stage_index": self.stage_index,
            "parameter_name": self.parameter_name,
            "range_arg_sources": list(self.range_arg_sources),
            "range_arg_asts": list(self.range_arg_asts),
            "coordinate_step": self.coordinate_step,
            "scan_direction": self.scan_direction,
            "materialization_extension_min": int(self.materialization_extension_min),
            "materialization_extension_max": int(self.materialization_extension_max),
            "evidence_uids": list(self.evidence_uids),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class StageBoundarySeed:
    uid: str
    stage_index: int
    value_uid: str
    coordinate: int
    source_access_uids: tuple[str, ...]
    proof_kind: str
    active_boundary_coordinates: tuple[int, ...] = ()
    replay_start_coordinate: int | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "stage_index": self.stage_index,
            "value_uid": self.value_uid,
            "coordinate": self.coordinate,
            "source_access_uids": list(self.source_access_uids),
            "proof_kind": self.proof_kind,
            "active_boundary_coordinates": list(self.active_boundary_coordinates),
            "replay_start_coordinate": self.replay_start_coordinate,
        }


@dataclass(frozen=True)
class StagePureMapABI:
    uid: str
    coordinate_parameter: str
    scalar_dependencies: tuple[str, ...]
    pure_map_dependencies: tuple[str, ...]
    proof_scope: str
    provenance: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "coordinate_parameter": self.coordinate_parameter,
            "scalar_dependencies": list(self.scalar_dependencies),
            "pure_map_dependencies": list(self.pure_map_dependencies),
            "proof_scope": self.proof_scope,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class StageAuxiliaryRecurrenceABI:
    uid: str
    scan_direction: str
    ring_depth: int
    offset_min: int
    offset_max: int

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "scan_direction": self.scan_direction,
            "ring_depth": self.ring_depth,
            "offset_min": self.offset_min,
            "offset_max": self.offset_max,
        }


@dataclass(frozen=True)
class StageExecutionBlock:
    uid: str
    stage_index: int
    kind: str
    iteration_domain_uid: str | None
    persistent_driver_uids: tuple[str, ...]
    pre_scalar_uids: tuple[str, ...]
    reduction_uids: tuple[str, ...]
    post_scalar_uids: tuple[str, ...]
    scan_direction: str
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "stage_index": self.stage_index,
            "kind": self.kind,
            "iteration_domain_uid": self.iteration_domain_uid,
            "persistent_driver_uids": list(self.persistent_driver_uids),
            "pre_scalar_uids": list(self.pre_scalar_uids),
            "reduction_uids": list(self.reduction_uids),
            "post_scalar_uids": list(self.post_scalar_uids),
            "scan_direction": self.scan_direction,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class StageDirectContract:
    iteration_domains: tuple[StageIterationDomain, ...]
    execution_blocks: tuple[StageExecutionBlock, ...]
    boundary_seeds: tuple[StageBoundarySeed, ...]
    pure_maps: tuple[StagePureMapABI, ...]
    auxiliary_recurrences: tuple[StageAuxiliaryRecurrenceABI, ...]
    barrier_uids: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.blockers and bool(self.iteration_domains) and bool(self.execution_blocks)

    def manifest(self) -> dict[str, Any]:
        return {
            "iteration_domains": [x.manifest() for x in self.iteration_domains],
            "execution_blocks": [x.manifest() for x in self.execution_blocks],
            "boundary_seeds": [x.manifest() for x in self.boundary_seeds],
            "pure_maps": [x.manifest() for x in self.pure_maps],
            "auxiliary_recurrences": [x.manifest() for x in self.auxiliary_recurrences],
            "barrier_uids": list(self.barrier_uids),
            "blockers": list(self.blockers),
            "supported": self.supported,
        }


def _fixed_coordinate_expr_value(
    node: ast.AST, variable: str, coordinate: int,
    bindings: Mapping[str, Any] | None = None,
) -> Any:
    bindings = bindings or {}
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == variable:
            return int(coordinate)
        if node.id in bindings:
            return bindings[node.id]
        raise ValueError(node.id)
    if isinstance(node, ast.UnaryOp):
        v = _fixed_coordinate_expr_value(node.operand, variable, coordinate, bindings)
        if isinstance(node.op, ast.Not): return not bool(v)
        if isinstance(node.op, ast.USub): return -v
        if isinstance(node.op, ast.UAdd): return +v
        raise ValueError(type(node.op).__name__)
    if isinstance(node, ast.BinOp):
        a = _fixed_coordinate_expr_value(node.left, variable, coordinate, bindings)
        b = _fixed_coordinate_expr_value(node.right, variable, coordinate, bindings)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
        if isinstance(node.op, ast.FloorDiv): return a // b
        if isinstance(node.op, ast.Mod): return a % b
        if isinstance(node.op, ast.Pow): return a ** b
        raise ValueError(type(node.op).__name__)
    if isinstance(node, ast.BoolOp):
        vals = [_fixed_coordinate_expr_value(x, variable, coordinate, bindings) for x in node.values]
        if isinstance(node.op, ast.And): return all(bool(x) for x in vals)
        if isinstance(node.op, ast.Or): return any(bool(x) for x in vals)
        raise ValueError(type(node.op).__name__)
    if isinstance(node, ast.Compare):
        left = _fixed_coordinate_expr_value(node.left, variable, coordinate, bindings)
        for op, rhs in zip(node.ops, node.comparators):
            right = _fixed_coordinate_expr_value(rhs, variable, coordinate, bindings)
            if isinstance(op, ast.Eq): ok = left == right
            elif isinstance(op, ast.NotEq): ok = left != right
            elif isinstance(op, ast.Lt): ok = left < right
            elif isinstance(op, ast.LtE): ok = left <= right
            elif isinstance(op, ast.Gt): ok = left > right
            elif isinstance(op, ast.GtE): ok = left >= right
            elif isinstance(op, ast.Is): ok = left is right
            elif isinstance(op, ast.IsNot): ok = left is not right
            else: raise ValueError(type(op).__name__)
            if not ok: return False
            left = right
        return True
    raise ValueError(type(node).__name__)


def _fixed_coordinate_seed_safe(
    function: ast.FunctionDef,
    coordinate: int,
    roles: Mapping[str, str],
    *,
    variants: Mapping[str, Any] | None = None,
    static_inputs: Mapping[str, Any] | None = None,
) -> bool:
    if not function.args.args:
        return False
    variable = function.args.args[0].arg
    unavailable = {"coordinate", "derived_state", "reduction", "vector"}

    def expr_value(node: ast.AST, bindings: Mapping[str, Any]) -> Any:
        if variants is not None:
            try:
                merged = dict(bindings)
                merged[variable] = int(coordinate)
                return _static_expr_value(
                    node,
                    variants=variants,
                    static_inputs={} if static_inputs is None else static_inputs,
                    bindings=merged,
                )
            except Exception:
                pass
        return _fixed_coordinate_expr_value(node, variable, coordinate, bindings)

    def state_call(node: ast.AST) -> bool:
        return any(
            isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
            and roles.get(x.func.id) in unavailable
            for x in ast.walk(node)
        )

    def walk(stmts: list[ast.stmt], bindings: dict[str, Any]) -> tuple[bool, bool]:
        for st in stmts:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                try:
                    bindings[st.targets[0].id] = expr_value(st.value, bindings)
                except Exception:
                    if state_call(st.value): return False, False
                    bindings.pop(st.targets[0].id, None)
                continue
            if isinstance(st, ast.If):
                try:
                    decision = bool(expr_value(st.test, bindings))
                except Exception:
                    decision = None
                if decision is None and state_call(st.test): return False, False
                branches = [st.body] if decision is True else [st.orelse] if decision is False else [st.body, st.orelse]
                results = [walk(branch, dict(bindings)) for branch in branches]
                if any(not safe for safe, _ in results): return False, False
                if results and all(term for _, term in results): return True, True
                continue
            if state_call(st): return False, False
            if isinstance(st, (ast.For, ast.While, ast.Try, ast.With, ast.Match)): return False, False
            if isinstance(st, (ast.Return, ast.Raise)): return True, True
        return True, False

    return bool(walk(function.body, {})[0])


def _fixed_coordinate_access_reachable(
    function: ast.FunctionDef,
    coordinate: int,
    call_ast: str,
    *,
    variants: Mapping[str, Any] | None = None,
    static_inputs: Mapping[str, Any] | None = None,
) -> bool:
    if not function.args.args:
        return True
    variable = function.args.args[0].arg

    def expr_value(node: ast.AST, bindings: Mapping[str, Any]) -> Any:
        if variants is not None:
            try:
                merged = dict(bindings)
                merged[variable] = int(coordinate)
                return _static_expr_value(
                    node,
                    variants=variants,
                    static_inputs={} if static_inputs is None else static_inputs,
                    bindings=merged,
                )
            except Exception:
                pass
        return _fixed_coordinate_expr_value(node, variable, coordinate, bindings)

    def matches(node: ast.AST) -> bool:
        return any(
            isinstance(x, ast.Call) and ast.dump(x, include_attributes=False) == call_ast
            for x in ast.walk(node)
        )

    def walk(stmts: list[ast.stmt], bindings: dict[str, Any]) -> tuple[bool, bool]:
        for st in stmts:
            if isinstance(st, ast.If):
                if matches(st.test): return True, False
                try:
                    decision = bool(expr_value(st.test, bindings))
                except Exception:
                    decision = None
                if decision is True:
                    hit, fall = walk(st.body, dict(bindings))
                    if hit or not fall: return hit, fall
                    continue
                if decision is False:
                    if not st.orelse: continue
                    hit, fall = walk(st.orelse, dict(bindings))
                    if hit or not fall: return hit, fall
                    continue
                bh, bf = walk(st.body, dict(bindings))
                eh, ef = walk(st.orelse, dict(bindings)) if st.orelse else (False, True)
                if bh or eh: return True, False
                if not (bf or ef): return False, False
                continue
            if matches(st): return True, False
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                try:
                    bindings[st.targets[0].id] = expr_value(st.value, bindings)
                except Exception:
                    bindings.pop(st.targets[0].id, None)
                continue
            if isinstance(st, (ast.Return, ast.Raise)): return False, False
        return False, True

    return bool(walk(function.body, {})[0])


def _range_step(args: tuple[ast.AST, ...]) -> int | None:
    if len(args) in {1, 2}: return 1
    if len(args) != 3: return None
    node = args[2]
    if not isinstance(node, ast.Constant) or not isinstance(node.value, int) or isinstance(node.value, bool):
        return None
    step = int(node.value)
    return step if step else None


def _static_expr_value(
    node: ast.AST, *, variants: Mapping[str, Any], static_inputs: Mapping[str, Any],
    active: set[str] | None = None, bindings: Mapping[str, Any] | None = None,
) -> Any:
    active = set() if active is None else active
    bindings = {} if bindings is None else bindings
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.Name):
        if node.id in bindings: return bindings[node.id]
        raise ValueError(node.id)
    if isinstance(node, ast.UnaryOp):
        v = _static_expr_value(node.operand, variants=variants, static_inputs=static_inputs, active=active, bindings=bindings)
        if isinstance(node.op, ast.USub): return -v
        if isinstance(node.op, ast.UAdd): return +v
        if isinstance(node.op, ast.Not): return not bool(v)
        raise ValueError(type(node.op).__name__)
    if isinstance(node, ast.BinOp):
        a = _static_expr_value(node.left, variants=variants, static_inputs=static_inputs, active=active, bindings=bindings)
        b = _static_expr_value(node.right, variants=variants, static_inputs=static_inputs, active=active, bindings=bindings)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
        if isinstance(node.op, ast.FloorDiv): return a // b
        if isinstance(node.op, ast.Mod): return a % b
        if isinstance(node.op, ast.Pow): return a ** b
        raise ValueError(type(node.op).__name__)
    if isinstance(node, ast.BoolOp):
        vals = [
            _static_expr_value(
                x, variants=variants, static_inputs=static_inputs,
                active=active, bindings=bindings,
            )
            for x in node.values
        ]
        if isinstance(node.op, ast.And): return all(bool(x) for x in vals)
        if isinstance(node.op, ast.Or): return any(bool(x) for x in vals)
        raise ValueError(type(node.op).__name__)
    if isinstance(node, ast.Compare):
        left = _static_expr_value(
            node.left, variants=variants, static_inputs=static_inputs,
            active=active, bindings=bindings,
        )
        for op, rhs in zip(node.ops, node.comparators):
            right = _static_expr_value(
                rhs, variants=variants, static_inputs=static_inputs,
                active=active, bindings=bindings,
            )
            if isinstance(op, ast.Eq): ok = left == right
            elif isinstance(op, ast.NotEq): ok = left != right
            elif isinstance(op, ast.Lt): ok = left < right
            elif isinstance(op, ast.LtE): ok = left <= right
            elif isinstance(op, ast.Gt): ok = left > right
            elif isinstance(op, ast.GtE): ok = left >= right
            elif isinstance(op, ast.Is): ok = left is right
            elif isinstance(op, ast.IsNot): ok = left is not right
            else: raise ValueError(type(op).__name__)
            if not ok: return False
            left = right
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name in {"point_input", "global_input"}:
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant): raise ValueError(name)
            key = str(node.args[0].value)
            if key not in static_inputs: raise ValueError(key)
            return static_inputs[key]
        if name in {"int", "float", "bool"} and len(node.args) == 1:
            v = _static_expr_value(node.args[0], variants=variants, static_inputs=static_inputs, active=active, bindings=bindings)
            return {"int": int, "float": float, "bool": bool}[name](v)
        if name in {"min", "max"} and node.args:
            vals = [_static_expr_value(x, variants=variants, static_inputs=static_inputs, active=active, bindings=bindings) for x in node.args]
            return min(vals) if name == "min" else max(vals)
        cv = variants.get(name)
        fn = getattr(cv, "function", None) if cv is not None else None
        if fn is not None and not node.args and not fn.args.args:
            if name in active: raise ValueError(f"static cycle:{name}")
            active.add(name)
            try:
                local: dict[str, Any] = {}
                for st in fn.body:
                    if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                        local[st.targets[0].id] = _static_expr_value(st.value, variants=variants, static_inputs=static_inputs, active=active, bindings=local)
                    elif isinstance(st, ast.Return):
                        return _static_expr_value(st.value, variants=variants, static_inputs=static_inputs, active=active, bindings=local)
                    else:
                        raise ValueError(type(st).__name__)
            finally:
                active.remove(name)
        raise ValueError(name)
    raise ValueError(type(node).__name__)


def _proven_range_boundary(
    args: tuple[ast.AST, ...], *, first: bool,
    variants: Mapping[str, Any], static_inputs: Mapping[str, Any],
) -> int | None:
    if not args: return None
    if len(args) == 1:
        start_node, stop_node, step_node = ast.Constant(0), args[0], ast.Constant(1)
    elif len(args) == 2:
        start_node, stop_node, step_node = args[0], args[1], ast.Constant(1)
    elif len(args) == 3:
        start_node, stop_node, step_node = args
    else:
        return None
    try:
        start = _static_expr_value(start_node, variants=variants, static_inputs=static_inputs)
        step = _static_expr_value(step_node, variants=variants, static_inputs=static_inputs)
    except Exception:
        return None
    if not isinstance(start, int) or isinstance(start, bool) or not isinstance(step, int) or isinstance(step, bool) or step == 0:
        return None
    if first: return int(start)
    try:
        stop = _static_expr_value(stop_node, variants=variants, static_inputs=static_inputs)
    except Exception:
        return None
    if not isinstance(stop, int) or isinstance(stop, bool): return None
    r = range(int(start), int(stop), int(step))
    return int(r[-1]) if r else None


def _task_stage_map(plan: StageExecutionPlan) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for task in plan.tasks:
        for uid in task.canonical_uids:
            out[uid] = tuple(task.stage_indices)
    return out


def _reduction_dependencies(uid: str, deps: Mapping[str, set[str]], reductions: set[str]) -> frozenset[str]:
    out: set[str] = set(); seen: set[str] = set()
    def visit(cur: str) -> None:
        if cur in seen: return
        seen.add(cur)
        for dep in deps.get(cur, ()):
            if dep == uid: continue
            if dep in reductions: out.add(dep)
            visit(dep)
    visit(uid); out.discard(uid)
    return frozenset(out)


def _boundary_replay_start(
    *,
    uid: str,
    coordinate: int,
    scan: str,
    graph: CanonicalSemanticGraph,
    nodes: Mapping[str, Any],
    variants: Mapping[str, Any],
    roles: Mapping[str, str],
    proof_rows: tuple[Mapping[str, Any], ...],
    static_input_values: Mapping[str, Any],
) -> int | None:
    """Prove a bounded self-causal replay for one non-source-safe seed.

    The initial subset intentionally admits only a single persistent family whose
    recursive self accesses are exactly one coordinate behind/ahead.  All other
    direct dependencies must be state-free.  This is sufficient to replay an
    in-force recurrence from a source-safe base without inventing model-specific
    initialization inputs, while rejecting coupled or algebraic recurrences.
    """
    fn = getattr(variants.get(uid), "function", None)
    if fn is None:
        return None
    direct_accesses = [a for a in graph.accesses if a.target_uid == uid]
    self_accesses = []
    for access in direct_accesses:
        source = nodes.get(access.source_uid)
        if source is None:
            return None
        if access.source_uid == uid:
            if access.min_offset is None or access.max_offset is None or access.min_offset != access.max_offset:
                return None
            offset = int(access.min_offset)
            if scan == "ascending" and offset != -1:
                return None
            if scan == "descending" and offset != 1:
                return None
            self_accesses.append(access)
            continue
        if source.state_semantic != "state_free":
            return None
    if not self_accesses:
        return None

    rows = proof_rows or ({},)
    direction = -1 if scan == "ascending" else 1
    # The replay search is a compile-time proof aid, not a runtime fallback.  The
    # finite cap keeps pathological/unbounded recurrence shapes fail-closed.
    for distance in range(1, 4097):
        candidate = int(coordinate + direction * distance)
        safe = True
        for row in rows:
            inputs = dict(row)
            inputs.update(static_input_values)
            if not _fixed_coordinate_seed_safe(
                fn, candidate, roles, variants=variants, static_inputs=inputs
            ):
                safe = False
                break
        if safe:
            return candidate
    return None


def _build_direct_stage_fragment(
    graph: CanonicalSemanticGraph,
    execution_plan: StageExecutionPlan,
    storage_plan: StageStoragePlan,
    variants: Mapping[str, Any],
    *,
    stage_index: int,
    static_input_values: Mapping[str, Any],
    proof_input_rows: tuple[Mapping[str, Any], ...],
) -> tuple[tuple[StageIterationDomain, ...], tuple[StageExecutionBlock, ...], tuple[StageBoundarySeed, ...], tuple[str, ...]]:
    """Lower one already-proven semantic Stage into direct physical scan blocks.

    This helper never derives stage placement.  ``StageExecutionPlan`` already owns
    that fact.  It only freezes the iteration/reduction geometry for the supplied
    stage and derives boundary evidence for persistent values produced there.
    """
    blockers: set[str] = set()
    nodes = {x.uid: x for x in graph.nodes}
    stages = _task_stage_map(execution_plan)
    scheduled = {uid for uid, ss in stages.items() if ss == (int(stage_index),)}
    reductions = {
        uid for uid in scheduled
        if uid in variants and getattr(variants[uid], "role", None) == "reduction"
    }
    persistent = tuple(sorted(
        uid for uid in scheduled if nodes[uid].state_semantic == "persistent_state"
    ))
    domain_by_owner = {x.owner_uid: x for x in execution_plan.domains}
    stage_row = next((x for x in execution_plan.stages if x.index == int(stage_index)), None)
    if stage_row is None:
        blockers.add(f"direct_stage_missing:{stage_index}")
        return (), (), (), tuple(sorted(blockers))

    # G2e admits one narrow rank-0 physical shape: a derived-only post Stage
    # whose exact scheduled values are scalar formulas with frozen rank-0
    # ExecutionDomains.  No fake coordinate axis or range(1) is introduced.
    post_stage_scalars: tuple[str, ...] = ()
    rank0_post_stage = False
    if not reductions and not persistent and scheduled:
        candidate_scalars = tuple(sorted(
            uid for uid in scheduled
            if uid in variants
            and getattr(variants[uid], "role", None) == "scalar"
            and nodes[uid].canonical_role == "scalar"
        ))
        if (
            candidate_scalars == (graph.output_uid,)
            and set(candidate_scalars) == set(scheduled)
            and not stage_row.component_task_uids
            and nodes[graph.output_uid].execution_semantic == "derived_task"
        ):
            rank0_ok = True
            for uid in candidate_scalars:
                domain = domain_by_owner.get(uid)
                if domain is None:
                    blockers.add(f"direct_post_stage_scalar_domain_missing:{stage_index}:{uid}")
                    rank0_ok = False
                    continue
                if domain.blockers:
                    blockers.update(
                        f"direct_post_stage_scalar_domain_blocker:{stage_index}:{uid}:{b}"
                        for b in domain.blockers
                    )
                    rank0_ok = False
                    continue
                if domain.rank != 0 or domain.parameter_names or domain.bounds:
                    blockers.add(
                        f"direct_post_stage_scalar_domain_rank_unsupported:{stage_index}:{uid}:{domain.rank}"
                    )
                    rank0_ok = False
            if rank0_ok and candidate_scalars:
                rank0_post_stage = True
                post_stage_scalars = candidate_scalars

    range_sigs: dict[tuple[str, ...], tuple[ast.AST, ...]] = {}
    range_evidence: dict[tuple[str, ...], list[str]] = {}
    if reductions:
        for uid in sorted(reductions):
            spec = getattr(variants[uid], "reduction", None)
            if spec is None:
                blockers.add(f"direct_reduction_spec_missing:{uid}")
                continue
            args = tuple(spec.range_args)
            sig = tuple(ast.dump(x, include_attributes=False) for x in args)
            range_sigs.setdefault(sig, args)
            range_evidence.setdefault(sig, []).append(uid)
    elif persistent:
        # A persistent-only semantic Stage still has a fully proven execution
        # domain even when no scalar reduction happens to provide a Python
        # ``range`` expression.  Lower the rank-1 domain envelope directly from
        # StageExecutionPlan instead of inventing another scheduler.
        domain_rows = []
        parameters: set[str] = set()
        for uid in persistent:
            domain = domain_by_owner.get(uid)
            if domain is None:
                blockers.add(f"direct_stage_execution_domain_missing:{stage_index}:{uid}")
                continue
            if domain.blockers:
                blockers.update(
                    f"direct_stage_execution_domain_blocker:{stage_index}:{uid}:{b}"
                    for b in domain.blockers
                )
                continue
            if domain.rank != 1 or len(domain.parameter_names) != 1 or len(domain.bounds) != 1:
                blockers.add(
                    f"direct_stage_execution_domain_rank_unsupported:{stage_index}:{uid}:{domain.rank}"
                )
                continue
            lo, hi = domain.bounds[0]
            if lo is None or hi is None:
                blockers.add(f"direct_stage_execution_domain_bounds_unproved:{stage_index}:{uid}")
                continue
            if int(lo) > int(hi):
                blockers.add(f"direct_stage_execution_domain_empty:{stage_index}:{uid}:{lo}:{hi}")
                continue
            parameters.add(str(domain.parameter_names[0]))
            domain_rows.append((int(lo), int(hi), str(domain.uid)))
        if domain_rows and len(domain_rows) == len(persistent):
            if len(parameters) != 1:
                blockers.add(
                    f"direct_stage_execution_domain_parameter_not_unique:{stage_index}:{sorted(parameters)!r}"
                )
            else:
                lo = min(x[0] for x in domain_rows)
                hi = max(x[1] for x in domain_rows)
                args = (ast.Constant(value=lo), ast.Constant(value=hi + 1))
                sig = tuple(ast.dump(x, include_attributes=False) for x in args)
                range_sigs[sig] = args
                range_evidence[sig] = [x[2] for x in domain_rows]
    elif not rank0_post_stage:
        blockers.add(f"direct_stage_iteration_domain_unproved:{stage_index}")
    if not rank0_post_stage and len(range_sigs) != 1:
        blockers.add(f"direct_iteration_range_not_unique:{stage_index}:{len(range_sigs)}")
    scan_rows = set(stage_row.scan_directions)
    concrete = {x for x in scan_rows if x not in {"any", "none"}}
    if len(concrete) == 1:
        scan = next(iter(concrete))
    elif rank0_post_stage:
        scan = "none"
    elif not concrete:
        # Completion-only/reduction stages have no persistent scan requirement.
        # Their reduction iteration order is ascending unless a semantic scan says
        # otherwise; this affects only associative scalar accumulation ordering and
        # matches Python range order.
        scan = "ascending"
    else:
        scan = "mixed"
        blockers.add(f"direct_stage_scan_direction_not_unique:{stage_index}")

    iteration_domains: list[StageIterationDomain] = []
    range_args: tuple[ast.AST, ...] = ()
    materialized_rows = [
        row for row in storage_plan.values
        if row.producer_stage_index == int(stage_index)
        and row.cross_stage_storage_kind == "materialized_history"
    ]
    extension_min = min(
        [0, *(int(row.retained_offset_min) for row in materialized_rows if row.retained_offset_min is not None)]
    )
    extension_max = max(
        [0, *(int(row.retained_offset_max) for row in materialized_rows if row.retained_offset_max is not None)]
    )
    absolute_materialized_rows = []
    for row in materialized_rows:
        if row.retained_offset_min is None or row.retained_offset_max is None:
            bounds = row.retained_bounds
            if (
                row.retained_offset_min is None
                and row.retained_offset_max is None
                and bounds is not None
                and bounds[0] is not None
                and bounds[1] is not None
            ):
                absolute_materialized_rows.append(row)
            else:
                blockers.add(f"direct_materialized_coordinate_envelope_unproved:{row.value_uid}")
    # Negative retained offsets are a normal producer-history requirement for
    # ascending recurrences (for example, an initial predecessor coordinate).
    # The runtime already scans the materialization-extended domain while keeping
    # reductions restricted to the original base range.  Nonascending extensions
    # remain fail-closed in source generation until their geometry is proven.
    if extension_min < 0 and scan != "ascending":
        blockers.add(
            f"direct_materialization_negative_extension_scan_unsupported:{stage_index}:{extension_min}:{scan}"
        )
    if len(range_sigs) == 1:
        sig, range_args = next(iter(range_sigs.items()))
        step = _range_step(range_args)
        if step is None:
            blockers.add(f"direct_iteration_step_not_literal:{stage_index}")
            step = 1
        if step == 0:
            blockers.add(f"direct_iteration_step_zero:{stage_index}")
            step = 1
        if extension_min % abs(int(step)) != 0 or extension_max % abs(int(step)) != 0:
            blockers.add(
                f"direct_materialization_extension_step_misaligned:{stage_index}:{extension_min}:{extension_max}:{step}"
            )
        if absolute_materialized_rows:
            base_first = _proven_range_boundary(
                range_args,
                first=True,
                variants=variants,
                static_inputs=static_input_values,
            )
            base_last = _proven_range_boundary(
                range_args,
                first=False,
                variants=variants,
                static_inputs=static_input_values,
            )
            if base_first is None or base_last is None:
                blockers.add(f"direct_absolute_materialized_stage_bounds_unproved:{stage_index}")
            else:
                base_lo = min(int(base_first), int(base_last))
                base_hi = max(int(base_first), int(base_last))
                abs_step = abs(int(step))
                for row in absolute_materialized_rows:
                    lo, hi = row.retained_bounds  # exact by admission above
                    lo = int(lo); hi = int(hi)
                    if lo > hi:
                        blockers.add(f"direct_materialized_absolute_bounds_empty:{row.value_uid}:{lo}:{hi}")
                        continue
                    if lo < base_lo or hi > base_hi:
                        blockers.add(
                            f"direct_materialized_absolute_bounds_outside_stage:{row.value_uid}:{lo}:{hi}:{base_lo}:{base_hi}"
                        )
                        continue
                    if (lo - int(base_first)) % abs_step != 0 or (hi - int(base_first)) % abs_step != 0:
                        blockers.add(
                            f"direct_materialized_absolute_bounds_step_misaligned:{row.value_uid}:{lo}:{hi}:{step}"
                        )
        loop_vars = {
            variants[uid].reduction.loop_var for uid in reductions
            if getattr(variants[uid], "reduction", None) is not None
        }
        if len(loop_vars) == 1:
            parameter_name = next(iter(loop_vars))
        elif not reductions and persistent:
            domain_by_owner = {x.owner_uid: x for x in execution_plan.domains}
            names = {
                str(domain_by_owner[uid].parameter_names[0])
                for uid in persistent
                if uid in domain_by_owner and len(domain_by_owner[uid].parameter_names) == 1
            }
            parameter_name = next(iter(names)) if len(names) == 1 else "t"
        else:
            parameter_name = "t"
        domain_uid = _stable_id(
            "stage_iteration_domain", (execution_plan.uid, stage_index, sig, step, scan)
        )
        iteration_domains.append(StageIterationDomain(
            uid=domain_uid,
            stage_index=int(stage_index),
            parameter_name=parameter_name,
            range_arg_sources=tuple(ast.unparse(x) for x in range_args),
            range_arg_asts=sig,
            coordinate_step=int(step),
            scan_direction=scan,
            materialization_extension_min=int(extension_min),
            materialization_extension_max=int(extension_max),
            evidence_uids=tuple(sorted(range_evidence[sig])),
        ))

    deps: dict[str, set[str]] = {uid: set() for uid in scheduled}
    for access in graph.accesses:
        if access.target_uid in deps and access.source_uid in nodes:
            deps[access.target_uid].add(access.source_uid)
    reduction_deps = {uid: _reduction_dependencies(uid, deps, reductions) for uid in scheduled}
    scalar_uids = {
        uid for uid in scheduled
        if uid in variants and getattr(variants[uid], "role", None) == "scalar"
    }
    pre_scalars = tuple(sorted(uid for uid in scalar_uids if not reduction_deps[uid]))
    post_scalars = tuple(sorted(uid for uid in scalar_uids if reduction_deps[uid]))
    layers: list[tuple[str, ...]] = []
    completed: set[str] = set()
    remaining = set(reductions)
    while remaining:
        ready = tuple(sorted(uid for uid in remaining if reduction_deps[uid].issubset(completed)))
        if not ready:
            blockers.add(
                f"direct_reduction_completion_cycle:{stage_index}:" + ",".join(sorted(remaining))
            )
            break
        layers.append(ready)
        completed.update(ready)
        remaining.difference_update(ready)

    roles: dict[str, str] = {}
    for uid, node in nodes.items():
        roles[uid] = "pure_map" if node.execution_semantic == "pure_map" else node.canonical_role

    seeds_by_key: dict[tuple[str, int], set[str]] = {}
    seed_boundaries_by_key: dict[tuple[str, int], set[int]] = {}
    seed_rows_by_key: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    if len(range_sigs) == 1 and scan in {"ascending", "descending"} and persistent:
        boundary = _proven_range_boundary(
            range_args,
            first=(scan == "ascending"),
            variants=variants,
            static_inputs=static_input_values,
        )
        boundary_rows: dict[int, list[Mapping[str, Any]]] = {}
        if boundary is not None:
            boundary_rows[int(boundary)] = list(proof_input_rows) or [{}]
        elif proof_input_rows:
            for row in proof_input_rows:
                candidate_inputs = dict(row)
                candidate_inputs.update(static_input_values)
                candidate = _proven_range_boundary(
                    range_args,
                    first=(scan == "ascending"),
                    variants=variants,
                    static_inputs=candidate_inputs,
                )
                if candidate is None:
                    boundary_rows = {}
                    break
                boundary_rows.setdefault(int(candidate), []).append(row)
        boundaries = tuple(sorted(boundary_rows))
        relevant = [
            a for a in graph.accesses
            if a.source_uid in persistent
            and stages.get(a.source_uid) == (int(stage_index),)
            and stages.get(a.target_uid) == (int(stage_index),)
            and a.min_offset is not None and a.max_offset is not None
            and a.min_offset == a.max_offset
            and (
                (scan == "ascending" and int(a.min_offset) < 0)
                or (scan == "descending" and int(a.max_offset) > 0)
            )
        ]
        if relevant and not boundaries:
            blockers.add(f"direct_boundary_coordinate_not_proven:{stage_index}")
        elif boundaries:
            for boundary_q in boundaries:
                for access in relevant:
                    target_q = int(boundary_q)
                    target_domain = nodes[access.target_uid].domain
                    if target_domain is not None:
                        if scan == "ascending" and target_domain[0] is not None:
                            target_q = max(target_q, int(target_domain[0]))
                        if scan == "descending" and target_domain[1] is not None:
                            target_q = min(target_q, int(target_domain[1]))
                    source_q = target_q + int(access.min_offset)
                    if scan == "ascending" and source_q >= int(boundary_q):
                        continue
                    if scan == "descending" and source_q <= int(boundary_q):
                        continue
                    target_fn = getattr(variants.get(access.target_uid), "function", None)
                    reachable_rows: list[Mapping[str, Any]] = []
                    for row in boundary_rows[boundary_q]:
                        inputs = dict(row)
                        inputs.update(static_input_values)
                        if target_fn is None or _fixed_coordinate_access_reachable(
                            target_fn,
                            target_q,
                            access.call_ast,
                            variants=variants,
                            static_inputs=inputs,
                        ):
                            reachable_rows.append(row)
                    if not reachable_rows:
                        continue
                    key = (access.source_uid, source_q)
                    seeds_by_key.setdefault(key, set()).add(access.uid)
                    seed_boundaries_by_key.setdefault(key, set()).add(int(boundary_q))
                    seed_rows_by_key.setdefault(key, []).extend(reachable_rows)

    boundary_seeds: list[StageBoundarySeed] = []
    for (uid, q), evidence in sorted(seeds_by_key.items()):
        fn = getattr(variants.get(uid), "function", None)
        proof_rows = tuple(seed_rows_by_key.get((uid, q), ())) or ({},)
        source_safe = bool(fn is not None) and all(
            _fixed_coordinate_seed_safe(
                fn,
                q,
                roles,
                variants=variants,
                static_inputs={**dict(row), **static_input_values},
            )
            for row in proof_rows
        )
        replay_start = None
        if not source_safe:
            replay_start = _boundary_replay_start(
                uid=uid,
                coordinate=q,
                scan=scan,
                graph=graph,
                nodes=nodes,
                variants=variants,
                roles=roles,
                proof_rows=proof_rows,
                static_input_values=static_input_values,
            )
            if replay_start is None:
                blockers.add(f"direct_boundary_seed_unproved:{uid}:{q}")
                continue
        active_boundaries = tuple(sorted(seed_boundaries_by_key.get((uid, q), ())))
        proof_kind = (
            "bounded_self_causal_replay_v1" if replay_start is not None
            else "finite_run_boundary_source_guard_v1" if proof_input_rows
            else "fixed_coordinate_source_guard_v1"
        )
        boundary_seeds.append(StageBoundarySeed(
            uid=_stable_id(
                "stage_boundary_seed",
                (
                    execution_plan.uid, stage_index, uid, q, active_boundaries,
                    replay_start, *sorted(evidence),
                ),
            ),
            stage_index=int(stage_index),
            value_uid=uid,
            coordinate=q,
            source_access_uids=tuple(sorted(evidence)),
            proof_kind=proof_kind,
            active_boundary_coordinates=active_boundaries,
            replay_start_coordinate=replay_start,
        ))

    blocks: list[StageExecutionBlock] = []
    if rank0_post_stage:
        blocks.append(StageExecutionBlock(
            uid=_stable_id(
                "stage_execution_block",
                (execution_plan.uid, stage_index, "post_stage_scalar", post_stage_scalars),
            ),
            stage_index=int(stage_index),
            kind="post_stage_scalar",
            iteration_domain_uid=None,
            persistent_driver_uids=(),
            pre_scalar_uids=post_stage_scalars,
            reduction_uids=(),
            post_scalar_uids=(),
            scan_direction="none",
            blockers=(),
        ))
    elif iteration_domains:
        domain = iteration_domains[0]
        block_blockers: list[str] = []
        if scan not in {"ascending", "descending"}:
            block_blockers.append(f"direct_scan_direction_unsupported:{stage_index}:{scan}")
            blockers.update(block_blockers)
        if layers:
            completed_before: set[str] = set()
            for i, layer in enumerate(layers):
                drivers = tuple(sorted(
                    uid for uid in persistent if reduction_deps[uid].issubset(completed_before)
                ))
                blocks.append(StageExecutionBlock(
                    uid=_stable_id(
                        "stage_execution_block",
                        (execution_plan.uid, stage_index, domain.uid, i, drivers, layer, scan),
                    ),
                    stage_index=int(stage_index),
                    kind="coordinate_scan",
                    iteration_domain_uid=domain.uid,
                    persistent_driver_uids=drivers,
                    pre_scalar_uids=pre_scalars if i == 0 else (),
                    reduction_uids=layer,
                    post_scalar_uids=post_scalars if i == len(layers) - 1 else (),
                    scan_direction=scan,
                    blockers=tuple(sorted(block_blockers)),
                ))
                completed_before.update(layer)
        elif persistent:
            drivers = tuple(sorted(persistent))
            blocks.append(StageExecutionBlock(
                uid=_stable_id(
                    "stage_execution_block",
                    (execution_plan.uid, stage_index, domain.uid, 0, drivers, (), scan),
                ),
                stage_index=int(stage_index),
                kind="coordinate_scan",
                iteration_domain_uid=domain.uid,
                persistent_driver_uids=drivers,
                pre_scalar_uids=pre_scalars,
                reduction_uids=(),
                post_scalar_uids=post_scalars,
                scan_direction=scan,
                blockers=tuple(sorted(block_blockers)),
            ))

    return (
        tuple(iteration_domains),
        tuple(blocks),
        tuple(boundary_seeds),
        tuple(sorted(blockers)),
    )


def _direct_barrier_blockers(
    execution_plan: StageExecutionPlan,
    storage_plan: StageStoragePlan,
) -> tuple[str, ...]:
    """Validate frozen full-stage fence contracts without backend rediscovery."""
    blockers: set[str] = set()
    stage_by_index = {int(x.index): x for x in execution_plan.stages}
    storage_stage_by_index = {int(x.index): x for x in storage_plan.stages}
    storage_barriers = {
        row.semantic_barrier_uid: row for row in storage_plan.fences
    }
    for barrier in execution_plan.barriers:
        uid = str(barrier.uid)
        for blocker in barrier.blockers:
            blockers.add(f"direct_barrier_semantic_blocker:{uid}:{blocker}")
        src = barrier.source_stage_index
        dst = barrier.target_stage_index
        if src is None or dst is None:
            blockers.add(f"direct_barrier_stage_unproved:{uid}")
            continue
        src = int(src); dst = int(dst)
        if src not in stage_by_index or dst not in stage_by_index:
            blockers.add(f"direct_barrier_stage_missing:{uid}:{src}:{dst}")
            continue
        if src >= dst:
            blockers.add(f"direct_barrier_order_unsupported:{uid}:{src}:{dst}")
        storage_fence = storage_barriers.get(uid)
        if storage_fence is None:
            blockers.add(f"direct_barrier_storage_identity_missing:{uid}")
        else:
            semantic_identity = (
                barrier.source_component_uid,
                barrier.target_component_uid,
                barrier.source_stage_index,
                barrier.target_stage_index,
                barrier.availability,
                barrier.proof_kind,
                tuple(barrier.evidence_uids),
            )
            storage_identity = (
                storage_fence.source_component_uid,
                storage_fence.target_component_uid,
                storage_fence.source_stage_index,
                storage_fence.target_stage_index,
                storage_fence.availability,
                storage_fence.semantic_proof_kind,
                tuple(storage_fence.semantic_evidence_uids),
            )
            if semantic_identity != storage_identity:
                blockers.add(f"direct_barrier_storage_semantic_identity_mismatch:{uid}")
            for blocker in storage_fence.blockers:
                blockers.add(f"direct_barrier_storage_blocker:{uid}:{blocker}")
            if storage_fence.execution_mode != "full_stage_materialization":
                blockers.add(
                    f"direct_barrier_execution_mode_unsupported:{uid}:"
                    f"{storage_fence.execution_mode}"
                )
        src_storage = storage_stage_by_index.get(src)
        dst_storage = storage_stage_by_index.get(dst)
        if src_storage is None or uid not in src_storage.outgoing_barrier_uids:
            blockers.add(f"direct_barrier_source_stage_link_missing:{uid}:{src}")
        if dst_storage is None or uid not in dst_storage.incoming_barrier_uids:
            blockers.add(f"direct_barrier_target_stage_link_missing:{uid}:{dst}")
    return tuple(sorted(blockers))


def build_stage_direct_contract(
    graph: CanonicalSemanticGraph,
    execution_plan: StageExecutionPlan,
    storage_plan: StageStoragePlan,
    variants: Mapping[str, Any],
    *,
    static_input_values: Mapping[str, Any] | None = None,
    proof_input_rows: tuple[Mapping[str, Any], ...] = (),
) -> StageDirectContract:
    """Freeze direct physical facts for an already-proven Stage plan.

    Stage order and barrier identity are consumed verbatim from ``StageExecutionPlan``.
    The admitted barrier subset is an upstream-proven full-stage materialization
    fence.  Semantic availability remains evidence only; no streaming or
    backend-side scheduling inference is performed here.
    """
    static_input_values = {} if static_input_values is None else static_input_values
    blockers: set[str] = set(_direct_barrier_blockers(execution_plan, storage_plan))
    if execution_plan.unplaced_task_uids:
        blockers.add(f"direct_unplaced_tasks:{len(execution_plan.unplaced_task_uids)}")

    iteration_domains: list[StageIterationDomain] = []
    blocks: list[StageExecutionBlock] = []
    seeds: list[StageBoundarySeed] = []
    if not blockers:
        for stage in execution_plan.stages:
            frag_domains, frag_blocks, frag_seeds, frag_blockers = _build_direct_stage_fragment(
                graph,
                execution_plan,
                storage_plan,
                variants,
                stage_index=stage.index,
                static_input_values=static_input_values,
                proof_input_rows=proof_input_rows,
            )
            iteration_domains.extend(frag_domains)
            blocks.extend(frag_blocks)
            seeds.extend(frag_seeds)
            blockers.update(frag_blockers)

    pure_map_rows: dict[str, StagePureMapABI] = {str(spec.uid): StagePureMapABI(
        uid=str(spec.uid),
        coordinate_parameter=str(spec.coordinate_parameter),
        scalar_dependencies=tuple(str(x) for x in spec.scalar_dependencies),
        pure_map_dependencies=tuple(str(x) for x in spec.pure_map_dependencies),
        proof_scope=str(spec.proof_scope),
        provenance=tuple(str(x) for x in spec.provenance),
    ) for spec in sorted(graph.pure_map_specs, key=lambda x: str(x.uid))}
    for proof in sorted(
        getattr(graph, "state_free_execution_proofs", ()),
        key=lambda x: str(x.value_uid),
    ):
        if not proof.approved:
            continue
        pure_map_rows.setdefault(str(proof.value_uid), StagePureMapABI(
            uid=str(proof.value_uid),
            coordinate_parameter=str(proof.coordinate_parameter),
            scalar_dependencies=tuple(str(x) for x in proof.scalar_dependencies),
            pure_map_dependencies=tuple(str(x) for x in proof.pure_map_dependencies),
            proof_scope="canonical_semantic_graph",
            provenance=tuple(str(x) for x in (*proof.provenance, proof.proof_kind, proof.proof_uid)),
        ))
    pure_maps = tuple(pure_map_rows[uid] for uid in sorted(pure_map_rows))

    aux_rows: list[StageAuxiliaryRecurrenceABI] = []
    for uid in sorted(getattr(graph, "auxiliary_recurrence_uids", ())):
        row = storage_plan.value(uid)
        if (
            row.working_storage_kind != "ring"
            or not row.ring_depth
            or row.working_offset_min is None
            or row.working_offset_max is None
        ):
            blockers.add(f"direct_auxiliary_recurrence_storage_unproved:{uid}")
            continue
        if row.working_offset_max <= 0 and row.working_offset_min < 0:
            aux_scan = "ascending"
        elif row.working_offset_min >= 0 and row.working_offset_max > 0:
            aux_scan = "descending"
        else:
            blockers.add(f"direct_auxiliary_recurrence_direction_unproved:{uid}")
            continue
        aux_rows.append(StageAuxiliaryRecurrenceABI(
            uid=uid,
            scan_direction=aux_scan,
            ring_depth=int(row.ring_depth),
            offset_min=int(row.working_offset_min),
            offset_max=int(row.working_offset_max),
        ))

    return StageDirectContract(
        iteration_domains=tuple(iteration_domains),
        execution_blocks=tuple(blocks),
        boundary_seeds=tuple(seeds),
        pure_maps=pure_maps,
        auxiliary_recurrences=tuple(aux_rows),
        barrier_uids=tuple(sorted(x.uid for x in execution_plan.barriers)),
        blockers=tuple(sorted(blockers)),
    )


def build_zero_barrier_direct_contract(
    graph: CanonicalSemanticGraph,
    execution_plan: StageExecutionPlan,
    storage_plan: StageStoragePlan,
    variants: Mapping[str, Any],
    *,
    static_input_values: Mapping[str, Any] | None = None,
    proof_input_rows: tuple[Mapping[str, Any], ...] = (),
) -> StageDirectContract:
    """Compatibility wrapper for callers that require exactly zero barriers."""
    if execution_plan.barriers:
        return StageDirectContract(
            (), (), (), (), (), (),
            (f"direct_zero_barrier_contract_requires_zero_barriers:{len(execution_plan.barriers)}",),
        )
    return build_stage_direct_contract(
        graph, execution_plan, storage_plan, variants,
        static_input_values=static_input_values,
        proof_input_rows=proof_input_rows,
    )


def build_one_stage_direct_contract(
    graph: CanonicalSemanticGraph,
    execution_plan: StageExecutionPlan,
    storage_plan: StageStoragePlan,
    variants: Mapping[str, Any],
    *,
    static_input_values: Mapping[str, Any] | None = None,
    proof_input_rows: tuple[Mapping[str, Any], ...] = (),
) -> StageDirectContract:
    """Compatibility wrapper retaining the established one-stage direct contract."""
    if tuple(x.index for x in execution_plan.stages) != (0,):
        return StageDirectContract(
            (), (), (), (), (), (), ("direct_one_stage_requires_exactly_stage_0",)
        )
    return build_zero_barrier_direct_contract(
        graph,
        execution_plan,
        storage_plan,
        variants,
        static_input_values=static_input_values,
        proof_input_rows=proof_input_rows,
    )

