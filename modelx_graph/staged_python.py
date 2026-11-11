from __future__ import annotations

"""Generic two-stage Python execution over :mod:`stage_execution_plan` semantics.

This module is deliberately narrower than the mature one-stage native backend.  It
executes a zero-barrier two-stage plan directly from canonical variants so
completed-history ``StateDerivedMap`` work does not have to masquerade as carried
state.  Concrete cross-stage storage comes only from ``ValueLifetime`` evidence.

The runtime also exposes an explicit external-value boundary for state-free
canonical values whose non-graph operator lowering is not yet supported.  That
boundary is an adapter, not execution permission inferred by name: callers must
provide the exact canonical UID and a resolver.  This lets graph/stage development
proceed without smuggling unrelated input/table normalization into scheduler code.
"""

import ast
import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .canonical_semantic_graph import CanonicalSemanticGraph
from .frontend import CanonicalProofSnapshot
from .stage_execution_plan import StageExecutionPlan
from .stage_storage_plan import StageStoragePlan, build_stage_storage_plan


class StagedPythonError(RuntimeError):
    pass


ExternalCallResolver = Callable[[str, tuple[Any, ...]], Any]
ExternalValueResolver = Callable[[str, tuple[Any, ...]], Any]


@dataclass(frozen=True)
class StagedPythonExecutionResult:
    output: Any
    stage_indices: tuple[int, ...]
    stage_program_uid: str | None
    coordinate_range: tuple[int, int, int]
    stage0_extension: int
    stage_storage_plan_uid: str
    materialized_uids: tuple[str, ...]
    materialized_lengths: tuple[tuple[str, int], ...]
    external_value_uids: tuple[str, ...]
    external_call_names: tuple[str, ...]
    executed_reduction_uids: tuple[str, ...]
    executed_scalar_uids: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "output": float(self.output) if isinstance(self.output, (float, np.floating)) else self.output,
            "stage_indices": list(self.stage_indices),
            "stage_program_uid": self.stage_program_uid,
            "coordinate_range": {
                "start": int(self.coordinate_range[0]),
                "stop": int(self.coordinate_range[1]),
                "step": int(self.coordinate_range[2]),
            },
            "stage0_extension": int(self.stage0_extension),
            "stage_storage_plan_uid": self.stage_storage_plan_uid,
            "materialized_uids": list(self.materialized_uids),
            "materialized_lengths": {
                uid: int(length) for uid, length in self.materialized_lengths
            },
            "external_value_uids": list(self.external_value_uids),
            "external_call_names": list(self.external_call_names),
            "executed_reduction_uids": list(self.executed_reduction_uids),
            "executed_scalar_uids": list(self.executed_scalar_uids),
        }


class StagedPythonExecutor:
    """Execute the first generic zero-barrier two-stage canonical subset.

    The executor intentionally does not interpret legacy ``coord_phase`` or
    ``DerivedHistoryRead`` metadata.  Persistent state executes only in its
    producing semantic stage.  A later-stage persistent read is legal only when
    the exact UID has a ``materialized_cross_stage`` lifetime.
    """

    _BUILTIN_CALL_NAMES = {
        "abs", "bool", "enumerate", "float", "int", "len", "list", "max",
        "min", "range", "round", "sum", "tuple", "zip",
    }

    def __init__(
        self,
        snapshot: CanonicalProofSnapshot,
        graph: CanonicalSemanticGraph,
        plan: StageExecutionPlan,
        *,
        storage_plan: StageStoragePlan | None = None,
        physical_values: tuple[Any, ...] = (),
        stage_program_uid: str | None = None,
        input_values: Mapping[str, Any] | None = None,
        external_call: ExternalCallResolver | None = None,
        external_value: ExternalValueResolver | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.graph = graph
        self.plan = plan
        self.storage_plan = storage_plan or build_stage_storage_plan(
            graph, plan, variants=snapshot.variants
        )
        if self.storage_plan.execution_plan_uid != plan.uid:
            raise StagedPythonError(
                "stage storage plan does not belong to the supplied StageExecutionPlan"
            )
        self.stage_program_uid = stage_program_uid
        self.physical_values = {x.value_uid: x for x in physical_values}
        self.variants = snapshot.variants
        self.nodes = {node.uid: node for node in graph.nodes}
        self.inputs = dict(input_values or {})
        self.external_call = external_call
        self.external_value = external_value

        if tuple(stage.index for stage in plan.stages) != (0, 1):
            raise StagedPythonError(
                "first staged Python executor requires exactly semantic stages 0 and 1"
            )
        if plan.barriers:
            raise StagedPythonError(
                "first staged Python executor does not execute completion barriers"
            )
        if plan.unplaced_task_uids:
            raise StagedPythonError(
                f"staged execution has unplaced tasks: {plan.unplaced_task_uids!r}"
            )

        self.task_stages: dict[str, tuple[int, ...]] = {}
        self.task_kind: dict[str, str] = {}
        for task in plan.tasks:
            for uid in task.canonical_uids:
                self.task_stages[uid] = tuple(task.stage_indices)
                self.task_kind[uid] = task.kind

        self.persistent_stage: dict[str, int] = {}
        for task in plan.tasks:
            if task.kind != "persistent_component":
                continue
            if len(task.stage_indices) != 1:
                raise StagedPythonError(
                    f"persistent component {task.uid} lacks one semantic stage"
                )
            for uid in task.canonical_uids:
                self.persistent_stage[uid] = int(task.stage_indices[0])

        self.materialized_uids = frozenset(self.storage_plan.materialized_value_uids)
        if any(self.nodes[uid].state_semantic != "persistent_state" for uid in self.materialized_uids):
            raise StagedPythonError("cross-stage materialization contains a nonpersistent value")

        if self.physical_values:
            physical_materialized = {
                uid for uid, row in self.physical_values.items()
                if getattr(row, "materialized_slot", None) is not None
            }
            if physical_materialized != set(self.materialized_uids):
                raise StagedPythonError(
                    "stage physical materialization does not match StageStoragePlan"
                )
            slots = sorted(
                int(self.physical_values[uid].materialized_slot)
                for uid in physical_materialized
            )
            if slots != list(range(len(slots))):
                raise StagedPythonError("stage materialized-history slots are not dense")
            scalar_slots = sorted(
                int(row.scalar_slot) for row in self.physical_values.values()
                if getattr(row, "scalar_slot", None) is not None
            )
            if scalar_slots != list(range(len(scalar_slots))):
                raise StagedPythonError("stage-carried scalar slots are not dense")

        self.external_value_uids = frozenset(
            blocker.split(":", 1)[1]
            for blocker in plan.capability_blockers
            if blocker.startswith("execution_semantic_unapproved:")
        )
        if self.external_value_uids and external_value is None:
            raise StagedPythonError(
                "plan contains state-free execution operators that require an explicit external_value resolver: "
                f"{sorted(self.external_value_uids)!r}"
            )
        for uid in self.external_value_uids:
            node = self.nodes.get(uid)
            if node is None or node.state_semantic != "state_free":
                raise StagedPythonError(
                    f"external execution boundary is not state-free: {uid}"
                )

        tolerated = {
            "backend_multistage_execution_not_supported:2",
        }
        remaining: list[str] = []
        for blocker in plan.capability_blockers:
            if blocker in tolerated:
                continue
            if blocker.startswith("backend_cross_stage_materialization_not_supported:"):
                continue
            if blocker.startswith("execution_semantic_unapproved:") and blocker.split(":", 1)[1] in self.external_value_uids:
                continue
            remaining.append(blocker)
        if remaining:
            raise StagedPythonError(
                "stage plan contains unsupported graph blockers outside the two-stage subset: "
                + "; ".join(sorted(remaining))
            )

        self._stage = 0
        self._stage0_closed = False
        self._persistent_cache: dict[str, dict[int, Any]] = {
            uid: {} for uid in self.persistent_stage
        }
        self._materialized: dict[str, dict[int, Any]] = {}
        materialized_slot_count = 1 + max(
            (int(row.materialized_slot) for row in self.physical_values.values()
             if getattr(row, "materialized_slot", None) is not None),
            default=-1,
        )
        carried_scalar_slot_count = 1 + max(
            (int(row.scalar_slot) for row in self.physical_values.values()
             if getattr(row, "scalar_slot", None) is not None),
            default=-1,
        )
        self._materialized_slots: list[dict[int, Any] | None] = [None] * materialized_slot_count
        self._stage_scalar_slots: list[Any] = [None] * carried_scalar_slot_count
        self._value_cache: dict[tuple[int, str, tuple[Any, ...]], Any] = {}
        self._scalar_cache: dict[str, Any] = {}
        self._reduction_cache: dict[str, Any] = {}
        self._active: set[tuple[int, str, tuple[Any, ...]]] = set()
        self._expr_code: dict[str, Any] = {}
        self._impl: dict[str, Callable[..., Any]] = {}
        self._executed_reductions: set[str] = set()
        self._executed_scalars: set[str] = set()
        self._external_call_names: set[str] = set()
        self._env = self._build_environment()
        self._compile_functions()

    # ---- public -----------------------------------------------------------------

    def run(self) -> StagedPythonExecutionResult:
        base_range = self._shared_reduction_range()
        if base_range.step == 0:
            raise StagedPythonError("zero coordinate step")
        if base_range.step < 0:
            raise StagedPythonError(
                "first two-stage executor currently requires an ascending primary range"
            )

        extension = self._required_positive_extension()
        if extension % base_range.step != 0:
            raise StagedPythonError(
                f"cross-stage coordinate extension {extension} is not aligned to step {base_range.step}"
            )
        extended_stop = base_range.stop + extension

        self._stage = 0
        for t in range(base_range.start, extended_stop, base_range.step):
            for uid in sorted(self.persistent_stage):
                if self.persistent_stage[uid] == 0:
                    self._call(uid, (t,))

        # Compute every scalar/reduction assigned to stage 0 while ordinary
        # non-materialized persistent state is still available.
        for uid in self._ordered_noncoordinate_uids(0):
            self._call(uid, ())

        self._materialized = {
            uid: dict(self._persistent_cache[uid]) for uid in sorted(self.materialized_uids)
        }
        if self.physical_values:
            for uid in sorted(self.materialized_uids):
                slot = int(self.physical_values[uid].materialized_slot)
                self._materialized_slots[slot] = self._materialized[uid]
        self._stage0_closed = True

        self._stage = 1
        for uid in self._ordered_noncoordinate_uids(1):
            self._call(uid, ())
        output = self._call(self.plan.output_uid, ())

        return StagedPythonExecutionResult(
            output=output,
            stage_indices=tuple(stage.index for stage in self.plan.stages),
            stage_program_uid=self.stage_program_uid,
            coordinate_range=(base_range.start, base_range.stop, base_range.step),
            stage0_extension=extension,
            stage_storage_plan_uid=self.storage_plan.uid,
            materialized_uids=tuple(sorted(self.materialized_uids)),
            materialized_lengths=tuple(
                (uid, len(self._materialized.get(uid, {})))
                for uid in sorted(self.materialized_uids)
            ),
            external_value_uids=tuple(sorted(self.external_value_uids)),
            external_call_names=tuple(sorted(self._external_call_names)),
            executed_reduction_uids=tuple(sorted(self._executed_reductions)),
            executed_scalar_uids=tuple(sorted(self._executed_scalars)),
        )

    # ---- canonical dispatch -------------------------------------------------------

    def _call(self, uid: str, args: tuple[Any, ...]) -> Any:
        if uid not in self.variants:
            raise StagedPythonError(f"canonical call references unknown variant {uid}")
        cv = self.variants[uid]
        node = self.nodes[uid]

        if uid in self.external_value_uids:
            key = (self._stage, uid, args)
            if key not in self._value_cache:
                assert self.external_value is not None
                self._value_cache[key] = self.external_value(uid, args)
            return self._value_cache[key]

        if cv.role == "reduction":
            return self._reduction(uid)
        if cv.role == "scalar":
            return self._scalar(uid)
        if cv.role == "vector":
            raise StagedPythonError(f"vector value is outside the first staged executor: {uid}")
        if cv.role != "coordinate" or len(args) != 1:
            raise StagedPythonError(f"unsupported canonical call shape: {uid}{args!r}")

        q = int(args[0])
        if node.state_semantic == "persistent_state":
            return self._persistent(uid, q)
        return self._coordinate_value(uid, q)

    def _persistent(self, uid: str, q: int) -> Any:
        producer = self.persistent_stage[uid]
        if self._stage < producer:
            raise StagedPythonError(
                f"persistent value {uid} requested before producing stage {producer}"
            )
        if self._stage > producer:
            if uid not in self.materialized_uids:
                raise StagedPythonError(
                    f"stage {self._stage} attempted non-materialized persistent read {uid}({q})"
                )
            try:
                if self.physical_values and getattr(self.physical_values[uid], "materialized_slot", None) is not None:
                    slot = int(self.physical_values[uid].materialized_slot)
                    history = self._materialized_slots[slot]
                    if history is None:
                        raise KeyError(q)
                    return history[q]
                return self._materialized[uid][q]
            except KeyError as exc:
                raise StagedPythonError(
                    f"materialized history {uid} has no completed-stage coordinate {q}"
                ) from exc

        cache = self._persistent_cache[uid]
        if q in cache:
            return cache[q]
        if self._stage0_closed:
            raise StagedPythonError(
                f"persistent stage is closed; late computation requested for {uid}({q})"
            )
        key = (self._stage, uid, (q,))
        if key in self._active:
            raise StagedPythonError(
                f"same-coordinate persistent recursion is not ordered by the current generic subset: {uid}({q})"
            )
        self._active.add(key)
        try:
            value = self._invoke_function(uid, (q,))
            cache[q] = value
            return value
        finally:
            self._active.remove(key)

    def _coordinate_value(self, uid: str, q: int) -> Any:
        stages = self.task_stages.get(uid, ())
        if stages and self._stage not in stages:
            raise StagedPythonError(
                f"coordinate value {uid} is not placed in semantic stage {self._stage}; placed={stages!r}"
            )
        key = (self._stage, uid, (q,))
        if key in self._value_cache:
            return self._value_cache[key]
        if key in self._active:
            raise StagedPythonError(f"nonpersistent coordinate recursion cycle: {uid}({q})")
        self._active.add(key)
        try:
            value = self._invoke_function(uid, (q,))
            self._value_cache[key] = value
            return value
        finally:
            self._active.remove(key)

    def _scalar(self, uid: str) -> Any:
        if uid in self._scalar_cache:
            return self._scalar_cache[uid]
        producer = self._single_stage(uid)
        physical = self.physical_values.get(uid)
        if self._stage > producer:
            if physical is not None and getattr(physical, "scalar_slot", None) is not None:
                value = self._stage_scalar_slots[int(physical.scalar_slot)]
                if value is not None:
                    self._scalar_cache[uid] = value
                    return value
            raise StagedPythonError(
                f"stage-{producer} scalar {uid} was not computed before stage {self._stage}"
            )
        if self._stage < producer:
            raise StagedPythonError(f"scalar {uid} requested before stage {producer}")
        value = self._invoke_function(uid, ())
        self._scalar_cache[uid] = value
        if physical is not None and getattr(physical, "scalar_slot", None) is not None:
            self._stage_scalar_slots[int(physical.scalar_slot)] = value
        self._executed_scalars.add(uid)
        return value

    def _reduction(self, uid: str) -> Any:
        if uid in self._reduction_cache:
            return self._reduction_cache[uid]
        producer = self._single_stage(uid)
        physical = self.physical_values.get(uid)
        if self._stage > producer:
            if physical is not None and getattr(physical, "scalar_slot", None) is not None:
                value = self._stage_scalar_slots[int(physical.scalar_slot)]
                if value is not None:
                    self._reduction_cache[uid] = value
                    return value
            raise StagedPythonError(
                f"stage-{producer} reduction {uid} was not completed before stage {self._stage}"
            )
        if self._stage < producer:
            raise StagedPythonError(f"reduction {uid} requested before stage {producer}")
        spec = self.variants[uid].reduction
        if spec is None:
            raise StagedPythonError(f"reduction variant {uid} has no ReductionSpec")
        local: dict[str, Any] = {}
        acc = self._eval_expr(spec.init, local)
        range_args = tuple(int(self._eval_expr(expr, local)) for expr in spec.range_args)
        for q in range(*range_args):
            local[spec.loop_var] = q
            local[spec.target] = acc
            if all(bool(self._eval_expr(filt, local)) for filt in spec.filters):
                acc = acc + self._eval_expr(spec.body_expr, local)
        self._reduction_cache[uid] = acc
        if physical is not None and getattr(physical, "scalar_slot", None) is not None:
            self._stage_scalar_slots[int(physical.scalar_slot)] = acc
        self._executed_reductions.add(uid)
        return acc

    # ---- stage geometry ----------------------------------------------------------

    def _single_stage(self, uid: str) -> int:
        stages = self.task_stages.get(uid, ())
        if len(stages) != 1:
            raise StagedPythonError(f"noncoordinate value {uid} lacks one execution stage: {stages!r}")
        return int(stages[0])

    def _ordered_noncoordinate_uids(self, stage: int) -> tuple[str, ...]:
        rows = [
            uid for uid, cv in self.variants.items()
            if cv.role in {"scalar", "reduction"}
            and self.task_stages.get(uid) == (stage,)
        ]
        # Recursive evaluation supplies the real dependency order.  Sorting keeps
        # audit/reproduction deterministic without turning names into semantics.
        return tuple(sorted(rows))

    def _shared_reduction_range(self) -> range:
        ranges: list[range] = []
        old_stage = self._stage
        self._stage = 0
        try:
            for uid, cv in self.variants.items():
                if cv.role != "reduction" or not self.task_stages.get(uid):
                    continue
                spec = cv.reduction
                if spec is None:
                    continue
                args = tuple(int(self._eval_expr(expr, {})) for expr in spec.range_args)
                ranges.append(range(*args))
        finally:
            self._stage = old_stage
        if not ranges:
            raise StagedPythonError("two-stage plan has no normalized reduction range")
        sig = (ranges[0].start, ranges[0].stop, ranges[0].step)
        if any((r.start, r.stop, r.step) != sig for r in ranges[1:]):
            raise StagedPythonError(
                "first generic two-stage executor requires one shared exact normalized range"
            )
        return ranges[0]

    def _required_positive_extension(self) -> int:
        max_extension = 0
        for uid in self.materialized_uids:
            row = self.storage_plan.value(uid)
            if row.retained_offset_max is None or row.retained_offset_min is None:
                raise StagedPythonError(
                    f"cross-stage materialized storage has no finite coordinate envelope: {uid}"
                )
            if int(row.retained_offset_min) < 0:
                raise StagedPythonError(
                    "first staged Python executor does not yet extend the primary range below its start: "
                    f"{uid} requires {row.retained_offset_min}"
                )
            max_extension = max(max_extension, int(row.retained_offset_max), 0)
        return max_extension

    # ---- formula environment -----------------------------------------------------

    def _build_environment(self) -> dict[str, Any]:
        env: dict[str, Any] = {
            "np": np,
            "math": math,
            "array_input": self._array_input,
            "table_input": self._table_input,
            "global_input": self._global_input,
            "point_input": self._point_input,
        }
        for uid in self.variants:
            env[uid] = self._make_uid_wrapper(uid)

        known = set(self.variants)
        external_names: set[str] = set()
        for cv in self.variants.values():
            roots: list[ast.AST] = []
            if cv.function is not None:
                roots.append(cv.function)
            if cv.reduction is not None:
                roots.extend([
                    cv.reduction.init,
                    cv.reduction.body_expr,
                    *cv.reduction.filters,
                    *cv.reduction.range_args,
                ])
            for root in roots:
                for node in ast.walk(root):
                    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                        continue
                    name = node.func.id
                    if name in known or name in self._BUILTIN_CALL_NAMES or name in env:
                        continue
                    external_names.add(name)
        for name in external_names:
            env[name] = self._make_external_wrapper(name)
        return env

    def _compile_functions(self) -> None:
        for uid, cv in self.variants.items():
            if cv.function is None:
                continue
            fn = copy.deepcopy(cv.function)
            impl_name = f"_stage_impl_{uid}"
            fn.name = impl_name
            module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
            local: dict[str, Any] = {}
            exec(compile(module, f"<stage:{uid}>", "exec"), self._env, local)
            self._impl[uid] = local[impl_name]

    def _make_uid_wrapper(self, uid: str) -> Callable[..., Any]:
        def wrapped(*args: Any) -> Any:
            return self._call(uid, tuple(args))
        return wrapped

    def _make_external_wrapper(self, name: str) -> Callable[..., Any]:
        def wrapped(*args: Any) -> Any:
            self._external_call_names.add(name)
            if self.external_call is None:
                raise StagedPythonError(
                    f"canonical formula requires unresolved external call {name!r}"
                )
            return self.external_call(name, tuple(args))
        return wrapped

    def _invoke_function(self, uid: str, args: tuple[Any, ...]) -> Any:
        try:
            fn = self._impl[uid]
        except KeyError as exc:
            raise StagedPythonError(f"canonical value {uid} has no executable function") from exc
        return fn(*args)

    def _eval_expr(self, expr: ast.AST, local: Mapping[str, Any]) -> Any:
        sig = ast.dump(expr, include_attributes=False)
        code = self._expr_code.get(sig)
        if code is None:
            code = compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(expr))), "<stage-expr>", "eval")
            self._expr_code[sig] = code
        return eval(code, self._env, dict(local))

    # ---- normalized inputs -------------------------------------------------------

    def _input(self, key: str) -> Any:
        if key not in self.inputs:
            raise StagedPythonError(f"normalized input {key!r} was not supplied")
        return self.inputs[key]

    def _global_input(self, key: str) -> Any:
        return self._input(key)

    def _array_input(self, key: str, index: Any) -> Any:
        return self._input(key)[int(index)]

    def _table_input(self, key: str, *indices: Any) -> Any:
        return self._input(key)[tuple(int(x) for x in indices)]

    def _point_input(self, key: str) -> Any:
        value = self._input(key)
        arr = np.asarray(value)
        if arr.ndim == 0:
            return value
        if len(arr) != 1:
            raise StagedPythonError(
                f"single-point staged execution expected one row for point input {key!r}"
            )
        return arr[0]
