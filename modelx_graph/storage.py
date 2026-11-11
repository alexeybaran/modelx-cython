from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import sys
from typing import Any

from .loop_recovery import LoopRecoveryError, StructuredSequentialProgram
from .sequential_ir import ReplayValidation, SequentialError


class StoragePlanError(SequentialError):
    """Raised when exact realized liveness/storage planning is inconsistent."""


@dataclass(frozen=True)
class NodeLifetime:
    node_id: int
    formula_op_id: int
    position: int
    last_use_position: int
    use_count: int
    shallow_value_bytes: int = 0

    @property
    def future_span(self) -> int:
        return self.last_use_position - self.position

    @property
    def pinned_to_result(self) -> bool:
        return self.last_use_position < 0


@dataclass(frozen=True)
class FormulaStorageSpec:
    formula_op_id: int
    name: str
    fullname: str
    produced_count: int
    dependency_use_count: int
    target_value_count: int
    immediate_value_count: int
    max_live_values: int
    max_future_span_ops: int
    baseline_shallow_value_bytes: int
    peak_live_shallow_value_bytes: int
    ring_compatible: bool
    storage_class: str

    @property
    def required_slots(self) -> int:
        return self.max_live_values

    @property
    def exact_slot_reduction(self) -> float:
        if not self.max_live_values:
            return 1.0
        return self.produced_count / self.max_live_values


@dataclass(frozen=True)
class StorageExecutionStats:
    operation_count: int
    peak_live_scheduled_values: int
    final_live_scheduled_values: int
    released_values: int
    planned_peak_live_values: int
    baseline_scheduled_values: int


@dataclass
class StoragePlan:
    """Exact last-use plan for one realized canonical execution schedule.

    The plan makes no claim about unseen arguments.  It proves only how many
    concrete cached values must coexist for the captured execution.  A bounded
    per-FormulaOp maximum is therefore an exact ring-capacity proof for a later
    generated backend, while the Python executor can implement the same semantics
    conservatively by evicting dead keys from the ordinary modelx cache.
    """

    structured: StructuredSequentialProgram = field(repr=False)
    lifetimes: tuple[NodeLifetime, ...]
    release_after: tuple[tuple[int, ...], ...]
    retained_node_ids: tuple[int, ...]
    formula_specs: tuple[FormulaStorageSpec, ...]
    peak_live_values: int
    baseline_scheduled_values: int
    baseline_shallow_value_bytes: int
    peak_live_shallow_value_bytes: int

    def __post_init__(self) -> None:
        self._lifetime_by_id = {x.node_id: x for x in self.lifetimes}
        self._formula_by_id = {x.formula_op_id: x for x in self.formula_specs}

    @property
    def lifetime_by_id(self) -> dict[int, NodeLifetime]:
        return self._lifetime_by_id

    @property
    def formula_by_id(self) -> dict[int, FormulaStorageSpec]:
        return self._formula_by_id

    @property
    def exact_cache_entry_reduction(self) -> float:
        if not self.peak_live_values:
            return 1.0
        return self.baseline_scheduled_values / self.peak_live_values

    @property
    def exact_shallow_value_reduction(self) -> float:
        if not self.peak_live_shallow_value_bytes:
            return 1.0
        return self.baseline_shallow_value_bytes / self.peak_live_shallow_value_bytes

    def storage_class_counts(self) -> Counter[str]:
        return Counter(x.storage_class for x in self.formula_specs)

    def validate(self) -> None:
        seq = self.structured.sequential
        expected = tuple(op.node_id for op in seq.ops)
        actual = tuple(x.node_id for x in self.lifetimes)
        if actual != expected:
            raise StoragePlanError("storage lifetime order differs from sequential schedule")
        if len(self.release_after) != len(expected):
            raise StoragePlanError("storage release schedule length mismatch")
        released = {nid for group in self.release_after for nid in group}
        retained = set(self.retained_node_ids)
        if released & retained:
            raise StoragePlanError("storage node is both released and retained")
        if released | retained != set(expected):
            raise StoragePlanError("storage release/retain partition is incomplete")


@dataclass
class StorageOptimizedProgram:
    structured: StructuredSequentialProgram = field(repr=False)
    plan: StoragePlan

    def execute(self, *, clear_first: bool = True, validate: bool = False, direct: bool = True):
        # recover_loops() already proved exact canonical expansion.  StoragePlan
        # validation plus the operation-by-operation runtime-node check below is
        # sufficient here and avoids rematerializing the full concrete schedule
        # on every replay.
        self.plan.validate()
        seq = self.structured.sequential
        if clear_first:
            if direct:
                seq._clear_scheduled_values()
            else:
                from .realized_trace import clear_realized_calculated
                clear_realized_calculated(seq.trace.model)

        system = seq.trace.model._impl.system
        raw_events = None
        if validate:
            with system.trace_stack(maxlen=None):
                stats = self._execute_and_release(direct=direct)
                result = seq._read_targets()
                raw_events = list(system.callstack.tracestack)
        else:
            stats = self._execute_and_release(direct=direct)
            result = seq._read_targets()

        if validate:
            replay = seq._validate_replay(raw_events or [], direct=direct, result=result)
            if not replay.ok:
                raise StoragePlanError(
                    "storage-optimized replay diverged from realized schedule: "
                    + "; ".join(replay.reasons)
                )
            return result, replay, stats
        return result, stats

    def _execute_and_release(self, *, direct: bool) -> StorageExecutionStats:
        if not direct:
            raise StoragePlanError("Stage-3a storage replay currently requires direct=True")

        seq = self.structured.sequential
        by_id = seq.node_by_id
        expected_ops = seq.ops
        lifetime_by_id = self.plan.lifetime_by_id
        live = 0
        peak = 0
        released = 0
        count = 0

        for pos, runtime_node in enumerate(self.structured.iter_canonical_runtime_nodes()):
            if pos >= len(expected_ops):
                raise StoragePlanError("canonical executor produced extra operation")
            current_id = expected_ops[pos].node_id
            expected = by_id[current_id]
            if runtime_node[0] is not expected.obj or runtime_node[1] != expected.args:
                raise StoragePlanError(f"canonical runtime node differs at operation {pos}")

            # During formula evaluation all predecessors whose last use is this
            # operation must still be present. The current value does not exist
            # yet, so this is the true pre-store cache peak.
            if live > peak:
                peak = live
            obj, key = runtime_node
            if key in obj.input_keys:
                raise StoragePlanError("scheduled formula changed Cell value into explicit input")
            value = obj.altfunc(*key)

            # Formula body has returned, so every predecessor with final consumer
            # at this operation can be released before the new value is stored.
            for nid in self.plan.release_after[pos]:
                if nid == current_id:
                    continue
                node = by_id[nid]
                try:
                    del node.obj.data[node.args]
                except KeyError as exc:
                    raise StoragePlanError(
                        f"planned live value missing before release at operation {pos}: {node.uid}"
                    ) from exc
                live -= 1
                released += 1

            # Use modelx's own cache-store helper so allow_none/error semantics are
            # exactly the same as CellsImpl.on_eval_formula().
            obj._store_value(key, value)
            live += 1
            count += 1
            if live > peak:
                peak = live

            # A current value with no future consumer is validated/stored once and
            # can then be removed immediately. Target values are retained by plan.
            lt = lifetime_by_id[current_id]
            if lt.last_use_position == pos:
                try:
                    del obj.data[key]
                except KeyError as exc:
                    raise StoragePlanError("newly stored immediate value is missing") from exc
                live -= 1
                released += 1

        if count != len(expected_ops):
            raise StoragePlanError(
                f"canonical executor produced {count} operations, expected {len(expected_ops)}"
            )
        if peak != self.plan.peak_live_values:
            raise StoragePlanError(
                f"runtime live-value peak {peak} differs from plan {self.plan.peak_live_values}"
            )
        if live != len(self.plan.retained_node_ids):
            raise StoragePlanError(
                f"runtime final live count {live} differs from plan {len(self.plan.retained_node_ids)}"
            )

        return StorageExecutionStats(
            operation_count=count,
            peak_live_scheduled_values=peak,
            final_live_scheduled_values=live,
            released_values=released,
            planned_peak_live_values=self.plan.peak_live_values,
            baseline_scheduled_values=self.plan.baseline_scheduled_values,
        )


def _shallow_value_size(node: Any) -> int:
    try:
        if node.args not in node.obj.data:
            return 0
        return int(sys.getsizeof(node.obj.data[node.args]))
    except Exception:
        return 0


def build_storage_plan(structured: StructuredSequentialProgram) -> StoragePlan:
    """Build an exact realized last-use/liveness plan.

    Dependencies are read from the authoritative realized graph.  No parameter
    name or argument position is treated as time.  A value is released only after
    its final realized consumer has finished.  Target values are retained so the
    normal modelx target read remains a cache hit.
    """

    # StructuredSequentialProgram produced by recover_loops() has already passed
    # exact expansion validation.  Repeating that O(N) materialization here would
    # make storage planning needlessly expensive on large realized traces.
    if structured.canonical_plan is None:
        raise StoragePlanError("canonical execution plan is required for Stage 3a storage planning")

    seq = structured.sequential
    trace = seq.trace
    ops = seq.ops
    nops = len(ops)
    positions = {op.node_id: i for i, op in enumerate(ops)}
    if len(positions) != nops:
        raise StoragePlanError("sequential schedule contains duplicate concrete node IDs")

    plan = structured.canonical_plan
    role_to_formula = {op.role: op.op_id for op in plan.formula_ops}
    formula_meta = {op.op_id: op for op in plan.formula_ops}
    by_id = seq.node_by_id

    last_use = list(range(nops))
    use_count = [0] * nops
    unknown_consumers: list[tuple[int, int]] = []
    for src, dst in trace.dependencies:
        src_pos = positions.get(src)
        if src_pos is None:
            continue
        dst_pos = positions.get(dst)
        if dst_pos is None:
            unknown_consumers.append((src, dst))
            continue
        if dst_pos <= src_pos:
            raise StoragePlanError(f"dependency order violation while planning storage: {src} -> {dst}")
        if dst_pos > last_use[src_pos]:
            last_use[src_pos] = dst_pos
        use_count[src_pos] += 1

    # Under the target-ancestor closure used by realized tracing this should be
    # empty.  If a future modelx graph contains an external consumer, retain the
    # source through result read rather than making an unsafe release decision.
    unknown_src = {src for src, _dst in unknown_consumers}

    target_ids = {tid for tid in trace.target_node_ids if tid is not None and tid in positions}
    runtime_to_id = trace.runtime_to_id
    for runtime in trace.target_runtime_nodes:
        tid = runtime_to_id.get(runtime)
        if tid is not None and tid in positions:
            target_ids.add(tid)

    retained = set(target_ids) | unknown_src
    for nid in retained:
        last_use[positions[nid]] = nops

    release_after: list[list[int]] = [[] for _ in range(nops)]
    for pos, op in enumerate(ops):
        if last_use[pos] < nops:
            release_after[last_use[pos]].append(op.node_id)

    formula_id_for_pos: list[int] = []
    sizes: list[int] = []
    for op in ops:
        node = by_id[op.node_id]
        try:
            fid = role_to_formula[node.shape_token]
        except KeyError as exc:
            raise StoragePlanError(f"formula registry missing realized role {node.shape_token!r}") from exc
        formula_id_for_pos.append(fid)
        sizes.append(_shallow_value_size(node))

    # Exact global and per-formula resident-set sweep for generated/direct
    # sequential execution. A predecessor whose final consumer is operation i is
    # released after that formula body returns but before result i is stored.
    live_total = 0
    live_bytes = 0
    peak_total = 0
    peak_bytes = 0
    live_by_formula: Counter[int] = Counter()
    peak_by_formula: Counter[int] = Counter()
    live_bytes_by_formula: Counter[int] = Counter()
    peak_bytes_by_formula: Counter[int] = Counter()

    node_formula = {ops[i].node_id: formula_id_for_pos[i] for i in range(nops)}
    node_size = {ops[i].node_id: sizes[i] for i in range(nops)}
    for pos, op in enumerate(ops):
        # Pre-store peak while the formula reads all currently live predecessors.
        peak_total = max(peak_total, live_total)
        peak_bytes = max(peak_bytes, live_bytes)

        # Release prior values whose final consumer is the current formula.
        for nid in release_after[pos]:
            if nid == op.node_id:
                continue
            rfid = node_formula[nid]
            rsize = node_size[nid]
            live_total -= 1
            live_bytes -= rsize
            live_by_formula[rfid] -= 1
            live_bytes_by_formula[rfid] -= rsize

        # Store the new current value.
        fid = formula_id_for_pos[pos]
        size = sizes[pos]
        live_total += 1
        live_bytes += size
        live_by_formula[fid] += 1
        live_bytes_by_formula[fid] += size
        peak_total = max(peak_total, live_total)
        peak_bytes = max(peak_bytes, live_bytes)
        if live_by_formula[fid] > peak_by_formula[fid]:
            peak_by_formula[fid] = live_by_formula[fid]
        if live_bytes_by_formula[fid] > peak_bytes_by_formula[fid]:
            peak_bytes_by_formula[fid] = live_bytes_by_formula[fid]

        # If the current value itself has no future consumer, remove it now.
        if last_use[pos] == pos:
            live_total -= 1
            live_bytes -= size
            live_by_formula[fid] -= 1
            live_bytes_by_formula[fid] -= size

    produced: Counter[int] = Counter(formula_id_for_pos)
    dep_uses: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    immediate: Counter[int] = Counter()
    max_span: Counter[int] = Counter()
    base_bytes: Counter[int] = Counter()

    lifetime_rows: list[NodeLifetime] = []
    for pos, op in enumerate(ops):
        fid = formula_id_for_pos[pos]
        nid = op.node_id
        span = last_use[pos] - pos
        dep_uses[fid] += use_count[pos]
        if nid in target_ids:
            target_counts[fid] += 1
        if last_use[pos] == pos:
            immediate[fid] += 1
        if span > max_span[fid]:
            max_span[fid] = span
        base_bytes[fid] += sizes[pos]
        lifetime_rows.append(
            NodeLifetime(
                node_id=nid,
                formula_op_id=fid,
                position=pos,
                last_use_position=(-1 if last_use[pos] == nops else last_use[pos]),
                use_count=use_count[pos],
                shallow_value_bytes=sizes[pos],
            )
        )

    positions_by_formula: dict[int, list[int]] = defaultdict(list)
    for pos, fid in enumerate(formula_id_for_pos):
        positions_by_formula[fid].append(pos)

    specs: list[FormulaStorageSpec] = []
    for fid in sorted(produced):
        count = produced[fid]
        max_live = peak_by_formula[fid]
        occurrence_positions = positions_by_formula[fid]
        ring_compatible = True
        if max_live > 0 and max_live < count:
            # A cyclic occurrence-indexed ring of capacity max_live is valid when
            # the value whose slot is reused has completed its final consumer by
            # the time the new occurrence finishes evaluation and is stored.
            for j in range(max_live, count):
                prior_pos = occurrence_positions[j - max_live]
                current_pos = occurrence_positions[j]
                if last_use[prior_pos] > current_pos:
                    ring_compatible = False
                    break
        if count == 1:
            storage_class = "scalar"
        elif immediate[fid] == count:
            storage_class = "immediate"
        elif max_live < count and ring_compatible:
            storage_class = "bounded_ring"
        elif max_live < count:
            storage_class = "bounded_slots"
        else:
            storage_class = "full_history"
        meta = formula_meta[fid]
        specs.append(
            FormulaStorageSpec(
                formula_op_id=fid,
                name=meta.name,
                fullname=meta.fullname,
                produced_count=count,
                dependency_use_count=dep_uses[fid],
                target_value_count=target_counts[fid],
                immediate_value_count=immediate[fid],
                max_live_values=max_live,
                max_future_span_ops=max_span[fid],
                baseline_shallow_value_bytes=base_bytes[fid],
                peak_live_shallow_value_bytes=peak_bytes_by_formula[fid],
                ring_compatible=ring_compatible,
                storage_class=storage_class,
            )
        )

    result = StoragePlan(
        structured=structured,
        lifetimes=tuple(lifetime_rows),
        release_after=tuple(tuple(x) for x in release_after),
        retained_node_ids=tuple(sorted(retained, key=positions.__getitem__)),
        formula_specs=tuple(specs),
        peak_live_values=peak_total,
        baseline_scheduled_values=nops,
        baseline_shallow_value_bytes=sum(sizes),
        peak_live_shallow_value_bytes=peak_bytes,
    )
    result.validate()
    return result


def optimize_storage(structured: StructuredSequentialProgram) -> StorageOptimizedProgram:
    return StorageOptimizedProgram(structured=structured, plan=build_storage_plan(structured))
