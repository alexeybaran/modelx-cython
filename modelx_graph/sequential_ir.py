from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .realized_trace import ConcreteNode, RealizedTrace, RealizedTraceError
from .realized_trace import clear_realized_calculated


def _value_equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        if hasattr(a, "equals") and callable(a.equals):
            return bool(a.equals(b))
    except Exception:
        pass
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))
        except Exception:
            return False
    try:
        v = a == b
    except Exception:
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return False


class SequentialError(RealizedTraceError):
    pass


@dataclass(frozen=True)
class EvalOp:
    node_id: int


@dataclass
class ReplayValidation:
    ok: bool
    expected_node_ids: tuple[int, ...]
    observed_node_ids: tuple[int, ...]
    unexpected_runtime_nodes: tuple[Any, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass
class SequentialProgram:
    trace: RealizedTrace = field(repr=False)
    ops: tuple[EvalOp, ...]
    _node_by_id: dict[int, ConcreteNode] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._node_by_id = {n.node_id: n for n in self.trace.nodes}

    @property
    def nodes(self) -> tuple[ConcreteNode, ...]:
        return self.trace.nodes

    @property
    def node_by_id(self) -> dict[int, ConcreteNode]:
        return self._node_by_id

    @property
    def operation_count(self) -> int:
        return len(self.ops)

    def runtime_nodes(self) -> tuple[tuple[Any, tuple[Any, ...]], ...]:
        by_id = self.node_by_id
        return tuple(by_id[op.node_id].runtime_node for op in self.ops)

    def execute(self, *, clear_first: bool = True, validate: bool = False, direct: bool = True):
        if clear_first:
            if direct:
                self._clear_scheduled_values()
            else:
                clear_realized_calculated(self.trace.model)

        system = self.trace.model._impl.system
        raw_events = None
        if validate:
            with system.trace_stack(maxlen=None):
                self._execute_ops(direct=direct)
                result = self._read_targets()
                raw_events = list(system.callstack.tracestack)
        else:
            self._execute_ops(direct=direct)
            result = self._read_targets()

        if validate:
            validation = self._validate_replay(raw_events or [], direct=direct, result=result)
            if not validation.ok:
                raise SequentialError("sequential replay diverged from realized schedule: " + "; ".join(validation.reasons))
            return result, validation
        return result


    def _clear_scheduled_values(self) -> None:
        """Clear only values calculated by this exact realized schedule.

        This is the realized-trace analogue of modelx ``clear`` actions.  It keeps
        the already-captured tracegraph intact and avoids a model-wide descendant
        traversal before every replay.  It is safe only for this exact schedule:
        explicit inputs are never removed, and a scheduled key that has since been
        converted to an input invalidates the artifact.
        """
        by_id = self.node_by_id
        for op in self.ops:
            node = by_id[op.node_id]
            try:
                if node.args in node.obj.input_keys:
                    raise SequentialError(
                        f"scheduled formula changed Cell value into explicit input: {node.uid}"
                    )
            except TypeError:
                # Non-hashable keys cannot be valid modelx Cell cache keys, but keep
                # the failure diagnostic local if an exotic implementation appears.
                raise SequentialError(f"unusable realized Cell key for {node.uid}: {node.args!r}")
            if node.args in node.obj.data:
                del node.obj.data[node.args]

    def _install_realized_graph(self) -> None:
        """Restore the captured dependency graph after modelx cache clearing.

        ``clear_all_values`` removes graph nodes/edges together with cached values.
        Direct replay bypasses modelx's executor, so it would not rebuild those
        edges itself.  Bulk NetworkX insertion is materially cheaper than one
        Python ``add_edge`` call per dependency on large traces.
        """
        graph = self.trace.model._impl.tracegraph
        by_id = self.node_by_id
        graph.add_nodes_from(node.runtime_node for node in self.trace.nodes)
        graph.add_edges_from(
            (by_id[src].runtime_node, by_id[dst].runtime_node)
            for src, dst in self.trace.dependencies
        )

    def _execute_ops(self, *, direct: bool) -> None:
        by_id = self.node_by_id
        for op in self.ops:
            node = by_id[op.node_id]
            if direct:
                if node.args in node.obj.input_keys:
                    raise SequentialError(
                        f"scheduled formula changed Cell value into explicit input: {node.uid}"
                    )
                node.obj.on_eval_formula(node.args)
            else:
                node.obj.get_value_from_key(node.args)

    def _read_targets(self):
        vals = tuple(obj.get_value_from_key(key) for obj, key in self.trace.target_runtime_nodes)
        return vals[0] if len(vals) == 1 else vals

    def _validate_replay(self, raw_events, *, direct: bool, result: Any) -> ReplayValidation:
        runtime_to_id = self.trace.runtime_to_id
        observed: list[int] = []
        unexpected: list[Any] = []
        cached_formula_events: list[Any] = []
        for sign, _depth, _stamp, item in raw_events:
            if sign != "EXIT" or not item[0].is_cached:
                continue
            runtime = (item[0], tuple(item[1]))
            cached_formula_events.append(runtime)
            nid = runtime_to_id.get(runtime)
            if nid is None:
                unexpected.append(runtime)
            else:
                observed.append(nid)

        expected = tuple(op.node_id for op in self.ops)
        reasons: list[str] = []
        if direct:
            # Scheduled formulas bypass modelx's executor deliberately. Any cached
            # formula evaluation visible here therefore came from an unexpected
            # nested cache miss and proves the flat schedule was insufficient.
            if cached_formula_events:
                reasons.append(f"direct replay triggered {len(cached_formula_events)} nested cached formula evaluations")
            for op in self.ops:
                node = self.node_by_id[op.node_id]
                try:
                    if node.args in node.obj.input_keys:
                        reasons.append(f"scheduled formula changed Cell value into explicit input: {node.uid}")
                        break
                except Exception:
                    pass
        else:
            if tuple(observed) != expected:
                reasons.append(f"cached EXIT order mismatch expected={expected!r} observed={tuple(observed)!r}")
            if unexpected:
                reasons.append(f"unexpected cached nodes evaluated: {len(unexpected)}")

        actual_targets = (result,) if len(self.trace.target_values) == 1 else tuple(result)
        if len(actual_targets) != len(self.trace.target_values) or not all(
            _value_equal(a, b) for a, b in zip(actual_targets, self.trace.target_values)
        ):
            reasons.append("target result differs from the captured realized result")

        return ReplayValidation(
            ok=not reasons,
            expected_node_ids=expected,
            observed_node_ids=tuple(observed),
            unexpected_runtime_nodes=tuple(unexpected),
            reasons=tuple(reasons),
        )


def build_sequential_program(trace: RealizedTrace) -> SequentialProgram:
    ids = set(trace.node_by_id)
    missing = [nid for nid in trace.schedule_node_ids if nid not in ids]
    if missing:
        raise SequentialError(f"schedule references non-Cells trace nodes: {missing!r}")

    positions = {nid: i for i, nid in enumerate(trace.schedule_node_ids)}
    for src, dst in trace.dependencies:
        if src in positions and dst in positions and positions[src] >= positions[dst]:
            raise SequentialError(f"dependency order violation {src} -> {dst}")

    return SequentialProgram(
        trace=trace,
        ops=tuple(EvalOp(nid) for nid in trace.schedule_node_ids),
    )
