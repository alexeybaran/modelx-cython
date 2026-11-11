from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .sequential_ir import EvalOp, SequentialError, SequentialProgram
from .realized_trace import clear_realized_calculated


class LoopRecoveryError(SequentialError):
    pass


def _safe_equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        v = a == b
    except Exception:
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return False


@dataclass(frozen=True)
class ConstantBinding:
    value: Any = field(compare=False)

    def value_at(self, iteration: int) -> Any:
        return self.value

    @property
    def kind(self) -> str:
        return "constant"


@dataclass(frozen=True)
class AffineIntBinding:
    base: int
    step: int

    def value_at(self, iteration: int) -> int:
        return self.base + self.step * iteration

    @property
    def kind(self) -> str:
        return "affine_int"


@dataclass(frozen=True)
class TableBinding:
    values: tuple[Any, ...] = field(compare=False)

    def value_at(self, iteration: int) -> Any:
        return self.values[iteration]

    @property
    def kind(self) -> str:
        return "table"


Binding = ConstantBinding | AffineIntBinding | TableBinding


def infer_binding(values: Sequence[Any]) -> Binding:
    if not values:
        raise LoopRecoveryError("cannot bind an empty value sequence")
    first = values[0]
    if all(_safe_equal(first, v) for v in values[1:]):
        return ConstantBinding(first)
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in values):
        ints = [int(v) for v in values]
        if len(ints) == 1:
            return ConstantBinding(ints[0])
        step = ints[1] - ints[0]
        if all(ints[i] == ints[0] + step * i for i in range(len(ints))):
            return AffineIntBinding(ints[0], step)
    return TableBinding(tuple(values))


@dataclass(frozen=True)
class EvalTemplate:
    schema_uid: str
    space_family_uid: str
    obj_binding: Binding = field(compare=False)
    arg_bindings: tuple[Binding, ...] = field(compare=False)

    def runtime_node(self, iteration: int) -> tuple[Any, tuple[Any, ...]]:
        obj = self.obj_binding.value_at(iteration)
        key = tuple(binding.value_at(iteration) for binding in self.arg_bindings)
        return obj, key


@dataclass(frozen=True)
class LiteralBlock:
    ops: tuple[EvalOp, ...]

    @property
    def operation_count(self) -> int:
        return len(self.ops)


@dataclass(frozen=True)
class LoopBlock:
    start: int
    body_size: int
    repetitions: int
    body: tuple[EvalTemplate, ...]
    dependency_signature: tuple[Any, ...] = field(compare=False)
    source_node_ids: tuple[int, ...] = field(compare=False, default=())

    @property
    def operation_count(self) -> int:
        return self.body_size * self.repetitions

    @property
    def saved_ops(self) -> int:
        return self.body_size * (self.repetitions - 1)

    def expand_runtime_nodes(self) -> tuple[tuple[Any, tuple[Any, ...]], ...]:
        out = []
        for i in range(self.repetitions):
            for op in self.body:
                out.append(op.runtime_node(i))
        return tuple(out)


Block = LiteralBlock | LoopBlock


@dataclass
class StructuredSequentialProgram:
    sequential: SequentialProgram = field(repr=False)
    blocks: tuple[Block, ...]

    @property
    def loop_count(self) -> int:
        return sum(isinstance(b, LoopBlock) for b in self.blocks)

    @property
    def operation_count(self) -> int:
        return sum(b.operation_count for b in self.blocks)

    @property
    def stored_body_ops(self) -> int:
        return sum(len(b.ops) if isinstance(b, LiteralBlock) else b.body_size for b in self.blocks)

    @property
    def compression_ratio(self) -> float:
        stored = self.stored_body_ops
        return self.operation_count / stored if stored else 1.0

    def expand_runtime_nodes(self) -> tuple[tuple[Any, tuple[Any, ...]], ...]:
        by_id = self.sequential.node_by_id
        out = []
        for block in self.blocks:
            if isinstance(block, LiteralBlock):
                out.extend(by_id[op.node_id].runtime_node for op in block.ops)
            else:
                out.extend(block.expand_runtime_nodes())
        return tuple(out)

    def validate_exact_expansion(self) -> None:
        expected = self.sequential.runtime_nodes()
        actual = self.expand_runtime_nodes()
        if len(expected) != len(actual):
            raise LoopRecoveryError(f"loop expansion length mismatch {len(actual)} != {len(expected)}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            if a[0] is not b[0] or a[1] != b[1]:
                raise LoopRecoveryError(f"loop expansion differs at operation {i}: {a!r} != {b!r}")

    def execute(self, *, clear_first: bool = True, validate: bool = False, direct: bool = True):
        self.validate_exact_expansion()
        if clear_first:
            if direct:
                self.sequential._clear_scheduled_values()
            else:
                clear_realized_calculated(self.sequential.trace.model)

        system = self.sequential.trace.model._impl.system
        raw_events = None
        if validate:
            with system.trace_stack(maxlen=None):
                self._execute_blocks(direct=direct)
                result = self.sequential._read_targets()
                raw_events = list(system.callstack.tracestack)
        else:
            self._execute_blocks(direct=direct)
            result = self.sequential._read_targets()

        if validate:
            replay = self.sequential._validate_replay(raw_events or [], direct=direct, result=result)
            if not replay.ok:
                raise LoopRecoveryError("structured replay diverged from realized schedule: " + "; ".join(replay.reasons))
            return result, replay
        return result

    def _execute_blocks(self, *, direct: bool) -> None:
        by_id = self.sequential.node_by_id
        for block in self.blocks:
            if isinstance(block, LiteralBlock):
                for op in block.ops:
                    node = by_id[op.node_id]
                    if direct:
                        if node.args in node.obj.input_keys:
                            raise LoopRecoveryError(
                                f"scheduled formula changed Cell value into explicit input: {node.uid}"
                            )
                        node.obj.on_eval_formula(node.args)
                    else:
                        node.obj.get_value_from_key(node.args)
            else:
                for i in range(block.repetitions):
                    for tmpl in block.body:
                        obj, key = tmpl.runtime_node(i)
                        if direct:
                            if key in obj.input_keys:
                                raise LoopRecoveryError(
                                    "scheduled loop formula changed Cell value into explicit input"
                                )
                            obj.on_eval_formula(key)
                        else:
                            obj.get_value_from_key(key)



@dataclass
class _LoopAnalysis:
    by_id: dict[int, Any]
    schedule: tuple[int, ...]
    pos_of: dict[int, int]
    incoming: dict[int, tuple[int, ...]]

    @classmethod
    def from_program(cls, program: SequentialProgram) -> "_LoopAnalysis":
        schedule = tuple(op.node_id for op in program.ops)
        incoming_lists: dict[int, list[int]] = defaultdict(list)
        for src, dst in program.trace.dependencies:
            incoming_lists[dst].append(src)
        return cls(
            by_id=program.node_by_id,
            schedule=schedule,
            pos_of={nid: i for i, nid in enumerate(schedule)},
            incoming={nid: tuple(srcs) for nid, srcs in incoming_lists.items()},
        )


@dataclass(frozen=True)
class LoopCandidate:
    start: int
    body_size: int
    repetitions: int
    dependency_signature: tuple[Any, ...] = ()

    @property
    def stop(self) -> int:
        return self.start + self.body_size * self.repetitions

    @property
    def saved_ops(self) -> int:
        return self.body_size * (self.repetitions - 1)


def _candidate_periods(
    token_ids: Sequence[int],
    *,
    max_candidates: int,
    max_gap_neighbors: int,
    max_body_size: int | None,
) -> list[int]:
    positions: dict[int, list[int]] = defaultdict(list)
    for i, token in enumerate(token_ids):
        positions[token].append(i)

    counts: Counter[int] = Counter()
    for pos in positions.values():
        n = len(pos)
        for i in range(n):
            upper = min(n, i + 1 + max_gap_neighbors)
            for j in range(i + 1, upper):
                gap = pos[j] - pos[i]
                if gap <= 0:
                    continue
                if max_body_size is not None and gap > max_body_size:
                    continue
                counts[gap] += 1

    # Shorter bodies win ties because they are more compact and tend to expose the
    # fundamental period rather than a multiple of it.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [gap for gap, _ in ranked[:max_candidates]]


def _runs_for_period(token_ids: Sequence[int], period: int, min_repetitions: int) -> list[tuple[int, int]]:
    n = len(token_ids)
    if period <= 0 or n < period * min_repetitions:
        return []
    limit = n - period
    runs = []
    i = 0
    minimum_true = period * (min_repetitions - 1)
    while i < limit:
        if token_ids[i] != token_ids[i + period]:
            i += 1
            continue
        start = i
        i += 1
        while i < limit and token_ids[i] == token_ids[i + period]:
            i += 1
        run_len = i - start
        if run_len >= minimum_true:
            reps = 1 + run_len // period
            if reps >= min_repetitions:
                runs.append((start, reps))
    return runs



def _coarse_rep_dependency_signatures(
    analysis: _LoopAnalysis, start: int, body_size: int, repetitions: int
) -> tuple[tuple[Any, ...], ...]:
    """Cheap per-repetition topology fingerprints used to split branch changes.

    Concrete values are intentionally omitted. Scheduled dependencies retain the
    source body slot and iteration delta, so a realized switch such as k-1 -> k-2
    produces a different fingerprint even when the Cell family is unchanged.
    """
    by_id = analysis.by_id
    schedule = analysis.schedule
    pos_of = analysis.pos_of
    incoming = analysis.incoming

    stop = start + body_size * repetitions
    result = []
    for rep in range(repetitions):
        rep_sig = []
        for slot in range(body_size):
            pos = start + rep * body_size + slot
            nid = schedule[pos]
            deps = []
            for src in incoming.get(nid, []):
                spos = pos_of.get(src)
                snode = by_id.get(src)
                relative = False
                if spos is not None and snode is not None:
                    rel = spos - start
                    src_rep = rel // body_size
                    src_slot = rel % body_size
                    template_nid = schedule[start + src_slot]
                    template_node = by_id[template_nid]
                    relative = snode.shape_token == template_node.shape_token
                if relative:
                    deps.append(("relative", src_slot, src_rep - rep))
                else:
                    if snode is None:
                        deps.append(("external_unknown",))
                    else:
                        deps.append((
                            "external", snode.schema_uid, snode.space_family_uid, len(snode.args)
                        ))
            deps.sort()
            rep_sig.append(tuple(deps))
        result.append(tuple(rep_sig))
    return tuple(result)


def _equal_signature_runs(signatures: Sequence[Any], min_repetitions: int) -> list[tuple[int, int]]:
    if not signatures:
        return []
    out = []
    start = 0
    for i in range(1, len(signatures) + 1):
        if i == len(signatures) or signatures[i] != signatures[start]:
            reps = i - start
            if reps >= min_repetitions:
                out.append((start, reps))
            start = i
    return out

def _dependency_signature(analysis: _LoopAnalysis, start: int, body_size: int, repetitions: int) -> tuple[Any, ...] | None:
    """Return a normalized exact-topology signature or None if topology varies.

    Dependencies to scheduled nodes are represented by body slot and repetition
    delta.  External/input dependencies must have a stable schema/space shape and
    constant or affine integer arguments; irregular external dependency routing is
    conservatively treated as a structural change and blocks loop recovery.
    """
    by_id = analysis.by_id
    schedule = analysis.schedule
    pos_of = analysis.pos_of
    incoming = analysis.incoming

    candidate_stop = start + body_size * repetitions
    per_rep: list[tuple[Any, ...]] = []
    external_series: dict[tuple[int, str, str, int], list[tuple[Any, tuple[Any, ...]]]] = defaultdict(list)

    for rep in range(repetitions):
        rep_sig = []
        for slot in range(body_size):
            pos = start + rep * body_size + slot
            nid = schedule[pos]
            deps = []
            external_grouped = []
            for src in incoming.get(nid, []):
                spos = pos_of.get(src)
                snode = by_id.get(src)
                relative = False
                if spos is not None and snode is not None:
                    rel = spos - start
                    src_rep = rel // body_size
                    src_slot = rel % body_size
                    template_nid = schedule[start + src_slot]
                    template_node = by_id[template_nid]
                    relative = snode.shape_token == template_node.shape_token
                if relative:
                    deps.append(("relative", src_slot, src_rep - rep))
                else:
                    if snode is None:
                        return None
                    shape = (slot, snode.schema_uid, snode.space_family_uid, len(snode.args))
                    external_grouped.append((shape, snode.runtime_node))
            deps.sort()
            external_grouped.sort(key=lambda x: x[0])
            for shape, runtime in external_grouped:
                external_series[shape].append(runtime)
                deps.append(("external", shape[1], shape[2], shape[3]))
            rep_sig.append(tuple(deps))
        per_rep.append(tuple(rep_sig))

    if any(sig != per_rep[0] for sig in per_rep[1:]):
        return None

    ext_signature = []
    for shape, runtimes in sorted(external_series.items(), key=lambda kv: kv[0]):
        # There may be more than one same-shaped external dependency per repetition.
        # If cardinality is not one per repetition, the simple correspondence is
        # ambiguous; reject rather than guess.
        if len(runtimes) != repetitions:
            return None
        objs = [r[0] for r in runtimes]
        keys = [r[1] for r in runtimes]
        obj_binding = infer_binding(objs)
        if isinstance(obj_binding, TableBinding):
            return None
        arg_kinds = []
        for j in range(len(keys[0])):
            b = infer_binding([key[j] for key in keys])
            if isinstance(b, TableBinding):
                return None
            if isinstance(b, ConstantBinding):
                arg_kinds.append((b.kind, repr(b.value)))
            else:
                arg_kinds.append((b.kind, b.step))
        ext_signature.append((shape, obj_binding.kind, tuple(arg_kinds)))

    return (per_rep[0], tuple(ext_signature))


def _make_loop_block(program: SequentialProgram, analysis: _LoopAnalysis, candidate: LoopCandidate) -> LoopBlock:
    by_id = analysis.by_id
    schedule = analysis.schedule
    body = []
    source_ids = tuple(schedule[candidate.start:candidate.stop])

    for slot in range(candidate.body_size):
        nodes = [
            by_id[schedule[candidate.start + rep * candidate.body_size + slot]]
            for rep in range(candidate.repetitions)
        ]
        first = nodes[0]
        if any(n.schema_uid != first.schema_uid or n.space_family_uid != first.space_family_uid or len(n.args) != len(first.args) for n in nodes[1:]):
            raise LoopRecoveryError("candidate token match failed structural node validation")
        obj_binding = infer_binding([n.obj for n in nodes])
        arg_bindings = tuple(infer_binding([n.args[j] for n in nodes]) for j in range(len(first.args)))
        body.append(
            EvalTemplate(
                schema_uid=first.schema_uid,
                space_family_uid=first.space_family_uid,
                obj_binding=obj_binding,
                arg_bindings=arg_bindings,
            )
        )

    block = LoopBlock(
        start=candidate.start,
        body_size=candidate.body_size,
        repetitions=candidate.repetitions,
        body=tuple(body),
        dependency_signature=candidate.dependency_signature,
        source_node_ids=source_ids,
    )

    expected = tuple(by_id[nid].runtime_node for nid in source_ids)
    actual = block.expand_runtime_nodes()
    if len(expected) != len(actual):
        raise LoopRecoveryError("candidate expansion length mismatch")
    for i, (a, b) in enumerate(zip(actual, expected)):
        if a[0] is not b[0] or a[1] != b[1]:
            raise LoopRecoveryError(f"candidate binding reconstruction mismatch at {i}")
    return block


def _select_non_overlapping(candidates: Sequence[LoopCandidate]) -> list[LoopCandidate]:
    if not candidates:
        return []
    cands = sorted(candidates, key=lambda c: (c.stop, c.start, c.body_size))
    ends = [c.stop for c in cands]
    prev = [bisect.bisect_right(ends, c.start) - 1 for c in cands]
    score = [0] * (len(cands) + 1)
    choose = [False] * len(cands)
    for i, cand in enumerate(cands, 1):
        take = cand.saved_ops + score[prev[i - 1] + 1]
        skip = score[i - 1]
        if take > skip:
            score[i] = take
            choose[i - 1] = True
        else:
            score[i] = skip

    selected = []
    i = len(cands)
    while i > 0:
        cand = cands[i - 1]
        take = cand.saved_ops + score[prev[i - 1] + 1]
        if take > score[i - 1]:
            selected.append(cand)
            i = prev[i - 1] + 1
        else:
            i -= 1
    selected.reverse()
    return selected


def recover_loops(
    program: SequentialProgram,
    *,
    min_repetitions: int = 3,
    max_candidates: int = 64,
    max_gap_neighbors: int = 8,
    max_body_size: int | None = 4096,
) -> StructuredSequentialProgram:
    if min_repetitions < 2:
        raise ValueError("min_repetitions must be at least 2")

    analysis = _LoopAnalysis.from_program(program)
    by_id = analysis.by_id
    tokens = [by_id[op.node_id].shape_token for op in program.ops]
    token_ids_by_shape: dict[Any, int] = {}
    token_ids = []
    for tok in tokens:
        if tok not in token_ids_by_shape:
            token_ids_by_shape[tok] = len(token_ids_by_shape)
        token_ids.append(token_ids_by_shape[tok])

    periods = _candidate_periods(
        token_ids,
        max_candidates=max_candidates,
        max_gap_neighbors=max_gap_neighbors,
        max_body_size=max_body_size,
    )

    candidates: dict[tuple[int, int], LoopCandidate] = {}
    for period in periods:
        for run_start, run_reps in _runs_for_period(token_ids, period, min_repetitions):
            coarse = _coarse_rep_dependency_signatures(analysis, run_start, period, run_reps)
            for rep_offset, reps in _equal_signature_runs(coarse, min_repetitions):
                start = run_start + rep_offset * period
                sig = _dependency_signature(analysis, start, period, reps)
                if sig is None:
                    continue
                cand = LoopCandidate(start, period, reps, sig)
                key = (cand.start, cand.stop)
                old = candidates.get(key)
                if old is None or cand.body_size < old.body_size:
                    candidates[key] = cand

    selected = _select_non_overlapping(tuple(candidates.values()))

    blocks: list[Block] = []
    cursor = 0
    for cand in selected:
        if cand.start > cursor:
            blocks.append(LiteralBlock(tuple(program.ops[cursor:cand.start])))
        blocks.append(_make_loop_block(program, analysis, cand))
        cursor = cand.stop
    if cursor < len(program.ops):
        blocks.append(LiteralBlock(tuple(program.ops[cursor:])))
    if not blocks and program.ops:
        blocks.append(LiteralBlock(program.ops))

    structured = StructuredSequentialProgram(program, tuple(blocks))
    structured.validate_exact_expansion()
    return structured
