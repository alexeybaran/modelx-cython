from __future__ import annotations

"""Exact recovered-loop evidence for the canonical NativeBatch scheduler.

The recovered program is evidence only.  Canonical source formulas, source-level
static/domain guards and the executable template remain runtime semantics.  This
module therefore carries structural facts that can be matched back to a guarded
canonical formula family without carrying realized runtime objects/bindings into
native execution.
"""

import hashlib
from dataclasses import dataclass
from typing import Iterable

from .loop_recovery import (
    AffineIntBinding,
    ArithmeticRunsIntBinding,
    LoopBlock,
    PhaseLoopBlock,
    StructuredSequentialProgram,
)


def _signature(*parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode("utf-8", "backslashreplace"))
        h.update(b"\0")
    return h.hexdigest()[:20]


@dataclass(frozen=True)
class RecoveredCoordinateDomain:
    fullname: str
    occurrence_count: int
    minimum: int
    maximum: int
    first: int
    last: int
    step: int | None
    binding_kind: str

    @property
    def span(self) -> int:
        return int(self.maximum) - int(self.minimum)

    def covers(self, other: "RecoveredCoordinateDomain") -> bool:
        return self.minimum <= other.minimum and self.maximum >= other.maximum

    def manifest(self) -> dict[str, object]:
        return {
            "fullname": self.fullname,
            "occurrence_count": int(self.occurrence_count),
            "minimum": int(self.minimum),
            "maximum": int(self.maximum),
            "first": int(self.first),
            "last": int(self.last),
            "step": None if self.step is None else int(self.step),
            "binding_kind": self.binding_kind,
        }


@dataclass(frozen=True)
class RecoveredLoopWitness:
    uid: str
    block_index: int
    kind: str
    operation_start: int
    operation_stop: int
    repetitions: int
    phase_count: int
    phase_head: tuple[int, ...]
    phase_tail: tuple[int, ...]
    grammar_signature: str
    formula_family_signature: str
    formula_fullnames: frozenset[str]
    formula_identities: tuple[tuple[str, str, int], ...]
    affine_coordinate_steps: tuple[tuple[str, int], ...]
    coordinate_domains: tuple[RecoveredCoordinateDomain, ...]

    def coordinate_domain(self, fullname: str) -> RecoveredCoordinateDomain | None:
        rows = [row for row in self.coordinate_domains if row.fullname == fullname]
        if not rows:
            return None
        # The same fullname may occur in several static variants/roles.  The widest
        # exact observed domain is the conservative structural witness.
        return max(rows, key=lambda row: (row.span, row.occurrence_count))

    def matching_fullnames(self, candidates: Iterable[str], stride: int) -> tuple[str, ...]:
        wanted = set(candidates)
        step = abs(int(stride))
        matches: set[str] = set()
        for fullname, observed_step in self.affine_coordinate_steps:
            if fullname not in wanted or abs(int(observed_step)) != step:
                continue
            domain = self.coordinate_domain(fullname)
            if domain is None:
                continue
            # The matched coordinate family must actually span the repeated loop,
            # not merely appear twice at a coincidental stride.
            min_span = max(0, (int(self.repetitions) - 1) * step)
            if domain.occurrence_count >= self.repetitions and domain.span >= min_span:
                matches.add(fullname)
        return tuple(sorted(matches))


@dataclass(frozen=True)
class RecoveredLoopMatch:
    witness: RecoveredLoopWitness
    matched_fullnames: tuple[str, ...]
    canonical_family_signature: str
    range_summary: str


@dataclass(frozen=True)
class RecoveredLoopEvidence:
    exact_expansion_validated: bool
    witnesses: tuple[RecoveredLoopWitness, ...]

    def matching_witnesses(
        self,
        candidates: Iterable[str],
        stride: int,
        *,
        required_fullnames: Iterable[str] = (),
        required_history_pairs: Iterable[tuple[str, str]] = (),
        canonical_family_signature: str = "",
    ) -> tuple[RecoveredLoopMatch, ...]:
        if not self.exact_expansion_validated:
            return ()
        required = set(required_fullnames)
        pairs = tuple(required_history_pairs)
        out: list[RecoveredLoopMatch] = []
        for witness in self.witnesses:
            matches = witness.matching_fullnames(candidates, stride)
            if not matches:
                continue
            if required and not required.issubset(witness.formula_fullnames):
                continue

            pair_summaries: list[str] = []
            valid_pairs = True
            for source, target in pairs:
                if source not in witness.formula_fullnames or target not in witness.formula_fullnames:
                    valid_pairs = False
                    break
                source_domain = witness.coordinate_domain(source)
                target_domain = witness.coordinate_domain(target)
                if source_domain is None or target_domain is None or not source_domain.covers(target_domain):
                    valid_pairs = False
                    break
                pair_summaries.append(
                    f"{source}[{source_domain.minimum}:{source_domain.maximum}]"
                    f">={target}[{target_domain.minimum}:{target_domain.maximum}]"
                )
            if not valid_pairs:
                continue

            range_summary = ";".join(pair_summaries) if pair_summaries else ",".join(
                f"{name}[{witness.coordinate_domain(name).minimum}:{witness.coordinate_domain(name).maximum}]"
                for name in matches
                if witness.coordinate_domain(name) is not None
            )
            out.append(
                RecoveredLoopMatch(
                    witness=witness,
                    matched_fullnames=matches,
                    canonical_family_signature=canonical_family_signature,
                    range_summary=range_summary,
                )
            )
        return tuple(out)


def _role_occurrence_counts(family, instance) -> list[int]:
    expanded = [
        family.grammar.expand_variant(i)
        for i in range(len(family.grammar.variant_programs))
    ]
    counts = [0] * len(family.formula_op_ids)
    for variant_id in instance.variant_ids:
        for role_id in expanded[int(variant_id)]:
            counts[int(role_id)] += 1
    return counts


def _coordinate_domain(fullname: str, binding, count: int) -> RecoveredCoordinateDomain | None:
    if binding is None or not binding.arg_bindings or count <= 0:
        return None
    first_binding = binding.arg_bindings[0]
    try:
        values = [first_binding.value_at(i) for i in range(int(count))]
    except Exception:
        return None
    if not values or not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return None
    ints = [int(v) for v in values]
    step: int | None = None
    if isinstance(first_binding, (AffineIntBinding, ArithmeticRunsIntBinding)):
        step = int(first_binding.step)
    return RecoveredCoordinateDomain(
        fullname=fullname,
        occurrence_count=int(count),
        minimum=min(ints),
        maximum=max(ints),
        first=ints[0],
        last=ints[-1],
        step=step,
        binding_kind=getattr(first_binding, "kind", type(first_binding).__name__),
    )


def build_recovered_loop_evidence(
    structured: StructuredSequentialProgram,
) -> RecoveredLoopEvidence:
    """Extract exact repeated-loop witnesses without carrying runtime bindings forward."""
    plan = structured.canonical_plan
    if plan is None:
        return RecoveredLoopEvidence(bool(structured.exact_expansion_validated), ())

    by_id = {op.op_id: op for op in plan.formula_ops}
    witnesses: list[RecoveredLoopWitness] = []
    for instance in plan.loop_instances:
        block = structured.blocks[instance.block_index]
        if not isinstance(block, (LoopBlock, PhaseLoopBlock)):
            continue
        if int(block.repetitions) < 2:
            continue
        family = plan.code_families[instance.family_id]
        counts = _role_occurrence_counts(family, instance)
        names: set[str] = set()
        identities: set[tuple[str, str, int]] = set()
        steps: set[tuple[str, int]] = set()
        domains: list[RecoveredCoordinateDomain] = []
        for op_id, binding, count in zip(
            family.formula_op_ids, instance.role_bindings, counts
        ):
            op = by_id[op_id]
            names.add(op.fullname)
            identities.add((op.fullname, op.schema_uid, int(op.arity)))
            domain = _coordinate_domain(op.fullname, binding, count)
            if domain is not None:
                domains.append(domain)
                if domain.step is not None and int(domain.step) != 0:
                    steps.add((op.fullname, int(domain.step)))

        if isinstance(block, PhaseLoopBlock):
            operation_start = int(block.start)
            operation_stop = int(block.stop)
            phase_ids = tuple(int(x) for x in block.phase_ids)
            kind = "phase"
            phase_count = int(block.phase_count)
        else:
            operation_start = int(block.start)
            operation_stop = int(block.start + block.operation_count)
            phase_ids = tuple(int(x) for x in instance.variant_ids)
            kind = "loop"
            phase_count = 1

        grammar_signature = _signature(
            family.grammar.terminal_roles,
            family.grammar.rules,
            family.grammar.variant_programs,
            phase_ids,
        )
        family_signature = _signature(tuple(sorted(identities)))
        witnesses.append(
            RecoveredLoopWitness(
                uid=f"recovered:block:{instance.block_index}:family:{instance.family_id}",
                block_index=int(instance.block_index),
                kind=kind,
                operation_start=operation_start,
                operation_stop=operation_stop,
                repetitions=int(block.repetitions),
                phase_count=phase_count,
                phase_head=phase_ids[:6],
                phase_tail=phase_ids[-6:],
                grammar_signature=grammar_signature,
                formula_family_signature=family_signature,
                formula_fullnames=frozenset(names),
                formula_identities=tuple(sorted(identities)),
                affine_coordinate_steps=tuple(sorted(steps)),
                coordinate_domains=tuple(sorted(domains, key=lambda x: (x.fullname, x.minimum, x.maximum))),
            )
        )
    return RecoveredLoopEvidence(
        exact_expansion_validated=bool(structured.exact_expansion_validated),
        witnesses=tuple(witnesses),
    )
