from __future__ import annotations

"""Optional, measured optimization passes over the executable graph IR."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PassReport:
    name: str
    applied: bool
    selected: tuple[str, ...]
    proofs: dict[str, str]
    notes: tuple[str, ...] = ()


class ReductionFusionPass:
    """Fuse only reductions that preserve the source traversal and FP order."""

    name = "reduction_fusion"

    def analyze(self, graph: Any) -> PassReport:
        selected = []
        proofs: dict[str, str] = {}
        for block in graph.loops:
            for uid in block.fusable_reductions:
                selected.append(uid)
                proofs[uid] = block.fusion_reasons.get(uid, "proved safe by executable template")
        return PassReport(
            name=self.name,
            applied=bool(selected),
            selected=tuple(selected),
            proofs=proofs,
            notes=(
                "fusion preserves the original ordered range and filter order",
                "no algebraic reassociation is performed",
            ),
        )


def reports_for_level(graph: Any, optimization_level: str) -> tuple[PassReport, ...]:
    level = optimization_level.upper()
    if level in {"O0", "O1"}:
        return ()
    if level == "O2":
        return (ReductionFusionPass().analyze(graph),)
    raise ValueError(f"unsupported optimization level {optimization_level!r}")


def fused_reductions_for_level(graph: Any, optimization_level: str) -> frozenset[str]:
    selected: set[str] = set()
    for report in reports_for_level(graph, optimization_level):
        if report.applied:
            selected.update(report.selected)
    return frozenset(selected)
