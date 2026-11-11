from __future__ import annotations

"""Generic graph IR for the modelx graph compiler.

The IR deliberately starts from concrete Cells values and dependency edges.  It
contains no required notion of a privileged coordinate, recurrence class, variable
role, or domain convention. Coordinate/loop structure is attached only as an
optional optimization result after the concrete graph has been built.
"""

import hashlib
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import networkx as nx
import numpy as np

from .trace import TraceCapture, _itemspace_key


def _stable(value: Any) -> str:
    """Stable-enough local representation for trace identities/manifests.

    Trace arguments may be arbitrary hashable Python objects.  Native eligibility
    is decided elsewhere; the IR only needs a deterministic identity within a
    compiler build.  repr is intentionally not interpreted semantically.
    """
    if isinstance(value, np.generic):
        value = value.item()
    return repr(value)


def _digest(parts: Iterable[str], prefix: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", "backslashreplace"))
        h.update(b"\0")
    return f"{prefix}_{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class ValueIdentity:
    schema_uid: str
    cell_fullname: str
    space_fullname: str
    arguments: tuple[Any, ...]
    itemspace_key: tuple[Any, ...]


@dataclass
class ConcreteValue:
    uid: str
    identity: ValueIdentity
    dtype: str
    shape: tuple[int, ...]
    variant_uid: str
    is_input: bool
    is_target: bool
    topo_index: int = -1
    last_use: int = -1
    slot: int = -1


@dataclass(frozen=True)
class IRDependency:
    source: str
    target: str


@dataclass(frozen=True)
class GuardSpec:
    kind: str
    subject: str
    expected: Any
    description: str


@dataclass
class LoopTemplate:
    """Repeated concrete topology under a consistent integer-coordinate shift.

    ``coordinate_positions`` maps variant identity to the argument position that
    shifted in the observed traces.  The name of that argument is irrelevant.
    ``generalization`` starts as ``observed_only``; a later static-analysis pass
    may strengthen it after proving a reusable range/branch structure.
    """

    uid: str
    signature: str
    coordinate_positions: dict[str, int]
    observed_spans: list[tuple[str, int, int]]
    repetitions: int
    coordinate_stride: int = 1
    boundary_coordinates: list[tuple[str, int]] = field(default_factory=list)
    generalization: str = "observed_only"
    proof_notes: list[str] = field(default_factory=list)

    @property
    def observed_min(self) -> int | None:
        return min((a for _, a, _ in self.observed_spans), default=None)

    @property
    def observed_max(self) -> int | None:
        return max((b for _, _, b in self.observed_spans), default=None)


@dataclass
class IRRegion:
    uid: str
    mode: str  # compiled_candidate | fallback_candidate
    values: list[str]
    incoming_boundaries: list[IRDependency]
    outgoing_boundaries: list[IRDependency]
    reasons: list[str] = field(default_factory=list)


@dataclass
class GraphIR:
    values: dict[str, ConcreteValue]
    dependencies: list[IRDependency]
    topological_order: list[str]
    formula_hash: str
    structural_hash: str
    concrete_slot_count: int
    peak_live_values: int
    variant_peak_live: dict[str, int]
    loop_templates: list[LoopTemplate]
    guards: list[GuardSpec]
    regions: list[IRRegion]
    diagnostics: list[str]
    sample_trace: TraceCapture = field(repr=False)

    def authorize_loop_generalization(self, reason: str) -> None:
        """Strengthen observed loop templates after a separate proof pass.

        This method does not itself prove anything.  It is intentionally called
        only by the static/formula frontend after exact range and formula lowering
        checks have succeeded.
        """
        loop_ids = {template.uid for template in self.loop_templates}
        for template in self.loop_templates:
            template.generalization = "static_formula_guarded"
            if reason not in template.proof_notes:
                template.proof_notes.append(reason)
        self.guards = [
            GuardSpec(
                g.kind,
                g.subject,
                ({**g.expected, "policy": "static_formula_guarded", "proof": reason}
                 if g.kind == "loop_extent" and g.subject in loop_ids and isinstance(g.expected, dict)
                 else g.expected),
                g.description,
            )
            for g in self.guards
        ]

    def manifest(self) -> dict[str, Any]:
        return {
            "formula_hash": self.formula_hash,
            "structural_hash": self.structural_hash,
            "value_count": len(self.values),
            "dependency_count": len(self.dependencies),
            "concrete_slot_count": self.concrete_slot_count,
            "peak_live_values": self.peak_live_values,
            "variant_peak_live": dict(sorted(self.variant_peak_live.items())),
            "loop_templates": [
                {
                    "uid": x.uid,
                    "signature": x.signature,
                    "coordinate_positions": dict(sorted(x.coordinate_positions.items())),
                    "observed_spans": list(x.observed_spans),
                    "repetitions": x.repetitions,
                    "coordinate_stride": x.coordinate_stride,
                    "boundary_coordinates": list(x.boundary_coordinates),
                    "generalization": x.generalization,
                    "proof_notes": list(x.proof_notes),
                }
                for x in self.loop_templates
            ],
            "guards": [g.__dict__.copy() for g in self.guards],
            "regions": [
                {
                    "uid": r.uid,
                    "mode": r.mode,
                    "value_count": len(r.values),
                    "incoming_boundaries": [e.__dict__.copy() for e in r.incoming_boundaries],
                    "outgoing_boundaries": [e.__dict__.copy() for e in r.outgoing_boundaries],
                    "reasons": list(r.reasons),
                }
                for r in self.regions
            ],
            "diagnostics": list(self.diagnostics),
        }


def _value_shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(np.asarray(value).shape) if isinstance(value, np.ndarray) else ()
    except Exception:
        return ()


def _node_uid(schema_uid: str, item_key: tuple[Any, ...], args: tuple[Any, ...]) -> str:
    return _digest(
        [schema_uid, _stable(item_key), *(_stable(x) for x in args)],
        "n",
    )


def _formula_hash(capture: TraceCapture) -> str:
    return _digest(
        [f"{uid}:{capture.schemas[uid].source}" for uid in sorted(capture.schemas)],
        "formula",
    )


def _allocate_slots(values: dict[str, ConcreteValue], order: list[str]) -> tuple[int, int]:
    """Greedy interval coloring over producer->last-consumer live ranges."""
    active: list[tuple[int, int]] = []  # (last_use, slot)
    free: list[int] = []
    next_slot = 0
    peak = 0
    for uid in order:
        cur = values[uid]
        while active and active[0][0] < cur.topo_index:
            _, slot = heapq.heappop(active)
            heapq.heappush(free, slot)
        if free:
            cur.slot = heapq.heappop(free)
        else:
            cur.slot = next_slot
            next_slot += 1
        heapq.heappush(active, (cur.last_use, cur.slot))
        peak = max(peak, len(active))
    return next_slot, peak


def _variant_peak_live(values: dict[str, ConcreteValue]) -> dict[str, int]:
    """Peak simultaneously-live concrete values per traced formula variant.

    Peaks are calculated independently per ItemSpace and then maxed, matching the
    per-model-point execution/storage problem rather than multiplying storage by
    the number of representative sample points.
    """
    grouped: dict[tuple[str, tuple[Any, ...]], list[ConcreteValue]] = defaultdict(list)
    for value in values.values():
        grouped[(value.variant_uid, value.identity.itemspace_key)].append(value)
    result: dict[str, int] = defaultdict(int)
    for (variant_uid, _), vals in grouped.items():
        events: list[tuple[int, int, int]] = []
        for v in vals:
            # start before end at the same index, conservatively treating the
            # value as live during its producing operation.
            events.append((v.topo_index, 0, +1))
            events.append((v.last_use, 1, -1))
        live = peak = 0
        for _, _, delta in sorted(events):
            live += delta
            peak = max(peak, live)
        result[variant_uid] = max(result[variant_uid], peak)
    return dict(result)


def _slice_signature(capture: TraceCapture, nodes: list[Any], coordinate: int) -> tuple[str, dict[str, int]]:
    entries = []
    positions: dict[str, int] = {}
    node_set = set(nodes)
    for node in nodes:
        vk = capture.node_variant.get(node)
        if vk is None or vk.time_pos is None:
            continue
        tr = capture.variants.get(vk)
        if tr is None:
            continue
        positions[tr.uid] = int(vk.time_pos)
        incoming = []
        for source in capture.graph.predecessors(node):
            svk = capture.node_variant.get(source)
            if svk is None:
                continue
            strace = capture.variants.get(svk)
            if strace is None:
                continue
            if svk.time_pos is None:
                incoming.append((strace.uid, "static"))
            else:
                try:
                    source_coordinate = int(source[1][svk.time_pos])
                except Exception:
                    incoming.append((strace.uid, "dynamic"))
                else:
                    incoming.append((strace.uid, source_coordinate - coordinate))
        entries.append((tr.uid, tuple(sorted(incoming, key=repr))))
    raw = repr(sorted(entries, key=repr))
    return hashlib.sha256(raw.encode()).hexdigest()[:20], positions


def _discover_loop_templates(capture: TraceCapture) -> list[LoopTemplate]:
    """Find repeated shift-isomorphic integer-coordinate slices.

    The repeated coordinate need not be zero-based or unit-spaced.  Concrete
    traces are sorted by coordinate value only to discover a constant absolute
    stride; static range analysis later establishes the signed execution step and
    exact Python ``range`` domain.  Names remain irrelevant.
    """
    per_signature: dict[tuple[str, tuple[tuple[str, int], ...], int], LoopTemplate] = {}
    dynamic_nodes: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for node, vk in capture.node_variant.items():
        if vk.time_pos is None:
            continue
        dynamic_nodes[_itemspace_key(node[0])].append(node)

    for item_key, nodes in dynamic_nodes.items():
        by_coordinate: dict[int, list[Any]] = defaultdict(list)
        for node in nodes:
            vk = capture.node_variant[node]
            try:
                by_coordinate[int(node[1][vk.time_pos])].append(node)
            except Exception:
                pass
        coords = sorted(by_coordinate)
        sigs: dict[int, tuple[str, dict[str, int]]] = {
            c: _slice_signature(capture, by_coordinate[c], c) for c in coords
        }
        used: set[int] = set()
        i = 0
        while i < len(coords):
            if i + 1 >= len(coords):
                break
            sig, positions = sigs[coords[i]]
            stride = coords[i + 1] - coords[i]
            if stride <= 0:
                i += 1
                continue
            j = i + 1
            while (
                j < len(coords)
                and coords[j] == coords[j - 1] + stride
                and sigs[coords[j]][0] == sig
                and sigs[coords[j]][1] == positions
            ):
                j += 1
            if j - i >= 2:
                a, b = coords[i], coords[j - 1]
                used.update(coords[i:j])
                poskey = tuple(sorted(positions.items()))
                key = (sig, poskey, stride)
                item_label = _stable(item_key)
                if key not in per_signature:
                    per_signature[key] = LoopTemplate(
                        uid=_digest([sig, repr(poskey), str(stride)], "loop"),
                        signature=sig,
                        coordinate_positions=dict(positions),
                        observed_spans=[(item_label, a, b)],
                        repetitions=j - i,
                        coordinate_stride=stride,
                    )
                else:
                    t = per_signature[key]
                    t.observed_spans.append((item_label, a, b))
                    t.repetitions += j - i
                i = j
            else:
                i += 1
        boundaries = [(_stable(item_key), c) for c in coords if c not in used]
        for template in per_signature.values():
            if any(label == _stable(item_key) for label, _, _ in template.observed_spans):
                template.boundary_coordinates.extend(boundaries)

    return sorted(per_signature.values(), key=lambda x: x.uid)


def _build_regions(values: dict[str, ConcreteValue], deps: list[IRDependency]) -> list[IRRegion]:
    mode = {
        uid: ("fallback_candidate" if value.dtype == "object" else "compiled_candidate")
        for uid, value in values.items()
    }
    graph = nx.Graph()
    graph.add_nodes_from(values)
    for dep in deps:
        if mode[dep.source] == mode[dep.target]:
            graph.add_edge(dep.source, dep.target)
    regions: list[IRRegion] = []
    for component in nx.connected_components(graph):
        component = set(component)
        first = next(iter(component))
        incoming = [d for d in deps if d.target in component and d.source not in component]
        outgoing = [d for d in deps if d.source in component and d.target not in component]
        reasons = []
        if mode[first] == "fallback_candidate":
            reasons.append("observed object/dynamic value requires frontend lowering or fallback")
        regions.append(
            IRRegion(
                uid=_digest(sorted(component), "region"),
                mode=mode[first],
                values=sorted(component),
                incoming_boundaries=incoming,
                outgoing_boundaries=outgoing,
                reasons=reasons,
            )
        )
    return sorted(regions, key=lambda x: x.uid)


def build_graph_ir(capture: TraceCapture) -> GraphIR:
    """Build the name-agnostic concrete graph IR from a representative trace."""
    raw_nodes = [n for n in capture.graph.nodes if n in capture.node_variant]
    subgraph = capture.graph.subgraph(raw_nodes).copy()
    if not nx.is_directed_acyclic_graph(subgraph):
        raise ValueError("traced dependency graph is cyclic at concrete-node level")
    raw_order = list(nx.topological_sort(subgraph))

    raw_to_uid: dict[Any, str] = {}
    values: dict[str, ConcreteValue] = {}
    targets = {tuple(x._impl) for x in capture.target_nodes}
    for idx, node in enumerate(raw_order):
        obj, key = node
        vk = capture.node_variant[node]
        tr = capture.variants[vk]
        item_key = _itemspace_key(obj)
        args = tuple(key)
        uid = _node_uid(tr.schema.uid, item_key, args)
        raw_to_uid[node] = uid
        try:
            value = obj.data[key]
        except Exception:
            value = None
        try:
            is_input = key in obj.input_keys
        except Exception:
            is_input = False
        identity = ValueIdentity(
            schema_uid=tr.schema.uid,
            cell_fullname=tr.schema.fullname,
            space_fullname=tr.schema.fullname.rsplit(".", 1)[0] if "." in tr.schema.fullname else "",
            arguments=args,
            itemspace_key=tuple(item_key),
        )
        values[uid] = ConcreteValue(
            uid=uid,
            identity=identity,
            dtype=tr.dtype,
            shape=_value_shape(value),
            variant_uid=tr.uid,
            is_input=bool(is_input),
            is_target=node in targets,
            topo_index=idx,
        )

    dependencies = [
        IRDependency(raw_to_uid[a], raw_to_uid[b])
        for a, b in subgraph.edges
        if a in raw_to_uid and b in raw_to_uid
    ]
    succ: dict[str, list[str]] = defaultdict(list)
    for dep in dependencies:
        succ[dep.source].append(dep.target)
    end_index = max(0, len(raw_order) - 1)
    for uid, value in values.items():
        consumers = succ.get(uid, [])
        value.last_use = max((values[c].topo_index for c in consumers), default=value.topo_index)
        if value.is_target:
            value.last_use = max(value.last_use, end_index)

    order = [raw_to_uid[n] for n in raw_order]
    slot_count, peak_live = _allocate_slots(values, order)
    variant_peak = _variant_peak_live(values)
    loops = _discover_loop_templates(capture)
    formula_hash = _formula_hash(capture)
    structural_hash = _digest(
        [
            formula_hash,
            *sorted(f"{values[d.source].variant_uid}->{values[d.target].variant_uid}" for d in dependencies),
            *sorted(f"{v.variant_uid}:{v.dtype}:{v.shape}" for v in values.values()),
        ],
        "graph",
    )

    guards: list[GuardSpec] = [
        GuardSpec(
            "formula_hash",
            "model",
            formula_hash,
            "compiled artifact is valid only for the formula sources hashed at build time",
        )
    ]
    seen_variant_guards = set()
    for value in values.values():
        token = (value.variant_uid, value.dtype, value.shape)
        if token in seen_variant_guards:
            continue
        seen_variant_guards.add(token)
        guards.append(
            GuardSpec(
                "observed_type_shape",
                value.variant_uid,
                {"dtype": value.dtype, "shape": value.shape},
                "sample-derived native type/shape evidence; frontend must prove or guard before reuse",
            )
        )
    for loop in loops:
        guards.append(
            GuardSpec(
                "loop_extent",
                loop.uid,
                {"spans": list(loop.observed_spans), "policy": "observed_only"},
                "representative repetition is not authorization for a longer coordinate domain",
            )
        )

    regions = _build_regions(values, dependencies)
    diagnostics = list(capture.diagnostics)
    if not loops:
        diagnostics.append("no repeated coordinate topology was proven from representative traces")
    if any(r.mode == "fallback_candidate" for r in regions):
        diagnostics.append("object/dynamic regions are represented explicitly and require lowering or fallback")

    return GraphIR(
        values=values,
        dependencies=dependencies,
        topological_order=order,
        formula_hash=formula_hash,
        structural_hash=structural_hash,
        concrete_slot_count=slot_count,
        peak_live_values=peak_live,
        variant_peak_live=variant_peak,
        loop_templates=loops,
        guards=guards,
        regions=regions,
        diagnostics=diagnostics,
        sample_trace=capture,
    )
