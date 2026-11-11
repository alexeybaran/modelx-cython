from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import networkx as nx

from .trace import CellSchema, TraceError, _schema_for_impl


def clear_realized_calculated(model: Any) -> None:
    """Clear calculated Cells values in static *and already-created dynamic Spaces*.

    modelx ``BaseSpace.clear_all_cells(..., recursive=True)`` walks named static
    subspaces but deliberately does not walk existing ItemSpaces.  Realized-trace
    replay must preserve those ItemSpace objects while clearing their calculated
    Cells, otherwise a previously evaluated target becomes an immediate cache hit
    and no trace is produced.
    """
    seen: set[int] = set()

    def walk(space: Any) -> None:
        if id(space) in seen:
            return
        seen.add(id(space))
        for cells in getattr(space, "cells", {}).values():
            cells.clear_all_values(clear_input=False)
        for child in getattr(space, "named_spaces", {}).values():
            walk(child)
        for child in getattr(space, "named_itemspaces", {}).values():
            walk(child)

    for space in model._impl.spaces.values():
        walk(space)


class RealizedTraceError(TraceError):
    """Raised when an exact realized modelx trace cannot be represented safely."""


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    depth: int
    sequence: int
    obj: Any = field(compare=False, repr=False)
    key: tuple[Any, ...]
    cached: bool

    @property
    def runtime_node(self) -> tuple[Any, tuple[Any, ...]]:
        return self.obj, self.key


@dataclass(frozen=True)
class ConcreteNode:
    """One exact cached Cells invocation from the realized trace."""

    node_id: int
    uid: str
    schema: CellSchema
    schema_uid: str
    space_instance_uid: str
    space_family_uid: str
    args: tuple[Any, ...]
    obj: Any = field(compare=False, repr=False)
    is_input: bool = False
    is_target: bool = False
    exit_rank: int | None = None

    @property
    def runtime_node(self) -> tuple[Any, tuple[Any, ...]]:
        return self.obj, self.args

    @property
    def shape_token(self) -> tuple[str, str, int]:
        return self.schema_uid, self.space_family_uid, len(self.args)


@dataclass
class RealizedTrace:
    model: Any = field(repr=False)
    events: tuple[TraceEvent, ...]
    nodes: tuple[ConcreteNode, ...]
    dependencies: tuple[tuple[int, int], ...]
    schedule_node_ids: tuple[int, ...]
    target_node_ids: tuple[int | None, ...]
    target_runtime_nodes: tuple[tuple[Any, tuple[Any, ...]], ...] = field(repr=False)
    target_values: tuple[Any, ...] = field(repr=False)
    diagnostics: tuple[str, ...] = ()

    @property
    def node_by_id(self) -> dict[int, ConcreteNode]:
        return {n.node_id: n for n in self.nodes}

    @property
    def runtime_to_id(self) -> dict[tuple[Any, tuple[Any, ...]], int]:
        return {n.runtime_node: n.node_id for n in self.nodes}

    @property
    def calculated_count(self) -> int:
        return len(self.schedule_node_ids)


def _normalize_targets(targets: Any) -> tuple[Any, ...]:
    if hasattr(targets, "_impl") and targets.__class__.__name__.endswith("Node"):
        return (targets,)
    if isinstance(targets, Sequence) and not isinstance(targets, (str, bytes)):
        out = tuple(targets)
        if out:
            return out
    return (targets,)


def _target_runtime_node(target: Any) -> tuple[Any, tuple[Any, ...]]:
    if hasattr(target, "_impl") and target.__class__.__name__.endswith("Node"):
        node = target._impl
        if len(node) != 2:
            raise RealizedTraceError("target must be a modelx ItemNode")
        return node[0], tuple(node[1])
    if hasattr(target, "node"):
        try:
            node = target.node()._impl
            return node[0], tuple(node[1])
        except Exception as exc:
            raise RealizedTraceError("target must be a modelx ItemNode or Cells-like scalar") from exc
    raise RealizedTraceError("target must be a modelx ItemNode or Cells-like scalar")


def _space_instance_uid(obj: Any) -> str:
    parent = getattr(obj, "parent", None)
    if parent is None:
        return "space:<unknown>"
    try:
        text = parent.get_repr(fullname=True, add_params=True)
    except Exception:
        text = repr(parent)
    return "space:" + hashlib.sha1(text.encode("utf-8", errors="backslashreplace")).hexdigest()[:16]


def _space_family_uid(obj: Any) -> str:
    """Identity shared by dynamic ItemSpace clones of one structural Space family."""
    parent = getattr(obj, "parent", None)
    if parent is None:
        return "family:<unknown>"

    # ItemSpace implementations expose their static base in ``bases``. Walk toward
    # the first non-dynamic structural Space.  This intentionally does not inspect
    # parameter names or values.
    seen: set[int] = set()
    cur = parent
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        try:
            dynamic = bool(cur.is_dynamic())
        except Exception:
            dynamic = False
        if not dynamic:
            break
        try:
            bases = list(cur.bases)
        except Exception:
            bases = []
        if bases:
            cur = bases[0]
        else:
            cur = getattr(cur, "parent", None)

    try:
        text = cur.get_repr(fullname=True, add_params=False)
    except Exception:
        try:
            text = cur.interface.fullname
        except Exception:
            text = repr(cur)
    return "family:" + hashlib.sha1(text.encode("utf-8", errors="backslashreplace")).hexdigest()[:16]


def _logical_uid(schema: CellSchema, obj: Any, key: tuple[Any, ...]) -> str:
    # The runtime tuple remains the authoritative identity.  This UID is diagnostic
    # only; repr is deliberately not used for equality or replay.
    try:
        space = obj.parent.get_repr(fullname=True, add_params=True)
    except Exception:
        space = repr(getattr(obj, "parent", None))
    payload = f"{schema.uid}|{space}|{key!r}"
    return "n_" + hashlib.sha1(payload.encode("utf-8", errors="backslashreplace")).hexdigest()[:16]


def _detect_recursive_uncached(events: Sequence[TraceEvent]) -> list[str]:
    active: list[TraceEvent] = []
    problems: list[str] = []
    seen_problem: set[int] = set()
    for ev in events:
        if ev.kind == "ENTER":
            if not ev.cached and any(a.obj is ev.obj for a in active):
                if id(ev.obj) not in seen_problem:
                    try:
                        name = ev.obj.get_repr(fullname=True, add_params=True)
                    except Exception:
                        name = repr(ev.obj)
                    problems.append(name)
                    seen_problem.add(id(ev.obj))
            active.append(ev)
        elif ev.kind == "EXIT":
            if not active:
                raise RealizedTraceError("malformed modelx trace: EXIT without ENTER")
            active.pop()
        else:
            raise RealizedTraceError(f"unknown modelx trace event {ev.kind!r}")
    if active:
        raise RealizedTraceError("malformed modelx trace: unterminated ENTER events")
    return problems


def capture_realized_trace(
    model: Any,
    targets: Any,
    *,
    clear_first: bool = True,
) -> RealizedTrace:
    """Capture one exact, realized modelx calculation.

    Unlike :func:`capture_trace`, this function performs no coordinate inference,
    dtype inference or static generalization.  The cold runtime trace is the source
    of truth.  Cached Cells EXIT order becomes the candidate Stage-1 schedule.
    """

    target_objs = _normalize_targets(targets)
    target_runtime = tuple(_target_runtime_node(t) for t in target_objs)

    if clear_first:
        clear_realized_calculated(model)

    system = model._impl.system
    raw_events = None
    values: list[Any] = []
    with system.trace_stack(maxlen=None):
        for obj, key in target_runtime:
            values.append(obj.get_value_from_key(key))
        # trace_stack() clears records in its finally block, so copy here.
        raw_events = list(system.callstack.tracestack)

    assert raw_events is not None
    events = tuple(
        TraceEvent(
            kind=str(sign),
            depth=int(depth),
            sequence=i,
            obj=item[0],
            key=tuple(item[1]),
            cached=bool(item[0].is_cached),
        )
        for i, (sign, depth, _stamp, item) in enumerate(raw_events)
    )

    recursive_uncached = _detect_recursive_uncached(events)
    if recursive_uncached:
        names = ", ".join(recursive_uncached)
        raise RealizedTraceError(
            "realized trace contains recursive uncached Cells; exact cache-driven "
            f"de-recursion is not semantics-preserving for: {names}"
        )

    exit_nodes: list[tuple[Any, tuple[Any, ...]]] = []
    seen_exit: set[tuple[Any, tuple[Any, ...]]] = set()
    for ev in events:
        if ev.kind == "EXIT" and ev.cached:
            node = ev.runtime_node
            if node in seen_exit:
                # A cold cached node should be formula-evaluated at most once.  If
                # modelx produces otherwise, preserve correctness by refusing to
                # pretend that simple cache replay is sufficient.
                raise RealizedTraceError(f"cached node evaluated more than once in cold trace: {node!r}")
            seen_exit.add(node)
            exit_nodes.append(node)

    # The persistent graph also contains explicit input nodes, which are cache hits
    # and therefore absent from the formula ENTER/EXIT trace.  Clearing calculated
    # values above removes stale calculated nodes and descendant edges.
    graph = model._impl.tracegraph
    keep: set[Any] = set(exit_nodes)
    for target in target_runtime:
        if target in graph:
            keep.add(target)
            keep.update(nx.ancestors(graph, target))
    subgraph = graph.subgraph(keep).copy()

    calculated_graph = subgraph.subgraph(exit_nodes).copy()
    if not nx.is_directed_acyclic_graph(calculated_graph):
        raise RealizedTraceError("realized cached Cell graph contains a concrete cycle")

    exit_rank = {node: i for i, node in enumerate(exit_nodes)}
    for src, dst in calculated_graph.edges:
        if exit_rank[src] >= exit_rank[dst]:
            raise RealizedTraceError(
                "cached EXIT order is not a dependency order: "
                f"{src!r} -> {dst!r}"
            )

    all_runtime_nodes: list[tuple[Any, tuple[Any, ...]]] = list(exit_nodes)
    for node in subgraph.nodes:
        if node not in seen_exit:
            all_runtime_nodes.append((node[0], tuple(node[1])))

    runtime_to_id: dict[tuple[Any, tuple[Any, ...]], int] = {
        node: i for i, node in enumerate(all_runtime_nodes)
    }
    target_set = set(target_runtime)
    concrete: list[ConcreteNode] = []
    diagnostics: list[str] = []
    for i, runtime_node in enumerate(all_runtime_nodes):
        obj, key = runtime_node
        schema = _schema_for_impl(obj)
        if schema is None:
            # TraceGraph can theoretically contain non-Cells trace objects.  They
            # are not executable Stage-1 Cell operations; retain a diagnostic and
            # exclude them from the compact node table.
            diagnostics.append(f"non-Cells trace node ignored: {runtime_node!r}")
            continue
        try:
            is_input = key in obj.input_keys
        except Exception:
            is_input = False
        concrete.append(
            ConcreteNode(
                node_id=i,
                uid=_logical_uid(schema, obj, key),
                schema=schema,
                schema_uid=schema.uid,
                space_instance_uid=_space_instance_uid(obj),
                space_family_uid=_space_family_uid(obj),
                args=tuple(key),
                obj=obj,
                is_input=bool(is_input),
                is_target=runtime_node in target_set,
                exit_rank=exit_rank.get(runtime_node),
            )
        )

    concrete_ids = {n.node_id for n in concrete}
    deps: list[tuple[int, int]] = []
    for src, dst in subgraph.edges:
        sid = runtime_to_id.get((src[0], tuple(src[1])))
        did = runtime_to_id.get((dst[0], tuple(dst[1])))
        if sid in concrete_ids and did in concrete_ids:
            deps.append((sid, did))

    schedule_ids = tuple(runtime_to_id[n] for n in exit_nodes if runtime_to_id[n] in concrete_ids)
    target_ids = tuple(runtime_to_id.get(n) if runtime_to_id.get(n) in concrete_ids else None for n in target_runtime)

    return RealizedTrace(
        model=model,
        events=events,
        nodes=tuple(concrete),
        dependencies=tuple(deps),
        schedule_node_ids=schedule_ids,
        target_node_ids=target_ids,
        target_runtime_nodes=target_runtime,
        target_values=tuple(values),
        diagnostics=tuple(diagnostics),
    )
