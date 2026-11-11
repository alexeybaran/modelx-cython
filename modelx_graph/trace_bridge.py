from __future__ import annotations

"""Adapters between the realized compiler and the reusable graph/codegen frontend.

This is intentionally a bridge, not a second graph analyzer.  The realized trace
has already executed modelx and captured concrete dependency edges.  We reuse the
single graph->TraceCapture lowering in :mod:`modelx_graph.trace` so NativeBatch can
consume that evidence without running ``capture_trace`` a second time.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx

from .trace import TraceCapture, build_trace_capture_from_graph


@dataclass(frozen=True)
class _TargetNodeProxy:
    _impl: tuple[Any, tuple[Any, ...]]


def trace_capture_from_realized(
    realized,
    *,
    sample_keys: Sequence[Any],
) -> TraceCapture:
    """Convert an already-captured ``RealizedTrace`` without evaluating modelx.

    No formula/value specialization occurs here.  Concrete values are consulted by
    the shared trace lowerer only for the same dtype/shape evidence used by the
    legacy frontend.  Dependency structure and target identity come directly from
    the realized compiler.
    """
    node_by_id = realized.node_by_id
    graph = nx.DiGraph()
    runtime_by_id: dict[int, tuple[Any, tuple[Any, ...]]] = {}
    for node in realized.nodes:
        runtime = (node.obj, tuple(node.args))
        runtime_by_id[node.node_id] = runtime
        graph.add_node(runtime)
    for source_id, target_id in realized.dependencies:
        source = runtime_by_id.get(source_id)
        target = runtime_by_id.get(target_id)
        if source is not None and target is not None:
            graph.add_edge(source, target)

    targets = tuple(
        _TargetNodeProxy((obj, tuple(key)))
        for obj, key in realized.target_runtime_nodes
    )
    return build_trace_capture_from_graph(
        graph,
        target_nodes=targets,
        sample_keys=tuple(sample_keys),
        diagnostics_prefix=tuple(realized.diagnostics) + (
            "TraceCapture reused from RealizedTrace; no second modelx trace executed",
        ),
    )
