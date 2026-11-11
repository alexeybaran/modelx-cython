from __future__ import annotations

import ast
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import networkx as nx
import numpy as np


class TraceError(RuntimeError):
    pass


@dataclass(frozen=True)
class CellSchema:
    """Identity of a Cells object independent of a concrete ItemSpace instance."""

    fullname: str
    name: str
    parameters: tuple[str, ...]
    source: str

    @property
    def uid(self) -> str:
        digest = hashlib.sha1((self.fullname + "\n" + self.source).encode()).hexdigest()[:12]
        return f"c_{digest}"


@dataclass(frozen=True)
class VariantKey:
    schema_uid: str
    time_pos: int | None
    aux_values: tuple[Any, ...] = ()


@dataclass
class VariantTrace:
    key: VariantKey
    schema: CellSchema
    dtype: str
    observed_times: tuple[int, ...] = ()
    observed_values: tuple[Any, ...] = ()
    has_input_values: bool = False

    @property
    def is_time(self) -> bool:
        return self.key.time_pos is not None

    @property
    def uid(self) -> str:
        aux = repr(self.key.aux_values)
        return "v_" + hashlib.sha1((self.key.schema_uid + "|" + aux).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class VariantEdge:
    source: VariantKey
    target: VariantKey
    offsets: tuple[int, ...] = ()  # source time - target time; empty for non-time edges
    count: int = 0


@dataclass
class TraceCapture:
    graph: nx.DiGraph
    schemas: dict[str, CellSchema]
    variants: dict[VariantKey, VariantTrace]
    edges: list[VariantEdge]
    node_variant: dict[Any, VariantKey]
    itemspace_keys: tuple[Any, ...]
    target_nodes: tuple[Any, ...]
    diagnostics: list[str] = field(default_factory=list)

    def variant(self, key: VariantKey) -> VariantTrace:
        return self.variants[key]


def _schema_for_impl(obj) -> CellSchema | None:
    """Return the base Cells schema for a trace node, not its ItemSpace clone."""
    try:
        interface = obj.interface
        if interface.__class__.__name__ != "Cells":
            return None
        params = tuple(interface.parameters)
        name = obj.name
    except Exception:
        return None

    base_cell = None
    try:
        parent = obj.parent
        bases = getattr(parent, "bases", None) or []
        if bases:
            base_space = bases[0].interface
            if hasattr(base_space, "cells") and name in base_space.cells:
                base_cell = base_space.cells[name]
    except Exception:
        base_cell = None

    cell = base_cell if base_cell is not None else obj.interface
    try:
        source = cell.formula.source
    except Exception:
        try:
            source = str(cell.formula)
        except Exception:
            return None
    fullname = getattr(cell, "fullname", None) or name
    return CellSchema(fullname=fullname, name=name, parameters=tuple(cell.parameters), source=source)


def _numeric_dtype(values: Sequence[Any]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "unknown"
    if all(isinstance(v, (bool, np.bool_)) for v in vals):
        return "bool"
    if all(isinstance(v, (int, np.integer, bool, np.bool_)) for v in vals):
        return "int64"
    if all(isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) for v in vals):
        return "float64"
    return "object"


def _candidate_time_positions(keys_by_item: dict[Any, list[tuple]], arity: int) -> list[tuple[float, int]]:
    scores: list[tuple[float, int]] = []
    for pos in range(arity):
        total_unique = 0
        continuity = []
        useful_groups = 0
        for keys in keys_by_item.values():
            vals = [k[pos] for k in keys]
            if not vals or not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in vals):
                continue
            uniq = sorted(set(int(v) for v in vals))
            if len(uniq) < 3:
                continue
            useful_groups += 1
            total_unique += len(uniq)
            if len(uniq) > 1:
                diffs = [b - a for a, b in zip(uniq, uniq[1:])]
                # Coordinate evidence is an integer arithmetic progression, not
                # necessarily a unit-spaced one.  Static formula/range proof later
                # establishes the signed executable step.
                counts = Counter(diffs)
                stride, freq = counts.most_common(1)[0]
                continuity.append(freq / len(diffs) if stride > 0 else 0.0)
        if useful_groups:
            c = sum(continuity) / len(continuity) if continuity else 0.0
            # Strongly prefer long, near-contiguous sequences within an ItemSpace.
            score = total_unique * (0.25 + 0.75 * c)
            if c >= 0.65:
                scores.append((score, pos))
    return sorted(scores, reverse=True)


def _source_recursive_coordinate_positions(schema: CellSchema) -> set[int]:
    """Return parameter positions with a proven affine self-recursive shift.

    This is only candidate evidence for coordinate-role propagation.  A caller/callee
    graph alignment with an already-proven coordinate is still required before the
    position is accepted, so unrelated recursive integer helpers are not promoted
    merely because they recurse.
    """
    try:
        tree = ast.parse(schema.source)
    except Exception:
        return set()
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return set()
    params = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
    if tuple(params) != tuple(schema.parameters):
        return set()

    def shifted(expr: ast.AST, param: str) -> bool:
        if not isinstance(expr, ast.BinOp) or not isinstance(expr.op, (ast.Add, ast.Sub)):
            return False
        if isinstance(expr.left, ast.Name) and expr.left.id == param and isinstance(expr.right, ast.Constant):
            value = expr.right.value
            return isinstance(value, int) and not isinstance(value, bool) and int(value) != 0
        if isinstance(expr.op, ast.Add) and isinstance(expr.right, ast.Name) and expr.right.id == param and isinstance(expr.left, ast.Constant):
            value = expr.left.value
            return isinstance(value, int) and not isinstance(value, bool) and int(value) != 0
        return False

    out: set[int] = set()
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == schema.name):
            continue
        if node.keywords or len(node.args) != len(params):
            continue
        for pos, param in enumerate(params):
            if not shifted(node.args[pos], param):
                continue
            others_ok = all(
                j == pos or (isinstance(node.args[j], ast.Name) and node.args[j].id == params[j])
                for j in range(len(params))
            )
            if others_ok:
                out.add(pos)
    return out


def _propagate_sparse_time_positions(
    graph: nx.DiGraph, schemas_by_obj: dict[Any, CellSchema], known: dict[Any, int]
) -> dict[Any, int]:
    """Propagate a known coordinate role to sparse recursive callees by graph alignment.

    At least two distinct aligned coordinates are required.  Only source parameters
    independently proven to participate in an affine self-recursive shift are
    eligible.  Evidence is monotonic and ambiguous candidates remain unknown.
    """
    result = dict(known)
    changed = True
    while changed:
        changed = False
        for obj, schema in schemas_by_obj.items():
            if obj in result or not schema.parameters:
                continue
            source_candidates = _source_recursive_coordinate_positions(schema)
            if not source_candidates:
                continue
            evidence: dict[int, set[int]] = {pos: set() for pos in source_candidates}
            contradicted: set[int] = set()
            for a, b in graph.edges:
                if a[0] is obj and b[0] in result:
                    unknown_node, known_node = a, b
                elif b[0] is obj and a[0] in result:
                    unknown_node, known_node = b, a
                else:
                    continue
                known_pos = result[known_node[0]]
                try:
                    known_coord = int(known_node[1][known_pos])
                except Exception:
                    continue
                for pos in source_candidates:
                    try:
                        value = unknown_node[1][pos]
                    except Exception:
                        contradicted.add(pos)
                        continue
                    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and int(value) == known_coord:
                        evidence[pos].add(known_coord)
                    else:
                        contradicted.add(pos)
            viable = [pos for pos, vals in evidence.items() if pos not in contradicted and len(vals) >= 2]
            if len(viable) == 1:
                result[obj] = viable[0]
                changed = True
    return result


def _itemspace_key(obj) -> Any:
    try:
        vals = obj.parent.argvalues
        out = []
        for v in vals:
            try:
                out.append(v.value)
            except Exception:
                out.append(v)
        return tuple(out)
    except Exception:
        return ()


def _choose_time_positions(nodes, schemas_by_obj, graph: nx.DiGraph) -> dict[Any, int]:
    """Infer a time parameter position from observed node keys and graph recurrences.

    Names are deliberately ignored.  A candidate must vary as a mostly contiguous
    integer sequence inside a single ItemSpace.  Recurring self-edges whose other
    arguments are unchanged provide a strong additional signal.
    """
    keys_by_obj_item: dict[Any, dict[Any, list[tuple]]] = defaultdict(lambda: defaultdict(list))
    for n in nodes:
        obj, key = n
        schema = schemas_by_obj.get(obj)
        if schema is None or len(key) != len(schema.parameters):
            continue
        keys_by_obj_item[obj][_itemspace_key(obj)].append(tuple(key))

    result: dict[Any, int] = {}
    for obj, groups in keys_by_obj_item.items():
        schema = schemas_by_obj[obj]
        candidates = _candidate_time_positions(groups, len(schema.parameters))
        if not candidates:
            continue
        rescored = []
        for base_score, pos in candidates:
            self_rec = 0
            for a, b in graph.edges:
                if a[0] is not obj or b[0] is not obj:
                    continue
                ka, kb = a[1], b[1]
                if len(ka) != len(schema.parameters) or len(kb) != len(schema.parameters):
                    continue
                if all(ka[j] == kb[j] for j in range(len(ka)) if j != pos):
                    d = int(ka[pos]) - int(kb[pos])
                    if d and abs(d) <= 8:
                        self_rec += 1
            rescored.append((base_score + 20.0 * self_rec, pos))
        rescored.sort(reverse=True)
        # Ambiguity is a compiler diagnostic, not an invitation to guess.
        if len(rescored) > 1 and rescored[1][0] >= 0.95 * rescored[0][0]:
            continue
        result[obj] = rescored[0][1]
    return result




def _clear_calculated_preserve_inputs(model) -> None:
    """Clear calculated Cell values without destroying explicit Cell inputs.

    ``Model.clear_all`` intentionally removes both inputs and calculations.  A
    compiler sample run must not mutate model semantics that way.  modelx already
    exposes the required distinction internally through ``clear_all_cells``.
    """
    for space in model.spaces.values():
        try:
            space._impl.clear_all_cells(clear_input=False, recursive=True, del_items=False)
        except Exception as exc:
            raise TraceError(
                f"could not clear calculated values while preserving inputs for {space.fullname}"
            ) from exc


def build_trace_capture_from_graph(
    graph: nx.DiGraph,
    *,
    target_nodes: Sequence[Any],
    sample_keys: Sequence[Any],
    diagnostics_prefix: Sequence[str] = (),
) -> TraceCapture:
    """Lower an already-evaluated concrete modelx graph into ``TraceCapture``.

    This is the single graph-to-variant/loop-evidence implementation used by both
    the legacy sampling frontend and the realized-trace compiler bridge.  It does
    not execute modelx and never infers constants from calculated values; values
    are observed only for numeric dtype/shape evidence, exactly as before.
    """
    schemas_by_obj: dict[Any, CellSchema] = {}
    schemas: dict[str, CellSchema] = {}
    for obj, key in graph.nodes:
        schema = _schema_for_impl(obj)
        if schema is None or len(key) != len(schema.parameters):
            continue
        schemas_by_obj[obj] = schema
        schemas[schema.uid] = schema

    time_pos_by_obj = _choose_time_positions(graph.nodes, schemas_by_obj, graph)
    time_pos_by_obj = _propagate_sparse_time_positions(graph, schemas_by_obj, time_pos_by_obj)

    # Dynamic ItemSpaces clone Cells implementations.  Propagate the inferred time
    # position across clones of the same base Cells schema, requiring agreement.
    schema_positions: dict[str, Counter] = defaultdict(Counter)
    for obj, pos in time_pos_by_obj.items():
        schema_positions[schemas_by_obj[obj].uid][pos] += 1
    inferred_schema_pos: dict[str, int] = {}
    diagnostics: list[str] = list(diagnostics_prefix)
    for uid, counts in schema_positions.items():
        if not counts:
            continue
        pos, n = counts.most_common(1)[0]
        if len(counts) > 1:
            diagnostics.append(f"time-axis votes for {schemas[uid].fullname}: {dict(counts)}; selected parameter index {pos}")
        inferred_schema_pos[uid] = pos

    node_variant: dict[Any, VariantKey] = {}
    values: dict[VariantKey, list[Any]] = defaultdict(list)
    times: dict[VariantKey, set[int]] = defaultdict(set)
    input_flags: dict[VariantKey, bool] = defaultdict(bool)
    for n in graph.nodes:
        obj, key = n
        schema = schemas_by_obj.get(obj)
        if schema is None or len(key) != len(schema.parameters):
            continue
        tpos = inferred_schema_pos.get(schema.uid)
        if tpos is None:
            vk = VariantKey(schema.uid, None, tuple(key)) if key else VariantKey(schema.uid, None, ())
        else:
            aux = tuple(v for j, v in enumerate(key) if j != tpos)
            vk = VariantKey(schema.uid, tpos, aux)
            try:
                times[vk].add(int(key[tpos]))
            except Exception:
                pass
        node_variant[n] = vk
        try:
            if key in obj.input_keys:
                input_flags[vk] = True
        except Exception:
            pass
        try:
            values[vk].append(obj.data[key])
        except Exception:
            pass

    variants: dict[VariantKey, VariantTrace] = {}
    for vk, vals in values.items():
        variants[vk] = VariantTrace(
            key=vk,
            schema=schemas[vk.schema_uid],
            dtype=_numeric_dtype(vals),
            observed_times=tuple(sorted(times.get(vk, set()))),
            observed_values=tuple(vals[:32]),
            has_input_values=bool(input_flags.get(vk, False)),
        )

    edge_counts: dict[tuple[VariantKey, VariantKey], Counter] = defaultdict(Counter)
    simple_counts: Counter = Counter()
    for a, b in graph.edges:
        if a not in node_variant or b not in node_variant:
            continue
        va, vb = node_variant[a], node_variant[b]
        if va.time_pos is not None and vb.time_pos is not None:
            try:
                delta = int(a[1][va.time_pos]) - int(b[1][vb.time_pos])
            except Exception:
                continue
            edge_counts[(va, vb)][delta] += 1
        else:
            simple_counts[(va, vb)] += 1

    edges: list[VariantEdge] = []
    for pair, counter in edge_counts.items():
        edges.append(VariantEdge(pair[0], pair[1], tuple(sorted(counter)), sum(counter.values())))
    for pair, count in simple_counts.items():
        edges.append(VariantEdge(pair[0], pair[1], (), count))

    return TraceCapture(
        graph=graph,
        schemas=schemas,
        variants=variants,
        edges=edges,
        node_variant=node_variant,
        itemspace_keys=tuple(sample_keys),
        target_nodes=tuple(target_nodes),
        diagnostics=diagnostics,
    )

def capture_trace(
    model,
    target_factory: Callable[[Any], Any],
    sample_keys: Sequence[Any],
    *,
    clear_first: bool = True,
) -> TraceCapture:
    """Run representative targets and capture the modelx calculation graph.

    ``target_factory(key)`` must return either an ItemNode or a zero-argument
    callable/Cells value associated with the ItemSpace represented by ``key``.
    The compiler never interprets the key or any Cell name semantically.
    """
    if clear_first:
        _clear_calculated_preserve_inputs(model)

    target_nodes = []
    for key in sample_keys:
        target = target_factory(key)
        # ItemNode has _impl.  If a caller supplied a Cells bound object instead,
        # execute it and then obtain node() if available.
        if hasattr(target, "_impl") and target.__class__.__name__.endswith("Node"):
            node = target
            impl = node._impl
            impl[0].get_value_from_key(impl[1])
        elif hasattr(target, "node"):
            node = target.node()
            impl = node._impl
            impl[0].get_value_from_key(impl[1])
        elif callable(target):
            target()
            try:
                node = target.node()
            except Exception as exc:
                raise TraceError("target_factory must return a modelx ItemNode or Cells-like object") from exc
        else:
            raise TraceError("target_factory must return a modelx ItemNode or Cells-like object")
        target_nodes.append(node)

    # ``model._impl.tracegraph`` is persistent across evaluations.  A compiler
    # invocation must consume only the precedent closure of the targets requested
    # in *this* capture; otherwise a previous trace can contaminate a later
    # regional/subgraph compile with unrelated downstream formulas.  modelx edges
    # are precedent -> dependent, so the exact captured closure is targets plus
    # their graph ancestors.
    full_graph = model._impl.tracegraph.copy()
    target_impls = tuple(node._impl for node in target_nodes)
    keep = set(target_impls)
    for target in target_impls:
        if target in full_graph:
            keep.update(nx.ancestors(full_graph, target))
    graph = full_graph.subgraph(keep).copy()

    return build_trace_capture_from_graph(
        graph, target_nodes=target_nodes, sample_keys=sample_keys
    )
