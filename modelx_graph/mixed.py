from __future__ import annotations

"""Automatic mixed native/Python regional execution.

The partitioner works at formula-Cell identity level, not at actuarial names.  It
uses static same-Space call edges plus representative trace type evidence to
separate maximal native-candidate and fallback regions.  Unsupported zero-
argument numeric Cells become explicit typed boundaries.  Fallback regions are
executed from the original formula sources in isolated per-point namespaces;
compiled state or modelx Cell caches never cross a boundary.

The first implementation intentionally fails closed for parameterized fallback
Cells and parameterized values crossing region boundaries.  Those cases need a
richer call-domain schema rather than an accidental approximation.
"""

import ast
import builtins
import hashlib
import importlib.util
import itertools
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from .frontend import GraphModelCompiler, FrontendError
from .python_codegen import PythonLoopGenerator
from .codegen import CythonGenerator
from .build import build_extension
from .safety import validate_formula_control_flow, SemanticSafetyError
from .trace import capture_trace


class PartitionError(RuntimeError):
    pass


def _digest(parts: Iterable[str], prefix: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "backslashreplace"))
        h.update(b"\0")
    return f"{prefix}_{h.hexdigest()[:16]}"


def _fn(cell) -> ast.FunctionDef | None:
    formula = getattr(cell, "formula", None)
    if formula is None:
        return None
    node = ast.parse(formula.source).body[0]
    return node if isinstance(node, ast.FunctionDef) else None


def _strip_doc(fn: ast.FunctionDef) -> list[ast.stmt]:
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def _same_space_calls(fn: ast.FunctionDef, names: set[str]) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names:
            out.add(n.func.id)
        elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in names:
            out.add(n.value.id)
    return out


def _looks_like_point_row(fn: ast.FunctionDef, space_params: tuple[str, ...], refs: Mapping[str, Any]) -> bool:
    if len(space_params) != 1:
        return False
    body = _strip_doc(fn)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    r = body[0].value
    return bool(
        isinstance(r, ast.Subscript)
        and isinstance(r.value, ast.Attribute)
        and r.value.attr == "loc"
        and isinstance(r.value.value, ast.Name)
        and r.value.value.id in refs
        and isinstance(r.slice, ast.Name)
        and r.slice.id == space_params[0]
    )


def _looks_like_vector_helper(fn: ast.FunctionDef, refs: Mapping[str, Any]) -> bool:
    body = _strip_doc(fn)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    r = body[0].value
    if not isinstance(r, ast.Call) or len(r.args) != 1:
        return False
    if not (isinstance(r.func, ast.Attribute) and isinstance(r.func.value, ast.Name) and r.func.attr == "array"):
        return False
    obj = refs.get(r.func.value.id)
    if obj is not np:
        return False
    arg = r.args[0]
    if isinstance(arg, ast.GeneratorExp):
        return True
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id in {"list", "tuple"} and arg.args and isinstance(arg.args[0], ast.GeneratorExp):
        return True
    return False


def _looks_like_reduction(fn: ast.FunctionDef) -> bool:
    body = _strip_doc(fn)
    if len(body) == 3 and isinstance(body[0], ast.Assign) and isinstance(body[1], ast.For) and isinstance(body[2], ast.Return):
        loop = body[1]
        return bool(
            isinstance(loop.iter, ast.Call)
            and isinstance(loop.iter.func, ast.Name)
            and loop.iter.func.id == "range"
            and len(loop.body) == 1
            and isinstance(loop.body[0], ast.AugAssign)
            and isinstance(loop.body[0].op, ast.Add)
        )
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    r = body[0].value
    return bool(
        isinstance(r, ast.Call)
        and isinstance(r.func, ast.Name)
        and r.func.id == "sum"
        and len(r.args) == 1
        and (
            isinstance(r.args[0], ast.GeneratorExp)
            or (
                isinstance(r.args[0], ast.Call)
                and isinstance(r.args[0].func, ast.Name)
                and r.args[0].func.id in {"list", "tuple"}
                and r.args[0].args
                and isinstance(r.args[0].args[0], ast.GeneratorExp)
            )
            or isinstance(r.args[0], ast.BinOp)
        )
    )


_ALLOWED_CALLS = {"max", "min", "abs", "round", "int", "float", "bool", "sum", "range", "list", "tuple"}
_ALLOWED_MATH = {"exp", "log", "sqrt", "sin", "cos", "floor", "ceil"}


def _native_expression_reason(fn: ast.FunctionDef, cell_names: set[str], refs: Mapping[str, Any]) -> str | None:
    """Cheap conservative pre-classifier.

    This is not the compiler proof.  A cell marked native here still has to pass
    the real frontend/template/codegen.  The classifier's job is to find obvious
    Python islands early enough to partition them automatically.
    """
    reduction = _looks_like_reduction(fn)
    if not reduction:
        try:
            validate_formula_control_flow(fn)
        except SemanticSafetyError as exc:
            return f"control_flow:{exc}"

    for n in ast.walk(fn):
        if isinstance(n, (ast.Dict, ast.Set, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.Yield, ast.YieldFrom, ast.Await)):
            return f"python_object_syntax:{type(n).__name__}"
        if isinstance(n, ast.List) and not reduction:
            return "python_object_syntax:List"
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                name = n.func.id
                if name in cell_names or name in _ALLOWED_CALLS:
                    continue
                # np.array vector helpers are classified before this function.
                return f"unsupported_call:{name}"
            if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
                receiver = n.func.value.id
                obj = refs.get(receiver)
                module_name = getattr(obj, "__name__", None)
                if module_name in ("math", "numpy") and n.func.attr in _ALLOWED_MATH:
                    continue
                # Calls such as local_list.sort() are intentionally fallback.
                return f"unsupported_method:{receiver}.{n.func.attr}"
            return "unsupported_dynamic_call"
    return None


def _dtype_from_values(values: Sequence[Any]) -> tuple[str, tuple[int, ...]]:
    if not values:
        return "object", ()
    shapes = {tuple(np.asarray(v).shape) for v in values}
    if len(shapes) != 1:
        return "object", ()
    shape = next(iter(shapes))
    kinds = {np.asarray(v).dtype.kind for v in values}
    if kinds <= {"b"}:
        return "bool", shape
    if kinds <= {"b", "i", "u"}:
        return "int64", shape
    if kinds <= {"b", "i", "u", "f"}:
        return "float64", shape
    return "object", shape


@dataclass(frozen=True)
class CellSupport:
    name: str
    mode: str  # compiled | fallback | helper
    reason: str
    dtype: str
    shape: tuple[int, ...]
    parameters: tuple[str, ...]
    observed: bool


@dataclass(frozen=True)
class BoundaryFieldSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    producer_region: str
    consumer_region: str

    @property
    def ndim(self) -> int:
        # leading point dimension is implicit in the schema
        return 1 + len(self.shape)


@dataclass
class MixedRegion:
    uid: str
    index: int
    mode: str
    cells: tuple[str, ...]
    inputs: tuple[BoundaryFieldSpec, ...] = ()
    outputs: tuple[BoundaryFieldSpec, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass
class MixedPartitionPlan:
    output: str
    regions: tuple[MixedRegion, ...]
    support: dict[str, CellSupport]
    static_edges: tuple[tuple[str, str], ...]  # dependency -> consumer
    sample_keys: tuple[Any, ...]
    run_keys: tuple[Any, ...]
    formula_hash: str
    diagnostics: tuple[str, ...] = ()

    @property
    def fallback_regions(self) -> tuple[MixedRegion, ...]:
        return tuple(r for r in self.regions if r.mode == "fallback")

    @property
    def compiled_regions(self) -> tuple[MixedRegion, ...]:
        return tuple(r for r in self.regions if r.mode == "compiled")

    @property
    def boundary_fields(self) -> tuple[BoundaryFieldSpec, ...]:
        seen: dict[tuple[str, str, str], BoundaryFieldSpec] = {}
        for r in self.regions:
            for f in r.outputs:
                seen[(f.name, f.producer_region, f.consumer_region)] = f
        return tuple(seen[k] for k in sorted(seen))

    def manifest(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "formula_hash": self.formula_hash,
            "sample_keys": [repr(x) for x in self.sample_keys],
            "run_key_count": len(self.run_keys),
            "regions": [
                {
                    "uid": r.uid,
                    "index": r.index,
                    "mode": r.mode,
                    "cells": list(r.cells),
                    "inputs": [f.__dict__.copy() for f in r.inputs],
                    "outputs": [f.__dict__.copy() for f in r.outputs],
                    "reasons": list(r.reasons),
                }
                for r in self.regions
            ],
            "support": {k: v.__dict__.copy() for k, v in sorted(self.support.items())},
            "static_edges": [list(x) for x in self.static_edges],
            "diagnostics": list(self.diagnostics),
        }


class BoundaryBatch:
    """Typed, ephemeral values crossing regional execution boundaries."""

    def __init__(self, npoints: int):
        self.npoints = int(npoints)
        self.values: dict[str, np.ndarray] = {}
        self.specs: dict[str, BoundaryFieldSpec] = {}

    def put(self, spec: BoundaryFieldSpec, value: Any) -> np.ndarray:
        if spec.dtype == "bool":
            dt = np.bool_
        elif spec.dtype == "int64":
            dt = np.int64
        elif spec.dtype == "float64":
            dt = np.float64
        else:
            raise PartitionError(f"boundary {spec.name} has non-numeric dtype {spec.dtype!r}")
        arr = np.ascontiguousarray(np.asarray(value, dtype=dt))
        expected = (self.npoints, *spec.shape)
        if arr.shape != expected:
            raise PartitionError(f"boundary {spec.name} shape {arr.shape}, expected {expected}")
        self.values[spec.name] = arr
        self.specs[spec.name] = spec
        return arr

    def get(self, name: str) -> np.ndarray:
        if name not in self.values:
            raise PartitionError(f"boundary value {name!r} is not available")
        return self.values[name]

    def manifest(self) -> dict[str, Any]:
        return {
            "npoints": self.npoints,
            "fields": {
                k: {"dtype": self.specs[k].dtype, "shape": list(v.shape), "bytes": int(v.nbytes)}
                for k, v in sorted(self.values.items())
            },
            "ownership": "ephemeral per-run boundary batch; no compiled/modelx cache ownership crosses regions",
        }


def _closure(output: str, calls: Mapping[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    stack = [output]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(calls.get(name, ()))
    return seen


def _formula_hash(space, names: Iterable[str]) -> str:
    parts = []
    for name in sorted(names):
        formula = getattr(space.cells[name], "formula", None)
        parts.append(f"{name}:{getattr(formula, 'source', '')}")
    return _digest(parts, "mixed_formula")


def _observed_cell_values(trace) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = defaultdict(list)
    for node, vk in trace.node_variant.items():
        tr = trace.variants[vk]
        try:
            value = node[0].data[node[1]]
        except Exception:
            continue
        out[tr.schema.name].append(value)
    return out


def build_mixed_partition(
    model,
    *,
    space,
    output: str,
    sample_keys: Sequence[Any],
    run_keys: Sequence[Any] | None = None,
) -> MixedPartitionPlan:
    """Automatically partition the static output closure into maximal regions."""
    if output not in space.cells:
        raise PartitionError(f"unknown output Cell {output!r}")
    sample_keys = tuple(sample_keys)
    if run_keys is None:
        # Reuse the ordinary frontend's key inference, but avoid asking it to lower
        # unsupported formulas.  DataFrame-row ItemSpaces are the common case.
        params = tuple(space.parameters or ())
        inferred = None
        if len(params) == 1:
            for name, cell in space.cells.items():
                fn = _fn(cell)
                if fn is not None and _looks_like_point_row(fn, params, space.refs):
                    body = _strip_doc(fn)
                    ret = body[0].value
                    refname = ret.value.value.id  # type: ignore[union-attr]
                    inferred = tuple(space.refs[refname].index)
                    break
        if inferred is None:
            raise PartitionError("run_keys are required when automatic mixed partitioning cannot infer the ItemSpace domain")
        run_keys = inferred
    run_keys = tuple(run_keys)

    names = set(space.cells)
    funcs = {name: _fn(cell) for name, cell in space.cells.items()}
    calls = {name: (_same_space_calls(fn, names) if fn is not None else set()) for name, fn in funcs.items()}
    needed = _closure(output, calls)

    trace = capture_trace(model, lambda k: getattr(space[k], output).node(), sample_keys)
    observed = _observed_cell_values(trace)
    params = tuple(space.parameters or ())
    support: dict[str, CellSupport] = {}
    diagnostics: list[str] = []

    for name in sorted(needed):
        cell = space.cells[name]
        fn = funcs[name]
        vals = observed.get(name, [])
        dtype, shape = _dtype_from_values(vals)
        cell_params = tuple(cell.parameters or ())
        if fn is None:
            support[name] = CellSupport(name, "fallback", "missing_formula", dtype, shape, cell_params, bool(vals))
            continue
        if _looks_like_point_row(fn, params, space.refs) or _looks_like_vector_helper(fn, space.refs):
            support[name] = CellSupport(name, "helper", "frontend_lowerable_object_helper", dtype, shape, cell_params, bool(vals))
            continue
        if not vals:
            support[name] = CellSupport(name, "fallback", "unobserved_static_branch", dtype, shape, cell_params, False)
            diagnostics.append(f"{name}: present in static closure but absent from representative trace")
            continue
        if dtype == "object":
            support[name] = CellSupport(name, "fallback", "observed_object_value", dtype, shape, cell_params, True)
            continue
        reason = _native_expression_reason(fn, names, space.refs)
        if reason is None:
            support[name] = CellSupport(name, "compiled", "numeric_formula_candidate", dtype, shape, cell_params, True)
        else:
            support[name] = CellSupport(name, "fallback", reason, dtype, shape, cell_params, True)

    # Build dependency -> consumer edges and transparently bypass known frontend
    # helpers.  A helper stays inside whichever runtime needs it and is not a typed
    # boundary by itself.
    raw_edges = {(dep, consumer) for consumer in needed for dep in calls.get(consumer, ()) if dep in needed}

    def upstream_nonhelpers(name: str, seen: set[str] | None = None) -> set[str]:
        seen = set() if seen is None else seen
        if name in seen:
            return set()
        seen.add(name)
        if support[name].mode != "helper":
            return {name}
        out = set()
        for dep in calls.get(name, ()):
            if dep in needed:
                out.update(upstream_nonhelpers(dep, seen))
        return out

    edges: set[tuple[str, str]] = set()
    for dep, consumer in raw_edges:
        if support[consumer].mode == "helper":
            continue
        if support[dep].mode == "helper":
            for root in upstream_nonhelpers(dep):
                if root != consumer:
                    edges.add((root, consumer))
        else:
            edges.add((dep, consumer))

    region_nodes = [name for name in needed if support[name].mode in {"compiled", "fallback"}]

    # A shared compiled ancestor (for example a scalar bound) must not collapse
    # compiled work on both sides of a fallback island into one region.  Assign a
    # generic boundary depth: crossing compiled<->fallback increments the depth;
    # same-mode dependencies preserve it.  This is a dependency barrier, not a
    # model-specific stage notion.
    depth = {name: 0 for name in region_nodes}
    for _ in range(max(8, 4 * max(1, len(region_nodes)))):
        changed = False
        for dep, consumer in edges:
            if dep not in depth or consumer not in depth or dep == consumer:
                continue
            req = depth[dep] + (1 if support[dep].mode != support[consumer].mode else 0)
            if req > depth[consumer]:
                depth[consumer] = req
                changed = True
        if not changed:
            break
    else:
        raise PartitionError("regional dependency barriers did not converge; cross-mode cycle requires explicit fallback")

    same = nx.Graph()
    same.add_nodes_from(region_nodes)
    for dep, consumer in edges:
        if (
            dep in same and consumer in same
            and support[dep].mode == support[consumer].mode
            and depth[dep] == depth[consumer]
        ):
            same.add_edge(dep, consumer)

    components = [set(c) for c in nx.connected_components(same)]
    cell_region: dict[str, int] = {}
    for idx, comp in enumerate(components):
        for name in comp:
            cell_region[name] = idx

    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(components)))
    for dep, consumer in edges:
        if dep not in cell_region or consumer not in cell_region:
            continue
        a, b = cell_region[dep], cell_region[consumer]
        if a != b:
            dag.add_edge(a, b)
    if not nx.is_directed_acyclic_graph(dag):
        raise PartitionError("regional dependency graph is cyclic")
    component_order = list(nx.topological_sort(dag))

    temp_uids = {
        idx: _digest([support[next(iter(comp))].mode, *sorted(comp)], "mixed_region")
        for idx, comp in enumerate(components)
    }
    boundary_map: dict[tuple[int, int, str], BoundaryFieldSpec] = {}
    for dep, consumer in sorted(edges):
        if dep not in cell_region or consumer not in cell_region:
            continue
        a, b = cell_region[dep], cell_region[consumer]
        if a == b or support[dep].mode == support[consumer].mode:
            # Same-mode dependencies across a fallback-depth barrier are safely
            # recomputed inside the downstream compiled artifact; they are not
            # marshaled as a runtime boundary.
            continue
        s = support[dep]
        if s.parameters:
            raise PartitionError(
                f"automatic regional boundary {dep!r}->{consumer!r} is parameterized; "
                "v0.7 requires an explicit call-domain schema for parameterized fallback boundaries"
            )
        if s.dtype not in {"float64", "int64", "bool"}:
            raise PartitionError(f"automatic regional boundary {dep!r} has unsupported dtype {s.dtype!r}")
        if s.shape:
            raise PartitionError(
                f"automatic compiled/fallback boundary {dep!r} has shape {s.shape}; "
                "fixed-size tensor BoundaryBatch is represented but compiled tensor Cells are not yet lowered"
            )
        boundary_map[(a, b, dep)] = BoundaryFieldSpec(
            dep, s.dtype, s.shape, temp_uids[a], temp_uids[b]
        )

    regions: list[MixedRegion] = []
    order_index = {comp: pos for pos, comp in enumerate(component_order)}
    for comp_idx in component_order:
        comp = components[comp_idx]
        uid = temp_uids[comp_idx]
        incoming = tuple(
            field for (a, b, _), field in boundary_map.items() if b == comp_idx
        )
        outgoing = tuple(
            field for (a, b, _), field in boundary_map.items() if a == comp_idx
        )
        # The final result is an implicit outgoing boundary owned by the terminal
        # region so the runtime knows which value to expose.
        if output in comp and not any(f.name == output for f in outgoing):
            s = support[output]
            if s.parameters or s.dtype not in {"float64", "int64", "bool"} or s.shape:
                raise PartitionError("mixed runtime currently requires a zero-argument scalar numeric final output")
            outgoing = outgoing + (
                BoundaryFieldSpec(output, s.dtype, s.shape, uid, "__result__"),
            )
        reasons = tuple(sorted({support[n].reason for n in comp if support[n].mode == "fallback"}))
        regions.append(
            MixedRegion(
                uid=uid,
                index=order_index[comp_idx],
                mode=support[next(iter(comp))].mode,
                cells=tuple(sorted(comp)),
                inputs=tuple(sorted(incoming, key=lambda f: (f.producer_region, f.name))),
                outputs=tuple(sorted(outgoing, key=lambda f: (f.consumer_region, f.name))),
                reasons=reasons,
            )
        )

    return MixedPartitionPlan(
        output=output,
        regions=tuple(regions),
        support=support,
        static_edges=tuple(sorted(edges)),
        sample_keys=sample_keys,
        run_keys=run_keys,
        formula_hash=_formula_hash(space, needed),
        diagnostics=tuple(diagnostics),
    )


class FallbackRegionExecutor:
    """Execute one fallback component from original formula sources.

    Only zero-argument Cells are admitted automatically.  Each point receives a
    fresh namespace and a fresh zero-argument Cell cache.  Boundary callables read
    only marshaled arrays, so modelx caches are never hidden state crossing the
    region boundary.
    """

    def __init__(self, *, space, region: MixedRegion, plan: MixedPartitionPlan):
        if region.mode != "fallback":
            raise ValueError("FallbackRegionExecutor requires a fallback region")
        self.space = space
        self.region = region
        self.plan = plan
        self.space_params = tuple(space.parameters or ())
        self.refs = dict(space.refs)
        self.boundary_names = {f.name for f in region.inputs}
        for name in region.cells:
            if tuple(space.cells[name].parameters or ()):
                raise PartitionError(
                    f"automatic fallback region contains parameterized Cell {name!r}; explicit call-domain fallback required"
                )

        # Include transparent helper dependencies recursively.  They execute in the
        # fallback namespace rather than crossing an object boundary.
        names = set(space.cells)
        funcs = {name: _fn(cell) for name, cell in space.cells.items()}
        calls = {name: (_same_space_calls(fn, names) if fn is not None else set()) for name, fn in funcs.items()}
        include = set(region.cells)
        stack = list(region.cells)
        while stack:
            name = stack.pop()
            for dep in calls.get(name, ()):
                support = plan.support.get(dep)
                if support is not None and support.mode == "helper" and dep not in include:
                    include.add(dep)
                    stack.append(dep)
        self.include = tuple(sorted(include))
        self.sources = {name: space.cells[name].formula.source for name in self.include}

    def _point_env(self, key: Any, boundaries: BoundaryBatch, point_index: int) -> dict[str, Any]:
        env: dict[str, Any] = {"__builtins__": builtins.__dict__}
        env.update(self.refs)
        if len(self.space_params) == 1:
            env[self.space_params[0]] = key
        elif self.space_params:
            if not isinstance(key, tuple) or len(key) != len(self.space_params):
                raise PartitionError(f"runtime key {key!r} does not match ItemSpace parameter arity")
            env.update(dict(zip(self.space_params, key)))

        # Boundaries are exposed with Cell-call syntax and nothing else.
        for name in self.boundary_names:
            value = boundaries.get(name)[point_index]
            env[name] = (lambda v=value: v.item() if isinstance(v, np.generic) else v)

        # Define original functions in a common namespace so same-region calls keep
        # their original Python semantics.  Then wrap zero-argument Cells with a
        # point-local memoizer matching modelx's one-value-per-key behavior.
        for name in self.include:
            exec(compile(self.sources[name], f"<fallback:{name}>", "exec"), env, env)
        cache: dict[str, Any] = {}
        for name in self.include:
            raw = env[name]
            if tuple(self.space.cells[name].parameters or ()):
                # Helpers may conceptually depend on ItemSpace params through globals,
                # but their Cell signature inside the ItemSpace is still zero-arg.
                raise PartitionError(f"fallback helper {name!r} unexpectedly has explicit Cell parameters")
            def wrapper(_name=name, _raw=raw):
                if _name not in cache:
                    cache[_name] = _raw()
                return cache[_name]
            env[name] = wrapper
        return env

    def run(self, keys: Sequence[Any], boundaries: BoundaryBatch) -> dict[str, np.ndarray]:
        keys = tuple(keys)
        output_names = sorted({f.name for f in self.region.outputs})
        collected: dict[str, list[Any]] = {name: [] for name in output_names}
        for i, key in enumerate(keys):
            env = self._point_env(key, boundaries, i)
            for name in output_names:
                if name not in env:
                    raise PartitionError(f"fallback output Cell {name!r} is not in extracted region namespace")
                collected[name].append(env[name]())
        result: dict[str, np.ndarray] = {}
        for name, vals in collected.items():
            s = self.plan.support[name]
            dt = np.bool_ if s.dtype == "bool" else np.int64 if s.dtype == "int64" else np.float64
            arr = np.ascontiguousarray(np.asarray(vals, dtype=dt))
            if arr.shape != (len(keys), *s.shape):
                raise PartitionError(f"fallback output {name!r} produced shape {arr.shape}, expected {(len(keys), *s.shape)}")
            result[name] = arr
        return result


@dataclass
class _CompiledOutputArtifact:
    name: str
    compiler: GraphModelCompiler
    run: Callable[[dict[str, Any], int], np.ndarray]
    logical_dtype: str
    backend: str
    build_fingerprint: str
    source_bytes: int = 0
    native_path: str | None = None


@dataclass
class MixedRuntimeStats:
    runs: int = 0
    points: int = 0
    compiled_region_runs: int = 0
    fallback_region_runs: int = 0
    boundary_values: int = 0
    boundary_bytes: int = 0

    def manifest(self) -> dict[str, Any]:
        total = self.compiled_region_runs + self.fallback_region_runs
        return {
            "runs": self.runs,
            "points": self.points,
            "compiled_region_runs": self.compiled_region_runs,
            "fallback_region_runs": self.fallback_region_runs,
            "region_native_coverage": self.compiled_region_runs / total if total else None,
            "boundary_values": self.boundary_values,
            "boundary_bytes": self.boundary_bytes,
        }


class AutomaticMixedRuntime:
    def __init__(
        self,
        *,
        model,
        space,
        plan: MixedPartitionPlan,
        artifacts: Mapping[str, Sequence[_CompiledOutputArtifact]],
        backend: str,
    ):
        self.model = model
        self.space = space
        self.plan = plan
        self.artifacts = {k: tuple(v) for k, v in artifacts.items()}
        self.backend = backend
        self.fallbacks = {
            r.uid: FallbackRegionExecutor(space=space, region=r, plan=plan)
            for r in plan.regions if r.mode == "fallback"
        }
        self.stats = MixedRuntimeStats()

    def run(self, keys: Sequence[Any] | None = None, *, threads: int = 1) -> np.ndarray:
        keys = tuple(self.plan.run_keys if keys is None else keys)
        boundaries = BoundaryBatch(len(keys))
        for region in self.plan.regions:
            if region.mode == "fallback":
                values = self.fallbacks[region.uid].run(keys, boundaries)
                self.stats.fallback_region_runs += 1
                for spec in region.outputs:
                    if spec.name in values:
                        arr = boundaries.put(spec, values[spec.name])
                        self.stats.boundary_values += arr.size
                        self.stats.boundary_bytes += arr.nbytes
                continue

            outputs = {a.name: a for a in self.artifacts.get(region.uid, ())}
            needed = sorted({f.name for f in region.outputs})
            missing = set(needed) - set(outputs)
            if missing:
                raise PartitionError(f"compiled region {region.uid} has no artifacts for outputs {sorted(missing)!r}")
            for name in needed:
                artifact = outputs[name]
                overrides: dict[str, Any] = {}
                for boundary_name, input_key in artifact.compiler.fallback_input_by_name.items():
                    overrides[input_key] = boundaries.get(boundary_name)
                inputs = artifact.compiler.canonical.bind_inputs(keys, overrides=overrides)
                raw = artifact.run(inputs, int(threads))
                s = self.plan.support[name]
                dt = np.bool_ if s.dtype == "bool" else np.int64 if s.dtype == "int64" else np.float64
                arr = np.ascontiguousarray(np.asarray(raw, dtype=dt))
                spec = next(f for f in region.outputs if f.name == name)
                arr = boundaries.put(spec, arr)
                self.stats.boundary_values += arr.size
                self.stats.boundary_bytes += arr.nbytes
            self.stats.compiled_region_runs += 1

        out = np.asarray(boundaries.get(self.plan.output))
        self.stats.runs += 1
        self.stats.points += len(keys)
        return out.copy()

    def manifest(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "plan": self.plan.manifest(),
            "artifacts": {
                uid: [
                    {
                        "output": a.name,
                        "backend": a.backend,
                        "logical_dtype": a.logical_dtype,
                        "build_fingerprint": a.build_fingerprint,
                        "source_bytes": a.source_bytes,
                        "native_path": a.native_path,
                    }
                    for a in arts
                ]
                for uid, arts in self.artifacts.items()
            },
            "cache_ownership": "compiled region-local temporaries + fallback point-local memoization; BoundaryBatch is sole cross-region state",
            "invalidation": "all boundary values are rebuilt on every run",
            "stats": self.stats.manifest(),
        }


class AutomaticMixedCompiler:
    """Build Python or native artifacts for an automatically partitioned graph."""

    def __init__(
        self,
        model,
        *,
        space,
        output: str,
        sample_keys: Sequence[Any],
        run_keys: Sequence[Any] | None = None,
        optimization_level: str = "O2",
    ):
        self.model = model
        self.space = space
        self.output = output
        self.optimization_level = optimization_level.upper()
        self.plan = build_mixed_partition(
            model, space=space, output=output, sample_keys=sample_keys, run_keys=run_keys
        )

    def _compiled_output_compiler(self, region: MixedRegion, output_name: str) -> GraphModelCompiler:
        fallback_inputs = tuple(sorted({f.name for f in region.inputs if self.plan.support[f.name].mode == "fallback"}))
        try:
            return GraphModelCompiler(
                self.model,
                space=self.space,
                output=output_name,
                sample_keys=self.plan.sample_keys,
                run_keys=self.plan.run_keys,
                fallback_cells=fallback_inputs,
            )
        except Exception as exc:
            raise PartitionError(
                f"automatic partition marked region {region.uid} compiled, but output {output_name!r} "
                f"failed the real compiler proof: {type(exc).__name__}: {exc}"
            ) from exc

    def build_python(self) -> AutomaticMixedRuntime:
        artifacts: dict[str, list[_CompiledOutputArtifact]] = defaultdict(list)
        for region in self.plan.compiled_regions:
            for output_name in sorted({f.name for f in region.outputs}):
                comp = self._compiled_output_compiler(region, output_name)
                gen = PythonLoopGenerator(comp.canonical, optimization_level=self.optimization_level)
                text = gen.emit(f"mixed_{region.index}_{output_name}")
                ns: dict[str, Any] = {}
                exec(compile(text, f"<mixed-python:{region.uid}:{output_name}>", "exec"), ns)
                artifacts[region.uid].append(
                    _CompiledOutputArtifact(
                        output_name,
                        comp,
                        lambda inputs, threads, _run=ns["run"]: _run(inputs),
                        self.plan.support[output_name].dtype,
                        "python",
                        ns["BUILD_FINGERPRINT"],
                        len(text.encode()),
                    )
                )
        return AutomaticMixedRuntime(
            model=self.model, space=self.space, plan=self.plan, artifacts=artifacts, backend="python"
        )

    def build_native(self, outdir: str | Path, *, openmp: bool = True, optimization: str = "O2") -> AutomaticMixedRuntime:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, list[_CompiledOutputArtifact]] = defaultdict(list)
        for region in self.plan.compiled_regions:
            for output_name in sorted({f.name for f in region.outputs}):
                comp = self._compiled_output_compiler(region, output_name)
                module_name = _digest([region.uid, output_name, self.optimization_level], "mixedmod")
                pyx = outdir / f"{module_name}.pyx"
                gen = CythonGenerator(comp.canonical, optimization_level=self.optimization_level)
                gen.write(pyx, module_name)
                mod, so, proc = build_extension(
                    pyx, module_name, outdir / f"build_{module_name}", openmp=openmp, optimization=optimization
                )
                order = list(comp.canonical.inputs)
                artifacts[region.uid].append(
                    _CompiledOutputArtifact(
                        output_name,
                        comp,
                        lambda inputs, threads, _m=mod, _order=order: _m.run(*(inputs[k] for k in _order), int(threads)),
                        self.plan.support[output_name].dtype,
                        "native",
                        getattr(mod, "BUILD_FINGERPRINT", ""),
                        pyx.stat().st_size,
                        str(so),
                    )
                )
        return AutomaticMixedRuntime(
            model=self.model, space=self.space, plan=self.plan, artifacts=artifacts, backend="native"
        )
