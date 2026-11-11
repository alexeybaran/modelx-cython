from __future__ import annotations

import ast
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .realized_compiler import RealizedTraceCompiler


@dataclass(frozen=True)
class FastRegionPlan:
    family_id: int
    formula_op_ids: tuple[int, ...]
    operation_count: int
    numeric_formula_ops: int
    existing_native_formula_ops: int
    register_candidate_operations: int
    spill_candidate_operations: int
    power_formula_ops: tuple[int, ...] = ()
    snapshot_formula_ops: tuple[int, ...] = ()
    sequence_formula_ops: tuple[int, ...] = ()
    reduction_formula_ops: tuple[int, ...] = ()


@dataclass(frozen=True)
class FastFoundationPlan:
    available: bool
    output_names: tuple[str, ...]
    base_space_name: str | None
    point_parameter_count: int
    operation_count: int
    regions: tuple[FastRegionPlan, ...]
    power_formula_ops: tuple[int, ...]
    snapshot_formula_ops: tuple[int, ...]
    sequence_formula_ops: tuple[int, ...]
    reduction_formula_ops: tuple[int, ...]
    reference_frontend_reason: str | None = None
    reference_frontend_manifest: dict[str, Any] | None = field(default=None, compare=False, repr=False)

    @property
    def fallback_required(self) -> bool:
        return not self.available


@dataclass(frozen=True)
class FastFoundationBuildReport:
    operation_count: int
    point_count: int
    output_count: int
    source_lines: int
    source_bytes: int
    pyx_path: str
    so_path: str
    build_seconds: float
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FastFoundationBenchmark:
    point_count: int
    prepare_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    per_point_seconds: float


class FastFoundationError(RuntimeError):
    pass


def _base_space_interface_from_target(compiler: RealizedTraceCompiler) -> Any:
    if len(compiler.trace.target_runtime_nodes) != 1:
        raise FastFoundationError("foundation fast backend currently requires one target")
    obj, key = compiler.trace.target_runtime_nodes[0]
    if key:
        raise FastFoundationError("foundation fast backend currently requires a zero-argument target")
    cur = getattr(obj, "parent", None)
    if cur is None:
        raise FastFoundationError("target has no containing Space")
    seen: set[int] = set()
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
        cur = bases[0] if bases else getattr(cur, "parent", None)
    try:
        return cur.interface
    except Exception as exc:
        raise FastFoundationError("could not resolve structural base Space interface") from exc


def _representative_nodes(compiler: RealizedTraceCompiler) -> dict[tuple[str, str, int], Any]:
    out = {}
    for node in compiler.trace.nodes:
        out.setdefault(node.shape_token, node)
    return out


def _parse_source(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    raise FastFoundationError("formula source has no function definition")


def _is_numeric_ref(value: Any) -> bool:
    try:
        import pandas as pd
        if isinstance(value, (pd.Series, pd.DataFrame)):
            arr = np.asarray(value)
            return arr.dtype.kind in "iufb"
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "iufb" and value.ndim in (1, 2)
    return False


def _formula_features(node: Any) -> tuple[bool, bool, bool, bool]:
    """Return power, numeric-snapshot, sequence-producer, reduction features."""
    try:
        fn = _parse_source(node.schema.source)
    except Exception:
        return False, False, False, False
    has_power = any(isinstance(x, ast.BinOp) and isinstance(x.op, ast.Pow) for x in ast.walk(fn))
    has_reduction = any(
        isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "sum"
        for x in ast.walk(fn)
    ) or any(isinstance(x, ast.AugAssign) and isinstance(x.op, ast.Add) for x in ast.walk(fn))
    has_sequence = False
    for x in ast.walk(fn):
        if isinstance(x, (ast.ListComp, ast.GeneratorExp)):
            has_sequence = True
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr in {"array", "asarray"}:
            has_sequence = True
    glb = node.obj.altfunc.__globals__
    names = set(x.id for x in ast.walk(fn) if isinstance(x, ast.Name))
    has_snapshot = any(name in glb and _is_numeric_ref(glb[name]) for name in names)
    return has_power, has_snapshot, has_sequence, has_reduction


def plan_fast_foundation(compiler: RealizedTraceCompiler) -> FastFoundationPlan:
    """Analyze the current canonical graph and probe the experimental fast backend.

    The canonical plan remains authoritative. The private reference frontend is
    queried only to answer the Part-1 question: can this already-captured model be
    lowered to a register-local, typed-snapshot batch kernel? A negative answer is
    a normal fast-path miss; the existing whole artifact remains the execution path.
    """
    if compiler.structured is None:
        compiler.recover_loops()
    analysis = compiler.native_plan or compiler.analyze_native()
    plan = compiler.structured.canonical_plan
    if plan is None:
        raise FastFoundationError("canonical execution plan is unavailable")
    reps = _representative_nodes(compiler)
    aby = {row.formula_op_id: row for row in analysis.formulae}

    power: list[int] = []
    snapshots: list[int] = []
    sequences: list[int] = []
    reductions: list[int] = []
    for fid, op in enumerate(plan.formula_ops):
        node = reps.get(op.role)
        if node is None:
            continue
        p, s, q, r = _formula_features(node)
        if p: power.append(fid)
        if s: snapshots.append(fid)
        if q: sequences.append(fid)
        if r: reductions.append(fid)

    # Weighted operation counts for each recovered family use the exact canonical
    # instance expansion; no name/coordinate interpretation is involved.
    regions: list[FastRegionPlan] = []
    from .native_plan import build_native_family_backend_plan, expand_native_family_instance
    for family in plan.code_families:
        backend = build_native_family_backend_plan(compiler, family.family_id)
        expansions = [
            expand_native_family_instance(compiler, backend, i)
            for i in range(len(backend.instances))
        ]
        op_count = sum(x.operation_count for x in expansions)
        fids = tuple(dict.fromkeys(family.formula_op_ids))
        numeric = sum(aby[fid].observed_return_type.dtype in {"float64", "int64", "bool"} for fid in fids)
        native = sum(bool(aby[fid].native_capable) for fid in fids)

        # Register-promotion opportunity is ordinary dataflow/liveness: if a
        # numeric realized value is consumed only inside this family region and is
        # not an externally visible target, a future backend may keep it in a C
        # local until a boundary.  This calculation deliberately ignores Cell
        # names and coordinate meanings.
        family_nodes = {nid for ex in expansions for _role, nid in ex.operation_sequence}
        consumers: dict[int, list[int]] = {}
        for src, dst in compiler.trace.dependencies:
            consumers.setdefault(src, []).append(dst)
        target_ids = {nid for nid in compiler.trace.target_node_ids if nid is not None}
        role_fid = tuple(family.formula_op_ids)
        node_to_role: dict[int, int] = {}
        for ex in expansions:
            for role, nid in ex.operation_sequence:
                node_to_role[nid] = role
        register_candidates = 0
        spill_candidates = 0
        for nid in family_nodes:
            role = node_to_role[nid]
            fid = role_fid[role]
            is_numeric = aby[fid].observed_return_type.dtype in {"float64", "int64", "bool"}
            escapes = nid in target_ids or any(dst not in family_nodes for dst in consumers.get(nid, ()))
            if is_numeric and not escapes:
                register_candidates += 1
            else:
                spill_candidates += 1
        regions.append(FastRegionPlan(
            family_id=family.family_id,
            formula_op_ids=fids,
            operation_count=op_count,
            numeric_formula_ops=numeric,
            existing_native_formula_ops=native,
            register_candidate_operations=register_candidates,
            spill_candidate_operations=spill_candidates,
            power_formula_ops=tuple(fid for fid in fids if fid in power),
            snapshot_formula_ops=tuple(fid for fid in fids if fid in snapshots),
            sequence_formula_ops=tuple(fid for fid in fids if fid in sequences),
            reduction_formula_ops=tuple(fid for fid in fids if fid in reductions),
        ))

    obj, _key = compiler.trace.target_runtime_nodes[0]
    output_names = (getattr(obj, "name", plan.formula_ops[-1].name),)
    base = None
    param_count = 0
    manifest = None
    reason = None
    available = False
    try:
        base = _base_space_interface_from_target(compiler)
        param_count = len(tuple(getattr(base, "parameters", ()) or ()))
        from ._fast_reference import ModelxModelCompiler
        frontend = ModelxModelCompiler(compiler.trace.model, space=base, outputs=list(output_names))
        # Construction alone is not enough: the prototype frontend can accept a
        # model and still fail when emitting one of its AST forms.  A cheap dry
        # source generation makes `available` mean buildable in principle rather
        # than merely parseable.
        with tempfile.TemporaryDirectory(prefix="mxg_fast_plan_") as td:
            frontend.generate_cython(Path(td) / "probe.pyx", module_name="mxg_fast_probe")
        manifest = frontend.manifest()
        available = True
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"

    return FastFoundationPlan(
        available=available,
        output_names=output_names,
        base_space_name=getattr(base, "fullname", None) if base is not None else None,
        point_parameter_count=param_count,
        operation_count=compiler.sequential.operation_count,
        regions=tuple(regions),
        power_formula_ops=tuple(power),
        snapshot_formula_ops=tuple(snapshots),
        sequence_formula_ops=tuple(sequences),
        reduction_formula_ops=tuple(reductions),
        reference_frontend_reason=reason,
        reference_frontend_manifest=manifest,
    )


@dataclass
class FastFoundationProgram:
    compiler: RealizedTraceCompiler | None
    plan: FastFoundationPlan
    frontend: Any | None
    backend: Any
    build_report: FastFoundationBuildReport
    _prepared_cache: dict[int | None, dict[str, np.ndarray]] = field(default_factory=dict, repr=False)
    _detached_inputs: dict[str, np.ndarray] | None = field(default=None, repr=False)
    _detached_point_count: int | None = field(default=None, repr=False)
    _detached_point_inputs: tuple[str, ...] = field(default=(), repr=False)

    @property
    def point_count(self) -> int:
        if self._detached_point_count is not None:
            return int(self._detached_point_count)
        if self.frontend is None:
            raise FastFoundationError("fast program has neither frontend nor detached inputs")
        return int(self.frontend.model_point_count)

    def prepare_inputs(self, limit: int | None = None) -> dict[str, np.ndarray]:
        if self._detached_inputs is not None:
            count = self.point_count if limit is None else min(int(limit), self.point_count)
            return {
                name: (value[:count] if name in self._detached_point_inputs else value)
                for name, value in self._detached_inputs.items()
            }
        if limit in self._prepared_cache:
            return self._prepared_cache[limit]
        if self.frontend is None:
            raise FastFoundationError("live frontend was detached before inputs were prepared")
        inputs = self.frontend.bind_inputs(limit=limit)
        self._prepared_cache[limit] = inputs
        return inputs

    def run_batch(self, limit: int | None = None, *, threads: int = 1) -> np.ndarray:
        inputs = self.prepare_inputs(limit)
        return np.asarray(self.backend.module.run(**inputs, threads=int(threads)))

    def execute_prefix(self, count: int) -> np.ndarray:
        """Execute the first ``count`` prepared model points."""
        return self.run_batch(min(int(count), self.point_count), threads=1)

    def detach_for_production(
        self, count: int | None = None, *, trim_allocator: bool = False
    ) -> dict[str, Any]:
        """Freeze compact numeric inputs and release the live model/compiler graph.

        The native extension and copied NumPy arrays are sufficient for execution.
        This keeps correctness proof machinery out of the production memory path.
        """
        count = self.point_count if count is None else min(int(count), self.point_count)
        raw = self.prepare_inputs(count)
        detached = {name: np.ascontiguousarray(value).copy() for name, value in raw.items()}
        point_inputs = tuple(
            name for name, value in detached.items()
            if name.startswith("mp__") and value.ndim and value.shape[0] == count
        )
        self._detached_inputs = detached
        self._detached_point_count = count
        self._detached_point_inputs = point_inputs
        self._prepared_cache.clear()
        self.compiler = None
        self.frontend = None
        trim_succeeded = False
        if trim_allocator:
            try:
                import ctypes
                libc = ctypes.CDLL(None)
                trim = getattr(libc, "malloc_trim", None)
                if trim is not None:
                    trim_succeeded = bool(trim(0))
            except Exception:
                trim_succeeded = False
        return {
            "detached": True,
            "point_count": count,
            "input_bytes": int(sum(value.nbytes for value in detached.values())),
            "point_input_names": point_inputs,
            "allocator_trim_attempted": bool(trim_allocator),
            "allocator_trim_succeeded": trim_succeeded,
        }

    def benchmark(self, point_counts=(1, 10, 100, 1000), *, repeats: int = 5) -> tuple[FastFoundationBenchmark, ...]:
        rows = []
        for count in point_counts:
            count = min(int(count), self.point_count)
            t0 = time.perf_counter()
            self._prepared_cache.pop(count, None)
            self.prepare_inputs(count)
            prep = time.perf_counter() - t0
            # warm once; runtime intentionally excludes input preparation
            self.run_batch(count, threads=1)
            samples = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                self.run_batch(count, threads=1)
                samples.append(time.perf_counter() - t0)
            med = statistics.median(samples)
            rows.append(FastFoundationBenchmark(
                point_count=count,
                prepare_seconds=prep,
                median_seconds=med,
                min_seconds=min(samples),
                max_seconds=max(samples),
                per_point_seconds=med / count,
            ))
        return tuple(rows)

    def validate_points(self, count: int = 20) -> dict[str, Any]:
        """Compare the fast batch with authoritative modelx for the first points.

        This helper is intentionally limited to the one-parameter model-point shape
        supported by the experimental reference backend. It is proof machinery,
        not a general portfolio compatibility mechanism.
        """
        if self.compiler is None or self.frontend is None:
            raise FastFoundationError("validation requires the live model before detachment")
        count = min(int(count), self.point_count)
        got = self.run_batch(count, threads=1)
        base = _base_space_interface_from_target(self.compiler)
        params = tuple(getattr(base, "parameters", ()) or ())
        if len(params) != 1:
            raise FastFoundationError("validation helper requires one base-Space parameter")
        mp_ref = getattr(self.frontend, "model_point_ref", None)
        if mp_ref:
            table = base.refs[mp_ref]
            labels = list(table.index[:count])
        else:
            # Some model variants expose the model-point axis directly as an
            # integer Space parameter and source all point data from external
            # numeric arrays. This is validation-only enumeration, not a compiler
            # semantic rule; failure simply means the proof helper cannot enumerate
            # that model's point domain automatically.
            labels = list(range(count))
        output = self.plan.output_names[0]
        expected = []
        for label in labels:
            item = base[label]
            expected.append(float(getattr(item, output)()))
        actual = np.asarray(got[:, 0], dtype=np.float64)
        exp = np.asarray(expected, dtype=np.float64)
        return {
            "count": count,
            "exact": bool(np.array_equal(actual, exp)),
            "max_abs_error": float(np.max(np.abs(actual-exp))) if count else 0.0,
            "actual": actual,
            "expected": exp,
        }


def build_fast_foundation(
    compiler: RealizedTraceCompiler,
    build_dir: str | Path,
    *,
    native_arch: bool = False,
) -> FastFoundationProgram:
    plan = plan_fast_foundation(compiler)
    if not plan.available:
        raise FastFoundationError(
            "experimental fast backend is not applicable; existing whole-artifact path remains valid: "
            + str(plan.reference_frontend_reason)
        )
    base = _base_space_interface_from_target(compiler)
    from ._fast_reference import ModelxModelCompiler, NativeBackendBuilder
    frontend = ModelxModelCompiler(compiler.trace.model, space=base, outputs=list(plan.output_names))
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    module_name = "mxg_fast_foundation_" + str(abs(hash((plan.base_space_name, plan.output_names))) & 0xffffffff)
    t0 = time.perf_counter()
    backend = NativeBackendBuilder(frontend).build(build_dir, module_name=module_name, native_arch=native_arch)
    build_seconds = time.perf_counter() - t0
    pyx = build_dir / f"{module_name}.pyx"
    source = pyx.read_text()
    so = next(iter(build_dir.glob(module_name + "*.so")), None)
    inputs = frontend.bind_inputs(limit=1)
    report = FastFoundationBuildReport(
        operation_count=compiler.sequential.operation_count,
        point_count=frontend.model_point_count,
        output_count=len(frontend.outputs),
        source_lines=source.count("\n"),
        source_bytes=len(source.encode()),
        pyx_path=str(pyx),
        so_path=str(so) if so is not None else "",
        build_seconds=build_seconds,
        input_names=tuple(inputs),
        input_shapes=tuple(tuple(arr.shape) for arr in inputs.values()),
    )
    return FastFoundationProgram(compiler, plan, frontend, backend, report)
