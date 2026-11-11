from __future__ import annotations

import hashlib
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .build import build_extension
from .native_bindings import (
    NativeBindingError,
    build_runtime_arg_sites,
    encode_runtime_bindings,
)
from .native_family import (
    NativeFamilyError,
    _CallResolver,
    _RuntimeFormulaEmitter,
    _RuntimeReferenceAddress,
    _authoritative_value,
    _emit_runtime_arg_expr,
    _emit_runtime_helpers,
    _emit_runtime_reference_helper,
    _exact_native_formula_ok,
    _initial_hybrid_pools,
    _typed_slot_plan,
)
from .native_plan import (
    NativePlanError,
    build_native_family_backend_plan,
    expand_native_family_instance,
)
from .native_references import (
    NativeReferenceError,
    build_reference_binding_plan,
    encode_reference_instance,
)
from .optimized_execution import OptimizedExecutionPlan, RegisterRegionPlan, RoleBackend
from .whole_artifact import _StrictCallbackFactory


class RegisterRegionError(RuntimeError):
    """Fail-closed error for the first executable canonical RegisterRegion slice."""


@dataclass(frozen=True)
class RegisterReferenceKernelSite:
    consumer_role: int
    formula_op_id: int
    site_id: int
    source_kind: str
    source_role: int | None
    lag: int | None
    pool: str


@dataclass(frozen=True)
class RegisterFamilyKernelCode:
    """Compile-time contract for one register-region kernel shape.

    This is intentionally narrower than the eventual reusable family backend.  It
    contains only family/code facts plus the structural local-ring contract proven
    by one RegisterRegionPlan.  Concrete node ids, slot offsets and itemspace
    objects are absent from generated source.
    """

    family_id: int
    code_signature: str
    variant_count: int
    role_count: int
    formula_op_ids: tuple[int, ...]
    ring_depth_by_role: tuple[int, ...]
    reference_sites: tuple[RegisterReferenceKernelSite, ...]


@dataclass(frozen=True)
class RegisterRegionBuildReport:
    block_index: int
    family_id: int
    instance_ordinal: int
    operation_count: int
    role_count: int
    variant_count: int
    ring_role_count: int
    ring_scalar_count: int
    slot_reference_site_count: int
    register_reference_site_count: int
    spill_count: int
    source_lines: int
    source_bytes: int
    build_seconds: float
    pyx_path: str
    so_path: str


@dataclass(frozen=True)
class RegisterRegionValidation:
    exact: bool
    max_abs_error: float
    spill_exact: bool
    modelx_trace_events: int


@dataclass(frozen=True)
class RegisterRegionBenchmark:
    repeats: int
    median_seconds: float
    min_seconds: float
    max_seconds: float
    ns_per_operation: float


@dataclass(frozen=True)
class RegisterRegionKernelBenchmark:
    repeats_per_sample: int
    samples: int
    median_seconds_per_region: float
    min_seconds_per_region: float
    max_seconds_per_region: float
    ns_per_operation: float


@dataclass(frozen=True)
class _ScalarReferenceAddress:
    expr_name: str

    def emit(self, arg: str | None = None) -> str:
        # The exact Cell argument was already used by the canonical realized graph
        # to prove this source relation.  Runtime source selection is the local
        # ring relation, not a semantic interpretation of that argument.
        return self.expr_name


class _RegisterFormulaEmitter(_RuntimeFormulaEmitter):
    """Runtime formula emitter with selected Cell sites supplied by local rings."""

    def __init__(
        self,
        compiler: Any,
        typed: Any,
        formula_op_id: int,
        reference_sites: tuple[Any, ...],
        register_reads: tuple[RegisterReferenceKernelSite, ...],
        role: int,
        frozen_sites: tuple[Any, ...] = (),
    ):
        super().__init__(compiler, typed, formula_op_id, reference_sites)
        self.frozen_sites = tuple(frozen_sites)
        self.role = role
        self.register_reads = tuple(x for x in register_reads if x.consumer_role == role)
        site_by_id = {x.site_id: x for x in reference_sites if x.formula_op_id == formula_op_id}
        for read in self.register_reads:
            if read.source_kind != "register_ring":
                continue
            site = site_by_id.get(read.site_id)
            if site is None:
                raise RegisterRegionError(
                    f"register reference site {read.site_id} is absent from runtime reference ABI"
                )
            param = self.register_param_name(read.site_id)
            self.resolvers[site.name] = _CallResolver(
                site.name, None, _ScalarReferenceAddress(param), site.arg_count
            )

    def expr(self, node: Any) -> str:
        if self.frozen_sites:
            from .frozen_references import frozen_lookup_expr
            frozen = frozen_lookup_expr(self, node, self.frozen_sites)
            if frozen is not None:
                return frozen
        return super().expr(node)

    @staticmethod
    def register_param_name(site_id: int) -> str:
        return f"rr_site_{site_id}"

    def emit_role_function(
        self,
        arg_types: list[tuple[str, str]],
        return_dtype: str,
    ) -> list[str]:
        rettype = {"float64": "double", "int64": "long long", "bool": "bint"}[return_dtype]
        parts = [f"{typ} {name}" for name, typ in arg_types]
        for read in self.register_reads:
            if read.source_kind != "register_ring":
                continue
            ctype = {"double": "double", "int64": "long long", "bool": "bint"}[read.pool]
            parts.append(f"{ctype} {self.register_param_name(read.site_id)}")
        parts.extend([
            "double[::1] dslots", "long long[::1] islots", "unsigned char[::1] bslots",
            "long long[::1] ref_kind", "long long[::1] ref_a", "long long[::1] ref_b",
            "long long[::1] ref_mod", "long long[::1] ref_lo",
            "long long[::1] ref_table_offset", "long long[::1] ref_table",
        ])
        lines = [f"cdef inline {rettype} f_role_{self.role}({', '.join(parts)}):"]
        for name in sorted(self.locals - self.formals):
            lines.append(f"    cdef double {name} = 0.0")
        lines.extend(self.statements(self.fn.body, "    "))
        return lines


def _selected_region(compiler: Any, block_index: int) -> tuple[OptimizedExecutionPlan, RegisterRegionPlan, Any]:
    optimized = compiler.optimized_plan or compiler.plan_optimized_execution()
    try:
        region = next(r for r in optimized.regions if r.block_index == block_index)
    except StopIteration as exc:
        raise RegisterRegionError(f"optimized plan has no block {block_index}") from exc
    if not isinstance(region, RegisterRegionPlan):
        raise RegisterRegionError(f"block {block_index} is {region.kind}, not a RegisterRegion")
    family = next(f for f in optimized.families if f.family_id == region.family_id)
    return optimized, region, family


def _kernel_contract(compiler: Any, region: RegisterRegionPlan, family: Any, refs: Any) -> RegisterFamilyKernelCode:
    if region.boundaries or region.slot_operation_count:
        raise RegisterRegionError(
            "first executable RegisterRegion proof requires a closed all-register region"
        )
    if set(region.register_roles) != set(range(len(family.role_plans))):
        raise RegisterRegionError("closed proof requires every realized family role to be register-owned")
    if any(rp.backend is not RoleBackend.REGISTER_NATIVE for rp in family.role_plans):
        raise RegisterRegionError("closed proof contains a non-register family role")

    site_by_id = {x.site_id: x for x in refs.sites}
    reads: list[RegisterReferenceKernelSite] = []
    for read in region.reference_reads:
        if read.source_kind == "dormant":
            continue
        site = site_by_id.get(read.site_id)
        if site is None:
            raise RegisterRegionError(f"reference site {read.site_id} missing from family reference ABI")
        reads.append(RegisterReferenceKernelSite(
            consumer_role=read.consumer_role,
            formula_op_id=read.formula_op_id,
            site_id=read.site_id,
            source_kind=read.source_kind,
            source_role=read.source_role,
            lag=read.lag_from_latest_produced,
            pool=site.pool,
        ))
    depth = region.register_ring_depth_by_role
    depths = tuple(int(depth.get(role, 0)) for role in range(len(family.role_plans)))
    payload = (
        family.code_signature,
        tuple((x.consumer_role, x.site_id, x.source_kind, x.source_role, x.lag) for x in reads),
        depths,
    )
    signature = hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()
    return RegisterFamilyKernelCode(
        family_id=family.family_id,
        code_signature=signature,
        variant_count=len(family.variants),
        role_count=len(family.role_plans),
        formula_op_ids=tuple(family.formula_op_ids),
        ring_depth_by_role=depths,
        reference_sites=tuple(reads),
    )


def _spill_payload(region: RegisterRegionPlan, expansion: Any, role_count: int) -> tuple[np.ndarray, np.ndarray]:
    by_node = {x.node_id: x for x in region.register_values}
    offsets = np.zeros(role_count, dtype=np.int64)
    table: list[int] = []
    for role in range(role_count):
        offsets[role] = len(table)
        for nid in expansion.role_node_ids[role]:
            value = by_node.get(nid)
            if value is None:
                raise RegisterRegionError(f"register role {role} output node {nid} has no RegisterValuePlan")
            table.append(int(value.slot_offset) if value.spill_required else -1)
    return offsets, np.asarray(table, dtype=np.int64)


def _ring_expr(read: RegisterReferenceKernelSite) -> str:
    if read.source_kind != "register_ring" or read.source_role is None or read.lag is None:
        raise RegisterRegionError("attempted to emit non-register reference as local ring")
    return f"ring_{read.source_role}_{read.lag}"


def _role_source_lines(
    role: int,
    family: Any,
    role_meta: list[Any],
    emitters: dict[int, _RegisterFormulaEmitter],
    kernel: RegisterFamilyKernelCode,
) -> list[str]:
    fid, args, pool = role_meta[role]
    if pool not in {"double", "int64", "bool"}:
        raise RegisterRegionError("first register kernel supports numeric output pools only")
    arg_exprs = [_emit_runtime_arg_expr(site, f"c_{role}") for _name, _typ, site in args]
    ring_reads = [
        x for x in kernel.reference_sites
        if x.consumer_role == role and x.source_kind == "register_ring"
    ]
    ring_args = [_ring_expr(x) for x in ring_reads]
    callargs = ", ".join(
        arg_exprs + ring_args + [
            "dslots", "islots", "bslots",
            "ref_kind", "ref_a", "ref_b", "ref_mod", "ref_lo", "ref_table_offset", "ref_table",
        ]
    )
    value_name = f"value_{role}"
    lines = [f"            {value_name} = f_role_{role}({callargs})"]

    depth = kernel.ring_depth_by_role[role]
    if depth:
        for lag in range(depth - 1, 0, -1):
            lines.append(f"            ring_{role}_{lag} = ring_{role}_{lag - 1}")
        lines.append(f"            ring_{role}_0 = {value_name}")

    lines.append(f"            spill_idx = spill_table[spill_offset[{role}] + c_{role}]")
    lines.append("            if spill_idx >= 0:")
    arr = {"double": "dslots", "int64": "islots", "bool": "bslots"}[pool]
    if pool == "bool":
        lines.append(f"                {arr}[spill_idx] = 1 if {value_name} else 0")
    else:
        lines.append(f"                {arr}[spill_idx] = {value_name}")
    lines.append("            if record_output:")
    lines.append(f"                result[out_pos] = <double>{value_name}")
    lines.append("                out_pos += 1")
    lines.append(f"            c_{role} += 1")
    return lines


@dataclass
class RegisterRegionProgram:
    compiler: Any
    region: RegisterRegionPlan
    family: Any
    kernel: RegisterFamilyKernelCode
    module: Any
    build_report: RegisterRegionBuildReport
    dslots: np.ndarray
    islots: np.ndarray
    bslots: np.ndarray
    variant_arr: np.ndarray
    binding_payload: Any
    reference_payload: Any
    spill_offset: np.ndarray
    spill_table: np.ndarray
    expected: np.ndarray
    expected_spills: tuple[tuple[str, int, Any], ...]
    _initial_dslots: np.ndarray
    _initial_islots: np.ndarray
    _initial_bslots: np.ndarray

    def reset(self) -> None:
        self.dslots[:] = self._initial_dslots
        self.islots[:] = self._initial_islots
        self.bslots[:] = self._initial_bslots
        self.binding_payload.cursor.fill(0)

    def _run_raw(self) -> np.ndarray:
        return np.asarray(self.module.run(
            self.dslots, self.islots, self.bslots,
            self.variant_arr,
            *self.binding_payload.as_call_args(),
            *self.reference_payload.as_call_args(),
            self.spill_offset, self.spill_table,
            self.region.operation_count,
        ), dtype=np.float64)

    def run(self, *, reset: bool = True) -> np.ndarray:
        if reset:
            self.reset()
        return self._run_raw()

    def validate(self, *, prove_no_modelx: bool = True) -> RegisterRegionValidation:
        self.reset()
        events = []
        if prove_no_modelx:
            # Expected values and external live-ins were captured before this call.
            # Clearing scheduled modelx values therefore makes any accidental
            # evaluator escape observable without affecting the C-only kernel.
            self.compiler.sequential._clear_scheduled_values()
            system = self.compiler.trace.model._impl.system
            with system.trace_stack(maxlen=None):
                actual = self._run_raw()
                events = list(system.callstack.tracestack)
        else:
            actual = self._run_raw()
        exact = bool(np.array_equal(actual, self.expected))
        max_abs = float(np.max(np.abs(actual - self.expected))) if actual.size else 0.0

        spill_exact = True
        for pool, offset, expected in self.expected_spills:
            if pool == "double":
                got = float(self.dslots[offset])
            elif pool == "int64":
                got = int(self.islots[offset])
            else:
                got = bool(self.bslots[offset])
            if got != expected:
                spill_exact = False
                break
        return RegisterRegionValidation(
            exact=exact,
            max_abs_error=max_abs,
            spill_exact=spill_exact,
            modelx_trace_events=len(events),
        )

    def benchmark(self, *, repeats: int = 200, reset_each: bool = False) -> RegisterRegionBenchmark:
        self.reset()
        self._run_raw()
        samples: list[float] = []
        for _ in range(repeats):
            if reset_each:
                self.reset()
            t0 = time.perf_counter()
            self._run_raw()
            samples.append(time.perf_counter() - t0)
        med = statistics.median(samples)
        return RegisterRegionBenchmark(
            repeats=repeats,
            median_seconds=med,
            min_seconds=min(samples),
            max_seconds=max(samples),
            ns_per_operation=(med * 1e9 / self.region.operation_count),
        )

    def benchmark_kernel(
        self, *, repeats_per_sample: int = 5000, samples: int = 7
    ) -> RegisterRegionKernelBenchmark:
        # This measures the internal Cython region call after ndarray/memoryview
        # conversion has happened once in the public benchmark wrapper.  It is the
        # relevant cost for eventual whole-artifact composition.  Stateful cursor
        # binding kinds are deliberately excluded from this first proof.
        stateful = {2, 6}  # arithmetic-runs / run-length descriptor kinds
        if any(int(x) in stateful for x in self.binding_payload.kind if int(x) >= 0):
            raise RegisterRegionError(
                "internal repeat benchmark does not yet reset stateful binding cursors"
            )
        values = []
        self.reset()
        self.module.benchmark_repeat(
            self.dslots, self.islots, self.bslots, self.variant_arr,
            *self.binding_payload.as_call_args(), *self.reference_payload.as_call_args(),
            self.spill_offset, self.spill_table, self.region.operation_count, 5,
        )
        for _ in range(samples):
            self.reset()
            t0 = time.perf_counter()
            self.module.benchmark_repeat(
                self.dslots, self.islots, self.bslots, self.variant_arr,
                *self.binding_payload.as_call_args(), *self.reference_payload.as_call_args(),
                self.spill_offset, self.spill_table, self.region.operation_count,
                repeats_per_sample,
            )
            values.append((time.perf_counter() - t0) / repeats_per_sample)
        med = statistics.median(values)
        return RegisterRegionKernelBenchmark(
            repeats_per_sample=repeats_per_sample, samples=samples,
            median_seconds_per_region=med, min_seconds_per_region=min(values),
            max_seconds_per_region=max(values),
            ns_per_operation=med * 1e9 / self.region.operation_count,
        )


def build_register_region_cython(
    compiler: Any,
    block_index: int,
    build_dir: str | Path,
    *,
    module_prefix: str = "mxg_register_region",
    optimization: str = "O2",
) -> RegisterRegionProgram:
    """Compile one closed canonical RegisterRegion using local C-role rings.

    This is deliberately the first executable proof only.  Python/slot operations
    inside the region, object outputs, snapshots, reductions and batching are not
    widened here; unsupported shapes fail closed and the established whole-artifact
    backend remains authoritative.
    """
    if compiler.slots is None:
        compiler.lower_slots()
    analysis = compiler.native_plan or compiler.analyze_native()
    typed = _typed_slot_plan(compiler, analysis, include_objects=True)
    optimized, region, family = _selected_region(compiler, block_index)
    if region.slot_operation_count:
        return _build_mixed_register_region_cython(
            compiler, block_index, build_dir, module_prefix=module_prefix + "_mixed",
            optimization=optimization,
        )

    try:
        backend = build_native_family_backend_plan(compiler, region.family_id)
        expansion = expand_native_family_instance(compiler, backend, region.instance_ordinal)
    except NativePlanError as exc:
        raise RegisterRegionError(str(exc)) from exc
    if tuple(nid for _role, nid in expansion.operation_sequence) != region.node_ids:
        raise RegisterRegionError("optimized region and native family expansion disagree")

    analysis_by_id = {x.formula_op_id: x for x in analysis.formulae}
    sites = build_runtime_arg_sites(backend)
    candidate_native = {
        fid: _exact_native_formula_ok(compiler, fid, analysis_by_id[fid])
        for fid in set(backend.kernel.formula_op_ids)
    }
    try:
        refs = build_reference_binding_plan(compiler, typed, backend, candidate_native)
    except NativeReferenceError as exc:
        raise RegisterRegionError(str(exc)) from exc
    kernel = _kernel_contract(compiler, region, family, refs)

    # Keep the first executable proof narrow: all emitted role values are numeric,
    # and every role must still pass the exact native formula gate used by the
    # established whole-artifact backend.
    for role, fid in enumerate(kernel.formula_op_ids):
        rp = family.role_plans[role]
        if not candidate_native.get(fid, False) or rp.backend is not RoleBackend.REGISTER_NATIVE:
            raise RegisterRegionError(f"role {role} is not exact-native in the closed register proof")
        if analysis_by_id[fid].observed_return_type.dtype not in {"float64", "int64", "bool"}:
            raise RegisterRegionError("first register proof supports scalar numeric role outputs only")

    site_rows_by_role: dict[int, list[Any]] = {role: [] for role in range(kernel.role_count)}
    for site in sites:
        site_rows_by_role[site.role].append(site)
    role_meta: list[tuple[int, list[tuple[str, str, Any]], str]] = []
    emitters: dict[int, _RegisterFormulaEmitter] = {}
    for role, fid in enumerate(kernel.formula_op_ids):
        emitter = _RegisterFormulaEmitter(
            compiler, typed, fid, refs.sites, kernel.reference_sites, role
        )
        emitters[role] = emitter
        rows = sorted(site_rows_by_role[role], key=lambda x: x.arg_index)
        formal_names = [a.arg for a in emitter.fn.args.args]
        if len(rows) != len(formal_names):
            raise RegisterRegionError("FormulaOp arity does not match register runtime binding ABI")
        args = [(name, site.cython_type, site) for name, site in zip(formal_names, rows)]
        role_meta.append((fid, args, typed.formula_by_id[fid].pool))

    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    module_name = f"{module_prefix}_{kernel.code_signature[:12]}"
    pyx = build_dir / f"{module_name}.pyx"
    lines = [
        "# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False",
        "import numpy as np",
        "cimport numpy as cnp",
        "",
    ]
    lines.extend(_emit_runtime_helpers())
    lines.extend(_emit_runtime_reference_helper())
    for role, fid in enumerate(kernel.formula_op_ids):
        args = [(name, typ) for name, typ, _site in role_meta[role][1]]
        lines.extend(emitters[role].emit_role_function(
            args, analysis_by_id[fid].observed_return_type.dtype
        ))
        lines.append("")

    public_args = [
        "cnp.ndarray[cnp.float64_t, ndim=1] d_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] i_arr",
        "cnp.ndarray[cnp.uint8_t, ndim=1] b_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] variant_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_kind_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_ia_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_ib_arr",
        "cnp.ndarray[cnp.float64_t, ndim=1] bind_da_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_payload_offset_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_payload_count_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_aux_offset_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_aux_count_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_i_payload_arr",
        "cnp.ndarray[cnp.float64_t, ndim=1] bind_d_payload_arr",
        "object bind_object_payload",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_run_ends_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_cursor_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_kind_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_a_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_b_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_mod_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_lo_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_table_offset_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_table_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] spill_offset_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] spill_table_arr",
        "Py_ssize_t out_count",
    ]
    internal_args = [
        "double[::1] dslots", "long long[::1] islots", "unsigned char[::1] bslots",
        "long long[::1] variants",
        "long long[::1] bind_kind", "long long[::1] bind_ia", "long long[::1] bind_ib",
        "double[::1] bind_da", "long long[::1] bind_payload_offset",
        "long long[::1] bind_payload_count", "long long[::1] bind_aux_offset",
        "long long[::1] bind_aux_count", "long long[::1] bind_i_payload",
        "double[::1] bind_d_payload", "object bind_object_payload",
        "long long[::1] bind_run_ends", "long long[::1] bind_cursor",
        "long long[::1] ref_kind", "long long[::1] ref_a", "long long[::1] ref_b",
        "long long[::1] ref_mod", "long long[::1] ref_lo",
        "long long[::1] ref_table_offset", "long long[::1] ref_table",
        "long long[::1] spill_offset", "long long[::1] spill_table",
        "Py_ssize_t out_count", "double[::1] result", "bint record_output",
    ]
    lines.append("cdef inline Py_ssize_t _run_region(")
    for i, arg in enumerate(internal_args):
        lines.append(f"    {arg}{',' if i + 1 < len(internal_args) else ''}")
    lines += [
        "):",
        "    cdef Py_ssize_t iteration, out_pos = 0, spill_idx",
        "    cdef long long variant",
    ]
    for role in range(kernel.role_count):
        lines.append(f"    cdef Py_ssize_t c_{role} = 0")
        dtype = analysis_by_id[kernel.formula_op_ids[role]].observed_return_type.dtype
        ctype = {"float64": "double", "int64": "long long", "bool": "bint"}[dtype]
        init = "0.0" if dtype == "float64" else "0"
        lines.append(f"    cdef {ctype} value_{role} = {init}")
        for lag in range(kernel.ring_depth_by_role[role]):
            lines.append(f"    cdef {ctype} ring_{role}_{lag} = {init}")
    lines += [
        "    for iteration in range(variants.shape[0]):",
        "        variant = variants[iteration]",
    ]
    for variant_id, pattern in enumerate(family.variants):
        lines.append(("        if" if variant_id == 0 else "        elif") + f" variant == {variant_id}:")
        for role in pattern:
            lines.extend(_role_source_lines(role, family, role_meta, emitters, kernel))
    lines += [
        "        else:",
        "            raise ValueError('variant id outside compiled register family grammar')",
        "    if record_output and out_pos != out_count:",
        "        raise ValueError('register region output count does not match runtime variant stream')",
        "    return out_pos",
        "",
    ]

    def _emit_public_header(name: str, extra: tuple[str, ...] = ()) -> None:
        lines.append(f"cpdef {('double' if name == 'benchmark_repeat' else 'cnp.ndarray')} {name}(")
        all_args = public_args + list(extra)
        for i, arg in enumerate(all_args):
            lines.append(f"    {arg}{',' if i + 1 < len(all_args) else ''}")
        lines.append("):")

    view_lines = [
        "    cdef double[::1] dslots = d_arr",
        "    cdef long long[::1] islots = i_arr",
        "    cdef unsigned char[::1] bslots = b_arr",
        "    cdef long long[::1] variants = variant_arr",
        "    cdef long long[::1] bind_kind = bind_kind_arr",
        "    cdef long long[::1] bind_ia = bind_ia_arr",
        "    cdef long long[::1] bind_ib = bind_ib_arr",
        "    cdef double[::1] bind_da = bind_da_arr",
        "    cdef long long[::1] bind_payload_offset = bind_payload_offset_arr",
        "    cdef long long[::1] bind_payload_count = bind_payload_count_arr",
        "    cdef long long[::1] bind_aux_offset = bind_aux_offset_arr",
        "    cdef long long[::1] bind_aux_count = bind_aux_count_arr",
        "    cdef long long[::1] bind_i_payload = bind_i_payload_arr",
        "    cdef double[::1] bind_d_payload = bind_d_payload_arr",
        "    cdef long long[::1] bind_run_ends = bind_run_ends_arr",
        "    cdef long long[::1] bind_cursor = bind_cursor_arr",
        "    cdef long long[::1] ref_kind = ref_kind_arr",
        "    cdef long long[::1] ref_a = ref_a_arr",
        "    cdef long long[::1] ref_b = ref_b_arr",
        "    cdef long long[::1] ref_mod = ref_mod_arr",
        "    cdef long long[::1] ref_lo = ref_lo_arr",
        "    cdef long long[::1] ref_table_offset = ref_table_offset_arr",
        "    cdef long long[::1] ref_table = ref_table_arr",
        "    cdef long long[::1] spill_offset = spill_offset_arr",
        "    cdef long long[::1] spill_table = spill_table_arr",
    ]
    internal_call = (
        "dslots, islots, bslots, variants, bind_kind, bind_ia, bind_ib, bind_da, "
        "bind_payload_offset, bind_payload_count, bind_aux_offset, bind_aux_count, "
        "bind_i_payload, bind_d_payload, bind_object_payload, bind_run_ends, bind_cursor, "
        "ref_kind, ref_a, ref_b, ref_mod, ref_lo, ref_table_offset, ref_table, "
        "spill_offset, spill_table, out_count"
    )

    _emit_public_header("run")
    lines.extend(view_lines)
    lines += [
        "    cdef cnp.ndarray[cnp.float64_t, ndim=1] result_arr = np.empty(out_count, dtype=np.float64)",
        "    cdef double[::1] result = result_arr",
        f"    _run_region({internal_call}, result, True)",
        "    return result_arr",
        "",
    ]

    _emit_public_header("benchmark_repeat", ("Py_ssize_t repeats",))
    lines.extend(view_lines)
    lines += [
        "    cdef cnp.ndarray[cnp.float64_t, ndim=1] result_arr = np.empty(1, dtype=np.float64)",
        "    cdef double[::1] result = result_arr",
        "    cdef Py_ssize_t rep",
        "    for rep in range(repeats):",
        f"        _run_region({internal_call}, result, False)",
        "    return dslots[0] if dslots.shape[0] else 0.0",
        "",
    ]
    source = "\n".join(lines) + "\n"
    pyx.write_text(source)
    t0 = time.perf_counter()
    module, so_path, _proc = build_extension(
        pyx, module_name, build_dir, openmp=False, optimization=optimization, native_arch=False
    )
    build_seconds = time.perf_counter() - t0

    family_node_ids = set(region.node_ids)
    dslots, islots, bslots, _oslots = _initial_hybrid_pools(
        compiler, typed, family_node_ids
    )
    initial_d = dslots.copy(); initial_i = islots.copy(); initial_b = bslots.copy()
    try:
        binding_payload = encode_runtime_bindings(backend.instances[region.instance_ordinal], sites)
    except NativeBindingError as exc:
        raise RegisterRegionError(str(exc)) from exc
    try:
        reference_payload = encode_reference_instance(refs, region.instance_ordinal)
    except NativeReferenceError as exc:
        raise RegisterRegionError(str(exc)) from exc
    spill_offset, spill_table = _spill_payload(region, expansion, kernel.role_count)
    variant_arr = np.asarray(region.variant_ids, dtype=np.int64)
    byid = compiler.sequential.node_by_id
    expected = np.asarray(
        [float(_authoritative_value(byid[nid])) for _role, nid in expansion.operation_sequence],
        dtype=np.float64,
    )
    spill_final: dict[tuple[str, int], Any] = {}
    value_by_node = {x.node_id: x for x in region.register_values}
    for _role, nid in expansion.operation_sequence:
        value = value_by_node.get(nid)
        if value is not None and value.spill_required:
            spill_final[(value.pool, value.slot_offset)] = _authoritative_value(byid[nid])
    expected_spills = tuple((pool, offset, val) for (pool, offset), val in spill_final.items())

    report = RegisterRegionBuildReport(
        block_index=region.block_index,
        family_id=region.family_id,
        instance_ordinal=region.instance_ordinal,
        operation_count=region.operation_count,
        role_count=kernel.role_count,
        variant_count=kernel.variant_count,
        ring_role_count=sum(x > 0 for x in kernel.ring_depth_by_role),
        ring_scalar_count=sum(kernel.ring_depth_by_role),
        slot_reference_site_count=sum(x.source_kind == "slot_reference" for x in kernel.reference_sites),
        register_reference_site_count=sum(x.source_kind == "register_ring" for x in kernel.reference_sites),
        spill_count=region.spill_count,
        source_lines=source.count("\n"),
        source_bytes=len(source.encode("utf-8")),
        build_seconds=build_seconds,
        pyx_path=str(pyx),
        so_path=str(so_path),
    )
    return RegisterRegionProgram(
        compiler=compiler, region=region, family=family, kernel=kernel, module=module,
        build_report=report,
        dslots=dslots, islots=islots, bslots=bslots,
        variant_arr=variant_arr,
        binding_payload=binding_payload,
        reference_payload=reference_payload,
        spill_offset=spill_offset, spill_table=spill_table,
        expected=expected, expected_spills=expected_spills,
        _initial_dslots=initial_d, _initial_islots=initial_i, _initial_bslots=initial_b,
    )

# ---------------------------------------------------------------------------
# v0.20.2 mixed RegisterRegion execution proof
# ---------------------------------------------------------------------------

_MIXED_POOL_CODE = {"double": 0, "int64": 1, "bool": 2, "object": 3}


@dataclass(frozen=True)
class MixedRegisterFamilyKernelCode:
    family_id: int
    code_signature: str
    variant_count: int
    used_variant_ids: tuple[int, ...]
    role_count: int
    formula_op_ids: tuple[int, ...]
    role_backends: tuple[str, ...]
    ring_depth_by_role: tuple[int, ...]
    reference_sites: tuple[RegisterReferenceKernelSite, ...]


@dataclass(frozen=True)
class MixedRegisterRegionBuildReport:
    block_index: int
    family_id: int
    instance_ordinal: int
    operation_count: int
    register_operation_count: int
    slot_operation_count: int
    python_boundary_count: int
    reload_value_count: int
    role_count: int
    used_variant_count: int
    ring_role_count: int
    ring_scalar_count: int
    slot_reference_site_count: int
    register_reference_site_count: int
    spill_count: int
    source_lines: int
    source_bytes: int
    build_seconds: float
    pyx_path: str
    so_path: str


@dataclass
class MixedRegisterRegionProgram:
    compiler: Any
    region: RegisterRegionPlan
    family: Any
    kernel: MixedRegisterFamilyKernelCode
    module: Any
    build_report: MixedRegisterRegionBuildReport
    dslots: np.ndarray
    islots: np.ndarray
    bslots: np.ndarray
    oslots: list[Any]
    downers: np.ndarray
    iowners: np.ndarray
    bowners: np.ndarray
    oowners: np.ndarray
    variant_arr: np.ndarray
    binding_payload: Any
    reference_payload: Any
    spill_offset: np.ndarray
    spill_table: np.ndarray
    role_base: np.ndarray
    node_ids: np.ndarray
    out_offsets: np.ndarray
    boundary_start: np.ndarray
    boundary_pool: np.ndarray
    boundary_slot: np.ndarray
    boundary_node: np.ndarray
    reload_start: np.ndarray
    reload_pool: np.ndarray
    reload_slot: np.ndarray
    reload_node: np.ndarray
    callbacks: tuple[Any, ...]
    expected: np.ndarray
    expected_spills: tuple[tuple[str, int, Any], ...]
    _initial_dslots: np.ndarray
    _initial_islots: np.ndarray
    _initial_bslots: np.ndarray
    _initial_oslots: list[Any]
    _initial_downers: np.ndarray
    _initial_iowners: np.ndarray
    _initial_bowners: np.ndarray
    _initial_oowners: np.ndarray

    def reset(self) -> None:
        self.dslots[:] = self._initial_dslots
        self.islots[:] = self._initial_islots
        self.bslots[:] = self._initial_bslots
        self.oslots[:] = self._initial_oslots
        self.downers[:] = self._initial_downers
        self.iowners[:] = self._initial_iowners
        self.bowners[:] = self._initial_bowners
        self.oowners[:] = self._initial_oowners
        self.binding_payload.cursor.fill(0)

    def _args(self) -> tuple[Any, ...]:
        return (
            self.dslots, self.islots, self.bslots, self.oslots,
            self.downers, self.iowners, self.bowners, self.oowners,
            self.variant_arr,
            *self.binding_payload.as_call_args(), *self.reference_payload.as_call_args(),
            self.spill_offset, self.spill_table,
            self.role_base, self.node_ids, self.out_offsets,
            self.boundary_start, self.boundary_pool, self.boundary_slot, self.boundary_node,
            self.reload_start, self.reload_pool, self.reload_slot, self.reload_node,
            self.callbacks, self.region.operation_count,
        )

    def _run_raw(self) -> np.ndarray:
        return np.asarray(self.module.run(*self._args()), dtype=np.float64)

    def run(self, *, reset: bool = True) -> np.ndarray:
        if reset:
            self.reset()
        return self._run_raw()

    def validate(self, *, prove_no_modelx: bool = True) -> RegisterRegionValidation:
        self.reset()
        events = []
        if prove_no_modelx:
            self.compiler.sequential._clear_scheduled_values()
            system = self.compiler.trace.model._impl.system
            with system.trace_stack(maxlen=None):
                actual = self._run_raw()
                events = list(system.callstack.tracestack)
        else:
            actual = self._run_raw()
        exact = bool(np.array_equal(actual, self.expected))
        max_abs = float(np.max(np.abs(actual - self.expected))) if actual.size else 0.0
        spill_exact = True
        for pool, offset, expected in self.expected_spills:
            if pool == "double":
                got = float(self.dslots[offset])
            elif pool == "int64":
                got = int(self.islots[offset])
            elif pool == "bool":
                got = bool(self.bslots[offset])
            else:
                got = self.oslots[offset]
            if got != expected:
                spill_exact = False
                break
        return RegisterRegionValidation(
            exact=exact,
            max_abs_error=max_abs,
            spill_exact=spill_exact,
            modelx_trace_events=len(events),
        )

    def benchmark(self, *, repeats: int = 200, reset_each: bool = False) -> RegisterRegionBenchmark:
        self.reset(); self._run_raw()
        samples: list[float] = []
        for _ in range(repeats):
            if reset_each:
                self.reset()
            t0 = time.perf_counter(); self._run_raw(); samples.append(time.perf_counter() - t0)
        med = statistics.median(samples)
        return RegisterRegionBenchmark(
            repeats=repeats,
            median_seconds=med,
            min_seconds=min(samples),
            max_seconds=max(samples),
            ns_per_operation=med * 1e9 / self.region.operation_count,
        )

    def benchmark_kernel(
        self, *, repeats_per_sample: int = 1000, samples: int = 7
    ) -> RegisterRegionKernelBenchmark:
        values: list[float] = []
        # Memoryviews and callback tuple are bound once by the generated wrapper;
        # this is the timing surface relevant to eventual whole-artifact cdef
        # composition.  Contract-owner checks remain enabled for public validation
        # but are disabled in the repeated production-style timing loop.
        self.module.benchmark_repeat(*self._args(), 3)
        for _ in range(samples):
            self.reset()
            t0 = time.perf_counter()
            self.module.benchmark_repeat(*self._args(), repeats_per_sample)
            values.append((time.perf_counter() - t0) / repeats_per_sample)
        med = statistics.median(values)
        return RegisterRegionKernelBenchmark(
            repeats_per_sample=repeats_per_sample,
            samples=samples,
            median_seconds_per_region=med,
            min_seconds_per_region=min(values),
            max_seconds_per_region=max(values),
            ns_per_operation=med * 1e9 / self.region.operation_count,
        )


def _mixed_kernel_contract(
    compiler: Any,
    region: RegisterRegionPlan,
    family: Any,
    refs: Any,
) -> MixedRegisterFamilyKernelCode:
    if not region.register_roles or not region.slot_operation_count:
        raise RegisterRegionError("mixed register proof requires both register and slot/Python operations")
    role_backends = tuple(rp.backend.value for rp in family.role_plans)
    for role in region.register_roles:
        if family.role_plans[role].backend is not RoleBackend.REGISTER_NATIVE:
            raise RegisterRegionError(f"role {role} is not owned by register backend")
    boundary_nodes = {x.node_id for x in region.boundaries}
    if len(boundary_nodes) != region.slot_operation_count:
        raise RegisterRegionError("mixed region boundaries do not cover every slot/Python operation")
    reload_nodes = {x.node_id for x in region.reload_values}
    value_by_node = {x.node_id: x for x in region.register_values}
    for boundary in region.boundaries:
        for nid in boundary.spill_before:
            value = value_by_node.get(nid)
            if value is None or not value.spill_required:
                raise RegisterRegionError(
                    f"boundary {boundary.node_id} requires unplanned register spill {nid}"
                )
        if boundary.register_consumers_after and boundary.node_id not in reload_nodes:
            raise RegisterRegionError(
                f"boundary {boundary.node_id} feeds register consumers but has no ReloadValuePlan"
            )

    site_by_id = {x.site_id: x for x in refs.sites}
    reads: list[RegisterReferenceKernelSite] = []
    for read in region.reference_reads:
        if read.source_kind == "dormant":
            continue
        site = site_by_id.get(read.site_id)
        if site is None:
            raise RegisterRegionError(f"reference site {read.site_id} missing from family reference ABI")
        reads.append(RegisterReferenceKernelSite(
            consumer_role=read.consumer_role,
            formula_op_id=read.formula_op_id,
            site_id=read.site_id,
            source_kind=read.source_kind,
            source_role=read.source_role,
            lag=read.lag_from_latest_produced,
            pool=site.pool,
        ))
    depth = region.register_ring_depth_by_role
    depths = tuple(int(depth.get(role, 0)) for role in range(len(family.role_plans)))
    used_variant_ids = tuple(sorted(set(int(x) for x in region.variant_ids)))
    payload = (
        family.code_signature,
        role_backends,
        used_variant_ids,
        tuple((x.consumer_role, x.site_id, x.source_kind, x.source_role, x.lag) for x in reads),
        depths,
    )
    return MixedRegisterFamilyKernelCode(
        family_id=family.family_id,
        code_signature=hashlib.sha1(repr(payload).encode("utf-8")).hexdigest(),
        variant_count=len(family.variants),
        used_variant_ids=used_variant_ids,
        role_count=len(family.role_plans),
        formula_op_ids=tuple(family.formula_op_ids),
        role_backends=role_backends,
        ring_depth_by_role=depths,
        reference_sites=tuple(reads),
    )


def _mixed_spill_payload(
    region: RegisterRegionPlan,
    expansion: Any,
    role_count: int,
    register_roles: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    by_node = {x.node_id: x for x in region.register_values}
    offsets = np.zeros(role_count, dtype=np.int64)
    table: list[int] = []
    for role in range(role_count):
        offsets[role] = len(table)
        for nid in expansion.role_node_ids[role]:
            value = by_node.get(nid)
            if role in register_roles:
                if value is None:
                    raise RegisterRegionError(
                        f"register role {role} output node {nid} has no RegisterValuePlan"
                    )
                table.append(int(value.slot_offset) if value.spill_required else -1)
            else:
                table.append(-1)
    return offsets, np.asarray(table, dtype=np.int64)


def _mixed_role_tables(
    compiler: Any,
    typed: Any,
    region: RegisterRegionPlan,
    expansion: Any,
    family: Any,
    callbacks_factory: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]:
    role_base = [0]
    node_ids: list[int] = []
    out_offsets: list[int] = []
    callbacks: list[Any] = []
    for role, stream in enumerate(expansion.role_node_ids):
        is_register = family.role_plans[role].backend is RoleBackend.REGISTER_NATIVE
        for nid in stream:
            addr = typed.node_by_id.get(nid)
            if addr is None:
                raise RegisterRegionError(f"mixed region node {nid} has no physical slot")
            node_ids.append(int(nid))
            out_offsets.append(int(addr.offset))
            callbacks.append(None if is_register else callbacks_factory.callback_for_obj(
                compiler.sequential.node_by_id[nid].obj
            ))
        role_base.append(len(node_ids))
    return (
        np.asarray(role_base, dtype=np.int64),
        np.asarray(node_ids, dtype=np.int64),
        np.asarray(out_offsets, dtype=np.int64),
        tuple(callbacks),
    )


def _guard_tables(
    compiler: Any,
    typed: Any,
    region: RegisterRegionPlan,
    expansion: Any,
    role_base: np.ndarray,
) -> tuple[np.ndarray, ...]:
    boundary_by_node = {x.node_id: x for x in region.boundaries}
    reload_by_node = {x.node_id: x for x in region.reload_values}
    deps_by_dst: dict[int, list[int]] = defaultdict(list)
    for src, dst in compiler.trace.dependencies:
        deps_by_dst[int(dst)].append(int(src))

    flat_nodes: list[int] = []
    for stream in expansion.role_node_ids:
        flat_nodes.extend(int(nid) for nid in stream)

    boundary_start = [0]
    boundary_pool: list[int] = []
    boundary_slot: list[int] = []
    boundary_node: list[int] = []
    reload_start = [0]
    reload_pool: list[int] = []
    reload_slot: list[int] = []
    reload_node: list[int] = []

    register_nodes = {x.node_id for x in region.register_values}
    for nid in flat_nodes:
        boundary = boundary_by_node.get(nid)
        if boundary is not None:
            for src in boundary.spill_before:
                addr = typed.node_by_id.get(src)
                if addr is None:
                    raise RegisterRegionError(f"boundary spill node {src} has no physical slot")
                boundary_pool.append(_MIXED_POOL_CODE[addr.pool])
                boundary_slot.append(int(addr.offset))
                boundary_node.append(int(src))
        boundary_start.append(len(boundary_node))

        # ReloadValuePlan counts only dependencies consumed by register-owned
        # occurrences. A slot/Python occurrence may read the same physical source,
        # but that read is handled by its strict callback and must not inflate the
        # register reload contract.
        if nid in register_nodes:
            for src in deps_by_dst.get(nid, ()):
                if src not in reload_by_node:
                    continue
                addr = typed.node_by_id.get(src)
                if addr is None:
                    raise RegisterRegionError(f"reload node {src} has no physical slot")
                reload_pool.append(_MIXED_POOL_CODE[addr.pool])
                reload_slot.append(int(addr.offset))
                reload_node.append(int(src))
        reload_start.append(len(reload_node))

    expected_reload_uses = sum(x.register_consumer_count for x in region.reload_values)
    if len(reload_node) != expected_reload_uses:
        raise RegisterRegionError(
            "mixed reload table does not match optimized ReloadValuePlan consumer counts"
        )
    return (
        np.asarray(boundary_start, dtype=np.int64),
        np.asarray(boundary_pool, dtype=np.int64),
        np.asarray(boundary_slot, dtype=np.int64),
        np.asarray(boundary_node, dtype=np.int64),
        np.asarray(reload_start, dtype=np.int64),
        np.asarray(reload_pool, dtype=np.int64),
        np.asarray(reload_slot, dtype=np.int64),
        np.asarray(reload_node, dtype=np.int64),
    )


def _seed_mixed_owners(
    compiler: Any,
    typed: Any,
    region_node_ids: set[int],
    downers: np.ndarray,
    iowners: np.ndarray,
    bowners: np.ndarray,
    oowners: np.ndarray,
) -> None:
    external: set[int] = set()
    for src, dst in compiler.trace.dependencies:
        if dst in region_node_ids and src not in region_node_ids:
            external.add(int(src))
    occupied: dict[tuple[str, int], int] = {}
    for nid in sorted(external):
        addr = typed.node_by_id.get(nid)
        if addr is None:
            continue
        key = (addr.pool, int(addr.offset))
        prior = occupied.get(key)
        if prior is not None and prior != nid:
            raise RegisterRegionError(
                "isolated mixed proof has colliding external live-ins; integrated predecessor execution required"
            )
        occupied[key] = nid
        if addr.pool == "double":
            downers[addr.offset] = nid
        elif addr.pool == "int64":
            iowners[addr.offset] = nid
        elif addr.pool == "bool":
            bowners[addr.offset] = nid
        else:
            oowners[addr.offset] = nid


def _mixed_register_role_source_lines(
    role: int,
    family: Any,
    role_meta: list[Any],
    kernel: MixedRegisterFamilyKernelCode,
) -> list[str]:
    fid, args, pool = role_meta[role]
    arg_exprs = [_emit_runtime_arg_expr(site, f"c_{role}") for _name, _typ, site in args]
    ring_reads = [
        x for x in kernel.reference_sites
        if x.consumer_role == role and x.source_kind == "register_ring"
    ]
    ring_args = [_ring_expr(x) for x in ring_reads]
    callargs = ", ".join(
        arg_exprs + ring_args + [
            "dslots", "islots", "bslots",
            "ref_kind", "ref_a", "ref_b", "ref_mod", "ref_lo", "ref_table_offset", "ref_table",
        ]
    )
    value_name = f"value_{role}"
    lines = [
        f"            pos = role_base[{role}] + c_{role}",
        "            if check_contract:",
        "                for guard_i in range(reload_start[pos], reload_start[pos + 1]):",
        "                    _guard_owner(reload_pool[guard_i], reload_slot[guard_i], reload_node[guard_i], downers, iowners, bowners, oowners)",
        f"            {value_name} = f_role_{role}({callargs})",
    ]
    depth = kernel.ring_depth_by_role[role]
    if depth:
        for lag in range(depth - 1, 0, -1):
            lines.append(f"            ring_{role}_{lag} = ring_{role}_{lag - 1}")
        lines.append(f"            ring_{role}_0 = {value_name}")
    lines += [
        f"            spill_idx = spill_table[spill_offset[{role}] + c_{role}]",
        "            if spill_idx >= 0:",
    ]
    arr = {"double": "dslots", "int64": "islots", "bool": "bslots"}[pool]
    owner = {"double": "downers", "int64": "iowners", "bool": "bowners"}[pool]
    if pool == "bool":
        lines.append(f"                {arr}[spill_idx] = 1 if {value_name} else 0")
    else:
        lines.append(f"                {arr}[spill_idx] = {value_name}")
    lines.append(f"                {owner}[spill_idx] = node_ids[pos]")
    lines += [
        "            if record_output:",
        f"                result[out_pos] = <double>{value_name}",
        "                out_pos += 1",
        f"            c_{role} += 1",
    ]
    return lines


def _mixed_slot_role_source_lines(role: int, role_meta: list[Any]) -> list[str]:
    _fid, args, pool = role_meta[role]
    arg_exprs = [_emit_runtime_arg_expr(site, f"c_{role}") for _name, _typ, site in args]
    pyargs = ", ".join(arg_exprs)
    call = f"callbacks[pos]({pyargs})" if pyargs else "callbacks[pos]()"
    lines = [
        f"            pos = role_base[{role}] + c_{role}",
        "            if check_contract:",
        "                for guard_i in range(boundary_start[pos], boundary_start[pos + 1]):",
        "                    _guard_owner(boundary_pool[guard_i], boundary_slot[guard_i], boundary_node[guard_i], downers, iowners, bowners, oowners)",
        f"            py_value = {call}",
        "            out_idx = out_offsets[pos]",
        "            nid = node_ids[pos]",
    ]
    if pool == "double":
        lines += ["            dslots[out_idx] = <double>py_value", "            downers[out_idx] = nid", "            numeric_value = dslots[out_idx]"]
    elif pool == "int64":
        lines += ["            islots[out_idx] = <long long>py_value", "            iowners[out_idx] = nid", "            numeric_value = <double>islots[out_idx]"]
    elif pool == "bool":
        lines += ["            bslots[out_idx] = 1 if bool(py_value) else 0", "            bowners[out_idx] = nid", "            numeric_value = <double>bslots[out_idx]"]
    else:
        raise RegisterRegionError("first mixed RegisterRegion proof supports numeric Python boundary outputs only")
    lines += [
        "            if record_output:",
        "                result[out_pos] = numeric_value",
        "                out_pos += 1",
        f"            c_{role} += 1",
    ]
    return lines


def _build_mixed_register_region_cython(
    compiler: Any,
    block_index: int,
    build_dir: str | Path,
    *,
    module_prefix: str,
    optimization: str,
) -> MixedRegisterRegionProgram:
    if compiler.slots is None:
        compiler.lower_slots()
    analysis = compiler.native_plan or compiler.analyze_native()
    typed = _typed_slot_plan(compiler, analysis, include_objects=True)
    _optimized, region, family = _selected_region(compiler, block_index)
    try:
        backend = build_native_family_backend_plan(compiler, region.family_id)
        expansion = expand_native_family_instance(compiler, backend, region.instance_ordinal)
    except NativePlanError as exc:
        raise RegisterRegionError(str(exc)) from exc
    if tuple(nid for _role, nid in expansion.operation_sequence) != region.node_ids:
        raise RegisterRegionError("optimized region and native family expansion disagree")

    analysis_by_id = {x.formula_op_id: x for x in analysis.formulae}
    sites = build_runtime_arg_sites(backend)
    register_roles = set(region.register_roles)
    roles_by_fid: dict[int, list[int]] = defaultdict(list)
    for role, fid in enumerate(family.formula_op_ids):
        roles_by_fid[fid].append(role)
    for fid, roles in roles_by_fid.items():
        ownership = {role in register_roles for role in roles}
        if len(ownership) > 1:
            raise RegisterRegionError(
                f"FormulaOp {fid} is shared by register and slot roles; first mixed proof keeps ownership FormulaOp-stable"
            )
    native_for_refs = {
        fid: any(
            role in register_roles and family.formula_op_ids[role] == fid
            for role in range(len(family.role_plans))
        ) and _exact_native_formula_ok(compiler, fid, analysis_by_id[fid])
        for fid in set(backend.kernel.formula_op_ids)
    }
    try:
        refs = build_reference_binding_plan(compiler, typed, backend, native_for_refs)
    except NativeReferenceError as exc:
        raise RegisterRegionError(str(exc)) from exc
    kernel = _mixed_kernel_contract(compiler, region, family, refs)

    site_rows_by_role: dict[int, list[Any]] = {role: [] for role in range(kernel.role_count)}
    for site in sites:
        site_rows_by_role[site.role].append(site)
    role_meta: list[tuple[int, list[tuple[str, str, Any]], str]] = []
    emitters: dict[int, _RegisterFormulaEmitter] = {}
    for role, fid in enumerate(kernel.formula_op_ids):
        rows = sorted(site_rows_by_role[role], key=lambda x: x.arg_index)
        pool = typed.formula_by_id[fid].pool
        if family.role_plans[role].backend is RoleBackend.REGISTER_NATIVE:
            if not _exact_native_formula_ok(compiler, fid, analysis_by_id[fid]):
                raise RegisterRegionError(f"register role {role} is no longer exact-native")
            if analysis_by_id[fid].observed_return_type.dtype not in {"float64", "int64", "bool"}:
                raise RegisterRegionError("mixed RegisterRegion supports scalar numeric register outputs only")
            emitter = _RegisterFormulaEmitter(
                compiler, typed, fid, refs.sites, kernel.reference_sites, role
            )
            emitters[role] = emitter
            formal_names = [a.arg for a in emitter.fn.args.args]
        else:
            if pool == "object":
                raise RegisterRegionError(
                    "first mixed RegisterRegion proof supports numeric Python boundary outputs only"
                )
            formal_names = [f"arg{i}" for i in range(len(rows))]
        if len(rows) != len(formal_names):
            raise RegisterRegionError("FormulaOp arity does not match mixed runtime binding ABI")
        role_meta.append((
            fid,
            [(name, site.cython_type, site) for name, site in zip(formal_names, rows)],
            pool,
        ))

    family_node_ids = set(region.node_ids)
    dslots, islots, bslots, oslots = _initial_hybrid_pools(compiler, typed, family_node_ids)
    downers = np.full(typed.double_count, -1, dtype=np.int64)
    iowners = np.full(typed.int_count, -1, dtype=np.int64)
    bowners = np.full(typed.bool_count, -1, dtype=np.int64)
    oowners = np.full(typed.object_count, -1, dtype=np.int64)
    _seed_mixed_owners(
        compiler, typed, family_node_ids, downers, iowners, bowners, oowners
    )
    callback_factory = _StrictCallbackFactory(
        compiler, typed, dslots, islots, bslots, oslots,
        downers, iowners, bowners, oowners,
    )
    role_base, node_ids, out_offsets, callbacks = _mixed_role_tables(
        compiler, typed, region, expansion, family, callback_factory
    )
    guards = _guard_tables(compiler, typed, region, expansion, role_base)
    (
        boundary_start, boundary_pool, boundary_slot, boundary_node,
        reload_start, reload_pool, reload_slot, reload_node,
    ) = guards

    try:
        binding_payload = encode_runtime_bindings(backend.instances[region.instance_ordinal], sites)
        reference_payload = encode_reference_instance(refs, region.instance_ordinal)
    except (NativeBindingError, NativeReferenceError) as exc:
        raise RegisterRegionError(str(exc)) from exc
    spill_offset, spill_table = _mixed_spill_payload(
        region, expansion, kernel.role_count, register_roles
    )
    variant_arr = np.asarray(region.variant_ids, dtype=np.int64)

    build_dir = Path(build_dir); build_dir.mkdir(parents=True, exist_ok=True)
    module_name = f"{module_prefix}_{kernel.code_signature[:12]}"
    pyx = build_dir / f"{module_name}.pyx"
    lines = [
        "# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False",
        "import numpy as np",
        "cimport numpy as cnp",
        "",
    ]
    lines.extend(_emit_runtime_helpers())
    lines.extend(_emit_runtime_reference_helper())
    lines += [
        "cdef inline void _guard_owner(long long pool, long long offset, long long nid, long long[::1] downers, long long[::1] iowners, long long[::1] bowners, long long[::1] oowners):",
        "    if pool == 0:",
        "        if downers[offset] != nid: raise ValueError('required double reload/spill is not materialized')",
        "    elif pool == 1:",
        "        if iowners[offset] != nid: raise ValueError('required int reload/spill is not materialized')",
        "    elif pool == 2:",
        "        if bowners[offset] != nid: raise ValueError('required bool reload/spill is not materialized')",
        "    else:",
        "        if oowners[offset] != nid: raise ValueError('required object reload/spill is not materialized')",
        "",
    ]
    for role in sorted(register_roles):
        fid = kernel.formula_op_ids[role]
        args = [(name, typ) for name, typ, _site in role_meta[role][1]]
        lines.extend(emitters[role].emit_role_function(
            args, analysis_by_id[fid].observed_return_type.dtype
        ))
        lines.append("")

    public_args = [
        "cnp.ndarray[cnp.float64_t, ndim=1] d_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] i_arr",
        "cnp.ndarray[cnp.uint8_t, ndim=1] b_arr",
        "object o_slots",
        "cnp.ndarray[cnp.int64_t, ndim=1] do_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] io_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bo_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] oo_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] variant_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_kind_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_ia_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_ib_arr",
        "cnp.ndarray[cnp.float64_t, ndim=1] bind_da_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_payload_offset_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_payload_count_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_aux_offset_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_aux_count_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_i_payload_arr",
        "cnp.ndarray[cnp.float64_t, ndim=1] bind_d_payload_arr",
        "object bind_object_payload",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_run_ends_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] bind_cursor_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_kind_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_a_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_b_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_mod_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_lo_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_table_offset_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] ref_table_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] spill_offset_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] spill_table_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] role_base_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] node_ids_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] out_offsets_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] boundary_start_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] boundary_pool_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] boundary_slot_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] boundary_node_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] reload_start_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] reload_pool_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] reload_slot_arr",
        "cnp.ndarray[cnp.int64_t, ndim=1] reload_node_arr",
        "object callbacks",
        "Py_ssize_t out_count",
    ]
    internal_args = [
        "double[::1] dslots", "long long[::1] islots", "unsigned char[::1] bslots", "object o_slots",
        "long long[::1] downers", "long long[::1] iowners", "long long[::1] bowners", "long long[::1] oowners",
        "long long[::1] variants",
        "long long[::1] bind_kind", "long long[::1] bind_ia", "long long[::1] bind_ib", "double[::1] bind_da",
        "long long[::1] bind_payload_offset", "long long[::1] bind_payload_count",
        "long long[::1] bind_aux_offset", "long long[::1] bind_aux_count",
        "long long[::1] bind_i_payload", "double[::1] bind_d_payload", "object bind_object_payload",
        "long long[::1] bind_run_ends", "long long[::1] bind_cursor",
        "long long[::1] ref_kind", "long long[::1] ref_a", "long long[::1] ref_b", "long long[::1] ref_mod",
        "long long[::1] ref_lo", "long long[::1] ref_table_offset", "long long[::1] ref_table",
        "long long[::1] spill_offset", "long long[::1] spill_table",
        "long long[::1] role_base", "long long[::1] node_ids", "long long[::1] out_offsets",
        "long long[::1] boundary_start", "long long[::1] boundary_pool", "long long[::1] boundary_slot", "long long[::1] boundary_node",
        "long long[::1] reload_start", "long long[::1] reload_pool", "long long[::1] reload_slot", "long long[::1] reload_node",
        "object callbacks", "Py_ssize_t out_count", "double[::1] result", "bint record_output", "bint check_contract",
    ]
    lines.append("cdef inline Py_ssize_t _run_region(")
    for i, arg in enumerate(internal_args):
        lines.append(f"    {arg}{',' if i + 1 < len(internal_args) else ''}")
    lines += [
        "):",
        "    cdef Py_ssize_t iteration, out_pos = 0, spill_idx, pos, out_idx, guard_i",
        "    cdef long long variant, nid",
        "    cdef object py_value",
        "    cdef double numeric_value = 0.0",
    ]
    for role in range(kernel.role_count):
        lines.append(f"    cdef Py_ssize_t c_{role} = 0")
        if role in register_roles:
            dtype = analysis_by_id[kernel.formula_op_ids[role]].observed_return_type.dtype
            ctype = {"float64": "double", "int64": "long long", "bool": "bint"}[dtype]
            init = "0.0" if dtype == "float64" else "0"
            lines.append(f"    cdef {ctype} value_{role} = {init}")
            for lag in range(kernel.ring_depth_by_role[role]):
                lines.append(f"    cdef {ctype} ring_{role}_{lag} = {init}")
    lines += [
        "    for iteration in range(variants.shape[0]):",
        "        variant = variants[iteration]",
    ]
    for j, variant_id in enumerate(kernel.used_variant_ids):
        pattern = family.variants[variant_id]
        lines.append(("        if" if j == 0 else "        elif") + f" variant == {variant_id}:")
        if not pattern:
            lines.append("            pass")
        for role in pattern:
            if role in register_roles:
                lines.extend(_mixed_register_role_source_lines(role, family, role_meta, kernel))
            else:
                lines.extend(_mixed_slot_role_source_lines(role, role_meta))
    lines += [
        "        else:",
        "            raise ValueError('variant id outside compiled mixed register grammar')",
    ]
    for role in range(kernel.role_count):
        lines.append(
            f"    if c_{role} != role_base[{role + 1}] - role_base[{role}]: raise ValueError('mixed register role occurrence count mismatch')"
        )
    lines += [
        "    if record_output and out_pos != out_count:",
        "        raise ValueError('mixed register output count does not match runtime variant stream')",
        "    return out_pos",
        "",
    ]

    view_lines = [
        "    cdef double[::1] dslots = d_arr",
        "    cdef long long[::1] islots = i_arr",
        "    cdef unsigned char[::1] bslots = b_arr",
        "    cdef long long[::1] downers = do_arr",
        "    cdef long long[::1] iowners = io_arr",
        "    cdef long long[::1] bowners = bo_arr",
        "    cdef long long[::1] oowners = oo_arr",
        "    cdef long long[::1] variants = variant_arr",
        "    cdef long long[::1] bind_kind = bind_kind_arr",
        "    cdef long long[::1] bind_ia = bind_ia_arr",
        "    cdef long long[::1] bind_ib = bind_ib_arr",
        "    cdef double[::1] bind_da = bind_da_arr",
        "    cdef long long[::1] bind_payload_offset = bind_payload_offset_arr",
        "    cdef long long[::1] bind_payload_count = bind_payload_count_arr",
        "    cdef long long[::1] bind_aux_offset = bind_aux_offset_arr",
        "    cdef long long[::1] bind_aux_count = bind_aux_count_arr",
        "    cdef long long[::1] bind_i_payload = bind_i_payload_arr",
        "    cdef double[::1] bind_d_payload = bind_d_payload_arr",
        "    cdef long long[::1] bind_run_ends = bind_run_ends_arr",
        "    cdef long long[::1] bind_cursor = bind_cursor_arr",
        "    cdef long long[::1] ref_kind = ref_kind_arr",
        "    cdef long long[::1] ref_a = ref_a_arr",
        "    cdef long long[::1] ref_b = ref_b_arr",
        "    cdef long long[::1] ref_mod = ref_mod_arr",
        "    cdef long long[::1] ref_lo = ref_lo_arr",
        "    cdef long long[::1] ref_table_offset = ref_table_offset_arr",
        "    cdef long long[::1] ref_table = ref_table_arr",
        "    cdef long long[::1] spill_offset = spill_offset_arr",
        "    cdef long long[::1] spill_table = spill_table_arr",
        "    cdef long long[::1] role_base = role_base_arr",
        "    cdef long long[::1] node_ids = node_ids_arr",
        "    cdef long long[::1] out_offsets = out_offsets_arr",
        "    cdef long long[::1] boundary_start = boundary_start_arr",
        "    cdef long long[::1] boundary_pool = boundary_pool_arr",
        "    cdef long long[::1] boundary_slot = boundary_slot_arr",
        "    cdef long long[::1] boundary_node = boundary_node_arr",
        "    cdef long long[::1] reload_start = reload_start_arr",
        "    cdef long long[::1] reload_pool = reload_pool_arr",
        "    cdef long long[::1] reload_slot = reload_slot_arr",
        "    cdef long long[::1] reload_node = reload_node_arr",
    ]
    internal_call = (
        "dslots, islots, bslots, o_slots, downers, iowners, bowners, oowners, variants, "
        "bind_kind, bind_ia, bind_ib, bind_da, bind_payload_offset, bind_payload_count, "
        "bind_aux_offset, bind_aux_count, bind_i_payload, bind_d_payload, bind_object_payload, "
        "bind_run_ends, bind_cursor, ref_kind, ref_a, ref_b, ref_mod, ref_lo, ref_table_offset, ref_table, "
        "spill_offset, spill_table, role_base, node_ids, out_offsets, boundary_start, boundary_pool, "
        "boundary_slot, boundary_node, reload_start, reload_pool, reload_slot, reload_node, callbacks, out_count"
    )

    def emit_header(name: str, extra: tuple[str, ...] = ()) -> None:
        lines.append(f"cpdef {('double' if name == 'benchmark_repeat' else 'cnp.ndarray')} {name}(")
        all_args = public_args + list(extra)
        for i, arg in enumerate(all_args):
            lines.append(f"    {arg}{',' if i + 1 < len(all_args) else ''}")
        lines.append("):")

    emit_header("run")
    lines.extend(view_lines)
    lines += [
        "    cdef cnp.ndarray[cnp.float64_t, ndim=1] result_arr = np.empty(out_count, dtype=np.float64)",
        "    cdef double[::1] result = result_arr",
        f"    _run_region({internal_call}, result, True, True)",
        "    return result_arr",
        "",
    ]
    emit_header("benchmark_repeat", ("Py_ssize_t repeats",))
    lines.extend(view_lines)
    lines += [
        "    cdef cnp.ndarray[cnp.float64_t, ndim=1] result_arr = np.empty(1, dtype=np.float64)",
        "    cdef double[::1] result = result_arr",
        "    cdef Py_ssize_t rep",
        "    for rep in range(repeats):",
        f"        _run_region({internal_call}, result, False, False)",
        "    return dslots[0] if dslots.shape[0] else 0.0",
        "",
    ]
    source = "\n".join(lines) + "\n"; pyx.write_text(source)
    t0 = time.perf_counter()
    module, so_path, _proc = build_extension(
        pyx, module_name, build_dir, openmp=False, optimization=optimization, native_arch=False
    )
    build_seconds = time.perf_counter() - t0

    byid = compiler.sequential.node_by_id
    expected = np.asarray(
        [float(_authoritative_value(byid[nid])) for _role, nid in expansion.operation_sequence],
        dtype=np.float64,
    )
    expected_final: dict[tuple[str, int], Any] = {}
    register_values = {x.node_id: x for x in region.register_values}
    for role, nid in expansion.operation_sequence:
        if role in register_roles:
            value = register_values[nid]
            if not value.spill_required:
                continue
            key = (value.pool, int(value.slot_offset))
        else:
            addr = typed.node_by_id[nid]
            key = (addr.pool, int(addr.offset))
        expected_final[key] = _authoritative_value(byid[nid])
    expected_spills = tuple((pool, offset, value) for (pool, offset), value in expected_final.items())

    report = MixedRegisterRegionBuildReport(
        block_index=region.block_index,
        family_id=region.family_id,
        instance_ordinal=region.instance_ordinal,
        operation_count=region.operation_count,
        register_operation_count=region.register_operation_count,
        slot_operation_count=region.slot_operation_count,
        python_boundary_count=len(region.boundaries),
        reload_value_count=len(region.reload_values),
        role_count=kernel.role_count,
        used_variant_count=len(kernel.used_variant_ids),
        ring_role_count=sum(x > 0 for x in kernel.ring_depth_by_role),
        ring_scalar_count=sum(kernel.ring_depth_by_role),
        slot_reference_site_count=sum(x.source_kind == "slot_reference" for x in kernel.reference_sites),
        register_reference_site_count=sum(x.source_kind == "register_ring" for x in kernel.reference_sites),
        spill_count=region.spill_count,
        source_lines=source.count("\n"),
        source_bytes=len(source.encode("utf-8")),
        build_seconds=build_seconds,
        pyx_path=str(pyx),
        so_path=str(so_path),
    )
    return MixedRegisterRegionProgram(
        compiler=compiler, region=region, family=family, kernel=kernel, module=module,
        build_report=report,
        dslots=dslots, islots=islots, bslots=bslots, oslots=oslots,
        downers=downers, iowners=iowners, bowners=bowners, oowners=oowners,
        variant_arr=variant_arr, binding_payload=binding_payload, reference_payload=reference_payload,
        spill_offset=spill_offset, spill_table=spill_table,
        role_base=role_base, node_ids=node_ids, out_offsets=out_offsets,
        boundary_start=boundary_start, boundary_pool=boundary_pool, boundary_slot=boundary_slot, boundary_node=boundary_node,
        reload_start=reload_start, reload_pool=reload_pool, reload_slot=reload_slot, reload_node=reload_node,
        callbacks=callbacks, expected=expected, expected_spills=expected_spills,
        _initial_dslots=dslots.copy(), _initial_islots=islots.copy(), _initial_bslots=bslots.copy(),
        _initial_oslots=list(oslots), _initial_downers=downers.copy(), _initial_iowners=iowners.copy(),
        _initial_bowners=bowners.copy(), _initial_oowners=oowners.copy(),
    )
