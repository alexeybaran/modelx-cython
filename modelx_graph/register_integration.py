from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .native_bindings import build_runtime_arg_sites, encode_runtime_bindings
from .native_family import (
    _emit_runtime_arg_expr,
    _exact_native_formula_ok,
    _typed_slot_plan,
)
from .native_plan import build_native_family_backend_plan, expand_native_family_instance
from .native_references import build_reference_binding_plan, encode_reference_instance
from .optimized_execution import RegisterRegionPlan, RoleBackend
from .register_region import (
    MixedRegisterFamilyKernelCode,
    RegisterRegionError,
    _RegisterFormulaEmitter,
    _guard_tables,
    _kernel_contract,
    _mixed_kernel_contract,
    _mixed_role_tables,
    _mixed_spill_payload,
    _ring_expr,
    _selected_region,
)


@dataclass(frozen=True)
class IntegratedRegisterRegionSpec:
    """One register-region instance prepared for whole-artifact composition.

    The generated function is internal Cython only. Runtime descriptor arrays and
    callbacks are converted to typed memoryviews once by ``_make_rr_context`` after
    extension build, so ``run_one`` pays neither a Python region call nor repeated
    ndarray->memoryview conversion for the region.
    """

    block_index: int
    family_id: int
    instance_ordinal: int
    function_name: str
    operation_count: int
    register_operation_count: int
    slot_operation_count: int
    python_boundary_count: int
    register_reference_site_count: int
    slot_reference_site_count: int
    spill_count: int
    source_lines: tuple[str, ...]
    runtime_data: tuple[Any, ...]
    binding_payload: Any


# Runtime-data tuple positions shared with the generated context factory.
_RR_VARIANTS = 0
_RR_BIND_START = 1
_RR_REF_START = 14
_RR_SPILL_OFFSET = 21
_RR_SPILL_TABLE = 22
_RR_ROLE_BASE = 23
_RR_NODE_IDS = 24
_RR_OUT_OFFSETS = 25
_RR_BOUNDARY_START = 26
_RR_BOUNDARY_POOL = 27
_RR_BOUNDARY_SLOT = 28
_RR_BOUNDARY_NODE = 29
_RR_RELOAD_START = 30
_RR_RELOAD_POOL = 31
_RR_RELOAD_SLOT = 32
_RR_RELOAD_NODE = 33
_RR_CALLBACKS = 34
_RR_DATA_LEN = 35
_RR_INLINE_FORMULA_ROLE_LIMIT = 32
_RR_INLINE_FORMULA_OPERATION_LIMIT = 128
_RR_INTEGRATION_SOURCE_LINE_LIMIT = 1500


def register_region_auto_cost_gate(region: RegisterRegionPlan) -> tuple[bool, str | None]:
    """First conservative whole-artifact composition cost gate.

    The generic mixed runner is already competent at native FormulaOps. Register
    composition earns its keep only when local/native work is not outnumbered by
    Python crossings and when enough values remain local to eliminate physical
    slot traffic. This is intentionally a coarse structural gate, not a final cost
    model; explicit block selection can still force a region for proof/debug.
    """
    if len(region.boundaries) > region.register_operation_count:
        return False, "cost_gate:python_boundaries_exceed_register_operations"
    spill_count = sum(1 for value in region.register_values if value.spill_required)
    if region.register_operation_count and spill_count * 4 > region.register_operation_count:
        return False, "cost_gate:register_spill_ratio_exceeds_quarter"
    return True, None


def emit_integrated_register_context_support() -> list[str]:
    """Emit one reusable typed context class for all integrated regions."""
    fields = [
        "long long[::1] variants",
        "long long[::1] bind_kind", "long long[::1] bind_ia", "long long[::1] bind_ib",
        "double[::1] bind_da",
        "long long[::1] bind_payload_offset", "long long[::1] bind_payload_count",
        "long long[::1] bind_aux_offset", "long long[::1] bind_aux_count",
        "long long[::1] bind_i_payload", "double[::1] bind_d_payload",
        "object bind_object_payload", "long long[::1] bind_run_ends", "long long[::1] bind_cursor",
        "long long[::1] ref_kind", "long long[::1] ref_a", "long long[::1] ref_b",
        "long long[::1] ref_mod", "long long[::1] ref_lo",
        "long long[::1] ref_table_offset", "long long[::1] ref_table",
        "long long[::1] spill_offset", "long long[::1] spill_table",
        "long long[::1] role_base", "long long[::1] node_ids", "long long[::1] out_offsets",
        "long long[::1] boundary_start", "long long[::1] boundary_pool",
        "long long[::1] boundary_slot", "long long[::1] boundary_node",
        "long long[::1] reload_start", "long long[::1] reload_pool",
        "long long[::1] reload_slot", "long long[::1] reload_node",
        "object callbacks",
    ]
    lines = ["cdef class _IntegratedRRContext:"]
    lines.extend(f"    cdef {field}" for field in fields)
    lines += [
        "",
        "cpdef object _make_rr_context(object data):",
        f"    if len(data) != {_RR_DATA_LEN}: raise ValueError('integrated RegisterRegion data shape mismatch')",
        "    cdef _IntegratedRRContext ctx = _IntegratedRRContext()",
        "    ctx.variants = data[0]",
        "    ctx.bind_kind = data[1]", "    ctx.bind_ia = data[2]", "    ctx.bind_ib = data[3]",
        "    ctx.bind_da = data[4]", "    ctx.bind_payload_offset = data[5]", "    ctx.bind_payload_count = data[6]",
        "    ctx.bind_aux_offset = data[7]", "    ctx.bind_aux_count = data[8]",
        "    ctx.bind_i_payload = data[9]", "    ctx.bind_d_payload = data[10]",
        "    ctx.bind_object_payload = data[11]", "    ctx.bind_run_ends = data[12]", "    ctx.bind_cursor = data[13]",
        "    ctx.ref_kind = data[14]", "    ctx.ref_a = data[15]", "    ctx.ref_b = data[16]",
        "    ctx.ref_mod = data[17]", "    ctx.ref_lo = data[18]", "    ctx.ref_table_offset = data[19]",
        "    ctx.ref_table = data[20]", "    ctx.spill_offset = data[21]", "    ctx.spill_table = data[22]",
        "    ctx.role_base = data[23]", "    ctx.node_ids = data[24]", "    ctx.out_offsets = data[25]",
        "    ctx.boundary_start = data[26]", "    ctx.boundary_pool = data[27]",
        "    ctx.boundary_slot = data[28]", "    ctx.boundary_node = data[29]",
        "    ctx.reload_start = data[30]", "    ctx.reload_pool = data[31]",
        "    ctx.reload_slot = data[32]", "    ctx.reload_node = data[33]",
        "    ctx.callbacks = data[34]",
        "    return ctx",
        "",
        "cdef inline void _rr_guard_owner(long long pool, long long offset, long long nid, long long[::1] downers, long long[::1] iowners, long long[::1] bowners, long long[::1] oowners):",
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
    return lines


def _integrated_kernel_contract(compiler: Any, region: Any, family: Any, refs: Any) -> MixedRegisterFamilyKernelCode:
    if region.slot_operation_count:
        return _mixed_kernel_contract(compiler, region, family, refs)
    closed = _kernel_contract(compiler, region, family, refs)
    return MixedRegisterFamilyKernelCode(
        family_id=closed.family_id,
        code_signature=closed.code_signature,
        variant_count=closed.variant_count,
        used_variant_ids=tuple(sorted(set(int(x) for x in region.variant_ids))),
        role_count=closed.role_count,
        formula_op_ids=closed.formula_op_ids,
        role_backends=tuple(x.backend.value for x in family.role_plans),
        ring_depth_by_role=closed.ring_depth_by_role,
        reference_sites=closed.reference_sites,
    )


def _register_role_lines(role: int, role_meta: list[Any], kernel: MixedRegisterFamilyKernelCode, fn_prefix: str) -> list[str]:
    _fid, args, pool = role_meta[role]
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
    value = f"value_{role}"
    lines = [
        f"            pos = role_base[{role}] + c_{role}",
        "            for guard_i in range(reload_start[pos], reload_start[pos + 1]):",
        "                _rr_guard_owner(reload_pool[guard_i], reload_slot[guard_i], reload_node[guard_i], downers, iowners, bowners, oowners)",
        f"            {value} = {fn_prefix}_f_role_{role}({callargs})",
    ]
    depth = kernel.ring_depth_by_role[role]
    if depth:
        for lag in range(depth - 1, 0, -1):
            lines.append(f"            ring_{role}_{lag} = ring_{role}_{lag - 1}")
        lines.append(f"            ring_{role}_0 = {value}")
    lines += [
        f"            spill_idx = spill_table[spill_offset[{role}] + c_{role}]",
        "            if spill_idx >= 0:",
    ]
    arr = {"double": "dslots", "int64": "islots", "bool": "bslots"}[pool]
    owner = {"double": "downers", "int64": "iowners", "bool": "bowners"}[pool]
    if pool == "bool":
        lines.append(f"                {arr}[spill_idx] = 1 if {value} else 0")
    else:
        lines.append(f"                {arr}[spill_idx] = {value}")
    lines += [
        f"                {owner}[spill_idx] = node_ids[pos]",
        f"            c_{role} += 1",
    ]
    return lines


def _slot_role_lines(role: int, role_meta: list[Any]) -> list[str]:
    _fid, args, pool = role_meta[role]
    arg_exprs = [_emit_runtime_arg_expr(site, f"c_{role}") for _name, _typ, site in args]
    pyargs = ", ".join(arg_exprs)
    call = f"callbacks[pos]({pyargs})" if pyargs else "callbacks[pos]()"
    lines = [
        f"            pos = role_base[{role}] + c_{role}",
        "            for guard_i in range(boundary_start[pos], boundary_start[pos + 1]):",
        "                _rr_guard_owner(boundary_pool[guard_i], boundary_slot[guard_i], boundary_node[guard_i], downers, iowners, bowners, oowners)",
        f"            py_value = {call}",
        "            out_idx = out_offsets[pos]",
        "            nid = node_ids[pos]",
    ]
    if pool == "double":
        lines += ["            dslots[out_idx] = <double>py_value", "            downers[out_idx] = nid"]
    elif pool == "int64":
        lines += ["            islots[out_idx] = <long long>py_value", "            iowners[out_idx] = nid"]
    elif pool == "bool":
        lines += ["            bslots[out_idx] = 1 if bool(py_value) else 0", "            bowners[out_idx] = nid"]
    elif pool == "object":
        lines += ["            oslots[out_idx] = py_value", "            oowners[out_idx] = nid"]
    else:
        raise RegisterRegionError(f"integrated RegisterRegion boundary uses unknown output pool {pool!r}")
    lines.append(f"            c_{role} += 1")
    return lines


def prepare_integrated_register_region(
    compiler: Any,
    block_index: int,
    typed: Any,
    callback_factory: Any,
    analysis: Any,
) -> IntegratedRegisterRegionSpec:
    """Prepare one proven RegisterRegion for in-module whole-artifact execution.

    No extension is compiled here. The returned source fragment and runtime data
    are composed into the existing whole-artifact module. Unsupported shapes raise
    ``RegisterRegionError`` so the caller can retain the generic family runner.
    """
    _optimized, region, family = _selected_region(compiler, block_index)
    if not isinstance(region, RegisterRegionPlan):
        raise RegisterRegionError("optimized block is not a RegisterRegion")
    if not region.register_roles:
        raise RegisterRegionError("integrated RegisterRegion has no register-owned role")

    backend = build_native_family_backend_plan(compiler, region.family_id)
    expansion = expand_native_family_instance(compiler, backend, region.instance_ordinal)
    if tuple(nid for _role, nid in expansion.operation_sequence) != region.node_ids:
        raise RegisterRegionError("optimized region and native family expansion disagree")

    analysis_by_id = {x.formula_op_id: x for x in analysis.formulae}
    sites = build_runtime_arg_sites(backend)
    register_roles = set(region.register_roles)
    frozen_plan = getattr(compiler, "frozen_reference_plan", None)
    frozen_formula_ids = set() if frozen_plan is None else set(frozen_plan.formula_op_ids)

    def exact_register_formula(fid: int) -> bool:
        if _exact_native_formula_ok(compiler, fid, analysis_by_id[fid]):
            return True
        if fid not in frozen_formula_ids:
            return False
        from .frozen_references import frozen_formula_ast_exact_ok
        return frozen_formula_ast_exact_ok(compiler, fid, analysis_by_id[fid])

    # A FormulaOp shared between register and slot roles would require role-level
    # reference/native ownership below the current FormulaOp-level reference plan.
    roles_by_fid: dict[int, list[int]] = defaultdict(list)
    for role, fid in enumerate(family.formula_op_ids):
        roles_by_fid[fid].append(role)
    for fid, roles in roles_by_fid.items():
        if len({role in register_roles for role in roles}) > 1:
            raise RegisterRegionError(
                f"FormulaOp {fid} is shared by register and slot roles; integrated first slice keeps ownership FormulaOp-stable"
            )

    native_for_refs = {
        fid: any(
            role in register_roles and family.formula_op_ids[role] == fid
            for role in range(len(family.role_plans))
        ) and exact_register_formula(fid)
        for fid in set(backend.kernel.formula_op_ids)
    }
    refs = build_reference_binding_plan(compiler, typed, backend, native_for_refs)
    kernel = _integrated_kernel_contract(compiler, region, family, refs)

    site_rows_by_role: dict[int, list[Any]] = {role: [] for role in range(kernel.role_count)}
    for site in sites:
        site_rows_by_role[site.role].append(site)
    role_meta: list[tuple[int, list[tuple[str, str, Any]], str]] = []
    emitters: dict[int, _RegisterFormulaEmitter] = {}
    for role, fid in enumerate(kernel.formula_op_ids):
        rows = sorted(site_rows_by_role[role], key=lambda x: x.arg_index)
        pool = typed.formula_by_id[fid].pool
        if family.role_plans[role].backend is RoleBackend.REGISTER_NATIVE:
            if not exact_register_formula(fid):
                raise RegisterRegionError(f"register role {role} is no longer exact-native")
            if analysis_by_id[fid].observed_return_type.dtype not in {"float64", "int64", "bool"}:
                raise RegisterRegionError("integrated RegisterRegion supports scalar numeric register outputs only")
            emitter = _RegisterFormulaEmitter(
                compiler, typed, fid, refs.sites, kernel.reference_sites, role,
                frozen_sites=(() if frozen_plan is None else frozen_plan.sites_for_formula(fid)),
            )
            emitters[role] = emitter
            formal_names = [a.arg for a in emitter.fn.args.args]
        else:
            # Python boundaries may produce object values in the integrated whole
            # artifact because the canonical object pool/owner arrays are already
            # part of the shared runtime. Register-owned outputs remain numeric.
            formal_names = [f"arg{i}" for i in range(len(rows))]
        if len(rows) != len(formal_names):
            raise RegisterRegionError("FormulaOp arity does not match integrated runtime binding ABI")
        role_meta.append((fid, [(n, site.cython_type, site) for n, site in zip(formal_names, rows)], pool))

    role_base, node_ids, out_offsets, callbacks = _mixed_role_tables(
        compiler, typed, region, expansion, family, callback_factory
    )
    (
        boundary_start, boundary_pool, boundary_slot, boundary_node,
        reload_start, reload_pool, reload_slot, reload_node,
    ) = _guard_tables(compiler, typed, region, expansion, role_base)
    binding_payload = encode_runtime_bindings(backend.instances[region.instance_ordinal], sites)
    reference_payload = encode_reference_instance(refs, region.instance_ordinal)
    spill_offset, spill_table = _mixed_spill_payload(region, expansion, kernel.role_count, register_roles)
    variant_arr = np.asarray(region.variant_ids, dtype=np.int64)

    runtime_data = (
        variant_arr,
        *binding_payload.as_call_args(),
        *reference_payload.as_call_args(),
        spill_offset, spill_table,
        role_base, node_ids, out_offsets,
        boundary_start, boundary_pool, boundary_slot, boundary_node,
        reload_start, reload_pool, reload_slot, reload_node,
        callbacks,
    )
    if len(runtime_data) != _RR_DATA_LEN:
        raise RegisterRegionError("integrated RegisterRegion runtime-data ABI mismatch")

    prefix = f"_rr_b{block_index}"
    lines: list[str] = []
    for role in sorted(register_roles):
        fid = kernel.formula_op_ids[role]
        args = [(name, typ) for name, typ, _site in role_meta[role][1]]
        emitted = emitters[role].emit_role_function(args, analysis_by_id[fid].observed_return_type.dtype)
        old = f"f_role_{role}("
        emitted = [line.replace(old, f"{prefix}_f_role_{role}(") for line in emitted]
        if kernel.role_count > _RR_INLINE_FORMULA_ROLE_LIMIT or region.operation_count > _RR_INLINE_FORMULA_OPERATION_LIMIT:
            emitted = [line.replace("cdef inline ", "cdef ", 1) if line.startswith("cdef inline ") else line for line in emitted]
        lines.extend(emitted)
        lines.append("")

    lines += [
        f"cdef void {prefix}(_IntegratedRRContext ctx, double[::1] dslots, long long[::1] islots, unsigned char[::1] bslots, object oslots, long long[::1] downers, long long[::1] iowners, long long[::1] bowners, long long[::1] oowners):",
        "    cdef long long[::1] variants = ctx.variants",
        "    cdef long long[::1] bind_kind = ctx.bind_kind",
        "    cdef long long[::1] bind_ia = ctx.bind_ia", "    cdef long long[::1] bind_ib = ctx.bind_ib",
        "    cdef double[::1] bind_da = ctx.bind_da",
        "    cdef long long[::1] bind_payload_offset = ctx.bind_payload_offset",
        "    cdef long long[::1] bind_payload_count = ctx.bind_payload_count",
        "    cdef long long[::1] bind_aux_offset = ctx.bind_aux_offset",
        "    cdef long long[::1] bind_aux_count = ctx.bind_aux_count",
        "    cdef long long[::1] bind_i_payload = ctx.bind_i_payload",
        "    cdef double[::1] bind_d_payload = ctx.bind_d_payload",
        "    cdef object bind_object_payload = ctx.bind_object_payload",
        "    cdef long long[::1] bind_run_ends = ctx.bind_run_ends",
        "    cdef long long[::1] bind_cursor = ctx.bind_cursor",
        "    cdef long long[::1] ref_kind = ctx.ref_kind", "    cdef long long[::1] ref_a = ctx.ref_a",
        "    cdef long long[::1] ref_b = ctx.ref_b", "    cdef long long[::1] ref_mod = ctx.ref_mod",
        "    cdef long long[::1] ref_lo = ctx.ref_lo",
        "    cdef long long[::1] ref_table_offset = ctx.ref_table_offset",
        "    cdef long long[::1] ref_table = ctx.ref_table",
        "    cdef long long[::1] spill_offset = ctx.spill_offset",
        "    cdef long long[::1] spill_table = ctx.spill_table",
        "    cdef long long[::1] role_base = ctx.role_base", "    cdef long long[::1] node_ids = ctx.node_ids",
        "    cdef long long[::1] out_offsets = ctx.out_offsets",
        "    cdef long long[::1] boundary_start = ctx.boundary_start",
        "    cdef long long[::1] boundary_pool = ctx.boundary_pool",
        "    cdef long long[::1] boundary_slot = ctx.boundary_slot",
        "    cdef long long[::1] boundary_node = ctx.boundary_node",
        "    cdef long long[::1] reload_start = ctx.reload_start",
        "    cdef long long[::1] reload_pool = ctx.reload_pool",
        "    cdef long long[::1] reload_slot = ctx.reload_slot",
        "    cdef long long[::1] reload_node = ctx.reload_node",
        "    cdef object callbacks = ctx.callbacks",
        "    cdef Py_ssize_t iteration, spill_idx, pos, out_idx, guard_i",
        "    cdef long long variant, nid",
        "    cdef object py_value",
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
    lines += ["    for iteration in range(variants.shape[0]):", "        variant = variants[iteration]"]
    for j, variant_id in enumerate(kernel.used_variant_ids):
        pattern = family.variants[variant_id]
        lines.append(("        if" if j == 0 else "        elif") + f" variant == {variant_id}:")
        if not pattern:
            lines.append("            pass")
        for role in pattern:
            if role in register_roles:
                lines.extend(_register_role_lines(role, role_meta, kernel, prefix))
            else:
                lines.extend(_slot_role_lines(role, role_meta))
    lines += ["        else:", "            raise ValueError('variant id outside integrated RegisterRegion grammar')"]
    for role in range(kernel.role_count):
        lines.append(
            f"    if c_{role} != role_base[{role + 1}] - role_base[{role}]: raise ValueError('integrated RegisterRegion role occurrence count mismatch')"
        )
    lines.append("")

    if len(lines) > _RR_INTEGRATION_SOURCE_LINE_LIMIT:
        raise RegisterRegionError(
            f"integrated RegisterRegion source complexity {len(lines)} lines exceeds first-slice limit {_RR_INTEGRATION_SOURCE_LINE_LIMIT}"
        )

    return IntegratedRegisterRegionSpec(
        block_index=block_index,
        family_id=region.family_id,
        instance_ordinal=region.instance_ordinal,
        function_name=prefix,
        operation_count=region.operation_count,
        register_operation_count=region.register_operation_count,
        slot_operation_count=region.slot_operation_count,
        python_boundary_count=len(region.boundaries),
        register_reference_site_count=region.register_reference_site_count,
        slot_reference_site_count=region.slot_reference_site_count,
        spill_count=region.spill_count,
        source_lines=tuple(lines),
        runtime_data=runtime_data,
        binding_payload=binding_payload,
    )
