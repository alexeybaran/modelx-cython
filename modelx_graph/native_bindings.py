from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numbers
import numpy as np

from .loop_recovery import (
    AffineIntBinding,
    ArithmeticRunsIntBinding,
    ConstantBinding,
    InterleavedAffineIntBinding,
    PeriodicBinding,
    RunLengthBinding,
    TableBinding,
)


class NativeBindingError(RuntimeError):
    """Fail-closed error while lowering canonical argument bindings."""


BIND_CONST = 0
BIND_AFFINE_INT = 1
BIND_ARITH_RUNS_INT = 2
BIND_PERIODIC = 3
BIND_INTERLEAVED_AFFINE_INT = 4
BIND_TABLE = 5
BIND_RUN_LENGTH = 6


@dataclass(frozen=True)
class RuntimeArgSite:
    """One formal argument site in a shared family kernel.

    ``binding_kind`` describes the representation when every observed instance
    agrees; ``dynamic`` means the generated kernel must use the generic descriptor
    helper. ``value_kind`` is the exact ABI family used to preserve Python value
    categories at the boundary.
    """

    site_id: int
    role: int
    arg_index: int
    cython_type: str
    value_kind: str
    binding_kind: str


@dataclass(frozen=True)
class EncodedRuntimeBindings:
    kind: np.ndarray
    ia: np.ndarray
    ib: np.ndarray
    da: np.ndarray
    payload_offset: np.ndarray
    payload_count: np.ndarray
    aux_offset: np.ndarray
    aux_count: np.ndarray
    i_payload: np.ndarray
    d_payload: np.ndarray
    object_payload: tuple[Any, ...]
    aux_i_payload: np.ndarray
    cursor: np.ndarray

    def as_call_args(self) -> tuple[Any, ...]:
        return (
            self.kind,
            self.ia,
            self.ib,
            self.da,
            self.payload_offset,
            self.payload_count,
            self.aux_offset,
            self.aux_count,
            self.i_payload,
            self.d_payload,
            list(self.object_payload),
            self.aux_i_payload,
            self.cursor,
        )


def _value_kind_from_values(values: Iterable[Any]) -> str:
    vals = list(values)
    if not vals:
        raise NativeBindingError("binding has no values")
    if all(isinstance(v, (bool, np.bool_)) for v in vals):
        return "bool"
    if all(
        isinstance(v, (numbers.Integral, np.integer))
        and not isinstance(v, (bool, np.bool_))
        for v in vals
    ):
        return "int64"
    if all(
        isinstance(v, (numbers.Real, np.floating))
        and not isinstance(v, (bool, np.bool_))
        for v in vals
    ):
        return "float64"
    return "object"


def binding_value_kind(binding: Any) -> str:
    if isinstance(binding, ConstantBinding):
        return _value_kind_from_values((binding.value,))
    if isinstance(binding, (AffineIntBinding, InterleavedAffineIntBinding, ArithmeticRunsIntBinding)):
        return "int64"
    if isinstance(binding, PeriodicBinding):
        return _value_kind_from_values(binding.pattern)
    if isinstance(binding, TableBinding):
        return _value_kind_from_values(binding.values)
    if isinstance(binding, RunLengthBinding):
        # The compressed run-value binding is authoritative. Inspect exactly one
        # value per run rather than expanding the full occurrence stream.
        return _value_kind_from_values(
            binding.run_values.value_at(i) for i in range(len(binding.run_ends))
        )
    raise NativeBindingError(f"unknown canonical binding {type(binding).__name__}")


def cython_type_for_value_kind(value_kind: str) -> str:
    return {
        "bool": "bint",
        "int64": "long long",
        "float64": "double",
        "object": "object",
    }[value_kind]


def build_runtime_arg_sites(backend: Any) -> tuple[RuntimeArgSite, ...]:
    sites: list[RuntimeArgSite] = []
    site_id = 0
    for role, _fid in enumerate(backend.kernel.formula_op_ids):
        observed: list[tuple[Any, ...]] = []
        for inst in backend.instances:
            rb = inst.role_bindings[role]
            if rb is not None:
                observed.append(tuple(rb.arg_bindings))
        if not observed:
            raise NativeBindingError(f"family role {role} has no realized argument binding")
        arity = len(observed[0])
        if any(len(row) != arity for row in observed):
            raise NativeBindingError("formal/binding arity changes across family instances")
        for arg_index in range(arity):
            bindings = [row[arg_index] for row in observed]
            value_kinds = {binding_value_kind(b) for b in bindings}
            # Do not silently widen Python int/float/bool semantics. A heterogeneous
            # argument site remains an object boundary until a dedicated proof exists.
            value_kind = next(iter(value_kinds)) if len(value_kinds) == 1 else "object"
            binding_kinds = {b.kind for b in bindings}
            structural_kind = next(iter(binding_kinds)) if len(binding_kinds) == 1 else "dynamic"
            sites.append(
                RuntimeArgSite(
                    site_id=site_id,
                    role=role,
                    arg_index=arg_index,
                    cython_type=cython_type_for_value_kind(value_kind),
                    value_kind=value_kind,
                    binding_kind=structural_kind,
                )
            )
            site_id += 1
    return tuple(sites)


def _append_values(
    values: Iterable[Any],
    value_kind: str,
    i_payload: list[int],
    d_payload: list[float],
    object_payload: list[Any],
) -> tuple[int, int]:
    vals = list(values)
    if value_kind in {"int64", "bool"}:
        off = len(i_payload)
        i_payload.extend(int(v) for v in vals)
    elif value_kind == "float64":
        off = len(d_payload)
        d_payload.extend(float(v) for v in vals)
    elif value_kind == "object":
        off = len(object_payload)
        object_payload.extend(vals)
    else:  # pragma: no cover
        raise NativeBindingError(f"unknown binding value kind {value_kind}")
    return off, len(vals)


def encode_runtime_bindings(inst: Any, sites: tuple[RuntimeArgSite, ...]) -> EncodedRuntimeBindings:
    n = len(sites)
    kind = np.full(n, -1, dtype=np.int64)
    ia = np.zeros(n, dtype=np.int64)
    ib = np.zeros(n, dtype=np.int64)
    da = np.zeros(n, dtype=np.float64)
    payload_offset = np.zeros(n, dtype=np.int64)
    payload_count = np.zeros(n, dtype=np.int64)
    aux_offset = np.zeros(n, dtype=np.int64)
    aux_count = np.zeros(n, dtype=np.int64)
    i_payload: list[int] = []
    d_payload: list[float] = []
    object_payload: list[Any] = []
    aux_i_payload: list[int] = []

    for site in sites:
        rb = inst.role_bindings[site.role]
        if rb is None:
            # Legal only for a role absent from this instance's exact variant stream.
            continue
        binding = rb.arg_bindings[site.arg_index]
        actual_kind = binding_value_kind(binding)
        if site.value_kind != "object" and actual_kind != site.value_kind:
            raise NativeBindingError(
                f"runtime binding value kind {actual_kind} violates site ABI {site.value_kind}"
            )
        value_kind = site.value_kind

        if isinstance(binding, ConstantBinding):
            kind[site.site_id] = BIND_CONST
            value = binding.value
            if value_kind == "float64":
                da[site.site_id] = float(value)
            elif value_kind in {"int64", "bool"}:
                ia[site.site_id] = int(value)
            else:
                off, count = _append_values((value,), value_kind, i_payload, d_payload, object_payload)
                payload_offset[site.site_id], payload_count[site.site_id] = off, count
        elif isinstance(binding, AffineIntBinding):
            kind[site.site_id] = BIND_AFFINE_INT
            ia[site.site_id] = int(binding.base)
            ib[site.site_id] = int(binding.step)
        elif isinstance(binding, PeriodicBinding):
            kind[site.site_id] = BIND_PERIODIC
            off, count = _append_values(binding.pattern, value_kind, i_payload, d_payload, object_payload)
            payload_offset[site.site_id], payload_count[site.site_id] = off, count
        elif isinstance(binding, InterleavedAffineIntBinding):
            kind[site.site_id] = BIND_INTERLEAVED_AFFINE_INT
            off, count = _append_values(binding.bases, "int64", i_payload, d_payload, object_payload)
            payload_offset[site.site_id], payload_count[site.site_id] = off, count
            aoff = len(aux_i_payload)
            aux_i_payload.extend(int(x) for x in binding.steps)
            aux_offset[site.site_id], aux_count[site.site_id] = aoff, len(binding.steps)
        elif isinstance(binding, ArithmeticRunsIntBinding):
            kind[site.site_id] = BIND_ARITH_RUNS_INT
            off, count = _append_values(binding.run_starts, "int64", i_payload, d_payload, object_payload)
            payload_offset[site.site_id], payload_count[site.site_id] = off, count
            aoff = len(aux_i_payload)
            aux_i_payload.extend(int(x) for x in binding.run_ends)
            aux_offset[site.site_id], aux_count[site.site_id] = aoff, len(binding.run_ends)
            ia[site.site_id] = int(binding.step)
        elif isinstance(binding, TableBinding):
            kind[site.site_id] = BIND_TABLE
            off, count = _append_values(binding.values, value_kind, i_payload, d_payload, object_payload)
            payload_offset[site.site_id], payload_count[site.site_id] = off, count
        elif isinstance(binding, RunLengthBinding):
            kind[site.site_id] = BIND_RUN_LENGTH
            run_values = [binding.run_values.value_at(i) for i in range(len(binding.run_ends))]
            off, count = _append_values(run_values, value_kind, i_payload, d_payload, object_payload)
            payload_offset[site.site_id], payload_count[site.site_id] = off, count
            aoff = len(aux_i_payload)
            aux_i_payload.extend(int(x) for x in binding.run_ends)
            aux_offset[site.site_id], aux_count[site.site_id] = aoff, len(binding.run_ends)
        else:  # pragma: no cover
            raise NativeBindingError(f"unsupported runtime binding {type(binding).__name__}")

    return EncodedRuntimeBindings(
        kind=kind,
        ia=ia,
        ib=ib,
        da=da,
        payload_offset=payload_offset,
        payload_count=payload_count,
        aux_offset=aux_offset,
        aux_count=aux_count,
        i_payload=np.asarray(i_payload, dtype=np.int64),
        d_payload=np.asarray(d_payload, dtype=np.float64),
        object_payload=tuple(object_payload),
        aux_i_payload=np.asarray(aux_i_payload, dtype=np.int64),
        cursor=np.zeros(n, dtype=np.int64),
    )


def python_binding_value(payload: EncodedRuntimeBindings, site: RuntimeArgSite, occurrence: int) -> Any:
    """Reference implementation of the runtime ABI used by tests and validation."""
    sid = site.site_id
    k = int(payload.kind[sid])
    if k < 0:
        raise NativeBindingError("executed role has no runtime binding descriptor")
    if k == BIND_CONST:
        if site.value_kind == "float64":
            return float(payload.da[sid])
        if site.value_kind == "int64":
            return int(payload.ia[sid])
        if site.value_kind == "bool":
            return bool(payload.ia[sid])
        return payload.object_payload[int(payload.payload_offset[sid])]
    if k == BIND_AFFINE_INT:
        return int(payload.ia[sid] + payload.ib[sid] * occurrence)
    poff = int(payload.payload_offset[sid]); pcount = int(payload.payload_count[sid])
    aoff = int(payload.aux_offset[sid]); acount = int(payload.aux_count[sid])
    if k == BIND_PERIODIC:
        pos = occurrence % pcount
    elif k == BIND_INTERLEAVED_AFFINE_INT:
        phase = occurrence % pcount; cycle = occurrence // pcount
        return int(payload.i_payload[poff + phase] + payload.aux_i_payload[aoff + phase] * cycle)
    elif k == BIND_ARITH_RUNS_INT:
        r = 0
        while r < acount and occurrence >= payload.aux_i_payload[aoff + r]:
            r += 1
        if r >= acount:
            raise IndexError(occurrence)
        start_pos = 0 if r == 0 else int(payload.aux_i_payload[aoff + r - 1])
        return int(payload.i_payload[poff + r] + payload.ia[sid] * (occurrence - start_pos))
    elif k == BIND_TABLE:
        if occurrence < 0 or occurrence >= pcount:
            raise IndexError(occurrence)
        pos = occurrence
    elif k == BIND_RUN_LENGTH:
        r = 0
        while r < acount and occurrence >= payload.aux_i_payload[aoff + r]:
            r += 1
        if r >= acount:
            raise IndexError(occurrence)
        pos = r
    else:
        raise NativeBindingError(f"unsupported descriptor kind {k}")
    if site.value_kind in {"int64", "bool"}:
        value: Any = int(payload.i_payload[poff + pos])
        return bool(value) if site.value_kind == "bool" else value
    if site.value_kind == "float64":
        return float(payload.d_payload[poff + pos])
    return payload.object_payload[poff + pos]
