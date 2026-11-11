from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .loop_recovery import LoopCodeFamily, LoopInstancePlan, RoleBindingTemplate


class NativePlanError(RuntimeError):
    """Fail-closed error while lowering canonical loop structure to native runtime plans."""


@dataclass(frozen=True)
class NativeFamilyKernelPlan:
    """Instance-independent code identity for one recovered loop family.

    This object deliberately contains no concrete Space objects, realized argument
    values, physical slot addresses or per-instance control stream.  Those belong
    to :class:`NativeFamilyInstanceRuntimePlan`.
    """

    family_id: int
    formula_op_ids: tuple[int, ...]
    variants: tuple[tuple[int, ...], ...]
    code_signature: str

    @property
    def role_count(self) -> int:
        return len(self.formula_op_ids)

    @property
    def variant_count(self) -> int:
        return len(self.variants)


@dataclass(frozen=True)
class NativeFamilyInstanceRuntimePlan:
    """Exact runtime data for one concrete occurrence of a shared family kernel."""

    instance_ordinal: int
    block_index: int
    family_id: int
    variant_ids: tuple[int, ...]
    role_bindings: tuple[RoleBindingTemplate | None, ...]


@dataclass(frozen=True)
class NativeFamilyBackendPlan:
    """Shared family code plus all exact concrete instances that consume it."""

    kernel: NativeFamilyKernelPlan
    instances: tuple[NativeFamilyInstanceRuntimePlan, ...]


@dataclass(frozen=True)
class NativeInstanceExpansion:
    """Exact FormulaOp-role/runtime-node stream for validation and address fitting."""

    family_id: int
    instance_ordinal: int
    role_node_ids: tuple[tuple[int, ...], ...]
    operation_sequence: tuple[tuple[int, int], ...]
    role_occurrence_counts: tuple[int, ...]

    @property
    def operation_count(self) -> int:
        return len(self.operation_sequence)


def _kernel_signature(family: LoopCodeFamily, formula_ops: tuple[Any, ...]) -> str:
    payload = {
        "abi": "native-family-kernel-v1",
        "family_roles": [
            {
                "schema_uid": formula_ops[fid].schema_uid,
                "space_family_uid": formula_ops[fid].space_family_uid,
                "arity": formula_ops[fid].arity,
            }
            for fid in family.formula_op_ids
        ],
        "variants": [list(v) for v in family.expanded_variants()],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_native_family_backend_plan(compiler: Any, family_id: int) -> NativeFamilyBackendPlan:
    """Lower one canonical family into shared-code and per-instance runtime plans.

    No semantic inference is performed here.  The canonical family grammar and
    exact observed instance streams remain authoritative.
    """

    structured = getattr(compiler, "structured", None)
    if structured is None or structured.canonical_plan is None:
        raise NativePlanError("canonical execution plan is required")
    plan = structured.canonical_plan
    try:
        family = plan.code_families[family_id]
    except IndexError as exc:
        raise NativePlanError(f"unknown family_id {family_id}") from exc

    variants = family.expanded_variants()
    if not variants:
        raise NativePlanError("loop family has no executable variants")
    role_count = len(family.formula_op_ids)
    for variant_id, variant in enumerate(variants):
        for role in variant:
            if role < 0 or role >= role_count:
                raise NativePlanError(
                    f"family {family_id} variant {variant_id} references invalid role {role}"
                )

    kernel = NativeFamilyKernelPlan(
        family_id=family_id,
        formula_op_ids=tuple(family.formula_op_ids),
        variants=tuple(tuple(v) for v in variants),
        code_signature=_kernel_signature(family, plan.formula_ops),
    )

    instances: list[NativeFamilyInstanceRuntimePlan] = []
    ordinal = 0
    for inst in plan.loop_instances:
        if inst.family_id != family_id:
            continue
        if len(inst.role_bindings) != role_count:
            raise NativePlanError("family instance role-binding width does not match kernel")
        for variant_id in inst.variant_ids:
            if variant_id < 0 or variant_id >= len(variants):
                raise NativePlanError(
                    f"family {family_id} instance uses unknown variant {variant_id}"
                )
        instances.append(
            NativeFamilyInstanceRuntimePlan(
                instance_ordinal=ordinal,
                block_index=inst.block_index,
                family_id=family_id,
                variant_ids=tuple(inst.variant_ids),
                role_bindings=tuple(inst.role_bindings),
            )
        )
        ordinal += 1

    if not instances:
        raise NativePlanError(f"family {family_id} has no concrete loop instances")
    return NativeFamilyBackendPlan(kernel=kernel, instances=tuple(instances))


def expand_native_family_instance(
    compiler: Any,
    backend_plan: NativeFamilyBackendPlan,
    instance_ordinal: int,
) -> NativeInstanceExpansion:
    """Expand one exact instance using canonical variants and binding occurrence counts.

    This is intentionally a proof/analysis operation.  Production generated code
    consumes the compact variant and binding streams directly rather than storing
    this operation-sized expansion.
    """

    if instance_ordinal < 0 or instance_ordinal >= len(backend_plan.instances):
        raise NativePlanError(f"unknown instance ordinal {instance_ordinal}")
    inst = backend_plan.instances[instance_ordinal]
    kernel = backend_plan.kernel
    counters = [0] * kernel.role_count
    role_node_ids: list[list[int]] = [[] for _ in range(kernel.role_count)]
    operation_sequence: list[tuple[int, int]] = []
    runtime_to_id = compiler.trace.runtime_to_id

    for variant_id in inst.variant_ids:
        for role in kernel.variants[variant_id]:
            rb = inst.role_bindings[role]
            if rb is None:
                raise NativePlanError(
                    f"family {kernel.family_id} instance {instance_ordinal} role {role} has no binding"
                )
            occurrence = counters[role]
            runtime = rb.runtime_node(occurrence)
            try:
                node_id = runtime_to_id[runtime]
            except KeyError as exc:
                raise NativePlanError(
                    "family binding runtime node is absent from the authoritative realized trace"
                ) from exc
            role_node_ids[role].append(node_id)
            operation_sequence.append((role, node_id))
            counters[role] += 1

    return NativeInstanceExpansion(
        family_id=kernel.family_id,
        instance_ordinal=instance_ordinal,
        role_node_ids=tuple(tuple(v) for v in role_node_ids),
        operation_sequence=tuple(operation_sequence),
        role_occurrence_counts=tuple(counters),
    )
