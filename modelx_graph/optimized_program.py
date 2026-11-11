from __future__ import annotations

"""Backend-neutral optimized program for the canonical native graph path.

This module is deliberately free of Cython/code-generation imports.  It freezes
semantic/storage decisions shared by generated Python and Cython while leaving
backend representation choices to their emitters.
"""

import ast
from dataclasses import dataclass
from typing import Any

from .frontend import CanonicalModel
from .program_inputs import OptimizedInputSpec
from .program_semantics import (
    GuardSpec, StaticScalarFact, ValidatedStaticFact, FiniteDomainFact,
    FixedCoordinateFact, RuntimeScalarHelperFact, SourceBackedScheduledFact,
    StepLookupDomainFact,
)
from .template_ir import TemplateError, TemplateStoragePlan, _offset, _range_parts
from .stage_storage_plan import StageStoragePlan, StageStorageCompatibility, validate_template_storage_compatibility


class OptimizedProgramError(RuntimeError):
    pass


def _local_names(fn: ast.FunctionDef | None) -> tuple[str, ...]:
    if fn is None:
        return ()
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return tuple(sorted(names))


@dataclass(frozen=True)
class OptimizedProgram:
    """Backend-neutral executable program after canonical storage optimization.

    The object intentionally keeps the canonical model while making all decisions
    previously borrowed by the Python emitter from ``CythonGenerator`` explicit:
    storage layout, optimization level, coordinate step, build fingerprint,
    input/order metadata and formula-local names.
    """

    canonical: CanonicalModel
    optimization_level: str
    full_array: bool
    layout: TemplateStoragePlan
    stage_storage_plan: StageStoragePlan | None
    stage_storage_compatibility: StageStorageCompatibility | None
    build_fingerprint: str
    coordinate_step: int
    input_order: tuple[str, ...]
    inputs: tuple[OptimizedInputSpec, ...]
    pre_static_order: tuple[str, ...]
    post_static_order: tuple[tuple[int, tuple[str, ...]], ...]
    local_names_by_uid: tuple[tuple[str, tuple[str, ...]], ...]
    guards: tuple[GuardSpec, ...]
    static_facts: tuple[StaticScalarFact, ...]
    validated_static_facts: tuple[ValidatedStaticFact, ...]
    finite_domain_facts: tuple[FiniteDomainFact, ...]
    fixed_coordinate_facts: tuple[FixedCoordinateFact, ...]
    runtime_scalar_helper_facts: tuple[RuntimeScalarHelperFact, ...]
    source_backed_scheduled_facts: tuple[SourceBackedScheduledFact, ...]
    step_lookup_domain_facts: tuple[StepLookupDomainFact, ...]

    @property
    def variants(self):
        return self.canonical.variants

    @property
    def schedule(self):
        return self.canonical.executable

    def position_shift(self, coordinate_offset: int) -> int:
        step = int(self.coordinate_step)
        if coordinate_offset % step != 0:
            raise OptimizedProgramError(
                f"coordinate offset {coordinate_offset} is not aligned to executable range step {step}"
            )
        return coordinate_offset // step

    def coordinate_offset(self, node: ast.AST, variable: str) -> int | None:
        return _offset(node, variable)

    def range_parts(self, args: tuple[ast.AST, ...]) -> tuple[ast.AST, ast.AST, int]:
        try:
            return _range_parts(args)
        except TemplateError as exc:
            raise OptimizedProgramError(str(exc)) from exc

    def local_names(self, uid: str) -> tuple[str, ...]:
        return dict(self.local_names_by_uid).get(uid, ())

    def input_spec(self, key: str) -> OptimizedInputSpec:
        for spec in self.inputs:
            if spec.key == key:
                return spec
        raise OptimizedProgramError(f"unknown optimized input key {key!r}")

    def guard(self, code: int) -> GuardSpec:
        for spec in self.guards:
            if spec.code == int(code):
                return spec
        raise OptimizedProgramError(f"unknown optimized guard code {code!r}")

    def manifest(self) -> dict[str, Any]:
        inputs = {spec.key: spec.manifest() for spec in self.inputs}
        storage = {
            "optimization_level": self.layout.optimization_level,
            "current_slots": dict(sorted(self.layout.current_slots.items())),
            "current_slot_count": int(self.layout.current_slot_count),
            "scalar_slots": dict(sorted(self.layout.scalar_slots.items())),
            "history_slots": dict(sorted(self.layout.history_slots.items())),
            "ring_bases": dict(sorted(self.layout.ring_bases.items())),
            "ring_depths": dict(sorted(self.layout.ring_depths.items())),
            "derived_state_slots": dict(sorted(self.layout.derived_state_slots.items())),
            "derived_region_state_slots": dict(sorted(self.layout.derived_region_state_slots.items())),
            "boundary_seed_slots": dict(sorted(self.layout.boundary_seed_slots.items())),
            "state_slot_count": int(self.layout.state_slot_count),
            "fused_reductions": sorted(self.layout.fused_reductions),
            "history_reasons": {k: list(v) for k, v in sorted(self.layout.history_reasons.items())},
        }
        return {
            "schema": "modelx_graph.optimized_program.v1",
            "output_name": self.canonical.output_name,
            "output_uid": self.canonical.output_uid,
            "optimization_level": self.optimization_level,
            "full_array": self.full_array,
            "coordinate_step": self.coordinate_step,
            "build_fingerprint": self.build_fingerprint,
            "formula_hash": self.schedule.formula_hash,
            "input_order": list(self.input_order),
            "inputs": inputs,
            "guards": [spec.manifest() for spec in self.guards],
            "static_facts": [spec.manifest() for spec in self.static_facts],
            "validated_static_facts": [spec.manifest() for spec in self.validated_static_facts],
            "finite_domain_facts": [spec.manifest() for spec in self.finite_domain_facts],
            "fixed_coordinate_facts": [spec.manifest() for spec in self.fixed_coordinate_facts],
            "runtime_scalar_helper_facts": [spec.manifest() for spec in self.runtime_scalar_helper_facts],
            "source_backed_scheduled_facts": [spec.manifest() for spec in self.source_backed_scheduled_facts],
            "step_lookup_domain_facts": [spec.manifest() for spec in self.step_lookup_domain_facts],
            "pre_static_order": list(self.pre_static_order),
            "post_static_order": {str(i): list(rows) for i, rows in self.post_static_order},
            "local_names_by_uid": {uid: list(names) for uid, names in self.local_names_by_uid},
            "storage": storage,
            "stage_storage": None if self.stage_storage_plan is None else self.stage_storage_plan.manifest(),
            "stage_storage_compatibility": None if self.stage_storage_compatibility is None else self.stage_storage_compatibility.manifest(),
            "executable": self.schedule.manifest(),
        }


def build_optimized_program(
    model: CanonicalModel,
    *,
    optimization_level: str | None = None,
    full_array: bool | None = None,
) -> OptimizedProgram:
    """Freeze backend-neutral optimization decisions for Python/Cython emitters."""
    if optimization_level is None:
        optimization_level = "O0" if full_array else "O2"
    level = optimization_level.upper()
    if full_array is not None and bool(full_array) != (level == "O0"):
        raise OptimizedProgramError(
            "full_array and optimization_level request conflicting storage policies"
        )
    try:
        layout = model.executable.storage_plan(level)
    except TemplateError as exc:
        raise OptimizedProgramError(str(exc)) from exc
    stage_storage_compatibility = None
    if model.stage_storage_plan is not None:
        stage_storage_compatibility = validate_template_storage_compatibility(
            model.stage_storage_plan, layout
        )
        # For the already-authoritative one-stage subset the legacy physical
        # layout is now an implementation of canonical storage semantics, not an
        # independent source of truth.  Contradiction is therefore a hard error.
        if (
            len(model.stage_storage_plan.stages) == 1
            and model.stage_storage_plan.geometry_proven
            and not stage_storage_compatibility.compatible
        ):
            raise OptimizedProgramError(
                "legacy storage layout contradicts canonical stage storage: "
                + "; ".join(stage_storage_compatibility.reasons)
            )
    steps = {int(block.coordinate_step) for block in model.executable.loops}
    if len(steps) != 1:
        raise OptimizedProgramError(
            "current optimized program requires executable loops to share one signed coordinate step"
        )
    locals_by_uid = tuple(
        (uid, _local_names(cv.function)) for uid, cv in sorted(model.variants.items())
    )
    optimized_inputs = tuple(
        OptimizedInputSpec(
            key=key, dtype=spec.dtype, ndim=int(spec.ndim), scope=spec.scope,
            description=spec.description, expected_shape=spec.expected_shape,
            domain_token=spec.domain_token, normalized_table=spec.normalized_table,
            point_row_source=spec.point_row_source, point_field=spec.point_field,
            normalized_structured_column=spec.normalized_structured_column,
            enum_labels=spec.enum_labels, category_validity_key=spec.category_validity_key,
            normalized_lookup_1d=spec.normalized_lookup_1d,
            lookup_1d_role=spec.lookup_1d_role,
        )
        for key, spec in model.inputs.items()
    )
    guards = tuple(
        sorted(
            (guard for cv in model.variants.values() for guard in cv.guards),
            key=lambda g: g.code,
        )
    )
    if len({g.code for g in guards}) != len(guards):
        raise OptimizedProgramError("optimized guard codes are not unique")
    return OptimizedProgram(
        canonical=model,
        optimization_level=level,
        full_array=(level == "O0"),
        layout=layout,
        stage_storage_plan=model.stage_storage_plan,
        stage_storage_compatibility=stage_storage_compatibility,
        build_fingerprint=model.executable.build_fingerprint(level),
        coordinate_step=next(iter(steps)),
        input_order=tuple(model.inputs),
        inputs=optimized_inputs,
        pre_static_order=tuple(model.executable.pre_scalars),
        post_static_order=tuple(
            (block.index, tuple(block.post_scalars)) for block in model.executable.loops
        ),
        local_names_by_uid=locals_by_uid,
        guards=guards,
        static_facts=tuple(model.static_facts),
        validated_static_facts=tuple(model.validated_static_facts),
        finite_domain_facts=tuple(model.finite_domain_facts),
        fixed_coordinate_facts=tuple(model.fixed_coordinate_facts),
        runtime_scalar_helper_facts=tuple(model.runtime_scalar_helper_facts),
        source_backed_scheduled_facts=tuple(model.source_backed_scheduled_facts),
        step_lookup_domain_facts=tuple(model.step_lookup_domain_facts),
    )
