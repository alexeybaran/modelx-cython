from __future__ import annotations

"""Graph-free formula/provider translation service used by fast recursion.

M7A Part 2 removes the temporary legacy adapter. Production callers now use a
preparation-only context that cannot construct execution graphs, schedules, Stage
plans or storage plans.
"""

from contextlib import contextmanager
from typing import Any, Mapping, Sequence

from .static_sequence_bridge import (
    FrozenCategorySequenceReductionNormalizer,
    FrozenIntegerSequenceReductionNormalizer,
    FrozenSequenceAccessNormalizer,
    FrozenSequenceMembershipNormalizer,
)

from .frontend import (
    CanonicalModel,
    FormulaTranslationContext,
    FrontendError,
    InputRegistry,
    InputSpec,
    ReductionSpec,
)


class FastFormulaTranslationService:
    """Stable graph-free interface consumed by modelx-cython fast recursion.

    All attributes and methods exposed here are translation evidence or prepared
    input concerns; callers receive no graph, schedule, Stage or storage authority.
    """

    _EXPOSED_TRANSLATION_API = frozenset({
        "refs", "space", "space_params", "run_keys", "point_keys",
        "point_frame", "model_point_row",
        "registry", "multiindex_axes", "base_funcs", "vector_elements",
        "static_category_result_domains", "static_category_parameter_domains",
        "static_category_nullable_parameters",
        "static_category_reachable_result_domains", "_normalized_operators",
        "_space_instance", "_discover_vector_helpers",
        "_inline_vector_subscripts", "_normalize_local_mapping_iteration", "_normalize_external_tables",
        "_expr_lowerer", "_normalize_reduction",
        "_normalize_scalar_intrinsics", "_normalize_simple_local_syntax",
        "_normalize_static_sequence_access", "_normalize_static_sequence_membership",
        "_normalize_static_integer_sequence_reductions",
        "_normalize_static_category_sequence_reductions",
        "_specialize_static_facts_and_domains", "_materialize_static_value_operators",
        "_reference_metadata_scalar_ok",
        # Context/static-value services are consumed by mature expression/table
        # lowerers.  They are translation evidence, not execution scheduling.
        "_context_cell_for_key", "_context_itemspace_cell", "_context_primitive", "_context_root_for_key",
        "_context_selected_itemspace", "_context_static_argument_is_fixed",
        "_context_static_value", "_hierarchical_itemspace_parameter_binding",
        "_validate_hierarchical_itemspace_binding",
        "_proven_static_reference_value",
        "proof_diagnostics",
        "coordinate_domains", "specialized_coordinate_domains", "variant_trace_by_uid",
        "modelx_cython_parameter_domains",
        "finite_symbolic_parameter_domains", "finite_symbolic_parameter_domain_sources",
        "source_guarded_parameter_lower_bounds",
        "source_guarded_parameter_lower_bound_sources",
    })

    def __init__(
        self,
        model: Any,
        *,
        space: Any,
        output: str,
        sample_keys: Sequence[Any],
        run_keys: Sequence[Any],
        point_keys: Sequence[Any] | None = None,
        point_frame: Any | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "_impl",
            FormulaTranslationContext(
                model,
                space=space,
                output=output,
                sample_keys=tuple(sample_keys),
                run_keys=tuple(run_keys),
                point_keys=None if point_keys is None else tuple(point_keys),
                point_frame=point_frame,
            ),
        )
        object.__setattr__(self, "_uid_to_source_name", {})

    def configure_formula_catalog(self, uid_to_source_name: Mapping[str, str]) -> None:
        mapping = dict(uid_to_source_name)
        object.__setattr__(self, "_uid_to_source_name", mapping)
        # Provider/table normalization is intentionally reused from the mature
        # frontend.  Give it the same canonical-UID bridge used by the fast
        # catalog so static source Cell proofs (for example a full-RunDomain fixed
        # numeric table prefix) survive modelx-cython call renaming.
        impl = object.__getattribute__(self, "_impl")
        impl.formula_uid_to_source_name = mapping

    @contextmanager
    def translation_refs(self, refs: Mapping[str, Any]):
        """Temporarily select provider refs for a metadata-resolved owning Space.

        Run-domain geometry and point coordinates intentionally remain rooted in
        the selected output Space.  M7C admits only target Spaces whose sampled
        parameters are a subset of those root coordinates.
        """
        impl = object.__getattribute__(self, "_impl")
        previous = impl.refs
        impl.refs = dict(refs)
        try:
            yield self
        finally:
            impl.refs = previous

    def _normalize_static_sequence_access(self, fn):
        normalizer = FrozenSequenceAccessNormalizer(
            uid_to_source_name=object.__getattribute__(self, "_uid_to_source_name"),
            run_keys=self.run_keys,
            space_instance=self._space_instance,
            register_point_input=self.registry.frozen_point_scalar,
        )
        return normalizer.visit(fn)

    def _normalize_static_sequence_membership(self, fn):
        normalizer = FrozenSequenceMembershipNormalizer(
            uid_to_source_name=object.__getattribute__(self, "_uid_to_source_name"),
            run_keys=self.run_keys,
            space_instance=self._space_instance,
            register_point_input=self.registry.frozen_point_scalar,
        )
        return normalizer.visit(fn)

    def _normalize_static_integer_sequence_reductions(self, fn):
        normalizer = FrozenIntegerSequenceReductionNormalizer(
            uid_to_source_name=object.__getattribute__(self, "_uid_to_source_name"),
            run_keys=self.run_keys,
            space_instance=self._space_instance,
        )
        return normalizer.visit(fn)

    def _normalize_static_category_sequence_reductions(self, fn, category_encoding):
        normalizer = FrozenCategorySequenceReductionNormalizer(
            uid_to_source_name=object.__getattribute__(self, "_uid_to_source_name"),
            run_keys=self.run_keys,
            space_instance=self._space_instance,
            register_point_input=self.registry.frozen_point_scalar,
            category_encoding=category_encoding,
        )
        return normalizer.visit(fn)

    @property
    def legacy_adapter_active(self) -> bool:
        return False

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "modelx_graph.fast_formula_translation_service.v1",
            "upstream_method_reuse_policy": "reuse_modelx_cython_unless_documented_reason",
            "graph_authority": False,
            "stage_authority": False,
            "schedule_authority": False,
            "legacy_adapter_active": False,
            "migration_state": "m7a_part2_graph_free_translation_context",
            "semantic_ir_schema": "modelx_graph.semantic_formula_ir.v1",
            "semantic_pass_boundary": "after_source_semantics_before_physical_expression_lowering",
            "exposed_translation_api": sorted(self._EXPOSED_TRANSLATION_API),
            "part2_exit_gate": "complete",
        }

    def __getattr__(self, name: str) -> Any:
        if name not in self._EXPOSED_TRANSLATION_API:
            raise AttributeError(
                f"{type(self).__name__} exposes translation evidence only; {name!r} is not part of the fast API"
            )
        return getattr(object.__getattribute__(self, "_impl"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_impl":
            object.__setattr__(self, name, value)
            return
        if name not in self._EXPOSED_TRANSLATION_API:
            raise AttributeError(
                f"{type(self).__name__} exposes translation evidence only; {name!r} is not part of the fast API"
            )
        setattr(object.__getattribute__(self, "_impl"), name, value)


__all__ = [
    "CanonicalModel",
    "FastFormulaTranslationService",
    "FrontendError",
    "InputRegistry",
    "InputSpec",
    "ReductionSpec",
]
