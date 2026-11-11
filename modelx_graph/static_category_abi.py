from __future__ import annotations

"""Static-proof adapter for the generic finite-category ABI.

The category ABI itself lives in :mod:`modelx_graph.category_abi`.  This module keeps
the dev80 static frontend API as a backwards-compatible diagnostic wrapper.
"""

from typing import Mapping

from .category_abi import (
    CategoryABIError as StaticCategoryABIError,
    CategoryDomain as StaticCategoryDomain,
    CategoryEncodingPlan as StaticCategoryEncodingPlan,
    attach_parameter_positions,
)
from .static_recursive_frontend import StaticRecursiveFormulaSeed
from .static_type_domain import StaticTypeDomainProof, StaticValueDomain


def _domain_from_static(domain: StaticValueDomain, provenance: str) -> StaticCategoryDomain | None:
    if not (domain.proven and domain.finite_literal):
        return None
    return StaticCategoryDomain.from_labels(domain.exact_values, provenance, *domain.provenance)


def _build_static_category_encoding_plan_impl(
    seed: StaticRecursiveFormulaSeed,
    type_proof: StaticTypeDomainProof,
    *,
    provider_result_domains: Mapping[str, StaticCategoryDomain] | None = None,
) -> StaticCategoryEncodingPlan:
    provider_result_domains = provider_result_domains or {}
    domains: dict[str, StaticCategoryDomain] = {}
    result_domains: dict[str, str] = {}
    parameter_domains: dict[tuple[str, str], str] = {}
    blockers: list[str] = []

    def register(domain: StaticCategoryDomain) -> str:
        prior = domains.get(domain.domain_id)
        if prior is not None and prior.labels != domain.labels:
            blockers.append(f"domain_hash_collision:{domain.domain_id}")
        elif prior is None:
            domains[domain.domain_id] = domain
        return domain.domain_id

    for uid, cell in seed.cells.items():
        proof = type_proof.cells[uid]
        static_result = _domain_from_static(proof.result_domain, f"static_result:{uid}")
        provider_result = provider_result_domains.get(uid)
        if static_result is not None and provider_result is not None:
            if set(static_result.labels) != set(provider_result.labels):
                blockers.append(
                    f"result_domain_contradiction:{uid}:static={static_result.labels!r}:provider={provider_result.labels!r}"
                )
                continue
            result_domains[uid] = register(static_result)
        elif static_result is not None:
            result_domains[uid] = register(static_result)
        elif provider_result is not None:
            result_domains[uid] = register(provider_result)

        for name, value_domain in proof.parameter_domains:
            category = _domain_from_static(value_domain, f"static_parameter:{uid}:{name}")
            if category is not None:
                parameter_domains[(uid, name)] = register(category)

    return StaticCategoryEncodingPlan(
        schema="modelx_graph.static_category_encoding.v1",
        domains=domains,
        result_domains=result_domains,
        parameter_domains=parameter_domains,
        blockers=tuple(sorted(set(blockers))),
    )


def build_static_category_encoding_plan(
    seed: StaticRecursiveFormulaSeed,
    type_proof: StaticTypeDomainProof,
    *,
    provider_result_domains: Mapping[str, StaticCategoryDomain] | None = None,
) -> StaticCategoryEncodingPlan:
    plan = _build_static_category_encoding_plan_impl(
        seed, type_proof, provider_result_domains=provider_result_domains
    )
    return attach_parameter_positions(
        plan, {uid: cell.parameters for uid, cell in seed.cells.items()}
    )


__all__ = [
    "StaticCategoryABIError", "StaticCategoryDomain", "StaticCategoryEncodingPlan",
    "build_static_category_encoding_plan",
]
