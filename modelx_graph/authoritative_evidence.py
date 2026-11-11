from __future__ import annotations

"""Demand-driven authoritative type/domain evidence for fast recursion.

The owner here is deliberately small. It reuses the existing local expression,
local-variable, join and result-domain semantics from ``static_type_domain`` but
runs them only on the semantic closure demanded by the selected output. Concrete
modelx-cython/TransSpec facts are anchors; only missing facts are inferred.
"""

import ast
from dataclasses import dataclass
from typing import Any, Mapping

from .static_type_domain import (
    StaticValueDomain,
    _infer_result_domain,
    _local_domains,
    _merge_domains,
    _resolved_cell_domain,
    _strip_ephemeral_blockers,
    infer_expression_domain,
    numeric_domain,
    unknown_domain,
)

_NUMERIC_DTYPES = frozenset({"bool", "int64", "float64"})


@dataclass(frozen=True)
class DemandEvidenceRequest:
    uid: str
    source_name: str
    parameters: tuple[str, ...]
    function: ast.FunctionDef


@dataclass(frozen=True)
class DemandEvidenceResolution:
    output_uid: str
    selected_uids: tuple[str, ...]
    result_domains: Mapping[str, StaticValueDomain]
    parameter_domains: Mapping[str, Mapping[str, StaticValueDomain]]
    iterations: int
    converged: bool

    def result_fallback(self, uid: str) -> tuple[str, str] | None:
        domain = self.result_domains[uid]
        if domain.proven and domain.numeric_dtype in _NUMERIC_DTYPES:
            return domain.numeric_dtype, "authoritative_selected_local_result"
        return None

    def parameter_fallback(self, uid: str, name: str) -> tuple[str, str] | None:
        domain = self.parameter_domains[uid][name]
        if domain.proven and domain.numeric_dtype in _NUMERIC_DTYPES:
            return domain.numeric_dtype, "authoritative_selected_callsite_parameter"
        return None

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "modelx_graph.demand_evidence_resolution.v2",
            "output_uid": self.output_uid,
            "selected_uids": list(self.selected_uids),
            "selected_cell_count": len(self.selected_uids),
            "iterations": self.iterations,
            "converged": self.converged,
            "global_fixed_point": False,
            "execution_graph_authority": False,
            "result_fallbacks": {
                uid: self.result_domains[uid].manifest()
                for uid in self.selected_uids
                if self.result_fallback(uid) is not None
            },
            "parameter_fallbacks": {
                uid: {
                    name: domain.manifest()
                    for name, domain in self.parameter_domains[uid].items()
                    if self.parameter_fallback(uid, name) is not None
                }
                for uid in self.selected_uids
                if any(
                    self.parameter_fallback(uid, name) is not None
                    for name in self.parameter_domains[uid]
                )
            },
        }


def _direct_dependencies(fn: ast.FunctionDef, all_uids: set[str]) -> tuple[str, ...]:
    return tuple(sorted({
        node.func.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in all_uids
    }))


def selected_semantic_closure(
    requests: Mapping[str, DemandEvidenceRequest], output_uid: str
) -> tuple[str, ...]:
    if output_uid not in requests:
        return ()
    all_uids = set(requests)
    selected: set[str] = set()
    queue = [output_uid]
    while queue:
        uid = queue.pop()
        if uid in selected or uid not in requests:
            continue
        selected.add(uid)
        queue.extend(
            dep for dep in _direct_dependencies(requests[uid].function, all_uids)
            if dep not in selected
        )
    return tuple(sorted(selected))


def _semantic_signature(domain: StaticValueDomain) -> tuple[Any, ...]:
    return (
        domain.numeric_dtype,
        domain.exact_values,
        domain.affine_integer_evidence,
        domain.blockers,
    )


def _enrich_anchored_result_domain(
    anchor: StaticValueDomain,
    inferred: StaticValueDomain,
) -> StaticValueDomain:
    """Keep modelx-cython's dtype authority while retaining compatible exact values.

    A result type anchor is stronger than static inference for dtype selection,
    but it need not erase a separate fail-closed finite-value proof.  Only a
    fully proven static domain with the same numeric dtype may enrich the
    anchored result; conflicts and partial proofs are ignored rather than
    weakening the anchor.
    """

    if not (
        inferred.proven
        and inferred.numeric_dtype == anchor.numeric_dtype
        and inferred.exact_values
    ):
        return anchor
    return StaticValueDomain(
        numeric_dtype=anchor.numeric_dtype,
        exact_values=inferred.exact_values,
        affine_integer_evidence=tuple(sorted(set(
            anchor.affine_integer_evidence
        ).union(inferred.affine_integer_evidence))),
        provenance=tuple(sorted(set(anchor.provenance).union(
            inferred.provenance,
            {"anchored_result_exact_domain_enrichment"},
        ))),
        blockers=(),
    )


def resolve_demand_driven_evidence(
    requests: Mapping[str, DemandEvidenceRequest],
    *,
    output_uid: str,
    formula_overrides: Mapping[str, ast.FunctionDef] | None = None,
    input_specs: Mapping[str, Any] | None = None,
    module_refs: Mapping[str, str] | None = None,
    result_anchors: Mapping[str, tuple[str, str]] | None = None,
    parameter_anchors: Mapping[str, Mapping[str, tuple[str, str]]] | None = None,
    reference_anchors: Mapping[str, Mapping[str, tuple[str, str]]] | None = None,
) -> DemandEvidenceResolution:
    """Infer only missing facts inside the selected semantic closure."""

    selected = selected_semantic_closure(requests, output_uid)
    selected_set = set(selected)
    overrides = {
        uid: fn for uid, fn in dict(formula_overrides or {}).items()
        if uid in selected_set
    }
    input_specs = dict(input_specs or {})
    module_refs = dict(module_refs or {})
    result_anchors = dict(result_anchors or {})
    parameter_anchors = {
        uid: dict(values) for uid, values in dict(parameter_anchors or {}).items()
    }
    reference_anchors = {
        uid: dict(values) for uid, values in dict(reference_anchors or {}).items()
        if uid in selected_set
    }
    reference_domains: dict[str, dict[str, StaticValueDomain]] = {
        uid: {
            name: numeric_domain(dtype, source)
            for name, (dtype, source) in reference_anchors.get(uid, {}).items()
            if dtype in _NUMERIC_DTYPES
        }
        for uid in selected
    }

    result_domains: dict[str, StaticValueDomain] = {}
    parameter_domains: dict[str, dict[str, StaticValueDomain]] = {}
    anchored_results: set[str] = set()
    anchored_parameters: set[tuple[str, str]] = set()
    for uid in selected:
        if uid in result_anchors and result_anchors[uid][0] in _NUMERIC_DTYPES:
            dtype, source = result_anchors[uid]
            result_domains[uid] = numeric_domain(dtype, source)
            anchored_results.add(uid)
        else:
            result_domains[uid] = unknown_domain(
                f"demand_result:{uid}", blocker=f"unproven_result:{uid}"
            )
        parameter_domains[uid] = {}
        for name in requests[uid].parameters:
            anchor = parameter_anchors.get(uid, {}).get(name)
            if anchor is not None and anchor[0] in _NUMERIC_DTYPES:
                parameter_domains[uid][name] = numeric_domain(anchor[0], anchor[1])
                anchored_parameters.add((uid, name))
            else:
                parameter_domains[uid][name] = unknown_domain(
                    f"demand_parameter:{uid}:{name}",
                    blocker=f"unproven_parameter:{uid}:{name}",
                )

    max_iterations = max(16, min(128, len(selected) * 2 + 8))
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        call_domains: dict[tuple[str, str], list[StaticValueDomain]] = {
            (uid, name): []
            for uid in selected
            for name in requests[uid].parameters
        }
        for caller_uid in selected:
            row = requests[caller_uid]
            fn = overrides.get(caller_uid, row.function)
            env = _local_domains(
                fn,
                parameter_domains=parameter_domains[caller_uid],
                result_domains=result_domains,
                input_specs=input_specs,
                all_uids=selected_set,
                module_refs=module_refs,
                reference_domains=reference_domains.get(caller_uid),
            )
            for node in ast.walk(fn):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in selected_set
                ):
                    continue
                callee_uid = node.func.id
                callee = requests[callee_uid]
                if len(node.args) != len(callee.parameters):
                    continue
                for name, arg in zip(callee.parameters, node.args):
                    call_domains[(callee_uid, name)].append(
                        infer_expression_domain(
                            arg,
                            env=env,
                            result_domains=result_domains,
                            input_specs=input_specs,
                            all_uids=selected_set,
                            module_refs=module_refs,
                        ).with_provenance(
                            f"authoritative_call:{caller_uid}->{callee_uid}:{name}"
                        )
                    )

        next_parameters = {uid: dict(values) for uid, values in parameter_domains.items()}
        for uid in selected:
            for name in requests[uid].parameters:
                if (uid, name) in anchored_parameters:
                    continue
                domains = call_domains[(uid, name)]
                if domains:
                    merged = _merge_domains(
                        domains, context=f"demand_parameter:{uid}:{name}"
                    )
                    # A recursive fixed point can retain an
                    # ``unresolved_cell_result`` placeholder from an earlier
                    # iteration even after the referenced Cell has become
                    # numeric.  Re-resolve those placeholders against the
                    # current result lattice before deciding the parameter is
                    # still blocked.
                    merged = _resolved_cell_domain(merged, result_domains)
                    next_parameters[uid][name] = _strip_ephemeral_blockers(merged)

        next_results = dict(result_domains)
        for uid in selected:
            row = requests[uid]
            fn = overrides.get(uid, row.function)
            inferred = _infer_result_domain(
                uid,
                fn,
                parameter_domains=next_parameters[uid],
                result_domains=result_domains,
                input_specs=input_specs,
                all_uids=selected_set,
                module_refs=module_refs,
                reference_domains=reference_domains.get(uid),
            )
            if uid in anchored_results:
                dtype, source = result_anchors[uid]
                next_results[uid] = _enrich_anchored_result_domain(
                    numeric_domain(dtype, source), inferred
                )
            else:
                next_results[uid] = inferred

        old_sig = (
            tuple((uid, _semantic_signature(result_domains[uid])) for uid in selected),
            tuple(
                (uid, name, _semantic_signature(parameter_domains[uid][name]))
                for uid in selected for name in requests[uid].parameters
            ),
        )
        new_sig = (
            tuple((uid, _semantic_signature(next_results[uid])) for uid in selected),
            tuple(
                (uid, name, _semantic_signature(next_parameters[uid][name]))
                for uid in selected for name in requests[uid].parameters
            ),
        )
        result_domains = next_results
        parameter_domains = next_parameters
        if new_sig == old_sig:
            converged = True
            break

    return DemandEvidenceResolution(
        output_uid=output_uid,
        selected_uids=selected,
        result_domains=result_domains,
        parameter_domains=parameter_domains,
        iterations=iterations,
        converged=converged,
    )


__all__ = [
    "DemandEvidenceRequest",
    "DemandEvidenceResolution",
    "selected_semantic_closure",
    "resolve_demand_driven_evidence",
]
