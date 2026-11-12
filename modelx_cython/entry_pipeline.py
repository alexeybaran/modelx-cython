"""Lower private Cell entries over shared syntax/facts, without disk rewrites."""
from collections import Counter
from dataclasses import replace

from .dense_proofs import _attribute_counts, prove_entries
from .dense_formulas import bind_entries
from .dense_one_axis import specialize_entries
from .cache_hits import expose_entries


def lower_entries(modules, dense, formulas, policy):
    uses = Counter()
    if policy.dense_bounds_proofs and dense:
        for state in modules.values():
            uses.update(_attribute_counts(state.syntax))
    result = {}
    for name, state in modules.items():
        layouts, entries = dense.get(name, {}), formulas.get(name, {})
        state = replace(state, cache_hits={key: entry.hit for key, entry in entries.items() if entry.hit})
        domains = {}
        if layouts and policy.dense_bounds_proofs:
            state = prove_entries(state, layouts, uses, domains)
        if entries and policy.dense_miss_binding:
            state = bind_entries(state, entries, policy.propagate_exact_receiver, domains)
        if layouts and policy.dense_one_axis:
            state = specialize_entries(state, layouts, policy.propagate_exact_receiver)
        if entries and policy.native_cache_hits != 'none':
            state = expose_entries(state, policy.native_cache_hits)
        result[name] = state
    return result
