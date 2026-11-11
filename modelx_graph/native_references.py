from __future__ import annotations

import ast
import inspect
import numbers
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .native_plan import NativeFamilyBackendPlan, expand_native_family_instance


class NativeReferenceError(RuntimeError):
    """Fail-closed error while lowering concrete Cell references to runtime data."""


@dataclass(frozen=True)
class ReferenceSitePlan:
    """Formula-code identity for one direct scheduled Cell call name.

    The site contains no concrete Cells object and no physical address. Those are
    per-instance data in :class:`ReferenceInstanceBinding`.
    """

    site_id: int
    formula_op_id: int
    name: str
    pool: str
    arg_count: int
    occurrence_addressed: bool = False
    call_lineno: int = -1
    call_col_offset: int = -1


@dataclass(frozen=True)
class ReferenceAddressBinding:
    site_id: int
    concrete_impl_token: int | None
    kind: int
    a: int = 0
    b: int = 0
    mod: int = 0
    table_lo: int = 0
    table_values: tuple[int, ...] = ()


@dataclass(frozen=True)
class ReferenceInstanceBinding:
    instance_ordinal: int
    sites: tuple[ReferenceAddressBinding, ...]


@dataclass(frozen=True)
class ReferenceBindingPlan:
    sites: tuple[ReferenceSitePlan, ...]
    instances: tuple[ReferenceInstanceBinding, ...]
    unsupported_formula_ops: tuple[int, ...]
    fallback_reasons: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class EncodedReferenceBindings:
    kind: np.ndarray
    a: np.ndarray
    b: np.ndarray
    mod: np.ndarray
    lo: np.ndarray
    table_offset: np.ndarray
    table: np.ndarray

    def as_call_args(self) -> tuple[np.ndarray, ...]:
        return (self.kind, self.a, self.b, self.mod, self.lo, self.table_offset, self.table)


def reference_structural_kinds(plan: ReferenceBindingPlan) -> tuple[int, ...]:
    """Return the per-site descriptor kind proven across realized instances.

    ``-2`` means compatible instances use different descriptor representations;
    ``-1`` means the site has no realized binding.  A non-negative kind is safe to
    specialize in generated source because only the *descriptor shape* is frozen:
    per-instance addresses, slopes, modulo parameters and tables remain runtime
    payload.  Instances in which a role is absent may carry ``kind=-1`` for its
    sites; those rows are ignored because that formula role cannot execute there.
    """
    observed: list[set[int]] = [set() for _ in plan.sites]
    for instance in plan.instances:
        for row in instance.sites:
            if row.site_id < 0 or row.site_id >= len(observed):
                raise NativeReferenceError("reference site id outside structural plan")
            if row.kind >= 0:
                observed[row.site_id].add(int(row.kind))
    result: list[int] = []
    for kinds in observed:
        if not kinds:
            result.append(-1)
        elif len(kinds) == 1:
            result.append(next(iter(kinds)))
        else:
            result.append(-2)
    return tuple(result)


def _parse_func(source: str) -> ast.FunctionDef:
    mod = ast.parse(textwrap.dedent(source))
    funcs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        raise NativeReferenceError("formula source has no function definition")
    return funcs[0]


def _impl_from_global(value: Any) -> Any | None:
    if inspect.ismethod(value):
        owner = getattr(value, "__self__", None)
        func = getattr(value, "__func__", None)
        if owner is not None and getattr(func, "__name__", None) == "call" and hasattr(owner, "data"):
            return owner
    if hasattr(value, "_impl") and value.__class__.__name__ == "Cells":
        return value._impl
    return None


def _cell_call_name(node: ast.Call) -> str | None:
    """Return a stable source identity for direct or Space-qualified calls."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return f"{node.func.value.id}.{node.func.attr}"
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Subscript)
        and isinstance(node.func.value.value, ast.Name)
    ):
        selector = ast.unparse(node.func.value.slice)
        return f"{node.func.value.value.id}[{selector}].{node.func.attr}"
    return None


def _static_selector(node: ast.AST, globals_: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in globals_:
        value = globals_[node.id]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
    if isinstance(node, ast.Tuple):
        return tuple(_static_selector(item, globals_) for item in node.elts)
    if isinstance(node, ast.UnaryOp):
        value = _static_selector(node.operand, globals_)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
    raise NativeReferenceError("qualified Space selector is not an immutable global")


def _impl_from_formula_global(
    globals_: dict[str, Any], name: str, call: ast.Call | None = None
) -> Any | None:
    """Resolve ``cell(...)`` or ``space.cell(...)`` without executing a formula."""
    if "." not in name:
        return _impl_from_global(globals_.get(name)) if name in globals_ else None
    if (
        call is not None
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Subscript)
        and isinstance(call.func.value.value, ast.Name)
    ):
        base_name = call.func.value.value.id
        if base_name not in globals_:
            return None
        try:
            selector = _static_selector(call.func.value.slice, globals_)
            value = getattr(globals_[base_name][selector], call.func.attr)
        except (AttributeError, KeyError, TypeError, IndexError, NativeReferenceError):
            return None
        return _impl_from_global(value)
    base_name, attr_name = name.split(".", 1)
    if base_name not in globals_:
        return None
    try:
        value = getattr(globals_[base_name], attr_name)
    except (AttributeError, KeyError, TypeError):
        return None
    return _impl_from_global(value)


def _fit_reference_address(mapping: dict[int, int]) -> tuple[int, int, int, int, int, tuple[int, ...]]:
    """Return kind/a/b/mod/lo/table for a one-integer-argument mapping."""
    keys = sorted(mapping)
    offs = [mapping[k] for k in keys]
    if not keys:
        raise NativeReferenceError("reference mapping is empty")
    if len(set(offs)) == 1:
        return 0, offs[0], 0, 0, 0, ()
    if len(keys) >= 2:
        dk = keys[1] - keys[0]
        do = offs[1] - offs[0]
        if dk != 0 and do % dk == 0:
            slope = do // dk
            intercept = offs[0] - slope * keys[0]
            if all(intercept + slope * k == o for k, o in zip(keys, offs)):
                return 1, intercept, slope, 0, 0, ()
    lo, hi = min(keys), max(keys)
    if keys == list(range(lo, hi + 1)):
        cap = len(set(offs)); base = min(offs)
        if cap > 0 and all(o == base + ((k - lo) % cap) for k, o in zip(keys, offs)):
            return 2, base, lo, cap, 0, ()
    if hi - lo > 100_000:
        raise NativeReferenceError("runtime reference table domain is too sparse/large")
    table = tuple(mapping.get(k, -1) for k in range(lo, hi + 1))
    return 3, 0, 0, 0, lo, table


def _call_shape(fn: ast.FunctionDef, name: str) -> tuple[int, bool] | None:
    calls = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call) and _cell_call_name(node) == name
    ]
    if not calls:
        return None
    shapes = {(len(n.args), bool(n.keywords)) for n in calls}
    if len(shapes) != 1:
        return None
    argc, has_keywords = next(iter(shapes))
    return argc, has_keywords


def _call_nodes(fn: ast.FunctionDef, name: str) -> list[ast.Call]:
    return sorted(
        (
            node for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and _cell_call_name(node) == name
        ),
        key=lambda node: (int(getattr(node, "lineno", -1)), int(getattr(node, "col_offset", -1))),
    )


def _safe_call_arg(
    node: ast.AST, env: dict[str, Any], resolve_cell: Any | None = None
) -> Any:
    """Evaluate a call argument from realized formula parameters only.

    This is not formula execution. It is a small structural evaluator used to
    associate each syntactic Cell callsite with the exact dependency edge already
    present in the realized graph. Unsupported syntax fails closed.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise NativeReferenceError(f"callsite argument name {node.id!r} is not a formula parameter")
        return env[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_call_arg(item, env, resolve_cell) for item in node.elts)
    if isinstance(node, ast.List):
        return [_safe_call_arg(item, env, resolve_cell) for item in node.elts]
    if isinstance(node, ast.UnaryOp):
        value = _safe_call_arg(node.operand, env, resolve_cell)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.Not):
            return not value
        raise NativeReferenceError(f"unsupported callsite unary operator {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _safe_call_arg(node.left, env, resolve_cell)
        right = _safe_call_arg(node.right, env, resolve_cell)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise NativeReferenceError(f"unsupported callsite binary operator {type(node.op).__name__}")
    if isinstance(node, ast.Call) and _cell_call_name(node) is not None:
        if node.keywords:
            raise NativeReferenceError("keyword callsite argument helpers are unsupported")
        args = [_safe_call_arg(arg, env, resolve_cell) for arg in node.args]
        name = _cell_call_name(node)
        assert name is not None
        if name == "int" and len(args) == 1:
            return int(args[0])
        if name == "max" and args:
            return max(args)
        if name == "min" and args:
            return min(args)
        if resolve_cell is not None:
            return resolve_cell(name, tuple(args))
    raise NativeReferenceError(
        f"unsupported callsite argument expression {ast.dump(node, include_attributes=False)}"
    )


def _args_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    if len(left) != len(right):
        return False
    try:
        return all(bool(a == b) for a, b in zip(left, right))
    except Exception:
        return False


def build_reference_binding_plan(
    compiler: Any,
    typed: Any,
    backend: NativeFamilyBackendPlan,
    candidate_native: dict[int, bool],
    *,
    formula_functions: dict[int, ast.FunctionDef] | None = None,
    occurrence_formula_ops: set[int] | None = None,
    reference_pool_hints: dict[tuple[int, str], str] | None = None,
) -> ReferenceBindingPlan:
    """Bind direct Cell calls to graph-owned physical slots.

    For formulas reused from modelx-cython, the preferred ABI is occurrence
    addressing: each realized FormulaOp occurrence already has an exact dependency
    edge to the Cell value it consumes.  We encode that address stream as runtime
    data instead of reconstructing modelx's key/cache lookup.  The source remains
    one shared formula body and the payload remains instance-specific.
    """
    plan = compiler.structured.canonical_plan
    if plan is None:
        raise NativeReferenceError("canonical execution plan is required")
    occurrence_formula_ops = set(occurrence_formula_ops or ())
    reference_pool_hints = reference_pool_hints or {}
    byid = compiler.sequential.node_by_id
    deps_by_dst: dict[int, list[int]] = defaultdict(list)
    for src, dst in compiler.trace.dependencies:
        deps_by_dst[dst].append(src)

    impl_pools: dict[int, set[str]] = defaultdict(set)
    for node in compiler.trace.nodes:
        addr = typed.node_by_id.get(node.node_id)
        if addr is not None:
            impl_pools[id(node.obj)].add(addr.pool)

    expansions = [expand_native_family_instance(compiler, backend, i) for i in range(len(backend.instances))]
    nodes_by_instance_fid: list[dict[int, list[int]]] = []
    for exp in expansions:
        rows: dict[int, list[int]] = defaultdict(list)
        for role, nids in enumerate(exp.role_node_ids):
            fid = backend.kernel.formula_op_ids[role]
            rows[fid].extend(nids)
        nodes_by_instance_fid.append(rows)

    sites: list[ReferenceSitePlan] = []
    per_instance: list[list[ReferenceAddressBinding]] = [[] for _ in backend.instances]
    unsupported: set[int] = set()
    reasons: list[tuple[int, str]] = []

    for fid in dict.fromkeys(backend.kernel.formula_op_ids):
        if not candidate_native.get(fid, False):
            continue
        op = plan.formula_ops[fid]
        all_nids = [nid for rows in nodes_by_instance_fid for nid in rows.get(fid, ())]
        if not all_nids:
            unsupported.add(fid); reasons.append((fid, "reference:no_family_nodes")); continue
        rep = byid[all_nids[0]]
        try:
            fn = (formula_functions or {}).get(fid)
            if fn is None:
                fn = _parse_func(rep.schema.source)
        except Exception:
            unsupported.add(fid); reasons.append((fid, "reference:unparseable_source")); continue
        call_names = sorted({
            name for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            for name in [_cell_call_name(node)]
            if name is not None
        })

        fid_sites: list[tuple[ReferenceSitePlan, list[ReferenceAddressBinding]]] = []
        fid_failed = False
        for name in call_names:
            calls = _call_nodes(fn, name)
            representative_call = calls[0] if calls else None
            concrete_impls = []
            some_noncell = False
            for nid in all_nids:
                glb = byid[nid].obj.altfunc.__globals__
                impl = _impl_from_formula_global(glb, name, representative_call)
                if impl is None:
                    some_noncell = True
                else:
                    concrete_impls.append(impl)
            if not concrete_impls:
                continue
            if some_noncell:
                fid_failed = True; reasons.append((fid, f"reference:environment_changes:{name}")); break
            occurrence_addressed = fid in occurrence_formula_ops
            if occurrence_addressed and len(calls) > 1:
                formal_names = [arg.arg for arg in fn.args.args]
                for call in calls:
                    if call.keywords:
                        fid_failed = True
                        reasons.append((fid, f"reference:callsite_keywords:{name}"))
                        break
                    site_id = len(sites) + len(fid_sites)
                    inst_bindings: list[ReferenceAddressBinding] = []
                    observed_pools: set[str] = set()
                    for ordinal, rows in enumerate(nodes_by_instance_fid):
                        dest_ids = rows.get(fid, [])
                        if not dest_ids:
                            inst_bindings.append(ReferenceAddressBinding(site_id, None, -1))
                            continue
                        impls = []
                        for nid in dest_ids:
                            glb = byid[nid].obj.altfunc.__globals__
                            impl = _impl_from_formula_global(glb, name, call)
                            if impl is None:
                                fid_failed = True
                                reasons.append((fid, f"reference:missing_instance_binding:{name}"))
                                break
                            impls.append(impl)
                        if fid_failed:
                            break
                        unique = {id(value): value for value in impls}
                        if len(unique) != 1:
                            fid_failed = True
                            reasons.append((fid, f"reference:multiple_impls_within_instance:{name}"))
                            break
                        impl = next(iter(unique.values()))
                        table: list[int] = []
                        for dst in dest_ids:
                            dest = byid[dst]
                            if len(dest.args) != len(formal_names):
                                fid_failed = True
                                reasons.append((fid, f"reference:formula_argument_arity:{name}"))
                                break
                            env = dict(zip(formal_names, dest.args))
                            # A call argument may combine formula parameters with
                            # an immutable scalar model reference (for example
                            # ``index_level(t - seg_term_mth)``).  Reading the
                            # representative globals here does not execute a
                            # formula; it merely resolves the already-realized
                            # dependency edge.  Non-scalars and Cells remain
                            # excluded and therefore fail closed.
                            for global_name, value in dest.obj.altfunc.__globals__.items():
                                if global_name in env or _impl_from_global(value) is not None:
                                    continue
                                if isinstance(value, np.generic):
                                    value = value.item()
                                if isinstance(value, (bool, int, float, str)) or value is None:
                                    env[global_name] = value
                            direct_dependencies = [byid[src] for src in deps_by_dst.get(dst, ())]

                            def resolve_nested_cell(cell_name: str, cell_args: tuple[Any, ...]) -> Any:
                                glb = dest.obj.altfunc.__globals__
                                nested_impl = _impl_from_formula_global(glb, cell_name)
                                if nested_impl is None:
                                    raise NativeReferenceError(
                                        f"nested call {cell_name!r} is not a Cell in the realized environment"
                                    )
                                nested_matches = [
                                    node for node in direct_dependencies
                                    if node.obj is nested_impl and _args_equal(tuple(node.args), cell_args)
                                ]
                                if not nested_matches:
                                    # A missing direct edge means this nested call belongs
                                    # to a dormant branch for the current occurrence.
                                    raise LookupError(cell_name)
                                if len(nested_matches) != 1:
                                    raise NativeReferenceError(
                                        f"nested call {cell_name!r} has ambiguous dependency edges"
                                    )
                                nested = nested_matches[0]
                                try:
                                    return nested.obj.data[tuple(nested.args)]
                                except Exception as exc:
                                    raise NativeReferenceError(
                                        f"nested call {cell_name!r} has no authoritative traced value"
                                    ) from exc

                            try:
                                expected_args = tuple(
                                    _safe_call_arg(arg, env, resolve_nested_cell) for arg in call.args
                                )
                            except LookupError:
                                table.append(-1)
                                continue
                            except Exception as exc:
                                fid_failed = True
                                reasons.append((fid, f"reference:callsite_argument:{name}:{type(exc).__name__}"))
                                break
                            matches = []
                            for src in deps_by_dst.get(dst, ()):
                                node = byid[src]
                                if (
                                    node.obj is impl
                                    and src in typed.node_by_id
                                    and _args_equal(tuple(node.args), expected_args)
                                ):
                                    matches.append(node)
                            if len(matches) > 1:
                                fid_failed = True
                                reasons.append((fid, f"reference:ambiguous_callsite:{name}"))
                                break
                            if matches:
                                address = typed.node_by_id[matches[0].node_id]
                                observed_pools.add(address.pool)
                                table.append(int(address.offset))
                            else:
                                # The syntactic callsite is dormant for this realized
                                # occurrence. _ref_index will fail if control flow ever
                                # reaches it, preserving the realized-graph contract.
                                table.append(-1)
                        if fid_failed:
                            break
                        inst_bindings.append(ReferenceAddressBinding(
                            site_id, id(impl), 4, 0, len(table), 0, 0, tuple(table)
                        ))
                    if fid_failed:
                        break
                    if not observed_pools:
                        hinted_pool = reference_pool_hints.get((fid, name))
                        if hinted_pool in {"double", "int64", "bool", "object"}:
                            observed_pools.add(hinted_pool)
                        else:
                            observed_pools.add("object")
                    if len(observed_pools) != 1:
                        fid_failed = True
                        reasons.append((fid, f"reference:pool_changes_across_callsites:{name}"))
                        break
                    fid_sites.append((
                        ReferenceSitePlan(
                            site_id, fid, name, next(iter(observed_pools)), len(call.args), True,
                            int(getattr(call, "lineno", -1)), int(getattr(call, "col_offset", -1)),
                        ),
                        inst_bindings,
                    ))
                if fid_failed:
                    break
                continue

            shape = _call_shape(fn, name)
            if shape is None or shape[1]:
                fid_failed = True; reasons.append((fid, f"reference:call_shape:{name}")); break
            arg_count = shape[0]
            if not occurrence_addressed and arg_count not in (0, 1):
                fid_failed = True; reasons.append((fid, f"reference:arity:{arg_count}:{name}")); break

            inst_bindings: list[ReferenceAddressBinding] = []
            observed_pools: set[str] = set()
            for ordinal, rows in enumerate(nodes_by_instance_fid):
                dest_ids = rows.get(fid, [])
                if not dest_ids:
                    inst_bindings.append(ReferenceAddressBinding(len(sites) + len(fid_sites), None, -1))
                    continue
                impls = []
                for nid in dest_ids:
                    glb = byid[nid].obj.altfunc.__globals__
                    impl = _impl_from_formula_global(glb, name, representative_call)
                    if impl is None:
                        fid_failed = True; reasons.append((fid, f"reference:missing_instance_binding:{name}")); break
                    impls.append(impl)
                if fid_failed:
                    break
                unique = {id(x): x for x in impls}
                if len(unique) != 1:
                    fid_failed = True; reasons.append((fid, f"reference:multiple_impls_within_instance:{name}")); break
                impl = next(iter(unique.values()))
                dep_rows: list[list[Any]] = []
                for dst in dest_ids:
                    deps = []
                    for src in deps_by_dst.get(dst, ()):
                        node = byid[src]
                        if node.obj is impl and src in typed.node_by_id:
                            deps.append(node)
                    dep_rows.append(deps)
                dep_nodes = [node for row in dep_rows for node in row]
                if not dep_nodes:
                    pools = impl_pools.get(id(impl), set())
                    if len(pools) == 1:
                        observed_pools.update(pools)
                    inst_bindings.append(ReferenceAddressBinding(
                        len(sites) + len(fid_sites), id(impl), -1
                    ))
                    continue
                addrs = [typed.node_by_id[n.node_id] for n in dep_nodes]
                pools = {a.pool for a in addrs}
                if len(pools) != 1:
                    fid_failed = True; reasons.append((fid, f"reference:pool_changes:{name}")); break
                pool = next(iter(pools)); observed_pools.add(pool)
                if pool == "object" and not occurrence_addressed:
                    fid_failed = True; reasons.append((fid, f"reference:object_pool:{name}")); break

                site_id = len(sites) + len(fid_sites)
                if occurrence_addressed:
                    if any(len(row) > 1 for row in dep_rows):
                        fid_failed = True
                        reasons.append((fid, f"reference:multiple_calls_per_occurrence:{name}"))
                        break
                    table = tuple(
                        -1 if not row else int(typed.node_by_id[row[0].node_id].offset)
                        for row in dep_rows
                    )
                    inst_bindings.append(ReferenceAddressBinding(
                        site_id, id(impl), 4, 0, len(table), 0, 0, table
                    ))
                elif arg_count == 0:
                    offsets = {a.offset for a in addrs}
                    if len(offsets) == 1:
                        inst_bindings.append(ReferenceAddressBinding(
                            site_id, id(impl), 0, a=next(iter(offsets))
                        ))
                    elif all(len(row) <= 1 for row in dep_rows):
                        table = tuple(
                            -1 if not row else int(typed.node_by_id[row[0].node_id].offset)
                            for row in dep_rows
                        )
                        occurrence_addressed = True
                        inst_bindings.append(ReferenceAddressBinding(
                            site_id, id(impl), 4, 0, len(table), 0, 0, table
                        ))
                    else:
                        fid_failed = True; reasons.append((fid, f"reference:zero_arg_multi_slot:{name}")); break
                else:
                    mapping: dict[int, int] = {}
                    integer_keys = True
                    for node, addr in zip(dep_nodes, addrs):
                        if len(node.args) != 1 or not isinstance(node.args[0], (numbers.Integral, np.integer)) or isinstance(node.args[0], (bool, np.bool_)):
                            integer_keys = False; break
                        key = int(node.args[0])
                        prev = mapping.setdefault(key, addr.offset)
                        if prev != addr.offset:
                            fid_failed = True; reasons.append((fid, f"reference:key_address_conflict:{name}")); break
                    if fid_failed:
                        break
                    if integer_keys:
                        try:
                            kind, a, b, mod, lo, table = _fit_reference_address(mapping)
                            inst_bindings.append(ReferenceAddressBinding(
                                site_id, id(impl), kind, a, b, mod, lo, table
                            ))
                        except NativeReferenceError:
                            integer_keys = False
                    if not integer_keys:
                        if any(len(row) > 1 for row in dep_rows):
                            fid_failed = True; reasons.append((fid, f"reference:non_integer_key:{name}")); break
                        occurrence_addressed = True
                        table = tuple(
                            -1 if not row else int(typed.node_by_id[row[0].node_id].offset)
                            for row in dep_rows
                        )
                        inst_bindings.append(ReferenceAddressBinding(
                            site_id, id(impl), 4, 0, len(table), 0, 0, table
                        ))
            if fid_failed:
                break
            if not observed_pools:
                hinted_pool = reference_pool_hints.get((fid, name))
                if hinted_pool in {"double", "int64", "bool", "object"}:
                    observed_pools.add(hinted_pool)
                else:
                    # Every instance descriptor for this site is already kind -1:
                    # no realized occurrence executed the syntactic call.  The
                    # generated helper will raise if a future invocation violates
                    # that realized-graph contract.  ``object`` is the only safe
                    # source-level placeholder because it does not invent a scalar
                    # conversion for the dormant value.
                    observed_pools.add("object")
            if len(observed_pools) != 1:
                fid_failed = True; reasons.append((fid, f"reference:pool_changes_across_instances:{name}")); break
            site = ReferenceSitePlan(
                len(sites) + len(fid_sites), fid, name, next(iter(observed_pools)),
                arg_count, occurrence_addressed,
            )
            # If any instance needed occurrence mode, all instance descriptors must
            # consume the formula occurrence. Re-encode direct rows as occurrence
            # streams only when they are already kind 4; mixed mode is rejected.
            if occurrence_addressed and any(b.kind not in (-1, 4) for b in inst_bindings):
                fid_failed = True; reasons.append((fid, f"reference:mixed_occurrence_mode:{name}")); break
            fid_sites.append((site, inst_bindings))

        if fid_failed:
            unsupported.add(fid)
            continue
        for site, bindings in fid_sites:
            sites.append(site)
            for ordinal, binding in enumerate(bindings):
                per_instance[ordinal].append(binding)

    instances = tuple(
        ReferenceInstanceBinding(i, tuple(rows)) for i, rows in enumerate(per_instance)
    )
    return ReferenceBindingPlan(
        sites=tuple(sites),
        instances=instances,
        unsupported_formula_ops=tuple(sorted(unsupported)),
        fallback_reasons=tuple(reasons),
    )


def encode_reference_instance(plan: ReferenceBindingPlan, instance_ordinal: int) -> EncodedReferenceBindings:
    if instance_ordinal < 0 or instance_ordinal >= len(plan.instances):
        raise NativeReferenceError(f"unknown reference instance {instance_ordinal}")
    n = len(plan.sites)
    kind = np.full(n, -1, dtype=np.int64)
    a = np.zeros(n, dtype=np.int64)
    b = np.zeros(n, dtype=np.int64)
    mod = np.zeros(n, dtype=np.int64)
    lo = np.zeros(n, dtype=np.int64)
    table_offset = np.zeros(n, dtype=np.int64)
    table: list[int] = []
    rows = {x.site_id: x for x in plan.instances[instance_ordinal].sites}
    for site in plan.sites:
        row = rows.get(site.site_id)
        if row is None:
            continue
        kind[site.site_id] = row.kind
        a[site.site_id] = row.a
        b[site.site_id] = row.b
        mod[site.site_id] = row.mod
        lo[site.site_id] = row.table_lo
        if row.kind in (3, 4):
            table_offset[site.site_id] = len(table)
            table.extend(row.table_values)
    return EncodedReferenceBindings(kind, a, b, mod, lo, table_offset, np.asarray(table, dtype=np.int64))
