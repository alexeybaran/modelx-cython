from __future__ import annotations

"""Backend-neutral static value-domain proof for qualified recursive Cells.

The type/domain layer intentionally sits between static graph/provider normalization
and physical execution.  It does not know products, Stage geometry, memo layouts or
trace variant identities.  Its authorities are:

* qualified static formulas and their complete static call graph;
* exact literals/defaults already materialized into those formulas;
* normalized runtime input specifications when available;
* conservative arithmetic/control typing rules shared with the native scalar ABI.

Representative trace evidence is *not* consumed here.  The provider-binding layer may
use trace evidence later as corroboration/fallback and must reject contradictions.
"""

import ast
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .static_recursive_frontend import StaticRecursiveFormulaSeed


_NUMERIC_DTYPES = {"bool", "int64", "float64"}
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_RUNTIME_INPUT_CALLS = {"point_input", "global_input", "array_input", "table_input"}
_FLOAT_INTRINSICS = {"__erf__"}
_BOOL_INTRINSICS = {"__is_missing__", "__is_inf__"}
_MATH_INTRINSICS = {"exp", "log", "sqrt", "sin", "cos", "floor", "ceil"}
_EXACT_NUMERIC_DOMAIN_LIMIT = 256


class StaticTypeDomainError(RuntimeError):
    pass


def _value_key(value: Any) -> tuple[str, str]:
    if value is None:
        return ("NoneType", "")
    return (type(value).__name__, repr(value))


def _ordered_values(values: Sequence[Any]) -> tuple[Any, ...]:
    unique: dict[tuple[str, str], Any] = {}
    for value in values:
        if isinstance(value, (bool, int, float, str)) or value is None:
            unique[_value_key(value)] = value
    return tuple(unique[key] for key in sorted(unique))


def _literal_dtype(value: Any) -> str | None:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "float64"
    return None


def _promote_numeric(kinds: Sequence[str]) -> str:
    values = set(kinds)
    if not values or not values <= _NUMERIC_DTYPES:
        raise StaticTypeDomainError(f"cannot promote non-numeric kinds: {sorted(values)}")
    if "float64" in values:
        return "float64"
    if "int64" in values:
        return "int64"
    return "bool"


def _compose_exact_binary_values(
    left: "StaticValueDomain",
    right: "StaticValueDomain",
    op: ast.operator,
) -> tuple[Any, ...]:
    """Compose a small, fully known numeric Cartesian product exactly.

    Static type/domain inference previously preserved finite guard domains but
    deliberately discarded them as soon as arithmetic was applied.  Local
    control progress proofs need the stronger value-domain fact.  Keep the
    operation set narrow, cap the Cartesian product, and fail closed on any
    exceptional/out-of-int64 result rather than approximating a range.
    """

    if not left.exact_values or not right.exact_values:
        return ()
    if len(left.exact_values) * len(right.exact_values) > _EXACT_NUMERIC_DOMAIN_LIMIT:
        return ()
    if not isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
        return ()
    values: list[Any] = []
    for a in left.exact_values:
        for b in right.exact_values:
            if not isinstance(a, (bool, int, float)) or not isinstance(b, (bool, int, float)):
                return ()
            try:
                if isinstance(op, ast.Add):
                    value = a + b
                elif isinstance(op, ast.Sub):
                    value = a - b
                elif isinstance(op, ast.Mult):
                    value = a * b
                elif isinstance(op, ast.FloorDiv):
                    value = a // b
                else:
                    value = a % b
            except (ArithmeticError, ValueError, TypeError):
                return ()
            if isinstance(value, int) and not (_INT64_MIN <= value <= _INT64_MAX):
                return ()
            values.append(value)
    return _ordered_values(values)


@dataclass(frozen=True)
class StaticValueDomain:
    """A conservative static value-domain fact.

    ``numeric_dtype`` is the directly native scalar type when proven.  A finite
    literal domain may instead contain strings/None and deliberately has no native
    dtype yet.  ``blockers`` means the proof is incomplete or contradictory.
    Numeric/exact facts may still be carried alongside blockers to support fixed-point
    convergence; admission requires ``proven``.
    """

    numeric_dtype: str | None = None
    exact_values: tuple[Any, ...] = ()
    affine_integer_evidence: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    @property
    def has_nonnumeric_literals(self) -> bool:
        return any(_literal_dtype(value) is None for value in self.exact_values)

    @property
    def finite_literal(self) -> bool:
        return bool(self.exact_values) and self.numeric_dtype is None and self.has_nonnumeric_literals

    @property
    def unknown(self) -> bool:
        return self.numeric_dtype is None and not self.exact_values and not self.blockers

    @property
    def conflict(self) -> bool:
        return any(item.startswith("conflict:") for item in self.blockers)

    @property
    def proven(self) -> bool:
        return not self.blockers and (self.numeric_dtype in _NUMERIC_DTYPES or self.finite_literal)

    @property
    def kind(self) -> str:
        if self.conflict:
            return "conflict"
        if self.numeric_dtype in _NUMERIC_DTYPES:
            return "numeric" if not self.blockers else "partial_numeric"
        if self.finite_literal:
            return "finite_literal" if not self.blockers else "partial_finite_literal"
        return "unknown" if not self.blockers else "blocked"

    def with_provenance(self, *items: str) -> "StaticValueDomain":
        return StaticValueDomain(
            numeric_dtype=self.numeric_dtype,
            exact_values=self.exact_values,
            affine_integer_evidence=self.affine_integer_evidence,
            provenance=tuple(sorted(set(self.provenance).union(x for x in items if x))),
            blockers=self.blockers,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "numeric_dtype": self.numeric_dtype,
            "exact_values": list(self.exact_values),
            "affine_integer_evidence": list(self.affine_integer_evidence),
            "provenance": list(self.provenance),
            "blockers": list(self.blockers),
            "proven": self.proven,
            "categorical_abi_required": self.finite_literal and not self.blockers,
        }


def unknown_domain(*provenance: str, blocker: str | None = None) -> StaticValueDomain:
    return StaticValueDomain(
        provenance=tuple(sorted(set(x for x in provenance if x))),
        blockers=() if blocker is None else (blocker,),
    )


def literal_domain(value: Any, provenance: str) -> StaticValueDomain:
    if not (isinstance(value, (bool, int, float, str)) or value is None):
        return unknown_domain(provenance, blocker=f"unsupported_literal:{type(value).__name__}")
    return StaticValueDomain(
        numeric_dtype=_literal_dtype(value),
        exact_values=(value,),
        provenance=(provenance,),
    )


def numeric_domain(dtype: str, provenance: str, *, affine: str | None = None) -> StaticValueDomain:
    if dtype not in _NUMERIC_DTYPES:
        raise StaticTypeDomainError(f"unsupported numeric domain dtype {dtype!r}")
    return StaticValueDomain(
        numeric_dtype=dtype,
        affine_integer_evidence=() if affine is None else (affine,),
        provenance=(provenance,),
    )


def _merge_domains(domains: Sequence[StaticValueDomain], *, context: str) -> StaticValueDomain:
    """Join domains conservatively.

    Unknown/blocked inputs keep a blocker so a currently unresolved fixed-point edge
    cannot accidentally be admitted.  Known type facts are retained, which allows a
    later iteration to resolve recursive cycles once the referenced Cell becomes known.
    """

    if not domains:
        return unknown_domain(context, blocker=f"unproven:{context}")

    numeric = [d.numeric_dtype for d in domains if d.numeric_dtype in _NUMERIC_DTYPES]
    exact: list[Any] = []
    provenance: set[str] = {context}
    affine: set[str] = set()
    blockers: set[str] = set()
    nonnumeric_literal = False
    for domain in domains:
        exact.extend(domain.exact_values)
        provenance.update(domain.provenance)
        affine.update(domain.affine_integer_evidence)
        blockers.update(domain.blockers)
        nonnumeric_literal = nonnumeric_literal or domain.has_nonnumeric_literals
        if domain.numeric_dtype is None and not domain.exact_values and not domain.blockers:
            blockers.add(f"unproven:{context}")

    # ``exact_values`` denotes a complete finite domain, not merely values seen
    # on some branches.  If any joined branch has only a type fact and no exact
    # values, the union is not finite/exact and must drop the partial literals.
    all_inputs_exact = all(bool(domain.exact_values) for domain in domains)
    exact_values = _ordered_values(exact) if all_inputs_exact else ()
    if numeric and nonnumeric_literal:
        blockers.add(f"conflict:mixed_numeric_and_literal:{context}")
        return StaticValueDomain(
            exact_values=exact_values,
            affine_integer_evidence=tuple(sorted(affine)),
            provenance=tuple(sorted(provenance)),
            blockers=tuple(sorted(blockers)),
        )
    if numeric:
        dtype = _promote_numeric([x for x in numeric if x is not None])
        return StaticValueDomain(
            numeric_dtype=dtype,
            exact_values=exact_values,
            affine_integer_evidence=tuple(sorted(affine)),
            provenance=tuple(sorted(provenance)),
            blockers=tuple(sorted(blockers)),
        )
    if nonnumeric_literal:
        return StaticValueDomain(
            exact_values=exact_values,
            affine_integer_evidence=tuple(sorted(affine)),
            provenance=tuple(sorted(provenance)),
            blockers=tuple(sorted(blockers)),
        )
    return StaticValueDomain(
        exact_values=exact_values,
        affine_integer_evidence=tuple(sorted(affine)),
        provenance=tuple(sorted(provenance)),
        blockers=tuple(sorted(blockers)) or (f"unproven:{context}",),
    )


def _domain_equal(a: StaticValueDomain, b: StaticValueDomain) -> bool:
    return a == b


def _strip_ephemeral_blockers(domain: StaticValueDomain) -> StaticValueDomain:
    """Remove fixed-point placeholders once a concrete type/domain fact exists.

    These blockers represent missing information from an earlier iteration, not an
    intrinsic semantic contradiction. Permanent blockers (unsupported names/calls,
    mixed domains, etc.) are retained.
    """
    if domain.numeric_dtype not in _NUMERIC_DTYPES and not domain.exact_values:
        return domain
    ephemeral_prefixes = (
        "unproven_parameter:",
        "unproven:static_parameter:",
    )
    blockers = tuple(
        item for item in domain.blockers
        if not item.startswith(ephemeral_prefixes)
    )
    return StaticValueDomain(
        numeric_dtype=domain.numeric_dtype,
        exact_values=domain.exact_values,
        affine_integer_evidence=domain.affine_integer_evidence,
        provenance=domain.provenance,
        blockers=blockers,
    )


def _resolved_cell_domain(
    domain: StaticValueDomain,
    result_domains: Mapping[str, StaticValueDomain],
) -> StaticValueDomain:
    """Drop only stale unresolved-Cell placeholders whose targets now have facts."""
    blockers: list[str] = []
    for item in domain.blockers:
        if item.startswith("unresolved_cell_result:"):
            target = item.split(":", 1)[1]
            other = result_domains.get(target)
            if other is not None and (
                other.numeric_dtype in _NUMERIC_DTYPES or other.exact_values
            ):
                # A later fixed-point iteration now has a concrete candidate for
                # this dependency. Permanent blockers on that dependency are
                # propagated when the call itself is re-evaluated.
                continue
        blockers.append(item)
    return StaticValueDomain(
        numeric_dtype=domain.numeric_dtype,
        exact_values=domain.exact_values,
        affine_integer_evidence=domain.affine_integer_evidence,
        provenance=domain.provenance,
        blockers=tuple(blockers),
    )


def _range_target_domain(iter_node: ast.AST, *, provenance: str) -> StaticValueDomain:
    if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name) and iter_node.func.id == "range":
        return numeric_domain("int64", provenance, affine="range_integer_axis")
    if isinstance(iter_node, (ast.Tuple, ast.List, ast.Set)):
        values: list[StaticValueDomain] = []
        for elt in iter_node.elts:
            if not isinstance(elt, ast.Constant):
                return unknown_domain(provenance, blocker=f"unproven_iterable:{ast.unparse(iter_node)}")
            values.append(literal_domain(elt.value, provenance))
        return _merge_domains(values, context=provenance)
    return unknown_domain(provenance, blocker=f"unproven_iterable:{ast.unparse(iter_node)}")


def _loop_domains(fn: ast.FunctionDef) -> dict[str, StaticValueDomain]:
    result: dict[str, StaticValueDomain] = {}
    for node in ast.walk(fn):
        generators: Sequence[ast.comprehension] = ()
        if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
            generators = node.generators
        for gen in generators:
            if isinstance(gen.target, ast.Name):
                result[gen.target.id] = _range_target_domain(
                    gen.iter, provenance=f"static_comprehension:{gen.target.id}"
                )
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            result[node.target.id] = _range_target_domain(
                node.iter, provenance=f"static_for:{node.target.id}"
            )
    return result


def _guard_restrictions(fn: ast.FunctionDef) -> dict[str, StaticValueDomain]:
    """Recognize a narrow fail-closed finite-domain guard.

    Pattern admitted:

    ``if v not in (<literal>...): __guard_fail__(...)`` followed by normal use of
    ``v``.  The guard proves that every non-error continuation has one of the listed
    values.  Nothing is inferred from an arbitrary comparison or from a guard whose
    body can continue normally.
    """

    result: dict[str, StaticValueDomain] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.NotIn)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], (ast.Tuple, ast.List, ast.Set))
        ):
            continue
        fail_only = bool(node.body)
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Name) and call.func.id == "__guard_fail__":
                    continue
            fail_only = False
            break
        if not fail_only:
            continue
        values: list[StaticValueDomain] = []
        for elt in test.comparators[0].elts:
            if not isinstance(elt, ast.Constant):
                values = []
                break
            values.append(literal_domain(elt.value, f"guard_finite_domain:{test.left.id}"))
        if values:
            result[test.left.id] = _merge_domains(
                values, context=f"guard_finite_domain:{test.left.id}"
            )
    return result


def _input_dtype(input_specs: Mapping[str, Any], key: Any) -> str | None:
    spec = input_specs.get(key)
    dtype = None if spec is None else getattr(spec, "dtype", None)
    return str(dtype) if dtype in _NUMERIC_DTYPES else None


def _infer_generator_domain(
    node: ast.GeneratorExp,
    *,
    env: Mapping[str, StaticValueDomain],
    result_domains: Mapping[str, StaticValueDomain],
    input_specs: Mapping[str, Any],
    all_uids: set[str],
    module_refs: Mapping[str, str],
) -> StaticValueDomain:
    local = dict(env)
    for gen in node.generators:
        if not isinstance(gen.target, ast.Name):
            return unknown_domain("generator", blocker="unproven_generator_target")
        local[gen.target.id] = _range_target_domain(
            gen.iter, provenance=f"static_generator:{gen.target.id}"
        )
    return infer_expression_domain(
        node.elt,
        env=local,
        result_domains=result_domains,
        input_specs=input_specs,
        all_uids=all_uids,
        module_refs=module_refs,
    )


def infer_expression_domain(
    node: ast.AST,
    *,
    env: Mapping[str, StaticValueDomain],
    result_domains: Mapping[str, StaticValueDomain],
    input_specs: Mapping[str, Any],
    all_uids: set[str],
    module_refs: Mapping[str, str] | None = None,
) -> StaticValueDomain:
    module_refs = module_refs or {}
    if isinstance(node, ast.Constant):
        return literal_domain(node.value, "static_literal")
    if isinstance(node, ast.Name):
        domain = env.get(node.id)
        if domain is None:
            return unknown_domain(f"name:{node.id}", blocker=f"unproven_name:{node.id}")
        return _strip_ephemeral_blockers(domain).with_provenance(f"name:{node.id}")
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return numeric_domain("bool", "static_unary_not")
        child = infer_expression_domain(
            node.operand, env=env, result_domains=result_domains,
            input_specs=input_specs, all_uids=all_uids, module_refs=module_refs,
        )
        if child.numeric_dtype in _NUMERIC_DTYPES:
            exact_values: tuple[Any, ...] = ()
            if child.exact_values and isinstance(node.op, (ast.UAdd, ast.USub)):
                values: list[Any] = []
                for value in child.exact_values:
                    if not isinstance(value, (bool, int, float)):
                        values = []
                        break
                    result = +value if isinstance(node.op, ast.UAdd) else -value
                    if isinstance(result, int) and not (_INT64_MIN <= result <= _INT64_MAX):
                        values = []
                        break
                    values.append(result)
                if values:
                    exact_values = _ordered_values(values)
            return StaticValueDomain(
                numeric_dtype=child.numeric_dtype,
                exact_values=exact_values,
                affine_integer_evidence=child.affine_integer_evidence,
                provenance=tuple(sorted(set(child.provenance) | {"static_unary_numeric"})),
                blockers=child.blockers,
            )
        return unknown_domain("static_unary_numeric", blocker=f"unproven_unary:{ast.unparse(node)}")
    if isinstance(node, ast.Compare):
        return numeric_domain("bool", "static_comparison")
    if isinstance(node, ast.BoolOp):
        return numeric_domain("bool", "static_boolean_operator")
    if isinstance(node, ast.IfExp):
        body = infer_expression_domain(
            node.body, env=env, result_domains=result_domains,
            input_specs=input_specs, all_uids=all_uids, module_refs=module_refs,
        )
        other = infer_expression_domain(
            node.orelse, env=env, result_domains=result_domains,
            input_specs=input_specs, all_uids=all_uids, module_refs=module_refs,
        )
        return _merge_domains((body, other), context="static_if_expression")
    if isinstance(node, ast.BinOp):
        left = infer_expression_domain(
            node.left, env=env, result_domains=result_domains,
            input_specs=input_specs, all_uids=all_uids, module_refs=module_refs,
        )
        right = infer_expression_domain(
            node.right, env=env, result_domains=result_domains,
            input_specs=input_specs, all_uids=all_uids, module_refs=module_refs,
        )
        blockers = tuple(sorted(set(left.blockers).union(right.blockers)))
        if left.numeric_dtype not in _NUMERIC_DTYPES or right.numeric_dtype not in _NUMERIC_DTYPES:
            return StaticValueDomain(
                provenance=tuple(sorted(set(left.provenance).union(right.provenance, {"static_binary"}))),
                blockers=blockers or (f"unproven_binary:{ast.unparse(node)}",),
            )
        if isinstance(node.op, ast.Div):
            dtype = "float64"
        elif isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow)):
            dtype = _promote_numeric((left.numeric_dtype, right.numeric_dtype))
            if isinstance(node.op, ast.Pow) and dtype == "bool":
                dtype = "int64"
        else:
            return unknown_domain("static_binary", blocker=f"unsupported_binary:{type(node.op).__name__}")
        affine: set[str] = set(left.affine_integer_evidence).union(right.affine_integer_evidence)
        if (
            dtype == "int64"
            and isinstance(node.left, ast.Name)
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, int)
            and isinstance(node.op, (ast.Add, ast.Sub))
        ):
            sign = 1 if isinstance(node.op, ast.Add) else -1
            affine.add(f"{node.left.id}{sign * node.right.value:+d}")
        return StaticValueDomain(
            numeric_dtype=dtype,
            exact_values=_compose_exact_binary_values(left, right, node.op),
            affine_integer_evidence=tuple(sorted(affine)),
            provenance=tuple(sorted(set(left.provenance).union(right.provenance, {"static_binary"}))),
            blockers=blockers,
        )
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in all_uids:
                domain = result_domains.get(name)
                if domain is None or (domain.numeric_dtype is None and not domain.exact_values):
                    return unknown_domain(
                        f"static_cell_call:{name}", blocker=f"unresolved_cell_result:{name}"
                    )
                return _resolved_cell_domain(domain, result_domains).with_provenance(
                    f"static_cell_call:{name}"
                )
            if name in _RUNTIME_INPUT_CALLS:
                if node.args and isinstance(node.args[0], ast.Constant):
                    dtype = _input_dtype(input_specs, node.args[0].value)
                    if dtype is not None:
                        return numeric_domain(dtype, f"runtime_input:{node.args[0].value}")
                return unknown_domain("runtime_input", blocker=f"unproven_runtime_input:{ast.unparse(node)}")
            if name == "int":
                return numeric_domain("int64", "numeric_cast:int")
            if name == "float":
                return numeric_domain("float64", "numeric_cast:float")
            if name == "bool":
                return numeric_domain("bool", "numeric_cast:bool")
            if name in _BOOL_INTRINSICS:
                return numeric_domain("bool", f"intrinsic:{name}")
            if name in {"__exact_axis_code__", "__exact_axis_code_valid__"}:
                return numeric_domain("int64", f"intrinsic:{name}")
            # These normalized provider helpers are emitted as double-valued
            # lookups by the source-preserving Cython backend.  Reuse that same
            # ABI fact in demand typing so an unobserved branch does not lose
            # numeric type evidence merely because normalization introduced a
            # compiler-owned lookup intrinsic.
            if name in {"__step_lookup_1d__", "__interval_lookup_1d__", "__interp_lookup_1d__"}:
                return numeric_domain("float64", f"intrinsic:{name}")
            if name in _FLOAT_INTRINSICS:
                return numeric_domain("float64", f"intrinsic:{name}")
            if name in {"min", "max", "abs"} and node.args:
                args = [
                    infer_expression_domain(
                        arg, env=env, result_domains=result_domains,
                        input_specs=input_specs, all_uids=all_uids, module_refs=module_refs,
                    )
                    for arg in node.args
                ]
                merged = _merge_domains(args, context=f"intrinsic:{name}")
                if merged.numeric_dtype in _NUMERIC_DTYPES:
                    return StaticValueDomain(
                        numeric_dtype=merged.numeric_dtype,
                        provenance=merged.provenance,
                        blockers=merged.blockers,
                    )
                return merged
            if name == "round" and node.args:
                child = infer_expression_domain(
                    node.args[0], env=env, result_domains=result_domains,
                    input_specs=input_specs, all_uids=all_uids, module_refs=module_refs,
                )
                if child.numeric_dtype in _NUMERIC_DTYPES:
                    # The shared Cython emitter uses rint/scale and therefore emits
                    # a double-valued expression for the normalized native subset.
                    return StaticValueDomain(
                        numeric_dtype="float64",
                        provenance=tuple(sorted(set(child.provenance) | {"intrinsic:round"})),
                        blockers=child.blockers,
                    )
                return unknown_domain("intrinsic:round", blocker=f"unproven_round:{ast.unparse(node)}")
            if name == "sum" and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
                domain = _infer_generator_domain(
                    node.args[0], env=env, result_domains=result_domains,
                    input_specs=input_specs, all_uids=all_uids, module_refs=module_refs,
                )
                if domain.numeric_dtype in _NUMERIC_DTYPES:
                    return domain.with_provenance("static_sum_reduction")
                return domain
            if name in {"any", "all"}:
                return numeric_domain("bool", f"static_reduction:{name}")
            if name == "__guard_fail__":
                return unknown_domain("guard_fail", blocker="nonreturning_guard")
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            receiver = node.func.value.id
            if module_refs.get(receiver) in {"math", "numpy"} and node.func.attr in _MATH_INTRINSICS:
                return numeric_domain("float64", f"math_intrinsic:{receiver}.{node.func.attr}")
        return unknown_domain("call", blocker=f"unproven_call:{ast.unparse(node)}")
    if isinstance(node, ast.GeneratorExp):
        return _infer_generator_domain(
            node, env=env, result_domains=result_domains,
            input_specs=input_specs, all_uids=all_uids, module_refs=module_refs,
        )
    return unknown_domain(
        f"expression:{type(node).__name__}", blocker=f"unsupported_expression:{ast.unparse(node)}"
    )


def _local_domains(
    fn: ast.FunctionDef,
    *,
    parameter_domains: Mapping[str, StaticValueDomain],
    result_domains: Mapping[str, StaticValueDomain],
    input_specs: Mapping[str, Any],
    all_uids: set[str],
    module_refs: Mapping[str, str],
    reference_domains: Mapping[str, StaticValueDomain] | None = None,
) -> dict[str, StaticValueDomain]:
    # modelx-cython traces refs independently from Cell arguments/results.  Feed
    # those authoritative scalar ref types into local expression inference when
    # available; parameters deliberately shadow same-named refs.
    env = dict(reference_domains or {})
    env.update(parameter_domains)
    env.update(_loop_domains(fn))
    restrictions = _guard_restrictions(fn)
    assignments: list[tuple[str, ast.AST]] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments.append((node.targets[0].id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            assignments.append((node.target.id, node.value))
    for _ in range(max(2, len(assignments) + 2)):
        changed = False
        for name, expr in assignments:
            inferred = infer_expression_domain(
                expr, env=env, result_domains=result_domains, input_specs=input_specs,
                all_uids=all_uids, module_refs=module_refs,
            )
            restriction = restrictions.get(name)
            if restriction is not None:
                # A fail-closed finite-domain guard dominates an otherwise unknown
                # provider/object source for normal continuation.  If the assignment
                # itself proves a conflicting numeric domain, preserve the conflict.
                if inferred.numeric_dtype in _NUMERIC_DTYPES:
                    if inferred.numeric_dtype == restriction.numeric_dtype:
                        inferred = StaticValueDomain(
                            numeric_dtype=inferred.numeric_dtype,
                            exact_values=restriction.exact_values,
                            affine_integer_evidence=tuple(sorted(set(
                                inferred.affine_integer_evidence
                            ).union(restriction.affine_integer_evidence))),
                            provenance=tuple(sorted(set(inferred.provenance).union(
                                restriction.provenance,
                                {f"guard_assignment:{name}"},
                            ))),
                            blockers=tuple(sorted(set(inferred.blockers).union(
                                restriction.blockers
                            ))),
                        )
                    else:
                        inferred = _merge_domains(
                            (inferred, restriction), context=f"guard_assignment:{name}"
                        )
                else:
                    inferred = restriction.with_provenance(*inferred.provenance)
            if env.get(name) != inferred:
                env[name] = inferred
                changed = True
        if not changed:
            break
    for name, restriction in restrictions.items():
        if name not in env:
            env[name] = restriction
    return env


def _call_argument_domains(
    seed: StaticRecursiveFormulaSeed,
    *,
    parameter_domains: Mapping[str, Mapping[str, StaticValueDomain]],
    result_domains: Mapping[str, StaticValueDomain],
    formula_overrides: Mapping[str, ast.FunctionDef],
    input_specs: Mapping[str, Any],
    module_refs: Mapping[str, str],
) -> dict[tuple[str, str], list[StaticValueDomain]]:
    all_uids = set(seed.cells)
    out: dict[tuple[str, str], list[StaticValueDomain]] = {
        (uid, name): []
        for uid, cell in seed.cells.items()
        for name in cell.parameters
    }
    for caller_uid, cell in seed.cells.items():
        fn = formula_overrides.get(caller_uid, cell.function)
        env = _local_domains(
            fn,
            parameter_domains=parameter_domains.get(caller_uid, {}),
            result_domains=result_domains,
            input_specs=input_specs,
            all_uids=all_uids,
            module_refs=module_refs,
        )
        for node in ast.walk(fn):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in all_uids
            ):
                continue
            callee = seed.cells[node.func.id]
            if len(node.args) != len(callee.parameters):
                # V02395F already materialized defaults/keywords.  Any mismatch is
                # a structural contradiction and must remain blocked.
                for name in callee.parameters:
                    out[(node.func.id, name)].append(
                        unknown_domain(
                            "static_call_arity",
                            blocker=f"conflict:call_arity:{caller_uid}->{node.func.id}",
                        )
                    )
                continue
            for name, arg in zip(callee.parameters, node.args):
                domain = infer_expression_domain(
                    arg, env=env, result_domains=result_domains, input_specs=input_specs,
                    all_uids=all_uids, module_refs=module_refs,
                ).with_provenance(f"static_call:{caller_uid}->{node.func.id}:{name}")
                out[(node.func.id, name)].append(domain)
    return out


def _infer_result_domain(
    uid: str,
    fn: ast.FunctionDef,
    *,
    parameter_domains: Mapping[str, StaticValueDomain],
    result_domains: Mapping[str, StaticValueDomain],
    input_specs: Mapping[str, Any],
    all_uids: set[str],
    module_refs: Mapping[str, str],
    reference_domains: Mapping[str, StaticValueDomain] | None = None,
) -> StaticValueDomain:
    env = _local_domains(
        fn,
        parameter_domains=parameter_domains,
        result_domains=result_domains,
        input_specs=input_specs,
        all_uids=all_uids,
        module_refs=module_refs,
        reference_domains=reference_domains,
    )
    returns: list[StaticValueDomain] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return):
            continue
        if node.value is None:
            returns.append(literal_domain(None, f"static_return:{uid}:None"))
        else:
            returns.append(
                infer_expression_domain(
                    node.value, env=env, result_domains=result_domains,
                    input_specs=input_specs, all_uids=all_uids, module_refs=module_refs,
                ).with_provenance(f"static_return:{uid}")
            )
    if not returns:
        return unknown_domain(f"static_result:{uid}", blocker=f"unproven_result:no_return:{uid}")
    merged = _merge_domains(returns, context=f"static_result:{uid}")
    # Guard failure is non-returning and must not pollute the result lattice.  It
    # may appear in local expression analysis but never as a Return value.
    blockers = tuple(x for x in merged.blockers if x != "nonreturning_guard")
    return StaticValueDomain(
        numeric_dtype=merged.numeric_dtype,
        exact_values=merged.exact_values,
        affine_integer_evidence=merged.affine_integer_evidence,
        provenance=merged.provenance,
        blockers=blockers,
    )


@dataclass(frozen=True)
class StaticCellTypeProof:
    uid: str
    source_name: str
    parameter_domains: tuple[tuple[str, StaticValueDomain], ...]
    result_domain: StaticValueDomain
    dependencies: tuple[str, ...]
    blockers: tuple[str, ...] = ()

    def parameter(self, name: str) -> StaticValueDomain:
        for key, domain in self.parameter_domains:
            if key == name:
                return domain
        raise KeyError(name)

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_name": self.source_name,
            "parameters": {
                name: domain.manifest() for name, domain in self.parameter_domains
            },
            "result": self.result_domain.manifest(),
            "dependencies": list(self.dependencies),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class StaticTypeDomainProof:
    schema: str
    cells: Mapping[str, StaticCellTypeProof]
    iterations: int

    def cell(self, uid: str) -> StaticCellTypeProof:
        return self.cells[uid]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "iterations": self.iterations,
            "cell_count": len(self.cells),
            "numeric_result_count": sum(
                c.result_domain.proven and c.result_domain.numeric_dtype in _NUMERIC_DTYPES
                for c in self.cells.values()
            ),
            "finite_literal_result_count": sum(
                c.result_domain.proven and c.result_domain.finite_literal
                for c in self.cells.values()
            ),
            "cells": {uid: cell.manifest() for uid, cell in sorted(self.cells.items())},
        }


def build_static_type_domain_proof(
    seed: StaticRecursiveFormulaSeed,
    *,
    formula_overrides: Mapping[str, ast.FunctionDef] | None = None,
    input_specs: Mapping[str, Any] | None = None,
    module_refs: Mapping[str, str] | None = None,
    materialized_uids: Sequence[str] = (),
) -> StaticTypeDomainProof:
    """Derive parameter/result domains to a fixed point from static formulas/calls.

    ``formula_overrides`` may contain provider-normalized versions of the same
    qualified static Cells.  It must not introduce a different Cell set or UID.
    """

    formula_overrides = dict(formula_overrides or {})
    unknown_override = set(formula_overrides) - set(seed.cells)
    if unknown_override:
        raise StaticTypeDomainError(
            "formula override contains non-static Cell UIDs: " + ", ".join(sorted(unknown_override))
        )
    input_specs = dict(input_specs or {})
    module_refs = dict(module_refs or {})
    materialized = set(materialized_uids)
    all_uids = set(seed.cells)

    parameter_domains: dict[str, dict[str, StaticValueDomain]] = {
        uid: {
            name: unknown_domain(
                f"static_parameter:{uid}:{name}",
                blocker=f"unproven_parameter:{uid}:{name}",
            )
            for name in cell.parameters
        }
        for uid, cell in seed.cells.items()
    }
    result_domains: dict[str, StaticValueDomain] = {
        uid: unknown_domain(
            f"static_result:{uid}", blocker=f"unproven_result:{uid}"
        )
        for uid in seed.cells
    }

    max_iter = max(12, len(seed.cells) * 4)
    iterations = 0
    for iteration in range(1, max_iter + 1):
        iterations = iteration
        changed = False
        call_domains = _call_argument_domains(
            seed,
            parameter_domains=parameter_domains,
            result_domains=result_domains,
            formula_overrides=formula_overrides,
            input_specs=input_specs,
            module_refs=module_refs,
        )
        next_params: dict[str, dict[str, StaticValueDomain]] = {}
        for uid, cell in seed.cells.items():
            next_params[uid] = {}
            for name in cell.parameters:
                domains = call_domains[(uid, name)]
                merged = _merge_domains(
                    domains, context=f"static_parameter:{uid}:{name}"
                )
                merged = _strip_ephemeral_blockers(merged)
                next_params[uid][name] = merged
                if not _domain_equal(merged, parameter_domains[uid][name]):
                    changed = True

        next_results: dict[str, StaticValueDomain] = {}
        for uid, cell in seed.cells.items():
            if uid in materialized:
                # Object/materialized helpers are semantic Cells but are not forced
                # into the numeric lattice.  Guard-restricted literal helpers can
                # still be proven if their static formula supplies such evidence.
                fn = formula_overrides.get(uid, cell.function)
                inferred = _infer_result_domain(
                    uid, fn,
                    parameter_domains=next_params[uid],
                    result_domains=result_domains,
                    input_specs=input_specs,
                    all_uids=all_uids,
                    module_refs=module_refs,
                )
                next_results[uid] = inferred
            else:
                fn = formula_overrides.get(uid, cell.function)
                next_results[uid] = _infer_result_domain(
                    uid, fn,
                    parameter_domains=next_params[uid],
                    result_domains=result_domains,
                    input_specs=input_specs,
                    all_uids=all_uids,
                    module_refs=module_refs,
                )
            if not _domain_equal(next_results[uid], result_domains[uid]):
                changed = True

        parameter_domains = next_params
        result_domains = next_results
        if not changed:
            break
    else:
        raise StaticTypeDomainError(
            f"static type/domain fixed point did not converge in {max_iter} iterations"
        )

    proofs: dict[str, StaticCellTypeProof] = {}
    for uid, cell in seed.cells.items():
        cell_blockers: set[str] = set(result_domains[uid].blockers)
        for domain in parameter_domains[uid].values():
            cell_blockers.update(domain.blockers)
        proofs[uid] = StaticCellTypeProof(
            uid=uid,
            source_name=cell.key.cell_name,
            parameter_domains=tuple(
                (name, parameter_domains[uid][name]) for name in cell.parameters
            ),
            result_domain=result_domains[uid],
            dependencies=cell.dependency_uids,
            blockers=tuple(sorted(cell_blockers)),
        )

    return StaticTypeDomainProof(
        schema="modelx_graph.static_type_domain.v1",
        cells=proofs,
        iterations=iterations,
    )


__all__ = [
    "StaticTypeDomainError",
    "StaticValueDomain",
    "StaticCellTypeProof",
    "StaticTypeDomainProof",
    "unknown_domain",
    "literal_domain",
    "numeric_domain",
    "infer_expression_domain",
    "build_static_type_domain_proof",
]
