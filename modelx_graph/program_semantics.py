from __future__ import annotations

"""Backend-neutral semantic operations shared by optimized Python and Cython."""

import ast
from dataclasses import dataclass
from typing import Any


class ProgramSemanticError(RuntimeError):
    pass




@dataclass(frozen=True)
class StaticScalarFact:
    """A scalar value proven invariant over the complete compiled run-key domain."""

    source_kind: str  # cell | reference
    source_name: str
    value: Any
    proof_domain_size: int

    def manifest(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_name": self.source_name,
            "value": self.value,
            "proof_domain_size": int(self.proof_domain_size),
        }


@dataclass(frozen=True)
class ValidatedStaticFact:
    """A static value whose source-domain validation remains explicit.

    This is used for small categorical selectors and exact numeric/bool point
    selectors whose guard is proven unreachable for the specialized value.
    Preparation proves one value across the compiled run-key domain while an
    explicit point input preserves the runtime validation contract.
    """

    source_name: str
    value: Any
    allowed_values: tuple[Any, ...]
    proof_domain_size: int
    validation_input_key: str

    def manifest(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "value": self.value,
            "allowed_values": list(self.allowed_values),
            "proof_domain_size": int(self.proof_domain_size),
            "validation_input_key": self.validation_input_key,
        }


@dataclass(frozen=True)
class RuntimeScalarHelperFact:
    """A pure scalar Cell inlined before executable scheduling.

    The helper disappears before scheduling/code generation; this record makes the
    upstream semantic decision auditable in ``OptimizedProgram.manifest()``.
    """

    source_name: str
    parameters: tuple[str, ...]
    observed_variant_count: int
    coordinate_classified: bool = False
    coordinate_recovered_from_callsite: bool = False
    source_backed_unobserved: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "parameters": list(self.parameters),
            "observed_variant_count": int(self.observed_variant_count),
            "coordinate_classified": bool(self.coordinate_classified),
            "coordinate_recovered_from_callsite": bool(self.coordinate_recovered_from_callsite),
            "source_backed_unobserved": bool(self.source_backed_unobserved),
        }


@dataclass(frozen=True)
class SourceBackedScheduledFact:
    """One scheduled canonical variant created from source rather than trace.

    Source-backed scheduled variants are ordinary graph operations after frontend
    construction; this record exists solely to keep their provenance explicit.
    In particular, ``observed`` is always false.  The compiler must never turn
    source availability into fictional representative-trace evidence.
    """

    source_name: str
    synthetic_uid: str
    parameters: tuple[str, ...]
    time_param: str
    aux_values: tuple[Any, ...] = ()
    dtype: str = "float64"
    observed: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "synthetic_uid": self.synthetic_uid,
            "parameters": list(self.parameters),
            "time_param": self.time_param,
            "aux_values": list(self.aux_values),
            "dtype": self.dtype,
            "observed": False,
            "source_backed_scheduled": True,
        }


@dataclass(frozen=True)
class StepLookupDomainFact:
    """Source/call-domain proof for an irregular right-continuous step lookup.

    The runtime lookup already implements ``max(anchor <= q)``.  This fact records
    why the source form with a bare ``max(keys)`` cannot observe an empty key set:
    every reachable specialized call proves ``q >= first_anchor``.  It is not a
    sampled-query fact and it does not define a synthetic below-first value.
    """

    source_uid: str
    source_name: str
    query_expr: str
    axis_min: float
    proven_lower_bound: float
    proven_upper_bound: float | None = None
    proof_kind: str = "specialized_call_domain"

    def manifest(self) -> dict[str, Any]:
        return {
            "source_uid": self.source_uid,
            "source_name": self.source_name,
            "query_expr": self.query_expr,
            "axis_min": float(self.axis_min),
            "proven_lower_bound": float(self.proven_lower_bound),
            "proven_upper_bound": (
                None if self.proven_upper_bound is None else float(self.proven_upper_bound)
            ),
            "proof_kind": self.proof_kind,
            "below_first": "unreachable_by_proof",
        }


@dataclass(frozen=True)
class FixedCoordinateFact:
    """One coordinate-family call specialized to a compile-time integer coordinate.

    The synthetic scalar remains part of the optimized program and is evaluated by
    generated Python/Cython.  This is not a modelx fallback or precomputed result.
    """

    source_uid: str
    source_name: str
    coordinate: int
    synthetic_uid: str

    def manifest(self) -> dict[str, Any]:
        return {
            "source_uid": self.source_uid,
            "source_name": self.source_name,
            "coordinate": int(self.coordinate),
            "synthetic_uid": self.synthetic_uid,
        }


@dataclass(frozen=True)
class CoordinateRecurrenceDomainFact:
    """Finite source-demand proof for one recurrent coordinate family.

    ``demand_*`` includes any terminal coordinate needed to evaluate the fixed
    OutputInvocation.  ``active_*`` is the subdomain on which the recursive self
    edge is reachable.  Boundary coordinates are therefore explicit semantic seed
    points rather than representative-trace extrema.
    """

    source_uid: str
    source_name: str
    coordinate_parameter: str
    demand_lo: int
    demand_hi: int
    active_lo: int
    active_hi: int
    recurrence_offsets: tuple[int, ...]
    boundary_coordinates: tuple[int, ...]
    scan_requirement: str
    output_invocation_uid: str
    proof_kind: str = "fixed_output_source_guarded_recurrence_domain_v1"
    blockers: tuple[str, ...] = ()

    @property
    def approved(self) -> bool:
        return not self.blockers

    def manifest(self) -> dict[str, Any]:
        return {
            "source_uid": self.source_uid,
            "source_name": self.source_name,
            "coordinate_parameter": self.coordinate_parameter,
            "demand_domain": [int(self.demand_lo), int(self.demand_hi)],
            "active_domain": [int(self.active_lo), int(self.active_hi)],
            "recurrence_offsets": [int(x) for x in self.recurrence_offsets],
            "boundary_coordinates": [int(x) for x in self.boundary_coordinates],
            "scan_requirement": self.scan_requirement,
            "output_invocation_uid": self.output_invocation_uid,
            "proof_kind": self.proof_kind,
            "approved": self.approved,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class FiniteDomainFact:
    """A small iterable domain proven invariant over all compiled run keys."""

    source_kind: str  # currently cell
    source_name: str
    values: tuple[Any, ...]
    proof_domain_size: int

    def manifest(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_name": self.source_name,
            "values": list(self.values),
            "proof_domain_size": int(self.proof_domain_size),
        }
@dataclass(frozen=True)
class GuardSpec:
    """Terminal fail-closed guard embedded in an optimized numeric formula.

    The frontend replaces supported ``raise`` statements with the synthetic
    ``__guard_fail__(code)`` statement.  The code is the backend-neutral semantic
    operation; Python renders the original exception while Cython records the same
    guard code in nogil execution and raises after returning to the GIL.
    """

    code: int
    uid: str
    exception_type: str
    message: str | None

    def manifest(self) -> dict[str, Any]:
        return {
            "code": int(self.code),
            "uid": self.uid,
            "exception_type": self.exception_type,
            "message": self.message,
        }


_SUPPORTED_EXCEPTIONS = {
    "ValueError",
    "RuntimeError",
    "TypeError",
    "AssertionError",
    "OverflowError",
    "ZeroDivisionError",
    "KeyError",
}


def _parse_raise(st: ast.Raise, uid: str) -> tuple[str, str | None]:
    if st.cause is not None:
        raise ProgramSemanticError(f"{uid}: raise ... from ... is outside the optimized guard subset")
    exc = st.exc
    if exc is None:
        raise ProgramSemanticError(f"{uid}: bare raise is outside the optimized guard subset")
    if isinstance(exc, ast.Name):
        name = exc.id
        args: list[ast.AST] = []
    elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and not exc.keywords:
        name = exc.func.id
        args = list(exc.args)
    else:
        raise ProgramSemanticError(f"{uid}: dynamic exception expression is outside the optimized guard subset")
    if name not in _SUPPORTED_EXCEPTIONS:
        raise ProgramSemanticError(f"{uid}: exception type {name!r} is outside the optimized guard subset")
    if not args:
        return name, None
    if len(args) == 1 and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
        return name, args[0].value
    raise ProgramSemanticError(f"{uid}: guard exception message must be a literal string")


def guard_code_from_stmt(st: ast.stmt) -> int | None:
    if not isinstance(st, ast.Expr) or not isinstance(st.value, ast.Call):
        return None
    call = st.value
    if not isinstance(call.func, ast.Name) or call.func.id != "__guard_fail__":
        return None
    if len(call.args) != 1 or call.keywords or not isinstance(call.args[0], ast.Constant):
        raise ProgramSemanticError("malformed optimized guard statement")
    code = call.args[0].value
    if not isinstance(code, int) or isinstance(code, bool) or code <= 0:
        raise ProgramSemanticError("optimized guard code must be a positive integer")
    return int(code)


def normalize_formula_guards(
    fn: ast.FunctionDef,
    *,
    uid: str,
    next_code: int,
) -> tuple[ast.FunctionDef, tuple[GuardSpec, ...], int]:
    """Replace supported Python raises with backend-neutral terminal guard ops."""

    specs: list[GuardSpec] = []
    code = int(next_code)

    class Normalize(ast.NodeTransformer):
        def visit_Raise(self, node: ast.Raise):
            nonlocal code
            exception_type, message = _parse_raise(node, uid)
            spec = GuardSpec(code=code, uid=uid, exception_type=exception_type, message=message)
            specs.append(spec)
            replacement = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="__guard_fail__", ctx=ast.Load()),
                    args=[ast.Constant(code)],
                    keywords=[],
                )
            )
            code += 1
            return ast.copy_location(replacement, node)

    out = Normalize().visit(ast.fix_missing_locations(fn))
    assert isinstance(out, ast.FunctionDef)
    return ast.fix_missing_locations(out), tuple(specs), code
