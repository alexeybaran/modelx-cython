from __future__ import annotations

"""Behavior-preserving semantic facade over the current fast compiler plan.

Phase A deliberately does *not* make compilation consume this module.  It creates
one explicit semantic contract from the facts already carried by the modelx-cython
frontend, source-transparent Cell catalog, category plan, ranked-output evidence,
and execution-class plan.  Every derived fact is checked against those legacy
representations before the fast plan is returned.

The separation is intentional:

* :class:`ValueSpec`, :class:`AxisDomain`, and :class:`CellSignature` describe
  logical model semantics.
* :class:`LoweringPlan` describes the currently selected/known physical ABI and
  memo facts.

Later cleanup phases can migrate consumers to these contracts one at a time while
retaining the old fields as differential assertions until they can be deleted.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class SemanticContractError(RuntimeError):
    """Raised when the semantic facade disagrees with existing compiler facts."""


_NATIVE_DTYPES = frozenset({"bool", "int64", "float64"})


def _semantic_dtype(dtype: str, category_domain_id: str | None) -> str:
    return "category" if category_domain_id is not None else str(dtype)


def _normalized_shape(ndim: int, shape: Sequence[int] | None) -> tuple[int, ...] | None:
    if int(ndim) == 0:
        return ()
    if shape is None:
        return None
    out = tuple(int(x) for x in shape)
    if len(out) != int(ndim):
        raise SemanticContractError(
            f"logical shape rank mismatch: ndim={ndim}, shape={out!r}"
        )
    return out


@dataclass(frozen=True)
class AxisDomain:
    """One logical axis of a ranked Cell result.

    ``axis_id`` is semantic identity, not merely shape.  Phase A only knows an
    authoritative identity for the selected traced point axis.  Other ranked
    axes therefore receive Cell-local unresolved IDs rather than being equated
    merely because their lengths happen to match.
    """

    axis_id: str
    kind: str  # point | unresolved
    size: int | None
    keys: tuple[Any, ...] = ()
    components: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "kind": self.kind,
            "size": self.size,
            "key_count": len(self.keys),
            "components": list(self.components),
            "provenance": list(self.provenance),
            "identity_is_not_shape_only": True,
        }


@dataclass(frozen=True)
class ValueSpec:
    """Logical value semantics independent from the physical recursive ABI."""

    logical_dtype: str
    axes: tuple[AxisDomain, ...] = ()
    category_domain_id: str | None = None

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...] | None:
        if not self.axes:
            return ()
        if any(axis.size is None for axis in self.axes):
            return None
        return tuple(int(axis.size) for axis in self.axes if axis.size is not None)

    def manifest(self) -> dict[str, Any]:
        return {
            "logical_dtype": self.logical_dtype,
            "category_domain_id": self.category_domain_id,
            "axes": [axis.axis_id for axis in self.axes],
            "ndim": self.ndim,
            "shape": None if self.shape is None else list(self.shape),
        }


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    value: ValueSpec

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value.manifest()}


@dataclass(frozen=True)
class CellSignature:
    """Semantic callable contract for one canonical Cell UID."""

    uid: str
    source_name: str
    source_fullname: str
    parameters: tuple[ParameterSpec, ...]
    result: ValueSpec

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_name": self.source_name,
            "source_fullname": self.source_fullname,
            "parameters": [row.manifest() for row in self.parameters],
            "result": self.result.manifest(),
        }


@dataclass(frozen=True)
class LoweringPlan:
    """Current physical lowering facts, kept separate from Cell semantics.

    Phase A intentionally mirrors the existing execution-class plan.  Later
    phases can replace individual fields with richer lowering choices while the
    semantic signature remains stable.
    """

    uid: str
    execution_class: str
    execution_backend: str  # native | python_semantics
    parameter_abi_dtypes: tuple[str, ...]
    result_abi_dtype: str
    result_value_abi: str
    native_scalar_abi: bool
    recursive_value_bridgeable: bool
    gil_required: bool
    memo_kind: str
    memo_eligible: bool
    memo_physicalization_ready: bool
    memo_bounds: tuple[tuple[int, int], ...] = ()
    memo_size: int = 0
    memo_hash_capacity: int = 0
    direct_blockers: tuple[tuple[str, str], ...] = ()
    classification_reasons: tuple[str, ...] = ()

    @property
    def key_abi(self) -> str:
        return (
            "native_tuple"
            if all(dtype in _NATIVE_DTYPES for dtype in self.parameter_abi_dtypes)
            else "python_object"
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "execution_class": self.execution_class,
            "execution_backend": self.execution_backend,
            "gil_required": self.gil_required,
            "parameter_abi_dtypes": list(self.parameter_abi_dtypes),
            "key_abi": self.key_abi,
            "result_abi_dtype": self.result_abi_dtype,
            "result_value_abi": self.result_value_abi,
            "native_scalar_abi": self.native_scalar_abi,
            "recursive_value_bridgeable": self.recursive_value_bridgeable,
            "memo": {
                "kind": self.memo_kind,
                "eligible": self.memo_eligible,
                "physicalization_ready": self.memo_physicalization_ready,
                "bounds": [list(x) for x in self.memo_bounds],
                "size": self.memo_size,
                "hash_capacity": self.memo_hash_capacity,
            },
            "direct_blockers": [
                {"kind": kind, "detail": detail}
                for kind, detail in self.direct_blockers
            ],
            "classification_reasons": list(self.classification_reasons),
        }


@dataclass(frozen=True)
class SemanticFacadePlan:
    schema: str
    axes: Mapping[str, AxisDomain]
    signatures: Mapping[str, CellSignature]
    lowerings: Mapping[str, LoweringPlan]
    validation_checks: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "role": "behavior_preserving_semantic_facade_not_yet_codegen_authority",
            "axes": {
                axis_id: axis.manifest()
                for axis_id, axis in sorted(self.axes.items())
            },
            "signatures": {
                uid: sig.manifest()
                for uid, sig in sorted(self.signatures.items())
            },
            "lowerings": {
                uid: lowering.manifest()
                for uid, lowering in sorted(self.lowerings.items())
            },
            "validation": {
                "legacy_consistency_asserted": True,
                "checks": list(self.validation_checks),
                "signature_count": len(self.signatures),
                "lowering_count": len(self.lowerings),
                "axis_count": len(self.axes),
            },
        }


def _frontend_cells(plan: Any) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for row in (*tuple(getattr(plan, "cells", ())), *tuple(getattr(plan, "cross_space_cells", ()))):
        prior = rows.get(row.uid)
        if prior is not None and prior != row:
            raise SemanticContractError(f"duplicate frontend Cell UID disagrees: {row.uid}")
        rows[row.uid] = row
    return rows


def _category_parameter_map(cell: Any) -> dict[str, str | None]:
    raw = getattr(cell, "parameter_category_domains", ())
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}
    raw_tuple = tuple(raw)
    if raw_tuple and isinstance(raw_tuple[0], tuple):
        return {str(name): domain for name, domain in raw_tuple}
    parameters = tuple(getattr(cell, "parameters", ()))
    names = tuple(
        p if isinstance(p, str) else str(getattr(p, "name"))
        for p in parameters
    )
    if len(raw_tuple) not in {0, len(names)}:
        raise SemanticContractError(
            f"{getattr(cell, 'uid', '<cell>')}: parameter category-domain arity mismatch"
        )
    return {
        name: (raw_tuple[pos] if raw_tuple else None)
        for pos, name in enumerate(names)
    }


def _cell_parameter_rows(cell: Any) -> tuple[tuple[str, str], ...]:
    parameters = tuple(getattr(cell, "parameters", ()))
    if not parameters:
        return ()
    if isinstance(parameters[0], str):
        names = tuple(str(x) for x in parameters)
        dtypes = tuple(str(x) for x in getattr(cell, "parameter_dtypes", ()))
        if len(names) != len(dtypes):
            raise SemanticContractError(
                f"{getattr(cell, 'uid', '<cell>')}: parameter dtype arity mismatch"
            )
        return tuple(zip(names, dtypes))
    return tuple((str(row.name), str(row.dtype)) for row in parameters)


def _cell_result_dtype(cell: Any) -> str:
    if hasattr(cell, "result_dtype"):
        return str(cell.result_dtype)
    return str(cell.dtype)


def _cell_result_geometry(cell: Any) -> tuple[int, tuple[int, ...] | None, str]:
    if hasattr(cell, "result_ndim"):
        ndim = int(cell.result_ndim)
        shape = getattr(cell, "result_shape", None)
        source = str(getattr(cell, "result_shape_source", "frontend"))
    else:
        ndim = int(getattr(cell, "logical_result_ndim", 0))
        shape = getattr(cell, "logical_result_shape", None)
        source = "NativeFormulaCell.logical_result_shape"
    return ndim, _normalized_shape(ndim, shape), source


def _make_axes_for_cell(
    *,
    plan: Any,
    uid: str,
    ndim: int,
    shape: tuple[int, ...] | None,
    shape_source: str,
    axes: dict[str, AxisDomain],
) -> tuple[AxisDomain, ...]:
    if ndim == 0:
        return ()

    ranked = getattr(plan, "ranked_output", None)
    if uid == getattr(plan, "output_uid", None) and ranked is not None:
        if int(ranked.ndim) != ndim:
            raise SemanticContractError(
                f"{uid}: ranked-output ndim {ranked.ndim} disagrees with Cell ndim {ndim}"
            )
        ranked_shape = _normalized_shape(int(ranked.ndim), ranked.shape)
        if ranked_shape != shape:
            raise SemanticContractError(
                f"{uid}: ranked-output shape {ranked_shape!r} disagrees with Cell shape {shape!r}"
            )
        if ndim != 1:
            raise SemanticContractError(
                f"{uid}: FastRankedOutput currently promises one point axis, got ndim={ndim}"
            )
        if len(tuple(ranked.axis_keys)) != int(ranked.shape[0]):
            raise SemanticContractError(
                f"{uid}: ranked-output axis key count does not match shape"
            )
        # Axis identity is semantic/provenance identity, not a hash of labels.
        # The current facade knows exactly one authoritative ranked axis: the
        # selected output point axis.  Keep its canonical Cell UID in the ID so
        # manifest generation remains O(1) in portfolio size.
        axis_id = f"axis_point_{uid}"
        axis = AxisDomain(
            axis_id=axis_id,
            kind="point",
            size=int(ranked.shape[0]),
            keys=tuple(ranked.axis_keys),
            provenance=tuple(
                x for x in (
                    str(ranked.authority),
                    None if ranked.point_frame_ref is None else f"point_frame:{ranked.point_frame_ref}",
                ) if x is not None
            ),
        )
        prior = axes.get(axis_id)
        if prior is not None and prior != axis:
            raise SemanticContractError(f"axis identity collision for {axis_id}")
        axes[axis_id] = axis
        return (axis,)

    result: list[AxisDomain] = []
    for pos in range(ndim):
        size = None if shape is None else int(shape[pos])
        axis_id = f"axis_unresolved_{uid}_{pos}"
        axis = AxisDomain(
            axis_id=axis_id,
            kind="unresolved",
            size=size,
            provenance=(
                f"legacy_rank:{ndim}",
                f"legacy_shape_source:{shape_source}",
                "identity_deliberately_not_inferred_from_equal_length",
            ),
        )
        axes[axis_id] = axis
        result.append(axis)
    return tuple(result)


def _signature_from_cell(plan: Any, cell: Any, axes: dict[str, AxisDomain]) -> CellSignature:
    uid = str(cell.uid)
    category_by_param = _category_parameter_map(cell)
    parameters = tuple(
        ParameterSpec(
            name=name,
            value=ValueSpec(
                logical_dtype=_semantic_dtype(dtype, category_by_param.get(name)),
                category_domain_id=category_by_param.get(name),
            ),
        )
        for name, dtype in _cell_parameter_rows(cell)
    )
    result_domain = getattr(cell, "result_category_domain", None)
    ndim, shape, shape_source = _cell_result_geometry(cell)
    result_axes = _make_axes_for_cell(
        plan=plan,
        uid=uid,
        ndim=ndim,
        shape=shape,
        shape_source=shape_source,
        axes=axes,
    )
    result = ValueSpec(
        logical_dtype=_semantic_dtype(_cell_result_dtype(cell), result_domain),
        axes=result_axes,
        category_domain_id=result_domain,
    )
    return CellSignature(
        uid=uid,
        source_name=str(cell.source_name),
        source_fullname=str(cell.source_fullname),
        parameters=parameters,
        result=result,
    )


def _lowering_from_execution_row(row: Any) -> LoweringPlan:
    memo = row.memo
    execution_class = str(row.execution_class)
    return LoweringPlan(
        uid=str(row.uid),
        execution_class=execution_class,
        execution_backend=("python_semantics" if execution_class == "P" else "native"),
        parameter_abi_dtypes=tuple(str(x) for x in row.parameter_dtypes),
        result_abi_dtype=str(row.result_dtype),
        result_value_abi=str(row.value_abi),
        native_scalar_abi=bool(row.native_scalar_abi),
        recursive_value_bridgeable=bool(row.bridgeable_python_fallback),
        gil_required=(execution_class == "P"),
        memo_kind=str(memo.kind),
        memo_eligible=bool(memo.eligible),
        memo_physicalization_ready=bool(memo.physicalization_ready),
        memo_bounds=tuple((int(lo), int(hi)) for lo, hi in memo.bounds),
        memo_size=int(memo.size),
        memo_hash_capacity=int(memo.hash_capacity),
        direct_blockers=tuple((str(k), str(v)) for k, v in row.direct_blockers),
        classification_reasons=tuple(str(x) for x in row.classification_reasons),
    )


def _validate_frontend_vs_catalog(frontend: Mapping[str, Any], catalog: Mapping[str, Any]) -> int:
    checked = 0
    for uid in sorted(set(frontend) & set(catalog)):
        left = frontend[uid]
        right = catalog[uid]
        if str(left.source_name) != str(right.source_name):
            raise SemanticContractError(f"{uid}: frontend/catalog source_name mismatch")
        if str(left.source_fullname) != str(right.source_fullname):
            raise SemanticContractError(f"{uid}: frontend/catalog source_fullname mismatch")
        if _cell_result_dtype(left) != _cell_result_dtype(right):
            raise SemanticContractError(
                f"{uid}: frontend/catalog result dtype mismatch: "
                f"{_cell_result_dtype(left)!r} != {_cell_result_dtype(right)!r}"
            )
        if _cell_parameter_rows(left) != _cell_parameter_rows(right):
            raise SemanticContractError(f"{uid}: frontend/catalog parameter signature mismatch")
        if getattr(left, "result_category_domain", None) != getattr(right, "result_category_domain", None):
            raise SemanticContractError(f"{uid}: frontend/catalog result category mismatch")
        if _category_parameter_map(left) != _category_parameter_map(right):
            raise SemanticContractError(f"{uid}: frontend/catalog parameter category mismatch")
        lndim, lshape, _ = _cell_result_geometry(left)
        rndim, rshape, _ = _cell_result_geometry(right)
        if lndim != rndim:
            raise SemanticContractError(f"{uid}: frontend/catalog ndim mismatch")
        if lshape != rshape:
            raise SemanticContractError(
                f"{uid}: frontend/catalog shape mismatch: {lshape!r} != {rshape!r}"
            )
        checked += 1
    return checked


def _validate_signature_against_cell(signature: CellSignature, cell: Any) -> None:
    expected_params = _cell_parameter_rows(cell)
    if tuple(p.name for p in signature.parameters) != tuple(name for name, _ in expected_params):
        raise SemanticContractError(f"{signature.uid}: semantic parameter names disagree with legacy Cell")
    domains = _category_parameter_map(cell)
    for param, (name, dtype) in zip(signature.parameters, expected_params):
        domain = domains.get(name)
        if param.value.category_domain_id != domain:
            raise SemanticContractError(f"{signature.uid}.{name}: semantic category domain mismatch")
        if param.value.logical_dtype != _semantic_dtype(dtype, domain):
            raise SemanticContractError(f"{signature.uid}.{name}: semantic dtype mismatch")
        if param.value.ndim != 0:
            raise SemanticContractError(f"{signature.uid}.{name}: Cell parameter unexpectedly ranked")
    result_domain = getattr(cell, "result_category_domain", None)
    if signature.result.category_domain_id != result_domain:
        raise SemanticContractError(f"{signature.uid}: semantic result category domain mismatch")
    if signature.result.logical_dtype != _semantic_dtype(_cell_result_dtype(cell), result_domain):
        raise SemanticContractError(f"{signature.uid}: semantic result dtype mismatch")
    ndim, shape, _ = _cell_result_geometry(cell)
    if signature.result.ndim != ndim:
        raise SemanticContractError(f"{signature.uid}: semantic result rank mismatch")
    if signature.result.shape != shape:
        raise SemanticContractError(
            f"{signature.uid}: semantic result shape mismatch: {signature.result.shape!r} != {shape!r}"
        )


def _validate_lowering(signature: CellSignature, lowering: LoweringPlan, row: Any) -> None:
    if lowering.uid != signature.uid or lowering.uid != str(row.uid):
        raise SemanticContractError(f"{signature.uid}: lowering UID mismatch")
    if lowering.parameter_abi_dtypes != tuple(str(x) for x in row.parameter_dtypes):
        raise SemanticContractError(f"{signature.uid}: lowering parameter ABI mismatch")
    if len(lowering.parameter_abi_dtypes) != len(signature.parameters):
        raise SemanticContractError(f"{signature.uid}: semantic/physical parameter arity mismatch")
    if lowering.result_abi_dtype != str(row.result_dtype):
        raise SemanticContractError(f"{signature.uid}: lowering result dtype mismatch")
    if lowering.result_value_abi != str(row.value_abi):
        raise SemanticContractError(f"{signature.uid}: lowering value ABI mismatch")
    if lowering.execution_class != str(row.execution_class):
        raise SemanticContractError(f"{signature.uid}: lowering execution-class mismatch")
    if lowering.memo_kind != str(row.memo.kind):
        raise SemanticContractError(f"{signature.uid}: lowering memo-kind mismatch")
    # Category semantics must stay distinct from their current int64 code ABI.
    for param, abi_dtype in zip(signature.parameters, lowering.parameter_abi_dtypes):
        if param.value.category_domain_id is not None and abi_dtype != "int64":
            raise SemanticContractError(
                f"{signature.uid}.{param.name}: category parameter is not lowered to int64"
            )
    if signature.result.category_domain_id is not None:
        if lowering.result_abi_dtype != "int64":
            raise SemanticContractError(
                f"{signature.uid}: category result is not lowered to int64"
            )


def build_semantic_facade(plan: Any) -> SemanticFacadePlan:
    """Derive and validate the Phase-A semantic facade for one fast plan."""
    frontend = _frontend_cells(plan)
    catalog = dict(getattr(plan, "source_transparent_cell_catalog", {}) or {})
    execution = getattr(plan, "execution_class_plan", None)

    checks: list[str] = []
    overlap = _validate_frontend_vs_catalog(frontend, catalog)
    checks.append(f"frontend_catalog_cells:{overlap}")

    axes: dict[str, AxisDomain] = {}
    signatures: dict[str, CellSignature] = {}
    all_uids = sorted(set(frontend) | set(catalog))
    for uid in all_uids:
        source = frontend.get(uid, catalog.get(uid))
        if source is None:  # pragma: no cover - set union makes this impossible.
            raise SemanticContractError(f"{uid}: missing semantic source Cell")
        signature = _signature_from_cell(plan, source, axes)
        _validate_signature_against_cell(signature, source)
        other = catalog.get(uid) if source is frontend.get(uid) else frontend.get(uid)
        if other is not None:
            _validate_signature_against_cell(signature, other)
        signatures[uid] = signature
    checks.append(f"signature_legacy_cells:{len(signatures)}")

    lowerings: dict[str, LoweringPlan] = {}
    if execution is not None:
        for uid, row in sorted(execution.cells.items()):
            signature = signatures.get(uid)
            if signature is None:
                raise SemanticContractError(
                    f"{uid}: execution-class Cell has no semantic signature"
                )
            lowering = _lowering_from_execution_row(row)
            _validate_lowering(signature, lowering, row)
            lowerings[uid] = lowering
        checks.append(f"execution_class_cells:{len(lowerings)}")
    else:
        checks.append("execution_class_cells:0")

    ranked = getattr(plan, "ranked_output", None)
    if ranked is not None:
        output = signatures.get(str(plan.output_uid))
        if output is None:
            raise SemanticContractError("ranked output has no semantic signature")
        if output.result.ndim != int(ranked.ndim):
            raise SemanticContractError("ranked output semantic rank mismatch")
        if output.result.shape != _normalized_shape(int(ranked.ndim), ranked.shape):
            raise SemanticContractError("ranked output semantic shape mismatch")
        if output.result.logical_dtype != _semantic_dtype(str(ranked.dtype), output.result.category_domain_id):
            raise SemanticContractError("ranked output semantic dtype mismatch")
        if output.result.ndim == 1:
            axis = output.result.axes[0]
            if axis.kind != "point" or axis.keys != tuple(ranked.axis_keys):
                raise SemanticContractError("ranked output point-axis evidence mismatch")
        checks.append("ranked_output_geometry:validated")
    else:
        checks.append("ranked_output_geometry:not_present")

    # Validate every referenced category domain exists in the current category
    # plan when such a plan is available.  We keep only IDs in ValueSpec so the
    # category ABI remains the single owner of labels/codes during migration.
    category_plan = getattr(plan, "category_plan", None)
    if category_plan is not None:
        known_domains = set(category_plan.encoding.domains)
        referenced = {
            value.category_domain_id
            for signature in signatures.values()
            for value in (
                signature.result,
                *(param.value for param in signature.parameters),
            )
            if value.category_domain_id is not None
        }
        missing = sorted(referenced - known_domains)
        if missing:
            raise SemanticContractError(
                "semantic facade references unknown category domains: " + ",".join(missing)
            )
        checks.append(f"category_domains:{len(referenced)}")
    else:
        checks.append("category_domains:0")

    return SemanticFacadePlan(
        schema="modelx_graph.semantic_facade.v1",
        axes=dict(axes),
        signatures=dict(signatures),
        lowerings=dict(lowerings),
        validation_checks=tuple(checks),
    )


__all__ = [
    "SemanticContractError",
    "AxisDomain",
    "ValueSpec",
    "ParameterSpec",
    "CellSignature",
    "LoweringPlan",
    "SemanticFacadePlan",
    "build_semantic_facade",
]
