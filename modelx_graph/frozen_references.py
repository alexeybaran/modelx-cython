from __future__ import annotations

import ast
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np

from .native_family import NativeFamilyError, _RuntimeFormulaEmitter, _impl_from_global


class FrozenReferenceError(RuntimeError):
    """Fail-closed error while freezing a numeric external reference."""


@dataclass(frozen=True)
class FrozenAxisPlan:
    """Exact finite integer-key -> physical-position mapping for one axis.

    ``key_mode`` describes only source representation, not domain semantics:
    ``int`` means the formula supplies an integer-like label directly;
    ``stringified_int`` means the source labels are strings whose full contents
    parse uniquely as base-10 integers and the formula supplies ``str(expr)``.
    ``position`` is used for NumPy positional indexing.
    """

    key_mode: str
    lo: int
    hi: int
    positions: np.ndarray
    size: int

    def position(self, key: int) -> int:
        key = int(key)
        if self.key_mode == "position":
            if key < 0:
                key += self.size
            if key < 0 or key >= self.size:
                raise IndexError(key)
            return key
        err_key: Any = str(key) if self.key_mode == "stringified_int" else key
        if key < self.lo or key > self.hi:
            raise KeyError(err_key)
        pos = int(self.positions[key - self.lo])
        if pos < 0:
            raise KeyError(err_key)
        return pos


@dataclass(frozen=True)
class FrozenNumericReferenceSite:
    site_id: int
    formula_op_id: int
    global_name: str
    source_kind: str  # ndarray_1d | ndarray_2d | series | dataframe
    dtype: str        # float64 | int64 | bool
    ndim: int
    shape: tuple[int, ...]
    lookup_ast: str
    axis0: FrozenAxisPlan
    axis1: FrozenAxisPlan | None
    values: np.ndarray
    source_object_id: int
    occurrence_count: int
    scope: str = "shared"


@dataclass(frozen=True)
class FrozenNumericReferencePlan:
    sites: tuple[FrozenNumericReferenceSite, ...]
    rejected: tuple[tuple[int, str, str], ...]

    @property
    def site_count(self) -> int:
        return len(self.sites)

    @property
    def formula_op_ids(self) -> tuple[int, ...]:
        return tuple(sorted({x.formula_op_id for x in self.sites}))

    @property
    def boundary_occurrences(self) -> int:
        return sum(x.occurrence_count for x in self.sites)

    def sites_for_formula(self, formula_op_id: int) -> tuple[FrozenNumericReferenceSite, ...]:
        return tuple(x for x in self.sites if x.formula_op_id == formula_op_id)

    def payload(self) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray | None], ...]:
        return tuple((x.values, x.axis0.positions, None if x.axis1 is None else x.axis1.positions) for x in self.sites)


@dataclass(frozen=True)
class _LookupCandidate:
    global_name: str
    node: ast.Subscript
    row_key: ast.AST
    col_key: ast.AST | None
    source_kind: str
    source: Any


def _dtype_name(dtype: np.dtype) -> str | None:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64):
        return "float64"
    if dtype == np.dtype(np.int64):
        return "int64"
    if dtype == np.dtype(np.bool_):
        return "bool"
    return None


def _parse_decimal_int_label(value: Any) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = int(value, 10)
    except Exception:
        return None
    # ``str(integer_expression)`` produces Python's canonical decimal spelling.
    # A label such as "01" therefore must not be silently treated as "1".
    if str(parsed) != value:
        return None
    return parsed


def _axis_plan(labels: list[Any], *, position: bool = False) -> FrozenAxisPlan:
    if position:
        if not labels:
            return FrozenAxisPlan("position", 0, -1, np.empty(0, dtype=np.int64), 0)
        arr = np.arange(len(labels), dtype=np.int64)
        return FrozenAxisPlan("position", 0, len(labels) - 1, arr, len(labels))

    if len(set(labels)) != len(labels):
        raise FrozenReferenceError("duplicate pandas labels are not supported by the first frozen-reference slice")
    if all(isinstance(x, (numbers.Integral, np.integer)) and not isinstance(x, (bool, np.bool_)) for x in labels):
        keys = [int(x) for x in labels]
        mode = "int"
    else:
        parsed = [_parse_decimal_int_label(x) for x in labels]
        if any(x is None for x in parsed):
            raise FrozenReferenceError("axis labels are not integer or decimal-integer strings")
        keys = [int(x) for x in parsed]
        if len(set(keys)) != len(keys):
            raise FrozenReferenceError("decimal-string labels collide after integer parsing")
        mode = "stringified_int"
    if not keys:
        return FrozenAxisPlan(mode, 0, -1, np.empty(0, dtype=np.int64), 0)
    lo, hi = min(keys), max(keys)
    span = hi - lo + 1
    # Avoid materializing absurdly sparse label maps. The fallback remains Python.
    if span > max(4096, len(keys) * 16):
        raise FrozenReferenceError("integer label domain is too sparse for the first dense map ABI")
    positions = np.full(span, -1, dtype=np.int64)
    for pos, key in enumerate(keys):
        positions[key - lo] = pos
    return FrozenAxisPlan(mode, lo, hi, positions, len(keys))


def _root_name(node: ast.AST) -> str | None:
    cur = node
    while isinstance(cur, ast.Subscript):
        cur = cur.value
    return cur.id if isinstance(cur, ast.Name) else None


def _find_lookup_candidates(fn: ast.FunctionDef, globals_dict: dict[str, Any]) -> list[_LookupCandidate]:
    try:
        import pandas as pd
    except Exception:  # pragma: no cover
        pd = None
    out: list[_LookupCandidate] = []
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(fn):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    for node in ast.walk(fn):
        if not isinstance(node, ast.Subscript):
            continue
        root = _root_name(node)
        if root is None or root not in globals_dict:
            continue
        source = globals_dict[root]
        # Ignore the inner half of a nested DataFrame[col][row] expression.
        parent = parents.get(id(node))
        if isinstance(parent, ast.Subscript) and _root_name(parent) == root:
            continue
        if isinstance(source, np.ndarray):
            if source.ndim == 1 and isinstance(node.value, ast.Name):
                out.append(_LookupCandidate(root, node, node.slice, None, "ndarray_1d", source))
            elif source.ndim == 2 and isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                out.append(_LookupCandidate(root, node, node.slice.elts[0], node.slice.elts[1], "ndarray_2d", source))
            continue
        if pd is not None and isinstance(source, pd.Series) and isinstance(node.value, ast.Name):
            out.append(_LookupCandidate(root, node, node.slice, None, "series", source))
            continue
        if pd is not None and isinstance(source, pd.DataFrame):
            # pandas expression df[column_label][row_label]
            if isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name) and node.value.value.id == root:
                out.append(_LookupCandidate(root, node, node.slice, node.value.slice, "dataframe", source))
    return out


def _formula_call_dtype(compiler: Any, globals_dict: dict[str, Any], name: str) -> str | None:
    impl = _impl_from_global(globals_dict.get(name))
    if impl is None or compiler.native_plan is None or compiler.structured is None:
        return None
    canonical = compiler.structured.canonical_plan
    if canonical is None:
        return None
    by_fid = {x.formula_op_id: x for x in compiler.native_plan.formulae}
    role_to_fid = {op.role: fid for fid, op in enumerate(canonical.formula_ops)}
    for node in compiler.trace.nodes:
        if node.obj is impl:
            fid = role_to_fid.get(node.shape_token)
            row = by_fid.get(fid)
            return None if row is None else row.observed_return_type.dtype
    return None


def _integer_expr_evidence(
    compiler: Any,
    formula_op_id: int,
    fn: ast.FunctionDef,
    globals_dict: dict[str, Any],
    node: ast.AST,
) -> bool:
    formals = [x.arg for x in fn.args.args]
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        if node.id in formals:
            index = formals.index(node.id)
            canonical = compiler.structured.canonical_plan
            if canonical is None:
                return False
            role = canonical.formula_ops[formula_op_id].role
            values = [n.args[index] for n in compiler.trace.nodes if n.shape_token == role and len(n.args) > index]
            return bool(values) and all(
                isinstance(v, (numbers.Integral, np.integer)) and not isinstance(v, (bool, np.bool_))
                for v in values
            )
        value = globals_dict.get(node.id, object())
        return isinstance(value, (numbers.Integral, np.integer)) and not isinstance(value, (bool, np.bool_))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name == "int" and len(node.args) == 1 and not node.keywords:
            return True
        if _formula_call_dtype(compiler, globals_dict, name) == "int64":
            return True
        if name in {"min", "max"} and node.args and not node.keywords:
            return all(_integer_expr_evidence(compiler, formula_op_id, fn, globals_dict, x) for x in node.args)
        return False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _integer_expr_evidence(compiler, formula_op_id, fn, globals_dict, node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
        return (
            _integer_expr_evidence(compiler, formula_op_id, fn, globals_dict, node.left)
            and _integer_expr_evidence(compiler, formula_op_id, fn, globals_dict, node.right)
        )
    if isinstance(node, ast.IfExp):
        return (
            _integer_expr_evidence(compiler, formula_op_id, fn, globals_dict, node.body)
            and _integer_expr_evidence(compiler, formula_op_id, fn, globals_dict, node.orelse)
        )
    return False


def _key_ast_compatible(
    compiler: Any,
    formula_op_id: int,
    fn: ast.FunctionDef,
    globals_dict: dict[str, Any],
    node: ast.AST,
    mode: str,
) -> bool:
    if mode in {"int", "position"}:
        return _integer_expr_evidence(compiler, formula_op_id, fn, globals_dict, node)
    if mode == "stringified_int":
        return (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "str" and len(node.args) == 1 and not node.keywords
            and _integer_expr_evidence(compiler, formula_op_id, fn, globals_dict, node.args[0])
        )
    return False

def _values_and_axes(candidate: _LookupCandidate) -> tuple[np.ndarray, FrozenAxisPlan, FrozenAxisPlan | None, str]:
    source = candidate.source
    if candidate.source_kind.startswith("ndarray"):
        arr = np.asarray(source)
        dtype = _dtype_name(arr.dtype)
        if dtype is None:
            raise FrozenReferenceError(f"unsupported numeric ndarray dtype {arr.dtype}")
        arr = np.array(arr, copy=True, order="C")
        axis0 = _axis_plan(list(range(arr.shape[0])), position=True)
        axis1 = _axis_plan(list(range(arr.shape[1])), position=True) if arr.ndim == 2 else None
        return arr, axis0, axis1, dtype
    if candidate.source_kind == "series":
        arr = np.asarray(source.to_numpy())
        dtype = _dtype_name(arr.dtype)
        if dtype is None:
            raise FrozenReferenceError(f"unsupported numeric Series dtype {arr.dtype}")
        axis0 = _axis_plan(list(source.index))
        return np.array(arr, copy=True, order="C"), axis0, None, dtype
    # DataFrame: runtime storage is [row, column], while source syntax is df[col][row].
    if not source.index.is_unique or not source.columns.is_unique:
        raise FrozenReferenceError("DataFrame index/columns must be unique")
    arr = np.asarray(source.to_numpy())
    dtype = _dtype_name(arr.dtype)
    if dtype is None:
        raise FrozenReferenceError(f"unsupported numeric DataFrame dtype {arr.dtype}")
    axis0 = _axis_plan(list(source.index))
    axis1 = _axis_plan(list(source.columns))
    return np.array(arr, copy=True, order="C"), axis0, axis1, dtype


def build_frozen_numeric_reference_plan(compiler: Any) -> FrozenNumericReferencePlan:
    """Discover narrow exact numeric external-reference lookups from FormulaOps.

    This pass never changes canonical execution. It snapshots only immutable runtime
    payload for lookups whose complete label mapping can be represented exactly.
    Unsupported references are recorded and remain ordinary Python boundaries.
    """
    if compiler.structured is None:
        compiler.recover_loops()
    analysis = compiler.native_plan or compiler.analyze_native()
    canonical = compiler.structured.canonical_plan
    if canonical is None:
        raise FrozenReferenceError("canonical execution plan is required")
    rows = {x.formula_op_id: x for x in analysis.formulae}
    reps: dict[Any, list[Any]] = {}
    for node in compiler.trace.nodes:
        reps.setdefault(node.shape_token, []).append(node)

    sites: list[FrozenNumericReferenceSite] = []
    rejected: list[tuple[int, str, str]] = []
    next_site = 0
    for fid, op in enumerate(canonical.formula_ops):
        row = rows.get(fid)
        if row is None or row.observed_return_type.dtype not in {"float64", "int64", "bool"}:
            continue
        nodes = reps.get(op.role, [])
        if not nodes:
            continue
        rep = nodes[0]
        try:
            fn = ast.parse(rep.schema.source).body[0]
            if not isinstance(fn, ast.FunctionDef):
                continue
        except Exception:
            continue
        glb = rep.obj.altfunc.__globals__
        candidates = _find_lookup_candidates(fn, glb)
        for candidate in candidates:
            try:
                # Shared code can freeze one runtime object only when every concrete
                # occurrence sees the same external object identity in this first slice.
                for other in nodes[1:]:
                    other_glb = other.obj.altfunc.__globals__
                    if candidate.global_name not in other_glb or other_glb[candidate.global_name] is not candidate.source:
                        raise FrozenReferenceError("external reference object changes across concrete formula environments")
                values, axis0, axis1, dtype = _values_and_axes(candidate)
                # Formula source key order is row, col after normalizing pandas syntax.
                if not _key_ast_compatible(compiler, fid, fn, glb, candidate.row_key, axis0.key_mode):
                    raise FrozenReferenceError("row/index key expression does not match frozen label representation")
                if axis1 is not None and (candidate.col_key is None or not _key_ast_compatible(compiler, fid, fn, glb, candidate.col_key, axis1.key_mode)):
                    raise FrozenReferenceError("column key expression does not match frozen label representation")
                sites.append(FrozenNumericReferenceSite(
                    site_id=next_site,
                    formula_op_id=fid,
                    global_name=candidate.global_name,
                    source_kind=candidate.source_kind,
                    dtype=dtype,
                    ndim=values.ndim,
                    shape=tuple(int(x) for x in values.shape),
                    lookup_ast=ast.dump(candidate.node, include_attributes=False),
                    axis0=axis0,
                    axis1=axis1,
                    values=values,
                    source_object_id=id(candidate.source),
                    occurrence_count=int(row.produced_count),
                    scope="shared",
                ))
                next_site += 1
            except Exception as exc:
                rejected.append((fid, candidate.global_name, f"{type(exc).__name__}: {exc}"))
    return FrozenNumericReferencePlan(tuple(sites), tuple(rejected))


def _extract_key_expr(emitter: Any, node: ast.AST, mode: str) -> str:
    if mode == "stringified_int":
        if not (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "str" and len(node.args) == 1 and not node.keywords
        ):
            raise NativeFamilyError("frozen string-label lookup requires str(numeric_expr)")
        node = node.args[0]
    return emitter.expr(node)


def frozen_lookup_expr(emitter: Any, node: ast.AST, sites: tuple[FrozenNumericReferenceSite, ...]) -> str | None:
    dump = ast.dump(node, include_attributes=False)
    for site in sites:
        if site.lookup_ast != dump:
            continue
        # Normalize source syntax to row/index then column.
        if site.source_kind == "dataframe":
            assert isinstance(node, ast.Subscript) and isinstance(node.value, ast.Subscript)
            row_node, col_node = node.slice, node.value.slice
        elif site.source_kind == "ndarray_2d":
            assert isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple)
            row_node, col_node = node.slice.elts
        else:
            assert isinstance(node, ast.Subscript)
            row_node, col_node = node.slice, None
        row = _extract_key_expr(emitter, row_node, site.axis0.key_mode)
        row_pos = f"_frozen_axis0_{site.site_id}(<long long>({row}))"
        if site.axis1 is None:
            return f"_frozen_values_{site.site_id}[{row_pos}]"
        assert col_node is not None
        col = _extract_key_expr(emitter, col_node, site.axis1.key_mode)
        col_pos = f"_frozen_axis1_{site.site_id}(<long long>({col}))"
        return f"_frozen_values_{site.site_id}[{row_pos}, {col_pos}]"
    return None


def frozen_formula_ast_exact_ok(compiler: Any, formula_op_id: int, analysis_row: Any, *, allow_pow: bool = False) -> bool:
    """Exactness guard for formulas promoted only because a frozen lookup is available."""
    if analysis_row.observed_return_type.dtype not in {"float64", "int64", "bool"}:
        return False
    canonical = compiler.structured.canonical_plan
    if canonical is None:
        return False
    op = canonical.formula_ops[formula_op_id]
    rep = next((n for n in compiler.trace.nodes if n.shape_token == op.role), None)
    if rep is None:
        return False
    try:
        fn = ast.parse(rep.schema.source).body[0]
    except Exception:
        return False
    if not isinstance(fn, ast.FunctionDef):
        return False
    if (not allow_pow) and any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) for n in ast.walk(fn)):
        return False
    # Keep the existing conservative local-temporary rule for non-float outputs.
    if analysis_row.observed_return_type.dtype != "float64":
        for n in ast.walk(fn):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(isinstance(t, ast.Name) for t in targets):
                    return False
    return True


class FrozenRuntimeFormulaEmitter(_RuntimeFormulaEmitter):
    def __init__(self, *args, frozen_sites: tuple[FrozenNumericReferenceSite, ...], **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_sites = frozen_sites

    def expr(self, node: ast.AST) -> str:
        frozen = frozen_lookup_expr(self, node, self.frozen_sites)
        if frozen is not None:
            return frozen
        return super().expr(node)


def emit_frozen_reference_support(plan: FrozenNumericReferencePlan) -> list[str]:
    if not plan.sites:
        return []
    lines: list[str] = []
    for site in plan.sites:
        ctype = {"float64": "double", "int64": "long long", "bool": "unsigned char"}[site.dtype]
        if site.ndim == 1:
            lines.append(f"cdef {ctype}[::1] _frozen_values_{site.site_id}")
        else:
            lines.append(f"cdef {ctype}[:, ::1] _frozen_values_{site.site_id}")
        lines.append(f"cdef long long[::1] _frozen_axis0_map_{site.site_id}")
        if site.axis1 is not None:
            lines.append(f"cdef long long[::1] _frozen_axis1_map_{site.site_id}")
        if site.axis0.key_mode == "position":
            lines += [
                f"cdef inline Py_ssize_t _frozen_axis0_{site.site_id}(long long key):",
                f"    if key < 0: key += {site.axis0.size}",
                f"    if key < 0 or key >= {site.axis0.size}: raise IndexError(key)",
                "    return key",
                "",
            ]
        else:
            err0 = "str(key)" if site.axis0.key_mode == "stringified_int" else "key"
            lines += [
                f"cdef inline Py_ssize_t _frozen_axis0_{site.site_id}(long long key):",
                f"    if key < {site.axis0.lo} or key > {site.axis0.hi}: raise KeyError({err0})",
                f"    cdef Py_ssize_t pos = _frozen_axis0_map_{site.site_id}[key - ({site.axis0.lo})]",
                f"    if pos < 0: raise KeyError({err0})",
                "    return pos",
                "",
            ]
        if site.axis1 is not None:
            if site.axis1.key_mode == "position":
                lines += [
                    f"cdef inline Py_ssize_t _frozen_axis1_{site.site_id}(long long key):",
                    f"    if key < 0: key += {site.axis1.size}",
                    f"    if key < 0 or key >= {site.axis1.size}: raise IndexError(key)",
                    "    return key",
                    "",
                ]
            else:
                err1 = "str(key)" if site.axis1.key_mode == "stringified_int" else "key"
                lines += [
                    f"cdef inline Py_ssize_t _frozen_axis1_{site.site_id}(long long key):",
                    f"    if key < {site.axis1.lo} or key > {site.axis1.hi}: raise KeyError({err1})",
                    f"    cdef Py_ssize_t pos = _frozen_axis1_map_{site.site_id}[key - ({site.axis1.lo})]",
                    f"    if pos < 0: raise KeyError({err1})",
                    "    return pos",
                    "",
                ]
    lines += ["cpdef _init_frozen_references(object payload):"]
    if not plan.sites:
        lines.append("    return None")
    for site in plan.sites:
        lines.append(f"    global _frozen_values_{site.site_id}, _frozen_axis0_map_{site.site_id}")
        if site.axis1 is not None:
            lines.append(f"    global _frozen_axis1_map_{site.site_id}")
        lines.append(f"    _frozen_values_{site.site_id} = payload[{site.site_id}][0]")
        lines.append(f"    _frozen_axis0_map_{site.site_id} = payload[{site.site_id}][1]")
        if site.axis1 is not None:
            lines.append(f"    _frozen_axis1_map_{site.site_id} = payload[{site.site_id}][2]")
    lines += ["    return None", ""]
    return lines


def frozen_formula_emitter_ok(
    compiler: Any,
    typed: Any,
    formula_op_id: int,
    reference_sites: tuple[Any, ...],
    plan: FrozenNumericReferencePlan,
    return_dtype: str,
    arg_types: list[tuple[str, str]],
) -> bool:
    sites = plan.sites_for_formula(formula_op_id)
    if not sites or return_dtype not in {"float64", "int64", "bool"}:
        return False
    try:
        emitter = FrozenRuntimeFormulaEmitter(
            compiler, typed, formula_op_id, reference_sites, frozen_sites=sites
        )
        emitter.emit_function(arg_types, return_dtype)
    except Exception:
        return False
    return True
