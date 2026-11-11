from __future__ import annotations

import ast
import builtins
import dis
import inspect
import textwrap
import numbers
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .slot_storage import FormulaSlotLayout


class NativeAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeTypeEvidence:
    dtype: str
    sample_count: int
    type_names: tuple[str, ...]


@dataclass(frozen=True)
class FormulaNativeAnalysis:
    formula_op_id: int
    name: str
    fullname: str
    produced_count: int
    physical_slots: int
    storage_class: str
    observed_return_type: NativeTypeEvidence
    native_capable: bool
    tier: str
    reasons: tuple[str, ...]
    ast_node_counts: tuple[tuple[str, int], ...]
    scheduled_dependency_calls: int = 0
    python_boundary_calls: int = 0


@dataclass(frozen=True)
class NativePlanAnalysis:
    formulae: tuple[FormulaNativeAnalysis, ...]
    total_formula_ops: int
    total_executed_ops: int
    native_formula_ops: int
    native_executed_ops: int
    python_formula_ops: int
    python_executed_ops: int
    reason_counts: tuple[tuple[str, int], ...]

    @property
    def weighted_native_fraction(self) -> float:
        return self.native_executed_ops / self.total_executed_ops if self.total_executed_ops else 0.0

    @property
    def formula_native_fraction(self) -> float:
        return self.native_formula_ops / self.total_formula_ops if self.total_formula_ops else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_formula_ops": self.total_formula_ops,
            "total_executed_ops": self.total_executed_ops,
            "native_formula_ops": self.native_formula_ops,
            "native_executed_ops": self.native_executed_ops,
            "python_formula_ops": self.python_formula_ops,
            "python_executed_ops": self.python_executed_ops,
            "weighted_native_fraction": self.weighted_native_fraction,
            "formula_native_fraction": self.formula_native_fraction,
            "reason_counts": dict(self.reason_counts),
            "formulae": [
                {
                    "formula_op_id": f.formula_op_id,
                    "name": f.name,
                    "fullname": f.fullname,
                    "produced_count": f.produced_count,
                    "physical_slots": f.physical_slots,
                    "storage_class": f.storage_class,
                    "observed_return_type": f.observed_return_type.dtype,
                    "native_capable": f.native_capable,
                    "tier": f.tier,
                    "reasons": list(f.reasons),
                    "scheduled_dependency_calls": f.scheduled_dependency_calls,
                    "python_boundary_calls": f.python_boundary_calls,
                    "ast_node_counts": dict(f.ast_node_counts),
                }
                for f in self.formulae
            ],
        }


_ALLOWED_BUILTIN_CALLS = {"abs", "min", "max", "float", "int", "bool"}
_ALLOWED_MATH_CALLS = {
    "ceil", "floor", "sqrt", "exp", "log", "log1p", "sin", "cos", "tan",
    "fabs", "pow",
}
_SUPPORTED_NODE_TYPES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg,
    ast.Return, ast.If, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr,
    ast.Load, ast.Store, ast.Name, ast.Constant,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.keyword,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
}
_UNSUPPORTED_STATEMENTS = (
    ast.For, ast.While, ast.With, ast.Try, ast.Raise, ast.Import, ast.ImportFrom,
    ast.Global, ast.Nonlocal, ast.Delete, ast.ClassDef, ast.AsyncFunctionDef,
)
_UNSUPPORTED_EXPRESSIONS = (
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda,
    ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr,
)


def _observed_dtype(values: Iterable[Any]) -> NativeTypeEvidence:
    vals = list(values)
    types = tuple(sorted({type(v).__name__ for v in vals}))
    if not vals:
        return NativeTypeEvidence("object", 0, ())
    if all(isinstance(v, (bool, np.bool_)) for v in vals):
        return NativeTypeEvidence("bool", len(vals), types)
    # Accept Python and NumPy scalar numerics through the abstract numeric ABCs.
    # bool/np.bool_ are deliberately excluded after the homogeneous-bool case.
    if all(isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_)) for v in vals):
        return NativeTypeEvidence("int64", len(vals), types)
    if all(isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)) for v in vals):
        return NativeTypeEvidence("float64", len(vals), types)
    return NativeTypeEvidence("object", len(vals), types)


def _parse_source(source: str) -> ast.FunctionDef | None:
    if not source or not source.strip():
        return None
    try:
        mod = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    funcs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return None
    return funcs[0]


def _function_def_for_obj(obj: Any) -> ast.FunctionDef | None:
    source = getattr(getattr(obj, "schema", None), "source", None)
    # ConcreteNode.obj is CellsImpl; source is attached to ConcreteNode.schema, not obj.
    return None


def _looks_like_modelx_cell(value: Any) -> bool:
    if inspect.ismethod(value):
        owner = getattr(value, "__self__", None)
        func = getattr(value, "__func__", None)
        return owner is not None and getattr(func, "__name__", None) == "call" and hasattr(owner, "data")
    if hasattr(value, "_impl") and value.__class__.__name__ == "Cells":
        return True
    return False


def _is_math_module(value: Any) -> bool:
    return getattr(value, "__name__", None) == "math"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return f"{node.func.value.id}.{node.func.attr}"
    return None


def _global_write_guard(fn: Any) -> bool:
    try:
        return any(ins.opname in {"STORE_GLOBAL", "DELETE_GLOBAL"} for ins in dis.get_instructions(fn))
    except Exception:
        return True


def _ast_capability(fn_def: ast.FunctionDef, globals_: dict[str, Any]) -> tuple[bool, tuple[str, ...], Counter[str], int, int]:
    reasons: list[str] = []
    counts: Counter[str] = Counter(type(n).__name__ for n in ast.walk(fn_def))
    scheduled_calls = 0
    python_boundary_calls = 0

    for n in ast.walk(fn_def):
        if type(n) not in _SUPPORTED_NODE_TYPES:
            if isinstance(n, _UNSUPPORTED_STATEMENTS):
                reasons.append(f"unsupported_statement:{type(n).__name__}")
            elif isinstance(n, _UNSUPPORTED_EXPRESSIONS):
                reasons.append(f"unsupported_expression:{type(n).__name__}")
            elif isinstance(n, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
                reasons.append(f"python_container:{type(n).__name__}")
            elif isinstance(n, ast.Subscript):
                reasons.append("subscript")
            elif isinstance(n, ast.Attribute):
                # math.foo is allowed only as the callee of a Call and handled below.
                reasons.append("attribute")
            else:
                reasons.append(f"unsupported_ast:{type(n).__name__}")

        if isinstance(n, ast.Call):
            name = _call_name(n)
            if name is None:
                reasons.append("dynamic_call")
                python_boundary_calls += 1
                continue
            if isinstance(n.func, ast.Name):
                callee = globals_.get(n.func.id, getattr(builtins, n.func.id, None))
                if n.func.id in _ALLOWED_BUILTIN_CALLS:
                    continue
                if _looks_like_modelx_cell(callee):
                    scheduled_calls += 1
                    continue
                reasons.append(f"python_call:{n.func.id}")
                python_boundary_calls += 1
                continue
            if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
                owner = globals_.get(n.func.value.id)
                if _is_math_module(owner) and n.func.attr in _ALLOWED_MATH_CALLS:
                    # Attribute node itself is not a reason in this special case.
                    try:
                        reasons.remove("attribute")
                    except ValueError:
                        pass
                    continue
                reasons.append(f"python_method_call:{name}")
                python_boundary_calls += 1

    # Compress reasons while preserving a stable order.
    seen: set[str] = set()
    unique = []
    for r in reasons:
        if r not in seen:
            unique.append(r); seen.add(r)
    return not unique, tuple(unique), counts, scheduled_calls, python_boundary_calls


def analyze_native_capability(compiler: Any) -> NativePlanAnalysis:
    # Native analysis needs the observed reference-cache values.  lower_slots()
    # builds the physical address plan without preparing/clearing the modelx
    # cache; lower_direct_slots() would intentionally clear it.
    if compiler.slots is None:
        compiler.lower_slots()
    assert compiler.structured is not None and compiler.storage is not None and compiler.slots is not None
    plan = compiler.structured.canonical_plan
    if plan is None:
        raise NativeAnalysisError("canonical execution plan is required")

    by_id = compiler.sequential.node_by_id
    node_by_formula: dict[int, list[Any]] = defaultdict(list)
    role_to_op_id = {op.role: op.op_id for op in plan.formula_ops}
    for op in compiler.sequential.ops:
        node = by_id[op.node_id]
        fid = role_to_op_id[node.shape_token]
        node_by_formula[fid].append(node)

    formula_layouts = compiler.slots.layout.formula_layout_by_id
    formulae: list[FormulaNativeAnalysis] = []
    reason_counter: Counter[str] = Counter()
    native_ops = 0

    for formula in plan.formula_ops:
        rows = node_by_formula.get(formula.op_id, [])
        values = []
        for node in rows:
            try:
                if node.args in node.obj.data:
                    values.append(node.obj.data[node.args])
            except Exception:
                pass
        evidence = _observed_dtype(values)
        layout: FormulaSlotLayout | None = formula_layouts.get(formula.op_id)
        storage_class = layout.storage_class if layout is not None else "unknown"
        slot_count = layout.slot_count if layout is not None else 0

        reasons: list[str] = []
        native = evidence.dtype in {"bool", "int64", "float64"}
        if not native:
            reasons.append("object_return")

        fn_def = None
        obj = rows[0].obj if rows else None
        globals_ = {}
        if obj is None:
            reasons.append("no_realized_nodes")
        else:
            source = rows[0].schema.source
            fn_def = _parse_source(source)
            if fn_def is None:
                reasons.append("unparseable_source")
            try:
                globals_ = obj.altfunc.__globals__
            except Exception:
                globals_ = {}
            if _global_write_guard(getattr(obj, "altfunc", None)):
                reasons.append("global_write_or_unknown_bytecode")

        counts: Counter[str] = Counter()
        scheduled_calls = 0
        py_calls = 0
        if fn_def is not None:
            ast_ok, ast_reasons, counts, scheduled_calls, py_calls = _ast_capability(fn_def, globals_)
            if not ast_ok:
                reasons.extend(ast_reasons)
                native = False
        else:
            native = False
        if reasons:
            native = False
        tier = "NATIVE" if native else "PYTHON"
        if native:
            native_ops += len(rows)
        for r in reasons:
            reason_counter[r] += len(rows) or 1
        formulae.append(
            FormulaNativeAnalysis(
                formula_op_id=formula.op_id,
                name=formula.name,
                fullname=formula.fullname,
                produced_count=len(rows),
                physical_slots=slot_count,
                storage_class=storage_class,
                observed_return_type=evidence,
                native_capable=native,
                tier=tier,
                reasons=tuple(dict.fromkeys(reasons)),
                ast_node_counts=tuple(sorted(counts.items())),
                scheduled_dependency_calls=scheduled_calls,
                python_boundary_calls=py_calls,
            )
        )

    total_ops = sum(f.produced_count for f in formulae)
    native_formula_ops = sum(1 for f in formulae if f.native_capable)
    return NativePlanAnalysis(
        formulae=tuple(formulae),
        total_formula_ops=len(formulae),
        total_executed_ops=total_ops,
        native_formula_ops=native_formula_ops,
        native_executed_ops=native_ops,
        python_formula_ops=len(formulae) - native_formula_ops,
        python_executed_ops=total_ops - native_ops,
        reason_counts=tuple(sorted(reason_counter.items())),
    )
