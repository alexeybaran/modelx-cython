from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .build import build_extension
from .realized_compiler import RealizedTraceCompiler
from .slot_storage import SlotStorageError


class NativeProofError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeSingleRecurrenceProof:
    formula_op_id: int
    formula_name: str
    produced_count: int
    slot_count: int
    pyx_path: str
    so_path: str
    result: float
    reference: float
    exact: bool


_BINOPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
}
_CMPOPS = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
}


def _parse_func(source: str) -> ast.FunctionDef:
    mod = ast.parse(textwrap.dedent(source))
    funcs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        raise NativeProofError("source has no function definition")
    return funcs[0]


def _offset_from_arg(node: ast.AST, arg_name: str) -> int | None:
    if isinstance(node, ast.Name) and node.id == arg_name:
        return 0
    if isinstance(node, ast.BinOp) and isinstance(node.left, ast.Name) and node.left.id == arg_name and isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
        if isinstance(node.op, ast.Sub):
            return -int(node.right.value)
        if isinstance(node.op, ast.Add):
            return int(node.right.value)
    return None


def _expr(node: ast.AST, *, func_name: str, arg_name: str, state_name: str) -> str:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "1" if node.value else "0"
        if isinstance(node.value, (int, float)):
            return repr(float(node.value)) if isinstance(node.value, float) else repr(int(node.value))
        raise NativeProofError(f"unsupported constant {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id == arg_name:
            return arg_name
        if node.id in ("True", "False"):
            return "1" if node.id == "True" else "0"
        raise NativeProofError(f"unsupported name {node.id}")
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return f"(-({_expr(node.operand, func_name=func_name, arg_name=arg_name, state_name=state_name)}))"
        if isinstance(node.op, ast.UAdd):
            return f"(+({_expr(node.operand, func_name=func_name, arg_name=arg_name, state_name=state_name)}))"
        if isinstance(node.op, ast.Not):
            return f"(not ({_expr(node.operand, func_name=func_name, arg_name=arg_name, state_name=state_name)}))"
        raise NativeProofError(f"unsupported unary {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise NativeProofError(f"unsupported binop {type(node.op).__name__}")
        left = _expr(node.left, func_name=func_name, arg_name=arg_name, state_name=state_name)
        right = _expr(node.right, func_name=func_name, arg_name=arg_name, state_name=state_name)
        if isinstance(node.op, ast.Div):
            return f"((<double>({left})) / (<double>({right})))"
        return f"(({left}) {op} ({right}))"
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NativeProofError("chained comparisons not supported in first native proof")
        op = _CMPOPS.get(type(node.ops[0]))
        if op is None:
            raise NativeProofError(f"unsupported comparison {type(node.ops[0]).__name__}")
        left = _expr(node.left, func_name=func_name, arg_name=arg_name, state_name=state_name)
        right = _expr(node.comparators[0], func_name=func_name, arg_name=arg_name, state_name=state_name)
        return f"(({left}) {op} ({right}))"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == func_name:
            if len(node.args) != 1 or node.keywords:
                raise NativeProofError("self call must have exactly one positional arg")
            off = _offset_from_arg(node.args[0], arg_name)
            if off != -1:
                raise NativeProofError("first native proof supports only previous-occurrence self calls")
            return state_name
        if node.func.id in {"float", "int"} and len(node.args) == 1 and not node.keywords:
            return f"(<double>({_expr(node.args[0], func_name=func_name, arg_name=arg_name, state_name=state_name)}))"
        if node.func.id == "abs" and len(node.args) == 1 and not node.keywords:
            inner = _expr(node.args[0], func_name=func_name, arg_name=arg_name, state_name=state_name)
            return f"(({inner}) if ({inner}) >= 0 else -({inner}))"
    raise NativeProofError(f"unsupported expression {ast.dump(node, include_attributes=False)}")


def _emit_assignment_from_return(expr_node: ast.AST, *, func_name: str, arg_name: str, state_name: str, indent: str = "        ") -> str:
    return indent + f"{state_name} = {_expr(expr_node, func_name=func_name, arg_name=arg_name, state_name=state_name)}"


def _emit_body(fn: ast.FunctionDef, *, arg_name: str, state_name: str) -> list[str]:
    body = fn.body
    if len(body) == 2 and isinstance(body[0], ast.If) and isinstance(body[1], ast.Return):
        first = body[0]
        if len(first.body) == 1 and isinstance(first.body[0], ast.Return) and not first.orelse:
            test = _expr(first.test, func_name=fn.name, arg_name=arg_name, state_name=state_name)
            return [
                f"        if {test}:",
                _emit_assignment_from_return(first.body[0].value, func_name=fn.name, arg_name=arg_name, state_name=state_name, indent="            "),
                "        else:",
                _emit_assignment_from_return(body[1].value, func_name=fn.name, arg_name=arg_name, state_name=state_name, indent="            "),
            ]
    if len(body) == 1 and isinstance(body[0], ast.If):
        first = body[0]
        if len(first.body) == 1 and isinstance(first.body[0], ast.Return) and len(first.orelse) == 1 and isinstance(first.orelse[0], ast.Return):
            test = _expr(first.test, func_name=fn.name, arg_name=arg_name, state_name=state_name)
            return [
                f"        if {test}:",
                _emit_assignment_from_return(first.body[0].value, func_name=fn.name, arg_name=arg_name, state_name=state_name, indent="            "),
                "        else:",
                _emit_assignment_from_return(first.orelse[0].value, func_name=fn.name, arg_name=arg_name, state_name=state_name, indent="            "),
            ]
    if len(body) == 1 and isinstance(body[0], ast.Return):
        return [_emit_assignment_from_return(body[0].value, func_name=fn.name, arg_name=arg_name, state_name=state_name)]
    raise NativeProofError("first native proof supports return or if-return/return formula bodies only")


def build_single_formula_recurrence_cython(
    compiler: RealizedTraceCompiler,
    build_dir: str | Path,
    *,
    module_prefix: str = "mxg_native_proof",
) -> NativeSingleRecurrenceProof:
    """Compile a tiny serial Cython proof for one simple realized recurrence.

    This is intentionally narrow. It proves that v0.15's realized FormulaOp and
    PhysicalSlotLayout can drive a C loop without hard-coded Cell names, modelx
    caches, or runtime deletion. Broader mixed lowering belongs to the next stage.
    """
    if compiler.direct_slots is None:
        compiler.lower_direct_slots()
    assert compiler.structured is not None and compiler.storage is not None and compiler.slots is not None
    if not compiler.trace.target_node_ids or compiler.trace.target_node_ids[0] is None:
        raise NativeProofError("single target node is required")
    target_id = compiler.trace.target_node_ids[0]
    target_node = compiler.sequential.node_by_id[target_id]
    plan = compiler.structured.canonical_plan
    if plan is None:
        raise NativeProofError("canonical plan is required")
    role_to_op = {op.role: op for op in plan.formula_ops}
    formula = role_to_op[target_node.shape_token]
    layout = compiler.slots.layout.formula_layout_by_id[formula.op_id]
    if layout.slot_count != 1 or layout.storage_class not in {"bounded_ring", "scalar"}:
        raise NativeProofError("first native proof requires a one-slot recurrence")
    rows = [lt for lt in compiler.storage.plan.lifetimes if lt.formula_op_id == formula.op_id]
    if len(rows) < 2:
        raise NativeProofError("recurrence proof needs at least two realized occurrences")
    nodes = [compiler.sequential.node_by_id[lt.node_id] for lt in rows]
    if any(len(n.args) != 1 or not isinstance(n.args[0], int) for n in nodes):
        raise NativeProofError("first native proof requires exactly one integer argument")
    args = [n.args[0] for n in nodes]
    step_set = {b - a for a, b in zip(args, args[1:])}
    if len(step_set) != 1:
        raise NativeProofError("argument stream must have one affine step")
    step = step_set.pop()
    if step == 0:
        raise NativeProofError("argument stream must move")
    fn = _parse_func(nodes[0].schema.source)
    if len(fn.args.args) != 1:
        raise NativeProofError("formula must have one formal argument")
    arg_name = fn.args.args[0].arg
    body = _emit_body(fn, arg_name=arg_name, state_name="state")
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1((formula.fullname + repr(args) + nodes[0].schema.source).encode()).hexdigest()[:12]
    module_name = f"{module_prefix}_{digest}"
    pyx_path = build_dir / f"{module_name}.pyx"
    lines = [
        "# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False",
        "cpdef double run():",
        "    cdef Py_ssize_t i",
        f"    cdef long {arg_name}",
        "    cdef double state = 0.0",
        f"    for i in range({len(args)}):",
        f"        {arg_name} = {int(args[0])} + i * {int(step)}",
    ]
    lines.extend(body)
    lines.append("    return state")
    pyx_path.write_text("\n".join(lines) + "\n")
    mod, so_path, _proc = build_extension(
        pyx_path,
        module_name,
        build_dir,
        openmp=False,
        optimization="O2",
        native_arch=False,
    )
    result = float(mod.run())
    reference = float(compiler.trace.target_values[0])
    return NativeSingleRecurrenceProof(
        formula_op_id=formula.op_id,
        formula_name=formula.name,
        produced_count=len(rows),
        slot_count=layout.slot_count,
        pyx_path=str(pyx_path),
        so_path=str(so_path),
        result=result,
        reference=reference,
        exact=(result == reference),
    )
