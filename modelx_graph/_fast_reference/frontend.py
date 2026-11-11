from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from dataclasses import dataclass
from types import ModuleType
from typing import Iterable, Any

import numpy as np
import pandas as pd

from .compiler import ModelCompiler, CompileError
from ..semantic_reuse import UnrollLiteralReductions


class ModelxCompileError(CompileError):
    pass


def _interface_value(value: Any) -> Any:
    """Unwrap modelx reference implementations without private imports."""
    try:
        return value.interface
    except Exception:
        return value


def _call_name(node: ast.AST) -> str | None:
    return node.func.id if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) else None


def _unwrap_list_generator(node: ast.AST) -> ast.GeneratorExp | None:
    if isinstance(node, ast.GeneratorExp):
        return node
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "list" and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
        return node.args[0]
    return None


def _time_vector_spec(fn: ast.FunctionDef) -> tuple[str, ast.AST] | None:
    rets = [n for n in fn.body if isinstance(n, ast.Return)]
    if len(rets) != 1:
        return None
    val = rets[0].value
    if not (
        isinstance(val, ast.Call)
        and isinstance(val.func, ast.Attribute)
        and isinstance(val.func.value, ast.Name)
        and val.func.value.id == "np"
        and val.func.attr == "array"
        and len(val.args) == 1
    ):
        return None
    gen = _unwrap_list_generator(val.args[0])
    if gen is None or len(gen.generators) != 1 or not isinstance(gen.generators[0].target, ast.Name):
        return None
    return gen.generators[0].target.id, gen.elt


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _safe_suffix(value: Any) -> str:
    if value is None:
        return "NONE"
    txt = str(value)
    return "".join(ch if ch.isalnum() else "_" for ch in txt).strip("_") or "X"


class _SubstituteNames(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, Any]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return ast.copy_location(ast.Constant(self.mapping[node.id]), node)
        return node




_UnrollLiteralReductions = UnrollLiteralReductions


class _SubstituteExprs(ast.NodeTransformer):
    """Substitute load-name parameters with arbitrary expression ASTs."""

    def __init__(self, mapping: dict[str, ast.AST]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return ast.copy_location(copy.deepcopy(self.mapping[node.id]), node)
        return node


def _primitive_literal(value: Any) -> Any:
    """Return an immutable scalar suitable for compile-time substitution."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return ...


def _linear_inline_expr(
    fn: ast.FunctionDef,
    bound_args: dict[str, ast.AST],
) -> ast.AST | None:
    """Inline a side-effect-free scalar helper into one expression.

    Supported bodies are assignments, a final return, and early-return guards
    (``if test: return value``). This covers the small parameterized arithmetic
    helpers that modelx-cython compiles as typed methods, while deliberately
    rejecting mutation, loops and exception-producing dynamic branches.
    """

    def substitute(node: ast.AST, env: dict[str, ast.AST]) -> ast.AST:
        value = _SubstituteExprs(env).visit(copy.deepcopy(node))
        return ast.fix_missing_locations(value)

    def branch_return(body: list[ast.stmt], env: dict[str, ast.AST]) -> ast.AST | None:
        local = {name: copy.deepcopy(value) for name, value in env.items()}
        meaningful = list(body)
        if (
            meaningful
            and isinstance(meaningful[0], ast.Expr)
            and isinstance(meaningful[0].value, ast.Constant)
            and isinstance(meaningful[0].value.value, str)
        ):
            meaningful = meaningful[1:]
        guards: list[tuple[ast.AST, ast.AST]] = []
        for index, stmt in enumerate(meaningful):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                local[stmt.targets[0].id] = substitute(stmt.value, local)
                continue
            if isinstance(stmt, ast.If):
                cond = substitute(stmt.test, local)
                then_value = branch_return(stmt.body, local)
                else_value = branch_return(stmt.orelse, local) if stmt.orelse else None
                if then_value is not None and else_value is None:
                    guards.append((cond, then_value))
                    continue
                if then_value is not None and else_value is not None and index == len(meaningful) - 1:
                    value: ast.AST = ast.IfExp(cond, then_value, else_value)
                    for guard, guarded_value in reversed(guards):
                        value = ast.IfExp(guard, guarded_value, value)
                    return ast.fix_missing_locations(value)
                return None
            if isinstance(stmt, ast.Return) and index == len(meaningful) - 1:
                value = substitute(stmt.value, local)
                for guard, guarded_value in reversed(guards):
                    value = ast.IfExp(guard, guarded_value, value)
                return ast.fix_missing_locations(value)
            return None
        return None

    env = {name: copy.deepcopy(expr) for name, expr in bound_args.items()}
    return branch_return(list(fn.body), env)

class _ConstantFolder(ast.NodeTransformer):
    """Small, intentionally conservative constant folder for specialized branches."""

    @staticmethod
    def _literal_node(node: ast.AST):
        try:
            value = ast.literal_eval(node)
        except Exception:
            try:
                expr = ast.Expression(body=copy.deepcopy(node))
                value = eval(
                    compile(ast.fix_missing_locations(expr), "<const>", "eval"),
                    {"__builtins__": {}},
                    {},
                )
            except Exception:
                return node
        if value is None or isinstance(value, (bool, int, float, str)):
            return ast.copy_location(ast.Constant(value), node)
        return node

    def visit_BinOp(self, node: ast.BinOp):
        node = self.generic_visit(node)
        return self._literal_node(node)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node = self.generic_visit(node)
        return self._literal_node(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        node = self.generic_visit(node)
        return self._literal_node(node)

    def visit_IfExp(self, node: ast.IfExp):
        node = self.generic_visit(node)
        try:
            test = ast.literal_eval(node.test)
        except Exception:
            return node
        return node.body if bool(test) else node.orelse

    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        try:
            test = ast.literal_eval(node.test)
        except Exception:
            return node
        chosen = node.body if bool(test) else node.orelse
        return chosen

    def visit_Compare(self, node: ast.Compare):
        node = self.generic_visit(node)
        try:
            expr = ast.Expression(body=node)
            val = eval(compile(ast.fix_missing_locations(expr), "<const>", "eval"), {"__builtins__": {}}, {})
            return ast.copy_location(ast.Constant(bool(val)), node)
        except Exception:
            return node


class _LiteralSpecializer:
    """Monomorphize non-time Cell parameters when calls supply literals.

    Example: pols_if_at(t, "BEF_MAT") becomes pols_if_at__BEF_MAT(t).
    The actuarial source remains untouched; specialization exists only in IR.
    """

    def __init__(self, funcs: dict[str, ast.FunctionDef]):
        self.funcs = funcs
        self.variants: dict[tuple[str, tuple[Any, ...]], str] = {}
        self.generated: dict[str, ast.FunctionDef] = {}
        self.queue: list[tuple[str, tuple[Any, ...]]] = []

    def _time_param(self, fn: ast.FunctionDef) -> str | None:
        if fn.args.args and fn.args.args[0].arg in ("t", "i"):
            return fn.args.args[0].arg
        return None

    def variant_name(self, name: str, literals: tuple[Any, ...]) -> str:
        if not literals:
            return name
        return name + "__" + "__".join(_safe_suffix(x) for x in literals)

    def request(self, name: str, literals: tuple[Any, ...]):
        key = (name, literals)
        if key not in self.variants:
            self.variants[key] = self.variant_name(name, literals)
            self.queue.append(key)
        return self.variants[key]

    def _defaults(self, fn: ast.FunctionDef) -> dict[str, Any]:
        args = [a.arg for a in fn.args.args]
        defaults = fn.args.defaults
        out = {}
        if defaults:
            for name, val in zip(args[-len(defaults):], defaults):
                lit = _literal(val)
                if lit is not None or isinstance(val, ast.Constant):
                    out[name] = lit
        return out

    def _rewrite_calls(self, node: ast.AST):
        outer = self
        class Rewriter(ast.NodeTransformer):
            def __init__(self):
                self.inline_stack: list[str] = []

            def visit_Call(self, n: ast.Call):
                # Inspect the original call before visiting children so a helper with
                # dynamic auxiliary arguments can be inlined as one expression.
                if not isinstance(n.func, ast.Name) or n.func.id not in outer.funcs:
                    return self.generic_visit(n)
                fn = outer.funcs[n.func.id]
                params = [a.arg for a in fn.args.args]
                time_param = outer._time_param(fn)
                start = 1 if time_param else 0
                defaults = outer._defaults(fn)
                keyword_values = {kw.arg: kw.value for kw in n.keywords if kw.arg is not None}
                bound_exprs: dict[str, ast.AST] = {}
                aux_values: list[Any] = []
                has_dynamic_aux = False
                for pos, pname in enumerate(params):
                    if pos < len(n.args):
                        expr = n.args[pos]
                    elif pname in keyword_values:
                        expr = keyword_values[pname]
                    elif pname in defaults:
                        expr = ast.Constant(defaults[pname])
                    else:
                        raise ModelxCompileError(f"{n.func.id}: missing argument {pname!r}")
                    bound_exprs[pname] = expr
                    if pos >= start:
                        lit = _literal(expr)
                        if lit is None and not isinstance(expr, ast.Constant):
                            has_dynamic_aux = True
                        aux_values.append(lit)

                if has_dynamic_aux:
                    if n.func.id in self.inline_stack:
                        raise ModelxCompileError(
                            f"{n.func.id}: recursive dynamic-argument helper cannot be inlined"
                        )
                    value = _linear_inline_expr(fn, bound_exprs)
                    if value is None:
                        dynamic = next(
                            (p for p in params[start:] if not isinstance(bound_exprs[p], ast.Constant) and _literal(bound_exprs[p]) is None),
                            params[start] if start < len(params) else "argument",
                        )
                        raise ModelxCompileError(
                            f"{n.func.id}: dynamic non-time parameter {dynamic!r} requires a non-linear helper body"
                        )
                    self.inline_stack.append(n.func.id)
                    try:
                        value = self.visit(value)
                    finally:
                        self.inline_stack.pop()
                    return ast.copy_location(value, n)

                n = self.generic_visit(n)
                if aux_values:
                    vname = outer.request(n.func.id, tuple(aux_values))
                    new_args = n.args[:start]
                    return ast.copy_location(
                        ast.Call(func=ast.Name(vname, ast.Load()), args=new_args, keywords=[]), n
                    )
                return n
        return Rewriter().visit(node)

    def build(self, roots: Iterable[str]) -> dict[str, ast.FunctionDef]:
        # Seed ordinary zero-param / time-only roots.
        for root in roots:
            if root not in self.funcs:
                raise ModelxCompileError(f"Unknown output Cell {root!r}")
            fn = self.funcs[root]
            params = [a.arg for a in fn.args.args]
            time_param = self._time_param(fn)
            aux = params[1:] if time_param else params
            if aux:
                defaults = self._defaults(fn)
                if all(p in defaults for p in aux):
                    self.request(root, tuple(defaults[p] for p in aux))
                else:
                    raise ModelxCompileError(f"Output {root!r} has required non-time parameters; specify a specialized output")
            else:
                self.request(root, ())

        while self.queue:
            name, literals = self.queue.pop(0)
            fn = copy.deepcopy(self.funcs[name])
            params = [a.arg for a in fn.args.args]
            time_param = self._time_param(fn)
            start = 1 if time_param else 0
            aux_params = params[start:]
            mapping = dict(zip(aux_params, literals))
            if mapping:
                fn = _SubstituteNames(mapping).visit(fn)
                fn = _ConstantFolder().visit(fn)
            fn.name = self.variants[(name, literals)]
            fn.args.args = fn.args.args[:start]
            fn.args.defaults = []
            fn = self._rewrite_calls(fn)
            ast.fix_missing_locations(fn)
            self.generated[fn.name] = fn

        # Closure: request ordinary called functions discovered from generated bodies.
        # _rewrite_calls leaves zero/time-only calls untouched, so collect recursively.
        changed = True
        while changed:
            changed = False
            for fn in list(self.generated.values()):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.funcs:
                        base = n.func.id
                        bfn = self.funcs[base]
                        params = [a.arg for a in bfn.args.args]
                        time_param = self._time_param(bfn)
                        aux = params[1:] if time_param else params
                        if not aux and base not in self.generated:
                            self.request(base, ())
                            changed = True
            while self.queue:
                name, literals = self.queue.pop(0)
                fn = copy.deepcopy(self.funcs[name])
                params = [a.arg for a in fn.args.args]
                time_param = self._time_param(fn)
                start = 1 if time_param else 0
                aux_params = params[start:]
                mapping = dict(zip(aux_params, literals))
                if mapping:
                    fn = _SubstituteNames(mapping).visit(fn)
                    fn = _ConstantFolder().visit(fn)
                fn.name = self.variants[(name, literals)]
                fn.args.args = fn.args.args[:start]
                fn.args.defaults = []
                fn = self._rewrite_calls(fn)
                ast.fix_missing_locations(fn)
                self.generated[fn.name] = fn
        return self.generated


@dataclass
class ExternalArraySpec:
    key: str
    value_getter: Any
    dtype: str
    ndim: int


class _ModelxExprLowerer(ast.NodeTransformer):
    def __init__(self, frontend: "ModelxModelCompiler", vector_helpers: set[str]):
        self.frontend = frontend
        self.vector_helpers = vector_helpers

    def visit_Name(self, node: ast.Name):
        # Numeric model/space references become literals. Space parameter itself is
        # only legal as an index and is handled by visit_Subscript.
        if isinstance(node.ctx, ast.Load) and node.id in self.frontend.refs:
            v = self.frontend.refs[node.id]
            if isinstance(v, (int, float, np.integer, np.floating, bool)) and node.id != self.frontend.space_param:
                return ast.copy_location(ast.Constant(v.item() if hasattr(v, "item") else v), node)
        return node

    def visit_Call(self, node: ast.Call):
        # Bypass trivial model-point accessor Cells.  modelx-cython obtains a
        # similar win by turning exported accessor methods into typed C calls;
        # the sequential backend can go one step further and read the prepared
        # structure-of-arrays column directly.
        if (
            isinstance(node.func, ast.Name)
            and not node.args
            and not node.keywords
            and node.func.id in self.frontend.model_point_accessors
        ):
            field = self.frontend.model_point_accessors[node.func.id]
            key = self.frontend.register_model_point_field(field)
            return ast.copy_location(
                ast.Call(ast.Name("input_scalar", ast.Load()), [ast.Constant(key)], []),
                node,
            )
        return self.generic_visit(node)

    @staticmethod
    def _input_key(node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "input_scalar"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return node.args[0].value
        return None

    def visit_Compare(self, node: ast.Compare):
        node = self.generic_visit(node)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return node
        left, right = node.left, node.comparators[0]
        left_key = self._input_key(left)
        right_key = self._input_key(right)
        if left_key and isinstance(right, ast.Constant) and isinstance(right.value, str):
            code = self.frontend.model_point_category_code(left_key, right.value)
            if code is not None:
                node.comparators[0] = ast.copy_location(ast.Constant(code), right)
        elif right_key and isinstance(left, ast.Constant) and isinstance(left.value, str):
            code = self.frontend.model_point_category_code(right_key, left.value)
            if code is not None:
                node.left = ast.copy_location(ast.Constant(code), left)
        return node

    def _external_cell(self, call: ast.Call):
        if not (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and not call.args):
            return None
        refname = call.func.value.id
        if refname not in self.frontend.refs:
            return None
        obj = self.frontend.refs[refname]
        # Duck-type modelx Space interface.
        if not hasattr(obj, "cells") or call.func.attr not in obj.cells:
            return None
        cellname = call.func.attr
        cell = obj.cells[cellname]
        return refname, cellname, cell

    def _external_call_key(self, call: ast.Call) -> str | None:
        resolved = self._external_cell(call)
        if resolved is None:
            return None
        refname, cellname, cell = resolved
        key = f"{refname}__{cellname}"
        self.frontend.register_external_cell_array(key, cell)
        return key

    def visit_Subscript(self, node: ast.Subscript):
        # model_point()["column"] -> one vector input by model point.
        if (
            self.frontend.model_point_cell
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == self.frontend.model_point_cell
            and not node.value.args
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            field = node.slice.value
            key = self.frontend.register_model_point_field(field)
            return ast.copy_location(ast.Call(ast.Name("input_scalar", ast.Load()), [ast.Constant(key)], []), node)

        # External DataFrame labelled lookup with a categorical model-point row,
        # e.g. ``data.class_factor_table().loc[rate_class(), "factor"]``.
        # Freeze the selected numeric column into the same integer code domain as
        # the prepared model-point field.  This removes pandas from the hot loop
        # without scalarising a downstream vector pipeline; the lookup feeds the
        # scalar recurrence directly.
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "loc"
            and isinstance(node.value.value, ast.Call)
        ):
            resolved = self._external_cell(node.value.value)
            if resolved is not None:
                refname, cellname, cell = resolved
                value = cell()
                if isinstance(value, pd.DataFrame) and isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                    row_expr = self.visit(copy.deepcopy(node.slice.elts[0]))
                    col_expr = node.slice.elts[1]
                    row_key = self._input_key(row_expr)
                    if row_key and isinstance(col_expr, ast.Constant):
                        key = self.frontend.register_external_loc_column(
                            refname,
                            cellname,
                            cell,
                            row_key,
                            col_expr.value,
                        )
                        return ast.copy_location(
                            ast.Call(
                                ast.Name("curve1d", ast.Load()),
                                [ast.Constant(key), row_expr],
                                [],
                            ),
                            node,
                        )

        # External Space cell returning ndarray, e.g. data.age_at_entry()[idx]
        if isinstance(node.value, ast.Call):
            key = self._external_call_key(node.value)
            if key:
                sl = node.slice
                if isinstance(sl, ast.Name) and sl.id == self.frontend.space_param:
                    return ast.copy_location(ast.Call(ast.Name("input_scalar", ast.Load()), [ast.Constant(key)], []), node)
                if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
                    return ast.copy_location(ast.Call(ast.Name("table2d", ast.Load()), [ast.Constant(key), self.visit(sl.elts[0]), self.visit(sl.elts[1])], []), node)
                return ast.copy_location(ast.Call(ast.Name("curve1d", ast.Load()), [ast.Constant(key), self.visit(sl)], []), node)

        # vector_cell()[t] helper
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            name = node.value.func.id
            if name in self.vector_helpers and not node.value.args and not isinstance(node.slice, ast.Slice):
                return ast.copy_location(ast.Call(ast.Name(name + "__at", ast.Load()), [self.visit(node.slice)], []), node)

        # Direct pandas Series/DataFrame refs.
        if isinstance(node.value, ast.Name) and node.value.id in self.frontend.refs:
            refname = node.value.id
            ref = self.frontend.refs[refname]
            if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                key = self.frontend.register_ref_array(refname, ref)
                return ast.copy_location(ast.Call(ast.Name("table2d", ast.Load()), [ast.Constant(key), self.visit(node.slice.elts[0]), self.visit(node.slice.elts[1])], []), node)
            if isinstance(ref, pd.Series):
                key = self.frontend.register_ref_array(refname, ref)
                return ast.copy_location(ast.Call(ast.Name("curve1d", ast.Load()), [ast.Constant(key), self.visit(node.slice)], []), node)

        # DataFrame column then row: table[col][row] -> dense table[row,col].
        if isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name):
            refname = node.value.value.id
            if refname in self.frontend.refs and isinstance(self.frontend.refs[refname], pd.DataFrame):
                col = node.value.slice
                if isinstance(col, ast.Call) and isinstance(col.func, ast.Name) and col.func.id == "str" and len(col.args) == 1:
                    col = col.args[0]
                key = self.frontend.register_ref_array(refname, self.frontend.refs[refname])
                return ast.copy_location(ast.Call(ast.Name("table2d", ast.Load()), [ast.Constant(key), self.visit(node.slice), self.visit(col)], []), node)

        return self.generic_visit(node)


class ModelxModelCompiler:
    """Compile a live modelx.Model into a Cython/OpenMP execution backend.

    The compiler consumes the live Model/Space/Cells/Refs objects directly.
    Formula strings are obtained from Cells.formula; numeric data are bound from
    live References. No lifelib folder, exported model or adapted actuarial source
    is required.
    """

    def __init__(self, model, space: str | Any = "Projection", outputs: Iterable[str] | None = None):
        self.model = model
        self.space = self._resolve_space(space)
        self.space_name = getattr(self.space, "fullname", None) or getattr(self.space, "name", str(space))
        self.space_params = tuple(self.space.parameters or ())
        if len(self.space_params) > 1:
            raise ModelxCompileError(
                f"Native model-point vectorization currently supports at most one Space parameter; {self.space_name} has {self.space_params}"
            )
        self.space_param = self.space_params[0] if self.space_params else None
        self.refs = dict(self.space.refs)
        self.raw_sources = {name: str(cell.formula) for name, cell in self.space.cells.items()}
        self.raw_funcs = {name: ast.parse(src).body[0] for name, src in self.raw_sources.items()}
        scalar_refs: dict[str, Any] = {}
        for ref_name, raw_value in self.refs.items():
            value = _interface_value(raw_value)
            literal = _primitive_literal(value)
            if literal is not ... and ref_name not in self.space_params:
                scalar_refs[ref_name] = literal
        if scalar_refs:
            prepared: dict[str, ast.FunctionDef] = {}
            for name, fn0 in self.raw_funcs.items():
                fn = copy.deepcopy(fn0)
                fn = _SubstituteNames(scalar_refs).visit(fn)
                fn = _ConstantFolder().visit(fn)
                fn = _UnrollLiteralReductions().visit(fn)
                if isinstance(fn, list):
                    raise ModelxCompileError(f"{name}: scalar-reference folding produced an invalid function")
                prepared[name] = ast.fix_missing_locations(fn)
            self.raw_funcs = prepared
        else:
            self.raw_funcs = {
                name: ast.fix_missing_locations(_UnrollLiteralReductions().visit(copy.deepcopy(fn)))
                for name, fn in self.raw_funcs.items()
            }
        (
            self.model_point_cell,
            self.model_point_ref,
            self._model_point_frame_getter,
        ) = self._detect_model_point_row_cell()
        self.model_point_accessors = self._detect_model_point_accessors()
        self.outputs = list(outputs) if outputs is not None else self._infer_outputs()
        if not self.outputs:
            raise ModelxCompileError("Could not infer numeric output Cells; pass outputs=[...]")

        self.external_specs: dict[str, ExternalArraySpec] = {}
        self.model_point_fields: dict[str, str] = {}
        self.model_point_key_fields: dict[str, str] = {}
        self.model_point_categories: dict[str, dict[Any, int]] = {}
        self.ref_array_keys: set[str] = set()

        specializer = _LiteralSpecializer(self.raw_funcs)
        self.specialized_funcs = specializer.build(self.outputs)
        self.specialization_map = dict(specializer.variants)

        self.vector_specs = {
            name: spec for name, fn in self.specialized_funcs.items() if (spec := _time_vector_spec(fn)) is not None
        }
        self.internal_text = self._build_internal_source()
        self.core = ModelCompiler(text=self.internal_text, source_label=f"modelx:{getattr(model,'name','Model')}:{self.space_name}")
        self.inputs = self.core.inputs

    def _resolve_space(self, space):
        if not isinstance(space, str):
            return space
        obj = self.model
        for part in space.split("."):
            if hasattr(obj, "spaces") and part in obj.spaces:
                obj = obj.spaces[part]
            else:
                obj = getattr(obj, part)
        return obj

    def _detect_model_point_row_cell(self):
        if not self.space_param:
            return None, None, None
        for name, fn in self.raw_funcs.items():
            if fn.args.args:
                continue
            rets = [n for n in fn.body if isinstance(n, ast.Return)]
            if len(rets) != 1:
                continue
            r = rets[0].value
            if (
                isinstance(r, ast.Subscript)
                and isinstance(r.value, ast.Attribute)
                and r.value.attr == "loc"
                and isinstance(r.slice, ast.Name)
                and r.slice.id == self.space_param
            ):
                table = r.value.value
                if (
                    isinstance(table, ast.Name)
                    and table.id in self.refs
                    and isinstance(_interface_value(self.refs[table.id]), pd.DataFrame)
                ):
                    refname = table.id
                    return name, refname, lambda n=refname: _interface_value(self.refs[n])

                # Modern lifelib product models commonly expose the table through
                # an attached Data Space Cell: ``data.model_point_table().loc[id]``.
                # Resolve that provider once during preparation and still lower all
                # downstream field reads to structure-of-arrays inputs.  This is a
                # structural provider rule, not a product or Cell-name exception.
                if (
                    isinstance(table, ast.Call)
                    and not table.args
                    and not table.keywords
                    and isinstance(table.func, ast.Attribute)
                    and isinstance(table.func.value, ast.Name)
                    and table.func.value.id in self.refs
                ):
                    refname = table.func.value.id
                    provider = _interface_value(self.refs[refname])
                    cellname = table.func.attr
                    if hasattr(provider, "cells") and cellname in provider.cells:
                        cell = provider.cells[cellname]
                        value = cell()
                        if isinstance(value, pd.DataFrame):
                            label = f"{refname}.{cellname}()"
                            return name, label, lambda c=cell: c()
        return None, None, None

    def _model_point_frame(self) -> pd.DataFrame:
        if self._model_point_frame_getter is None:
            raise ModelxCompileError("No model-point DataFrame provider was detected")
        value = self._model_point_frame_getter()
        if not isinstance(value, pd.DataFrame):
            raise ModelxCompileError(
                f"Model-point provider {self.model_point_ref!r} no longer returns a DataFrame"
            )
        return value

    def _detect_model_point_accessors(self) -> dict[str, str]:
        if not self.model_point_cell:
            return {}
        result: dict[str, str] = {}
        for name, fn in self.raw_funcs.items():
            if fn.args.args:
                continue
            body = list(fn.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            expr = body[0].value
            if (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Name)
                and expr.func.id in {"bool", "float", "int", "str"}
                and len(expr.args) == 1
            ):
                expr = expr.args[0]
            if (
                isinstance(expr, ast.Subscript)
                and isinstance(expr.value, ast.Call)
                and isinstance(expr.value.func, ast.Name)
                and expr.value.func.id == self.model_point_cell
                and not expr.value.args
                and isinstance(expr.slice, ast.Constant)
                and isinstance(expr.slice.value, str)
            ):
                result[name] = expr.slice.value
        return result

    def _infer_outputs(self):
        preferred = ["pv_premiums", "pv_claims", "pv_expenses", "pv_commissions", "pv_net_cf"]
        out = [n for n in preferred if n in self.raw_funcs and len(self.raw_funcs[n].args.args) == 0]
        if out:
            return out
        return sorted(n for n, fn in self.raw_funcs.items() if n.startswith("pv_") and not fn.args.args)

    def register_model_point_field(self, field: str) -> str:
        if not self.model_point_ref:
            raise ModelxCompileError(f"model_point field {field!r} used but no model-point row Cell was detected")
        df = self._model_point_frame()
        if field not in df.columns:
            raise ModelxCompileError(f"{self.model_point_ref} has no field {field!r}")
        key = f"mp__{field}"
        self.model_point_fields[field] = key
        self.model_point_key_fields[key] = field
        values = df[field]
        if not (
            pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)
        ):
            if field not in self.model_point_categories:
                uniques = [value for value in pd.unique(values) if not pd.isna(value)]
                self.model_point_categories[field] = {
                    value: index for index, value in enumerate(uniques)
                }
        return key

    def model_point_category_code(self, key: str, value: str) -> int | None:
        field = self.model_point_key_fields.get(key)
        if field is None or field not in self.model_point_categories:
            return None
        # -1 is a safe missing-category sentinel because encoded categories are
        # always non-negative.  Equality and inequality therefore preserve the
        # natural result for literals absent from the prepared portfolio.
        return self.model_point_categories[field].get(value, -1)

    def _model_point_field_dtype(self, field: str) -> str:
        values = self._model_point_frame()[field]
        if field in self.model_point_categories:
            return "int64"
        return self._dtype_for_vector(values.to_numpy())

    def _encode_model_point_field(self, field: str) -> np.ndarray:
        values = self._model_point_frame()[field]
        if field in self.model_point_categories:
            mapping = self.model_point_categories[field]
            encoded = np.fromiter(
                (mapping.get(value, -1) for value in values),
                dtype=np.int64,
                count=len(values),
            )
            return np.ascontiguousarray(encoded)
        dtype = np.int64 if self._model_point_field_dtype(field) == "int64" else np.float64
        return np.ascontiguousarray(values.to_numpy(dtype=dtype))

    def register_ref_array(self, refname: str, value) -> str:
        key = f"ref__{refname}"
        self.ref_array_keys.add(refname)
        return key

    def register_external_cell_array(self, key: str, cell):
        if key in self.external_specs:
            return
        def getter(c=cell):
            return c()
        value = np.asarray(getter())
        if value.dtype.kind not in "iufb" or value.ndim not in (1, 2):
            raise ModelxCompileError(f"External Cell {cell} must return numeric 1-D/2-D array, got {value.dtype} shape={value.shape}")
        dtype = "int64" if value.dtype.kind in "iub" else "float64"
        self.external_specs[key] = ExternalArraySpec(key, getter, dtype, value.ndim)

    def register_external_loc_column(
        self,
        refname: str,
        cellname: str,
        cell,
        row_key: str,
        column: Any,
    ) -> str:
        field = self.model_point_key_fields.get(row_key)
        if field is None or field not in self.model_point_categories:
            raise ModelxCompileError(
                f"Labelled lookup {refname}.{cellname} requires a categorical prepared row key"
            )
        digest = hashlib.sha256(
            repr((refname, cellname, field, column)).encode()
        ).hexdigest()[:10]
        key = f"loc__{_safe_suffix(refname)}__{_safe_suffix(cellname)}__{digest}"
        if key in self.external_specs:
            return key
        mapping = dict(self.model_point_categories[field])

        def getter(c=cell, labels=mapping, col=column):
            frame = c()
            if not isinstance(frame, pd.DataFrame):
                raise ModelxCompileError(
                    f"Labelled provider {refname}.{cellname} no longer returns a DataFrame"
                )
            result = np.empty(len(labels), dtype=np.float64)
            for label, code in labels.items():
                result[code] = float(frame.loc[label, col])
            return np.ascontiguousarray(result)

        # Validate at preparation/build time so missing labels fail closed rather
        # than becoming a mysterious generated-kernel memory access later.
        getter()
        self.external_specs[key] = ExternalArraySpec(key, getter, "float64", 1)
        return key

    def _dtype_for_vector(self, values) -> str:
        arr = np.asarray(values)
        if arr.dtype.kind in "iub":
            return "int64"
        if arr.dtype.kind in "fc":
            return "float64"
        raise ModelxCompileError(f"Non-numeric model-point input dtype {arr.dtype}")

    def _dense_series(self, series: pd.Series):
        if isinstance(series.index, pd.MultiIndex):
            if series.index.nlevels != 2:
                raise ModelxCompileError("Only 2-level numeric MultiIndex Series are supported")
            coords = [(int(a), int(b)) for a, b in series.index]
            arr = np.zeros((max(a for a,_ in coords)+1, max(b for _,b in coords)+1), dtype=np.float64)
            for (a,b), v in series.items():
                arr[int(a), int(b)] = float(v)
            return np.ascontiguousarray(arr)
        idx = [int(x) for x in series.index]
        arr = np.zeros(max(idx)+1, dtype=np.float64)
        for x, v in series.items():
            arr[int(x)] = float(v)
        return np.ascontiguousarray(arr)

    def _dense_dataframe(self, df: pd.DataFrame):
        rows = [int(x) for x in df.index]
        cols = [int(x) for x in df.columns]
        arr = np.zeros((max(rows)+1, max(cols)+1), dtype=np.float64)
        for csrc, c in zip(df.columns, cols):
            for r, v in df[csrc].items():
                arr[int(r), int(c)] = float(v)
        return np.ascontiguousarray(arr)

    def _build_internal_source(self):
        lowerer = _ModelxExprLowerer(self, set(self.vector_specs))
        body: list[ast.stmt] = [
            ast.ImportFrom(module="modelx_native.dsl", names=[ast.alias("cell"), ast.alias("input_scalar"), ast.alias("curve1d"), ast.alias("table2d")], level=0)
        ]

        # First transform all formulas so input registrations occur.
        transformed = []
        vector_needed = set()
        for name, fn0 in self.specialized_funcs.items():
            if self.model_point_cell and name == self.model_point_cell:
                continue
            if name in self.vector_specs:
                vector_needed.add(name)
                continue
            fn = copy.deepcopy(fn0)
            fn = self._inline_fixed_time_calls(fn)
            fn = self._normalize_vector_reduction(fn)
            fn.body = [lowerer.visit(copy.deepcopy(st)) for st in fn.body]
            fn = self._strip_docstring(fn)
            fn.decorator_list = [self._cell_decorator(name)]
            # Normalize explicit accumulator loops used by BasicTerm_SC.
            fn = self._normalize_explicit_reduction(fn)
            transformed.append(ast.fix_missing_locations(fn))

        # Vector helpers are lowered into scalar __at(t) Cells.
        vector_fns = []
        for name in vector_needed:
            tname, elt = self.vector_specs[name]
            elt = lowerer.visit(copy.deepcopy(elt))
            class Rename(ast.NodeTransformer):
                def visit_Name(self, n):
                    if n.id == tname:
                        return ast.copy_location(ast.Name("t", n.ctx), n)
                    return n
            elt = Rename().visit(elt)
            vf = ast.FunctionDef(
                name=name+"__at",
                args=ast.arguments(posonlyargs=[], args=[ast.arg("t")], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=[ast.Return(elt)], decorator_list=[self._cell_decorator(name)]
            )
            ast.fix_missing_locations(vf)
            vector_fns.append(vf)

        # Rewrite calls/slices to vector helper names after variants exist.
        class VecCallRewrite(ast.NodeTransformer):
            def visit_Call(self, n):
                n = self.generic_visit(n)
                if isinstance(n.func, ast.Name) and n.func.id in vector_needed and len(n.args)==1:
                    n.func.id = n.func.id + "__at"
                return n
        transformed = [ast.fix_missing_locations(VecCallRewrite().visit(f)) for f in transformed]

        inputs: dict[str, dict[str, object]] = {}
        if self.model_point_fields:
            df = self._model_point_frame()
            for field, key in self.model_point_fields.items():
                inputs[key] = {"dtype": self._model_point_field_dtype(field), "ndim": 1}
        for refname in sorted(self.ref_array_keys):
            value = self.space.refs[refname]
            key = f"ref__{refname}"
            if isinstance(value, pd.Series):
                nd = 2 if isinstance(value.index, pd.MultiIndex) else 1
            elif isinstance(value, pd.DataFrame):
                nd = 2
            else:
                arr = np.asarray(value); nd = arr.ndim
            inputs[key] = {"dtype": "float64", "ndim": nd}
        for key, spec in self.external_specs.items():
            inputs[key] = {"dtype": spec.dtype, "ndim": spec.ndim}

        # Outputs may have been specialized only if defaults existed. Current test
        # models have ordinary zero-param outputs, so map them directly.
        out_names = []
        for out in self.outputs:
            key = (out, ())
            out_names.append(self.specialization_map.get(key, out))

        body += [
            ast.Assign([ast.Name("INPUTS", ast.Store())], ast.parse(repr(inputs), mode="eval").body),
            ast.Assign([ast.Name("OUTPUTS", ast.Store())], ast.parse(repr(out_names), mode="eval").body),
        ]
        body.extend(transformed)
        body.extend(vector_fns)
        mod = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(mod)
        return ast.unparse(mod) + "\n"



    def _inline_fixed_time_calls(self, fn: ast.FunctionDef):
        frontend = self
        class Inliner(ast.NodeTransformer):
            depth = 0
            def visit_Call(self, n: ast.Call):
                n = self.generic_visit(n)
                if self.depth > 8:
                    return n
                if not (isinstance(n.func, ast.Name) and n.func.id in frontend.specialized_funcs and len(n.args)==1 and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, int)):
                    return n
                target = copy.deepcopy(frontend.specialized_funcs[n.func.id])
                if not target.args.args or target.args.args[0].arg not in ("t", "i"):
                    return n
                tname = target.args.args[0].arg
                target = _SubstituteNames({tname: int(n.args[0].value)}).visit(target)
                target = _ConstantFolder().visit(target)
                if isinstance(target, list):
                    return n
                target = frontend._strip_docstring(target)
                meaningful = target.body
                if len(meaningful)==1 and isinstance(meaningful[0], ast.Return):
                    self.depth += 1
                    value = self.visit(copy.deepcopy(meaningful[0].value))
                    self.depth -= 1
                    return ast.copy_location(value, n)
                # Reuse the same conservative straight-line helper inliner used
                # for non-time literal specialization.  Absolute-time calls such
                # as ``premium_pp(1)`` are common in issue-cost formulas.  Keeping
                # them as random-access time Cells would force a full historical
                # cache, while inlining the side-effect-free body preserves the
                # sequential O(1) execution model.
                value = _linear_inline_expr(target, {})
                if value is not None:
                    # Prune constant guards before recursively visiting nested
                    # calls.  Otherwise a dead recursive branch such as
                    # ``idx_factor(t - 1)`` is visited even when the fixed t is
                    # already at the base case.
                    value = _ConstantFolder().visit(value)
                    value = ast.fix_missing_locations(value)
                    self.depth += 1
                    value = self.visit(copy.deepcopy(value))
                    self.depth -= 1
                    return ast.copy_location(value, n)
                return n
        return ast.fix_missing_locations(Inliner().visit(fn))

    def _strip_docstring(self, fn: ast.FunctionDef):
        if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) and isinstance(fn.body[0].value.value, str):
            fn.body = fn.body[1:]
        return fn

    def _cell_decorator(self, name: str):
        base = name.split("__", 1)[0]
        int_names = {
            "age", "age_at_entry", "duration", "duration_mth", "duration_yr",
            "policy_term", "proj_len", "max_proj_len", "issue_age", "entry_age"
        }
        if base in int_names:
            return ast.Call(ast.Name("cell", ast.Load()), [], [ast.keyword("dtype", ast.Constant("long"))])
        return ast.Name("cell", ast.Load())

    def _normalize_vector_reduction(self, fn: ast.FunctionDef):
        # lifelib BasicTerm form:
        # return sum(list(claims(t) for t in range(proj_len())) * disc_factors()[:proj_len()])
        body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            return fn
        ret = body[0].value
        if not (isinstance(ret, ast.Call) and isinstance(ret.func, ast.Name) and ret.func.id == "sum" and len(ret.args)==1):
            return fn
        arg = ret.args[0]
        if not (isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mult)):
            return fn
        for gen_side, vec_side in ((arg.left,arg.right),(arg.right,arg.left)):
            gen = _unwrap_list_generator(gen_side)
            if gen is None or len(gen.generators)!=1 or not isinstance(gen.generators[0].target, ast.Name):
                continue
            if isinstance(vec_side, ast.Subscript) and isinstance(vec_side.value, ast.Call) and isinstance(vec_side.value.func, ast.Name):
                helper = vec_side.value.func.id
                if helper in self.vector_specs:
                    tname = gen.generators[0].target.id
                    new_elt = ast.BinOp(
                        left=gen.elt, op=ast.Mult(),
                        right=ast.Call(ast.Name(helper+"__at", ast.Load()), [ast.Name(tname, ast.Load())], [])
                    )
                    fn.body = [ast.Return(ast.Call(ast.Name("sum", ast.Load()), [ast.GeneratorExp(new_elt, copy.deepcopy(gen.generators))], []))]
                    return ast.fix_missing_locations(fn)
        return fn

    def _normalize_explicit_reduction(self, fn: ast.FunctionDef):
        # Canonicalize: x=0; for t in range(...): x += EXPR; return x
        meaningful = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
        if len(meaningful) == 3 and isinstance(meaningful[0], ast.Assign) and isinstance(meaningful[1], ast.For) and isinstance(meaningful[2], ast.Return):
            a, loop, ret = meaningful
            if len(a.targets)==1 and isinstance(a.targets[0], ast.Name) and isinstance(ret.value, ast.Name) and ret.value.id == a.targets[0].id:
                acc = a.targets[0].id
                if len(loop.body)==1 and isinstance(loop.body[0], ast.AugAssign) and isinstance(loop.body[0].target, ast.Name) and loop.body[0].target.id==acc and isinstance(loop.body[0].op, ast.Add):
                    gen = ast.GeneratorExp(elt=loop.body[0].value, generators=[ast.comprehension(target=loop.target, iter=loop.iter, ifs=[], is_async=0)])
                    fn.body = [ast.Return(ast.Call(ast.Name("sum", ast.Load()), [gen], []))]
        return fn

    @property
    def model_point_count(self) -> int:
        if self.model_point_ref:
            return len(self._model_point_frame())
        sizes = []
        for spec in self.external_specs.values():
            arr = np.asarray(spec.value_getter())
            if arr.ndim == 1:
                sizes.append(arr.shape[0])
        if not sizes:
            raise ModelxCompileError("Could not infer model-point count from live model")
        return max(sizes)

    def bind_inputs(self, limit: int | None = None) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if self.model_point_fields:
            for field, key in self.model_point_fields.items():
                out[key] = self._encode_model_point_field(field)
        for refname in self.ref_array_keys:
            value = self.space.refs[refname]
            key = f"ref__{refname}"
            if isinstance(value, pd.Series):
                out[key] = self._dense_series(value)
            elif isinstance(value, pd.DataFrame):
                out[key] = self._dense_dataframe(value)
            else:
                out[key] = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        for key, spec in self.external_specs.items():
            arr = np.asarray(spec.value_getter())
            dt = np.int64 if spec.dtype == "int64" else np.float64
            out[key] = np.ascontiguousarray(arr, dtype=dt)
        if limit is not None:
            n = self.model_point_count
            limit = min(int(limit), n)
            for key, arr in list(out.items()):
                if arr.ndim == 1 and arr.shape[0] == n:
                    out[key] = np.ascontiguousarray(arr[:limit])
        return out

    def pass_manifest(self):
        return self.core.pass_manifest()

    def manifest(self):
        m = self.core.manifest()
        m.update({
            "frontend": "live-modelx",
            "model_name": getattr(self.model, "name", None),
            "space": self.space_name,
            "space_parameters": list(self.space_params),
            "source_kind": "Cells.formula",
            "formula_sha256": {
                n: hashlib.sha256(src.encode()).hexdigest() for n, src in self.raw_sources.items() if n in self.raw_sources
            },
            "model_point_cell": self.model_point_cell,
            "model_point_reference": self.model_point_ref,
            "model_point_accessors": dict(sorted(self.model_point_accessors.items())),
            "model_point_categories": {
                field: {str(value): code for value, code in mapping.items()}
                for field, mapping in sorted(self.model_point_categories.items())
            },
            "specializations": {f"{n}{args}": v for (n,args),v in self.specialization_map.items() if args},
            "external_arrays": sorted(self.external_specs),
        })
        return m

    def explain(self, name: str):
        # Original modelx source remains the explanation surface.
        base = name.split("__", 1)[0]
        lowered = name
        if name in self.vector_specs:
            lowered = name + "__at"
        if lowered in self.core.formulas:
            info = self.core.explain(lowered)
        elif name in self.core.formulas:
            info = self.core.explain(name)
        else:
            # find specialization
            candidates = [v for (n,_),v in self.specialization_map.items() if n==name]
            if len(candidates)==1 and candidates[0] in self.core.formulas:
                info = self.core.explain(candidates[0])
            else:
                raise KeyError(name)
        if base in self.raw_sources:
            info["source"] = self.raw_sources[base]
            info["source_origin"] = f"{self.space_name}.{base}.formula"
        return info

    def generate_cython(self, path, module_name="modelx_compiled"):
        return self.core.generate_cython(path, module_name=module_name)

    def write_manifest(self, path):
        with open(path, "w") as f:
            json.dump(self.manifest(), f, indent=2)

    def make_reference_module(self, name="_modelx_lowered_reference") -> ModuleType:
        mod = ModuleType(name)
        exec(compile(self.internal_text, f"modelx:{self.space_name}", "exec"), mod.__dict__)
        return mod
