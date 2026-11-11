from __future__ import annotations

"""Typed formula sources for the graph-owned Cython execution backend.

Production compilation derives formula bodies and ABI types from the graph
compiler's own realized trace. A translated modelx-cython package can still be
read as an independent coverage/type oracle, but it is not required by the graph
compiler and is never imported or executed. In both cases formula bodies are
re-emitted against physical slots owned by the graph/loop-recovery backend; the
modelx-cython ``_v_*``/``_has_*`` historical caches are never used.
"""

import ast
import copy
import numbers
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


class ModelxCythonFormulaError(RuntimeError):
    """Fail-closed error while reading translated modelx-cython formula source."""


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _decorator_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _annotation_name(node: ast.AST | None) -> str:
    if node is None:
        return "object"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _annotation_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    try:
        return ast.unparse(node)
    except Exception:
        return "object"


def cython_type_from_annotation(node: ast.AST | None) -> str:
    """Map modelx-cython trace annotations to the graph runtime ABI."""
    name = _annotation_name(node).lower()
    leaf = name.rsplit(".", 1)[-1]
    if leaf in {"double", "float", "float64"}:
        return "double"
    if leaf in {"longlong", "long", "int", "int64", "py_ssize_t"}:
        return "long long"
    if leaf in {"bint", "bool"}:
        return "bint"
    if leaf in {"str", "unicode"}:
        return "str"
    return "object"




class _ReplaceLoadedName(ast.NodeTransformer):
    """Replace one comprehension induction variable with a literal value."""

    def __init__(self, name: str, value: object):
        self.name = name
        self.value = value

    def visit_Name(self, node: ast.Name):  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id == self.name:
            return ast.copy_location(ast.Constant(value=self.value), node)
        return node


class _ReplaceLoadedNameExpr(ast.NodeTransformer):
    """Replace a comprehension induction variable with a copied expression."""

    def __init__(self, name: str, expression: ast.expr):
        self.name = name
        self.expression = expression

    def visit_Name(self, node: ast.Name):  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id == self.name:
            return ast.copy_location(copy.deepcopy(self.expression), node)
        return node


class _FiniteLiteralGeneratorUnroller(ast.NodeTransformer):
    """Unroll only tiny finite-literal ``sum`` generator expressions.

    This is deliberately not a general comprehension compiler.  It targets the
    structural case where a traced formula sums a Cell/helper call over a small
    tuple/list of literal alternatives.  Turning the generator into distinct
    syntactic callsites lets the existing realized-edge reference planner bind
    each call directly to its already-traced dependency.
    """

    MAX_ITEMS = 16

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "sum"
            or len(node.args) != 1
            or node.keywords
            or not isinstance(node.args[0], ast.GeneratorExp)
        ):
            return node
        gen = node.args[0]
        if len(gen.generators) != 1:
            return node
        comp = gen.generators[0]
        if comp.ifs or comp.is_async or not isinstance(comp.target, ast.Name):
            return node
        if not isinstance(comp.iter, (ast.Tuple, ast.List)):
            return node
        values = []
        for item in comp.iter.elts:
            if not isinstance(item, ast.Constant):
                return node
            values.append(item.value)
        if len(values) > self.MAX_ITEMS:
            return node

        # Python sum starts from integer zero and accumulates left-to-right.  Keep
        # that ordering so this rewrite does not silently become a reduction-tree
        # numerical optimization.
        acc: ast.expr = ast.copy_location(ast.Constant(value=0), node)
        for ordinal, value in enumerate(values):
            term = copy.deepcopy(gen.elt)
            term = _ReplaceLoadedName(comp.target.id, value).visit(term)
            # Distinct synthetic coordinates matter to the graph reference planner:
            # it keys multiple same-Cell callsites by (name, line, column).
            for sub in ast.walk(term):
                if isinstance(sub, ast.Call):
                    base_col = int(getattr(sub, "col_offset", getattr(node, "col_offset", 0)))
                    sub.col_offset = base_col + (ordinal + 1) * 1000
                    if hasattr(sub, "end_col_offset"):
                        sub.end_col_offset = max(int(getattr(sub, "end_col_offset", base_col)), sub.col_offset + 1)
            acc = ast.copy_location(ast.BinOp(left=acc, op=ast.Add(), right=term), node)
        return ast.copy_location(acc, node)


def _realized_scalar_expr(node: ast.AST, env: dict[str, object]) -> object:
    """Evaluate the integer-only subset used to bound a realized ``range``."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ModelxCythonFormulaError(f"unbound realized range name {node.id!r}")
        value = env[node.id]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (bool, int)):
            return value
        raise ModelxCythonFormulaError(f"non-integral realized range name {node.id!r}")
    if isinstance(node, ast.UnaryOp):
        value = _realized_scalar_expr(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
    if isinstance(node, ast.BinOp):
        left = _realized_scalar_expr(node.left, env)
        right = _realized_scalar_expr(node.right, env)
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
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and not node.keywords
    ):
        values = [_realized_scalar_expr(arg, env) for arg in node.args]
        if node.func.id == "max" and values:
            return max(values)
        if node.func.id == "min" and values:
            return min(values)
        if node.func.id == "int" and len(values) == 1:
            return int(values[0])
    raise ModelxCythonFormulaError("unsupported realized range expression")


class _FiniteRealizedRangeGeneratorUnroller(ast.NodeTransformer):
    """Turn a bounded traced ``sum(Cell(...) for i in range(...))`` into sites.

    A generator can invoke one syntactic Cell call several times per formula
    occurrence.  The graph already contains every concrete dependency edge, but
    the reference planner deliberately binds one address per syntactic callsite.
    Expanding the finite realized range creates those callsites without executing
    or rediscovering a modelx dependency at artifact runtime.  Each term retains a
    membership guard, so shorter ranges and dormant branches keep Python's exact
    iteration semantics.
    """

    # Very large tuple expressions exhaust Cython's recursive parser.  These
    # ranges remain safely on the Python boundary until loop-form C emission is
    # available; ordinary actuarial lookbacks (including IUL's 11 months) fit.
    MAX_ITEMS = 128

    def __init__(
        self, rows: Iterable[object], formal_names: tuple[str, ...],
        function: ast.FunctionDef,
    ):
        self.rows = tuple(rows)
        self.formal_names = formal_names
        self.local_exprs = {
            statement.targets[0].id: statement.value
            for statement in function.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        }

    def _sequences(self, iterator: ast.AST) -> tuple[tuple[int, ...], ...] | None:
        if (
            not isinstance(iterator, ast.Call)
            or not isinstance(iterator.func, ast.Name)
            or iterator.func.id != "range"
            or iterator.keywords
            or not (1 <= len(iterator.args) <= 3)
        ):
            return None
        sequences: list[tuple[int, ...]] = []
        for row in self.rows:
            env = dict(zip(self.formal_names, tuple(getattr(row, "args", ()))))
            glb = getattr(getattr(row, "obj", None), "altfunc", None)
            glb = {} if glb is None else getattr(glb, "__globals__", {})
            for name, value in glb.items():
                if name in env:
                    continue
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, (bool, int)):
                    env[name] = value
            pending = dict(self.local_exprs)
            while pending:
                progressed = False
                for name, expression in tuple(pending.items()):
                    try:
                        env[name] = _realized_scalar_expr(expression, env)
                    except Exception:
                        continue
                    del pending[name]
                    progressed = True
                if not progressed:
                    break
            try:
                args = [int(_realized_scalar_expr(arg, env)) for arg in iterator.args]
                seq = tuple(range(*args))
            except Exception:
                return None
            if len(seq) > self.MAX_ITEMS:
                return None
            sequences.append(seq)
        return tuple(sequences)

    def _relative_candidates(
        self, iterator: ast.AST, sequences: tuple[tuple[int, ...], ...]
    ) -> tuple[ast.expr, ...] | None:
        """Express a sliding realized range as bounded offsets from its stop."""
        if not isinstance(iterator, ast.Call) or not iterator.args:
            return None
        stop = iterator.args[0] if len(iterator.args) == 1 else iterator.args[1]
        if not isinstance(stop, ast.Name) or stop.id not in self.formal_names:
            return None
        formal_index = self.formal_names.index(stop.id)
        relative: list[tuple[int, ...]] = []
        for row, sequence in zip(self.rows, sequences):
            args = tuple(getattr(row, "args", ()))
            if formal_index >= len(args) or not isinstance(args[formal_index], (int, np.integer)):
                return None
            anchor = int(args[formal_index])
            relative.append(tuple(anchor - value for value in sequence))
        longest = max(relative, key=len)
        if any(tuple(value for value in longest if value in seq) != seq for seq in relative):
            return None
        return tuple(
            ast.BinOp(
                left=ast.Name(id=stop.id, ctx=ast.Load()),
                op=ast.Sub(),
                right=ast.Constant(value=offset),
            )
            for offset in longest
        )

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "sum"
            or len(node.args) != 1
            or node.keywords
            or not isinstance(node.args[0], ast.GeneratorExp)
        ):
            return node
        gen = node.args[0]
        if len(gen.generators) != 1:
            return node
        comp = gen.generators[0]
        if comp.ifs or comp.is_async or not isinstance(comp.target, ast.Name):
            return node
        sequences = self._sequences(comp.iter)
        if not sequences:
            return node
        longest = max(sequences, key=len)
        if any(tuple(value for value in longest if value in seq) != seq for seq in sequences):
            candidates = self._relative_candidates(comp.iter, sequences)
            if candidates is None:
                return node
        else:
            candidates = tuple(ast.Constant(value=value) for value in longest)
        if len(candidates) > self.MAX_ITEMS:
            return node

        terms: list[ast.expr] = []
        for ordinal, candidate in enumerate(candidates):
            term = _ReplaceLoadedNameExpr(comp.target.id, candidate).visit(copy.deepcopy(gen.elt))
            for sub in ast.walk(term):
                if isinstance(sub, ast.Call):
                    base_col = int(getattr(sub, "col_offset", getattr(node, "col_offset", 0)))
                    sub.col_offset = base_col + (ordinal + 1) * 1000
                    if hasattr(sub, "end_col_offset"):
                        sub.end_col_offset = sub.col_offset + 1
            guard = ast.Compare(
                left=copy.deepcopy(candidate),
                ops=[ast.In()],
                comparators=[copy.deepcopy(comp.iter)],
            )
            terms.append(ast.IfExp(test=guard, body=term, orelse=ast.Constant(value=0)))
        replacement = ast.Call(
            func=ast.Name(id="sum", ctx=ast.Load()),
            args=[ast.Tuple(elts=terms, ctx=ast.Load())],
            keywords=[],
        )
        return ast.copy_location(replacement, node)


class _StripSelf(ast.NodeTransformer):
    """Convert translated ``self.foo`` references back to formula globals.

    The private modelx-cython formula is a method only because its generated Space
    owns references and public Cell methods.  The graph backend supplies those
    references independently, so the leading ``self`` is removed while preserving
    the rest of an attribute chain (``self.data.table`` -> ``data.table``).
    """

    def visit_Attribute(self, node: ast.Attribute):  # noqa: N802
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def unroll_finite_literal_generators(fn: ast.FunctionDef) -> ast.FunctionDef:
    """Expand only small literal ``sum(... for x in (...))`` generators upstream.

    This is semantic normalization, not a Cython optimization, so both optimized
    Python and Cython consume the same distinct literal callsites.
    """
    out = _FiniteLiteralGeneratorUnroller().visit(copy.deepcopy(fn))
    assert isinstance(out, ast.FunctionDef)
    return ast.fix_missing_locations(out)


def normalize_formula_function(fn: ast.FunctionDef, *, cell_name: str) -> ast.FunctionDef:
    out = copy.deepcopy(fn)
    out.decorator_list = []
    out.name = cell_name
    if not out.args.args or out.args.args[0].arg != "self":
        raise ModelxCythonFormulaError(f"translated formula _f_{cell_name} has no self argument")
    out.args.args = out.args.args[1:]
    out = _StripSelf().visit(out)
    out = unroll_finite_literal_generators(out)
    ast.fix_missing_locations(out)
    return out


def normalize_graph_formula_function(fn: ast.FunctionDef, *, cell_name: str) -> ast.FunctionDef:
    """Normalize an original modelx formula without a generated ``self`` ABI."""
    out = copy.deepcopy(fn)
    out.decorator_list = []
    out.name = cell_name
    out = unroll_finite_literal_generators(out)
    ast.fix_missing_locations(out)
    return out


def _source_function(source: str, *, cell_name: str) -> ast.FunctionDef:
    try:
        module = ast.parse(textwrap.dedent(source))
    except SyntaxError as exc:
        raise ModelxCythonFormulaError(
            f"cannot parse realized formula {cell_name}: {exc}"
        ) from exc
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise ModelxCythonFormulaError(
            f"realized formula {cell_name} must contain one function definition"
        )
    return normalize_graph_formula_function(functions[0], cell_name=cell_name)


def _observed_cython_type(values: Iterable[object]) -> str:
    rows = tuple(values)
    if not rows:
        return "object"
    if all(isinstance(value, (bool, np.bool_)) for value in rows):
        return "bint"
    if all(
        isinstance(value, (numbers.Integral, np.integer))
        and not isinstance(value, (bool, np.bool_))
        for value in rows
    ):
        return "long long"
    if all(
        isinstance(value, (numbers.Real, np.floating))
        and not isinstance(value, (bool, np.bool_))
        for value in rows
    ):
        return "double"
    if all(isinstance(value, str) for value in rows):
        return "str"
    return "object"


def source_uses_modelx_caches(node: ast.AST | str) -> bool:
    """Detect accidental coupling to modelx-cython's historical Cell caches."""
    if isinstance(node, str):
        # Generated Cython is not valid Python syntax because of cdef/cimports.
        # Scan explicit cache identifiers first; parse as Python only when possible.
        if re.search(r"\b(?:self\.)?_(?:v|has)_[A-Za-z_][A-Za-z0-9_]*", node):
            return True
        try:
            node = ast.parse(node)
        except SyntaxError:
            return False
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and (item.id.startswith("_v_") or item.id.startswith("_has_")):
            return True
        if isinstance(item, ast.Attribute) and (item.attr.startswith("_v_") or item.attr.startswith("_has_")):
            return True
    return False


@dataclass(frozen=True)
class ModelxCythonFormula:
    source_path: str
    class_name: str
    space_name: str
    cell_name: str
    function: ast.FunctionDef
    arg_types: tuple[tuple[str, str], ...]
    return_type: str
    formula_op_id: int | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.space_name, self.cell_name)

    @property
    def source(self) -> str:
        return ast.unparse(self.function)


@dataclass(frozen=True)
class ModelxCythonFormulaCatalog:
    formulas: tuple[ModelxCythonFormula, ...]
    package_root: str
    source_kind: str = "modelx_cython_translated"

    @classmethod
    def from_package(cls, package_root: str | Path) -> "ModelxCythonFormulaCatalog":
        root = Path(package_root)
        if not root.exists():
            raise ModelxCythonFormulaError(f"translated modelx-cython package does not exist: {root}")
        formulas: list[ModelxCythonFormula] = []
        for path in sorted(root.rglob("_mx_classes.py")):
            formulas.extend(_parse_translated_file(path))
        if not formulas:
            raise ModelxCythonFormulaError(f"no private modelx-cython formulas found under {root}")
        return cls(tuple(formulas), str(root.resolve()), "modelx_cython_translated")

    @classmethod
    def from_compiler(cls, compiler, typed) -> "ModelxCythonFormulaCatalog":
        """Build a typed catalog from graph-owned realized evidence.

        This mirrors modelx-cython's broad formula policy: ordinary function
        bodies remain eligible even when they return Python objects. Only true
        generator functions are excluded because Cython cannot make them cdef.
        """
        structured = getattr(compiler, "structured", None)
        plan = None if structured is None else structured.canonical_plan
        if plan is None:
            raise ModelxCythonFormulaError(
                "canonical execution plan is required for the internal formula catalog"
            )

        nodes_by_role: dict[object, list[object]] = {}
        for node in compiler.trace.nodes:
            nodes_by_role.setdefault(node.shape_token, []).append(node)
        dependencies_by_dst: dict[int, list[int]] = {}
        for src, dst in compiler.trace.dependencies:
            dependencies_by_dst.setdefault(dst, []).append(src)

        formulas: list[ModelxCythonFormula] = []
        for op in plan.formula_ops:
            rows = nodes_by_role.get(op.role, ())
            output = typed.formula_by_id.get(op.op_id)
            if not rows or output is None:
                continue
            try:
                function = _source_function(rows[0].schema.source, cell_name=op.name)
            except ModelxCythonFormulaError:
                continue
            if any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(function)):
                continue

            formals = [*function.args.posonlyargs, *function.args.args]
            if len(formals) != op.arity or any(len(node.args) != op.arity for node in rows):
                continue
            function = _FiniteRealizedRangeGeneratorUnroller(
                rows, tuple(formal.arg for formal in formals), function
            ).visit(function)
            function = ast.fix_missing_locations(function)
            arg_types = tuple(
                (
                    formal.arg,
                    _observed_cython_type(node.args[index] for node in rows),
                )
                for index, formal in enumerate(formals)
            )
            return_type = {
                "double": "double",
                "int64": "long long",
                "bool": "bint",
            }.get(output.pool)
            if return_type == "long long" and any(
                isinstance(node, (ast.Div, ast.Pow))
                or isinstance(node, ast.Constant) and isinstance(node.value, float)
                for node in ast.walk(function)
            ):
                # The realized path can return only integral sentinels while a
                # dormant branch is fractional.  Widen the function ABI as
                # modelx-cython does; narrowing at an integral traced slot is
                # still governed by the realized artifact's typed slot plan.
                return_type = "double"
            if return_type == "long long" and any(
                getattr(typed.node_by_id.get(src), "pool", None) == "double"
                for row in rows
                for src in dependencies_by_dst.get(row.node_id, ())
            ):
                # A realized integral sentinel can hide floating result branches
                # selected by string/kind arguments.  A double function ABI is
                # safe for the realized integer slot and lets Cython type-check
                # every branch (CashValue claim_pp is the corpus example).
                return_type = "double"
            if return_type is None:
                returned = []
                for node in rows:
                    try:
                        returned.append(node.obj.data[tuple(node.args)])
                    except Exception:
                        returned = []
                        break
                return_type = "str" if returned and all(
                    isinstance(value, str) for value in returned
                ) else "object"

            parts = str(op.fullname).split(".")
            space_name = parts[-2] if len(parts) >= 2 else ""
            formulas.append(
                ModelxCythonFormula(
                    source_path=f"<realized:{op.fullname}>",
                    class_name=f"_c_{space_name}" if space_name else "",
                    space_name=space_name,
                    cell_name=op.name,
                    function=function,
                    arg_types=arg_types,
                    return_type=return_type,
                    formula_op_id=op.op_id,
                )
            )
        return cls(tuple(formulas), "<realized-trace>", "internal_realized_trace")

    def candidates(self, space_name: str, cell_name: str) -> tuple[ModelxCythonFormula, ...]:
        return tuple(x for x in self.formulas if x.space_name == space_name and x.cell_name == cell_name)

    def match(self, space_name: str, cell_name: str) -> ModelxCythonFormula | None:
        rows = self.candidates(space_name, cell_name)
        if not rows:
            return None
        # Inherited/exported Space classes can repeat an identical formula.  Reuse
        # is safe only when the normalized body and traced ABI agree exactly.
        sigs = {
            (
                ast.dump(x.function, include_attributes=False),
                x.arg_types,
                x.return_type,
            )
            for x in rows
        }
        if len(sigs) != 1:
            return None
        return rows[0]

    def match_formula_op(self, formula_op_id: int) -> ModelxCythonFormula | None:
        rows = tuple(x for x in self.formulas if x.formula_op_id == formula_op_id)
        return rows[0] if len(rows) == 1 else None

    def by_cell_name(self, cell_name: str) -> tuple[ModelxCythonFormula, ...]:
        return tuple(x for x in self.formulas if x.cell_name == cell_name)


def _parse_translated_file(path: Path) -> Iterable[ModelxCythonFormula]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ModelxCythonFormulaError(f"cannot parse translated source {path}: {exc}") from exc
    for cls_node in module.body:
        if not isinstance(cls_node, ast.ClassDef) or not cls_node.name.startswith("_c_"):
            continue
        space_name = cls_node.name[3:]
        for fn in cls_node.body:
            if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("_f_"):
                continue
            if not any(_decorator_name(d).endswith(".cfunc") or _decorator_name(d) == "cfunc" for d in fn.decorator_list):
                continue
            cell_name = fn.name[3:]
            normalized = normalize_formula_function(fn, cell_name=cell_name)
            if source_uses_modelx_caches(normalized):
                raise ModelxCythonFormulaError(
                    f"private formula unexpectedly references generated cache state: {path}:{cls_node.name}.{fn.name}"
                )
            arg_types = tuple(
                (arg.arg, cython_type_from_annotation(arg.annotation))
                for arg in fn.args.args[1:]
            )
            yield ModelxCythonFormula(
                source_path=str(path),
                class_name=cls_node.name,
                space_name=space_name,
                cell_name=cell_name,
                function=normalized,
                arg_types=arg_types,
                return_type=cython_type_from_annotation(fn.returns),
            )
