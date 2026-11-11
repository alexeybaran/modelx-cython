"""Pure-Python reference emitter for :mod:`fast_recursive_graph`.

The emitted module is intentionally simple: one direct recursive function per exported
Cell plus one memo dictionary per Cell, keyed by the complete Cell argument tuple.  It
is an executable proof of the fast-recursive graph ABI before Cython/type/provider
lowering.  Non-Cell source syntax is preserved rather than interpreted by the graph
layer.
"""
from __future__ import annotations

from pathlib import Path
import ast

from .fast_recursive_graph import FastRecursiveGraphError, FastRecursiveProgram


class _FormulaTransformer(ast.NodeTransformer):
    def __init__(self, cell_names: set[str]):
        self.cell_names = cell_names

    def visit_Call(self, node: ast.Call):  # noqa: N802
        # Rewrite direct Cell calls before replacing generic self references.
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in self.cell_names
        ):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id=f"fr_{node.func.attr}", ctx=ast.Load()),
                    args=[ast.Name(id="rt", ctx=ast.Load())] + [self.visit(a) for a in node.args],
                    keywords=[ast.keyword(arg=k.arg, value=self.visit(k.value)) for k in node.keywords],
                ),
                node,
            )
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):  # noqa: N802
        if node.id == "self" and isinstance(node.ctx, ast.Load):
            return ast.copy_location(
                ast.Attribute(
                    value=ast.Name(id="rt", ctx=ast.Load()),
                    attr="ctx",
                    ctx=ast.Load(),
                ),
                node,
            )
        return node


def _arg_node(name: str) -> ast.arg:
    return ast.arg(arg=name, annotation=None)


def _runtime_class(cache_count: int) -> ast.ClassDef:
    src = f'''\nclass FastRecursiveRuntime:\n    def __init__(self, ctx=None):\n        self.ctx = ctx\n        self.cache = [dict() for _ in range({cache_count})]\n\n    def bind(self, ctx):\n        self.ctx = ctx\n        for c in self.cache:\n            c.clear()\n        return self\n'''
    return ast.parse(src).body[0]  # type: ignore[return-value]


def _formula_nodes(program: FastRecursiveProgram) -> dict[str, ast.FunctionDef]:
    module = ast.parse(Path(program.source_path).read_text(encoding="utf-8"), filename=program.source_path)
    cls = next(
        (
            n for n in module.body
            if isinstance(n, ast.ClassDef) and n.name == program.class_name
        ),
        None,
    )
    if cls is None:
        raise FastRecursiveGraphError(f"exported class {program.class_name!r} disappeared from source")
    return {
        n.name[3:]: n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_f_") and n.name[3:] in program.cells
    }


def emit_fast_recursive_python(program: FastRecursiveProgram) -> str:
    """Emit an executable recursive Python module for a static Cell program."""
    if not program.graph_supported:
        raise FastRecursiveGraphError("cannot emit graph with blockers: " + "; ".join(program.graph_blockers))
    formulas = _formula_nodes(program)
    if set(formulas) != set(program.cells):
        missing = sorted(set(program.cells) - set(formulas))
        raise FastRecursiveGraphError(f"formula nodes missing for {missing}")
    names = set(program.cells)
    tx = _FormulaTransformer(names)
    order = sorted(program.cells)
    cache_index = {name: i for i, name in enumerate(order)}
    body: list[ast.stmt] = [
        ast.Expr(value=ast.Constant(value="Generated static fast-recursive reference runtime.")),
        _runtime_class(len(order)),
    ]

    for name in order:
        original = formulas[name]
        # Calculate function: preserve the original formula body, defaults and control flow,
        # replacing only direct Cell dispatch and self -> runtime context.
        args = [_arg_node("rt")] + [_arg_node(p.name) for p in program.cells[name].parameters]
        defaults = [ast.parse(p.default_source, mode="eval").body for p in program.cells[name].parameters if p.default_source is not None]
        calc = ast.FunctionDef(
            name=f"_calc_{name}",
            args=ast.arguments(
                posonlyargs=[], args=args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=defaults
            ),
            body=[tx.visit(ast.fix_missing_locations(ast.parse(ast.unparse(stmt)).body[0])) for stmt in original.body],
            decorator_list=[], returns=None, type_comment=None,
        )
        body.append(calc)

        params = [p.name for p in program.cells[name].parameters]
        if not params:
            key_expr: ast.expr = ast.Constant(value=None)
        elif len(params) == 1:
            key_expr = ast.Name(id=params[0], ctx=ast.Load())
        else:
            key_expr = ast.Tuple(elts=[ast.Name(id=p, ctx=ast.Load()) for p in params], ctx=ast.Load())
        call_calc = ast.Call(
            func=ast.Name(id=f"_calc_{name}", ctx=ast.Load()),
            args=[ast.Name(id="rt", ctx=ast.Load())] + [ast.Name(id=p, ctx=ast.Load()) for p in params],
            keywords=[],
        )
        wrapper_src = f'''\ndef fr_{name}({', '.join(['rt'] + [p.name + (('=' + p.default_source) if p.default_source is not None else '') for p in program.cells[name].parameters])}):\n    c = rt.cache[{cache_index[name]}]\n    key = None\n    if key in c:\n        return c[key]\n    val = None\n    c[key] = val\n    return val\n'''
        wrapper = ast.parse(wrapper_src).body[0]
        assert isinstance(wrapper, ast.FunctionDef)
        # Replace the placeholder key / calc while keeping the simple, easy-to-inspect body.
        assign_key = wrapper.body[1]
        assert isinstance(assign_key, ast.Assign)
        assign_key.value = key_expr
        assign_val = wrapper.body[3]
        assert isinstance(assign_val, ast.Assign)
        assign_val.value = call_calc
        body.append(wrapper)

    target_cell = program.cells[program.target]
    if target_cell.parameters:
        # Direct functions are still emitted; only the convenience evaluate() helper needs
        # a zero-argument target, which is true for all current benchmark roots.
        helper_body: list[ast.stmt] = [
            ast.Raise(
                exc=ast.Call(func=ast.Name(id="TypeError", ctx=ast.Load()), args=[ast.Constant("target requires arguments")], keywords=[]),
                cause=None,
            )
        ]
    else:
        helper_body = [
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id="rt", ctx=ast.Load()), attr="bind", ctx=ast.Load()),
                    args=[ast.Name(id="ctx", ctx=ast.Load())], keywords=[],
                )
            ),
            ast.Return(
                value=ast.Call(func=ast.Name(id=f"fr_{program.target}", ctx=ast.Load()), args=[ast.Name(id="rt", ctx=ast.Load())], keywords=[])
            ),
        ]
    body.append(
        ast.FunctionDef(
            name="evaluate",
            args=ast.arguments(posonlyargs=[], args=[_arg_node("rt"), _arg_node("ctx")], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=helper_body,
            decorator_list=[], returns=None, type_comment=None,
        )
    )
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"


def write_fast_recursive_python(program: FastRecursiveProgram, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(emit_fast_recursive_python(program), encoding="utf-8")
    return out


__all__ = ["emit_fast_recursive_python", "write_fast_recursive_python"]
