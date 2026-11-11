"""Static fast-recursive graph IR for exported modelx packages.

This module deliberately separates *graph admissibility* from formula/data lowering.
It reads exported ``_mx_classes.py`` source and builds a direct recursive Cell program
without using realized modelx traces, recurrence schedules, stages, or storage fences.

Unsupported formula syntax (pandas, Python containers, module calls, etc.) is recorded
as a formula/provider concern and does not prevent graph construction.  The graph layer
only rejects cases where Cell dependencies themselves cannot be resolved/bound
statically (for example a first-class/dynamic Cell reference).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
from typing import Iterable, Mapping, Sequence


class FastRecursiveGraphError(RuntimeError):
    """Raised when the recursive Cell graph itself cannot be represented safely."""


@dataclass(frozen=True)
class FastRecursiveParameter:
    name: str
    default_source: str | None = None

    @property
    def required(self) -> bool:
        return self.default_source is None


@dataclass(frozen=True)
class FastRecursiveBoundArg:
    parameter: str
    source: str
    origin: str  # explicit | keyword | default
    relation: str


@dataclass(frozen=True)
class FastRecursiveCallSite:
    caller: str
    callee: str
    source: str
    lineno: int
    col_offset: int
    bound_args: tuple[FastRecursiveBoundArg, ...]

    @property
    def arity(self) -> int:
        return len(self.bound_args)


@dataclass(frozen=True)
class FastRecursiveExternalCall:
    caller: str
    source: str
    root: str
    kind: str  # provider | module | builtin | unknown
    lineno: int


@dataclass(frozen=True)
class FastRecursiveCell:
    name: str
    formula_name: str
    parameters: tuple[FastRecursiveParameter, ...]
    source: str
    lineno: int
    dependency_calls: tuple[FastRecursiveCallSite, ...]
    external_calls: tuple[FastRecursiveExternalCall, ...]
    syntax_features: tuple[str, ...]

    @property
    def cache_key(self) -> tuple[str, ...]:
        # Semantics-preserving default: memoization is keyed by every Cell argument.
        # Physical cache layout is a later lowering decision.
        return tuple(p.name for p in self.parameters)


@dataclass(frozen=True)
class FastRecursiveSCC:
    cells: tuple[str, ...]
    recursive: bool


@dataclass(frozen=True)
class FastRecursiveProgram:
    source_path: str
    class_name: str
    space_name: str
    module_key: str
    target: str
    cells: Mapping[str, FastRecursiveCell]
    call_sites: tuple[FastRecursiveCallSite, ...]
    external_calls: tuple[FastRecursiveExternalCall, ...]
    sccs: tuple[FastRecursiveSCC, ...]
    graph_blockers: tuple[str, ...] = ()
    formula_notes: tuple[str, ...] = ()

    @property
    def graph_supported(self) -> bool:
        return not self.graph_blockers

    @property
    def recursive_sccs(self) -> tuple[FastRecursiveSCC, ...]:
        return tuple(s for s in self.sccs if s.recursive)

    @property
    def unique_dependency_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted({(c.caller, c.callee) for c in self.call_sites}))

    @property
    def max_call_arity(self) -> int:
        return max((c.arity for c in self.call_sites), default=0)

    @property
    def max_scc_size(self) -> int:
        return max((len(s.cells) for s in self.recursive_sccs), default=0)

    def manifest(self) -> dict:
        return {
            "source_path": self.source_path,
            "class_name": self.class_name,
            "space_name": self.space_name,
            "module_key": self.module_key,
            "target": self.target,
            "graph_supported": self.graph_supported,
            "graph_blockers": list(self.graph_blockers),
            "cell_count": len(self.cells),
            "call_site_count": len(self.call_sites),
            "unique_dependency_count": len(self.unique_dependency_pairs),
            "external_call_count": len(self.external_calls),
            "max_call_arity": self.max_call_arity,
            "scc_count": len(self.sccs),
            "recursive_scc_count": len(self.recursive_sccs),
            "max_recursive_scc_size": self.max_scc_size,
            "recursive_sccs": [list(s.cells) for s in self.recursive_sccs],
            "cells": {
                name: {
                    "parameters": [
                        {"name": p.name, "default": p.default_source}
                        for p in cell.parameters
                    ],
                    "cache_key": list(cell.cache_key),
                    "dependencies": sorted({c.callee for c in cell.dependency_calls}),
                    "dependency_call_sites": [
                        {
                            "callee": c.callee,
                            "source": c.source,
                            "bound_args": [
                                {
                                    "parameter": a.parameter,
                                    "source": a.source,
                                    "origin": a.origin,
                                    "relation": a.relation,
                                }
                                for a in c.bound_args
                            ],
                        }
                        for c in cell.dependency_calls
                    ],
                    "external_calls": [x.source for x in cell.external_calls],
                    "syntax_features": list(cell.syntax_features),
                }
                for name, cell in sorted(self.cells.items())
            },
            "formula_notes": list(self.formula_notes),
        }


_BUILTINS = {
    "abs", "all", "any", "bool", "enumerate", "float", "int", "len", "list",
    "max", "min", "range", "round", "str", "sum", "tuple", "zip",
}


def _source(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _parameters(fn: ast.FunctionDef) -> tuple[FastRecursiveParameter, ...]:
    args = list(fn.args.args)
    if not args or args[0].arg != "self":
        raise FastRecursiveGraphError(f"{fn.name}: exported Cell formula does not start with self")
    args = args[1:]
    defaults = list(fn.args.defaults)
    first_default = len(args) - len(defaults)
    result: list[FastRecursiveParameter] = []
    for i, a in enumerate(args):
        default = None
        if i >= first_default:
            default = _source(defaults[i - first_default])
        result.append(FastRecursiveParameter(a.arg, default))
    if fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
        raise FastRecursiveGraphError(
            f"{fn.name}: varargs/kwonly/posonly exported Cell signatures are not yet supported"
        )
    return tuple(result)


def _relation(expr: ast.AST, caller_params: set[str]) -> str:
    if isinstance(expr, ast.Constant):
        return "literal"
    if isinstance(expr, ast.Name) and expr.id in caller_params:
        return f"param:{expr.id}"
    if (
        isinstance(expr, ast.BinOp)
        and isinstance(expr.left, ast.Name)
        and expr.left.id in caller_params
        and isinstance(expr.right, ast.Constant)
        and isinstance(expr.right.value, (int, float))
        and isinstance(expr.op, (ast.Add, ast.Sub))
    ):
        sign = 1 if isinstance(expr.op, ast.Add) else -1
        return f"affine:{expr.left.id}:{sign * expr.right.value:+g}"
    names = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
    if names and names <= caller_params:
        return "caller-expression"
    return "dynamic-expression"


def _bind_call(
    caller: str,
    caller_params: Sequence[FastRecursiveParameter],
    callee: str,
    callee_params: Sequence[FastRecursiveParameter],
    call: ast.Call,
) -> tuple[FastRecursiveBoundArg, ...]:
    if any(k.arg is None for k in call.keywords):
        raise FastRecursiveGraphError(f"{caller}->{callee}: **kwargs call is not statically bindable")
    if len(call.args) > len(callee_params):
        raise FastRecursiveGraphError(
            f"{caller}->{callee}: {len(call.args)} positional args for {len(callee_params)} parameters"
        )
    caller_names = {p.name for p in caller_params}
    values: dict[str, tuple[str, str, str]] = {}
    for p, arg in zip(callee_params, call.args):
        values[p.name] = (_source(arg), "explicit", _relation(arg, caller_names))
    known = {p.name: p for p in callee_params}
    for kw in call.keywords:
        assert kw.arg is not None
        if kw.arg not in known:
            raise FastRecursiveGraphError(f"{caller}->{callee}: unknown keyword {kw.arg!r}")
        if kw.arg in values:
            raise FastRecursiveGraphError(f"{caller}->{callee}: duplicate argument {kw.arg!r}")
        values[kw.arg] = (_source(kw.value), "keyword", _relation(kw.value, caller_names))
    result: list[FastRecursiveBoundArg] = []
    for p in callee_params:
        if p.name in values:
            src, origin, rel = values[p.name]
        elif p.default_source is not None:
            src, origin, rel = p.default_source, "default", "default"
        else:
            raise FastRecursiveGraphError(f"{caller}->{callee}: missing required argument {p.name!r}")
        result.append(FastRecursiveBoundArg(p.name, src, origin, rel))
    return tuple(result)


def _nested_root(attr: ast.Attribute) -> str | None:
    cur: ast.AST = attr
    chain: list[str] = []
    while isinstance(cur, ast.Attribute):
        chain.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name) and cur.id == "self" and chain:
        return chain[-1]
    return None


def _syntax_features(fn: ast.FunctionDef) -> tuple[str, ...]:
    kinds: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.For): kinds.add("for")
        elif isinstance(n, ast.While): kinds.add("while")
        elif isinstance(n, ast.Raise): kinds.add("raise")
        elif isinstance(n, ast.ListComp): kinds.add("listcomp")
        elif isinstance(n, ast.GeneratorExp): kinds.add("generator")
        elif isinstance(n, ast.DictComp): kinds.add("dictcomp")
        elif isinstance(n, ast.SetComp): kinds.add("setcomp")
        elif isinstance(n, (ast.List, ast.Dict, ast.Set)): kinds.add("container-literal")
        elif isinstance(n, ast.Subscript): kinds.add("subscript")
        elif isinstance(n, ast.Slice): kinds.add("slice")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            root = _nested_root(n.func)
            if root in {"data", "Data"}: kinds.add("provider-call")
            elif root in {"pd", "np", "math"}: kinds.add("module-call")
    return tuple(sorted(kinds))


def _tarjan(
    cells: Iterable[str], dependency_pairs: Iterable[tuple[str, str]]
) -> tuple[FastRecursiveSCC, ...]:
    nodes = tuple(sorted(cells))
    adj = {n: set() for n in nodes}
    for a, b in dependency_pairs:
        if a in adj and b in adj:
            adj[a].add(b)
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    comps: list[FastRecursiveSCC] = []

    def strong(v: str) -> None:
        nonlocal index
        indices[v] = low[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(adj[v]):
            if w not in indices:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            cc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                cc.append(w)
                if w == v:
                    break
            ordered = tuple(sorted(cc))
            recursive = len(ordered) > 1 or (len(ordered) == 1 and ordered[0] in adj[ordered[0]])
            comps.append(FastRecursiveSCC(ordered, recursive))

    for n in nodes:
        if n not in indices:
            strong(n)
    return tuple(sorted(comps, key=lambda s: (min(s.cells), len(s.cells))))


def build_fast_recursive_program(
    classes_path: str | Path,
    *,
    target: str,
    class_name: str | None = None,
    space_name: str = "Projection",
    module_key: str | None = None,
) -> FastRecursiveProgram:
    """Build a static recursive Cell program from a modelx exported classes module.

    The result contains the target dependency closure only.  It deliberately does not
    inspect realized modelx nodes/traces and does not derive any sequential execution
    stages.  Every Cell call is bound against the callee's exported Python signature;
    defaults and multi-argument calls are retained as part of the recursive ABI.
    """
    path = Path(classes_path)
    module_key = module_key or path.parent.name
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [n for n in module.body if isinstance(n, ast.ClassDef)]
    if class_name is None:
        suffix = space_name
        matches = [c for c in classes if c.name == f"_c_{space_name}" or c.name.endswith(suffix)]
        if len(matches) != 1:
            raise FastRecursiveGraphError(
                f"expected one exported class for space {space_name!r}, found {[c.name for c in matches]}"
            )
        cls = matches[0]
    else:
        matches = [c for c in classes if c.name == class_name]
        if len(matches) != 1:
            raise FastRecursiveGraphError(f"exported class {class_name!r} not found uniquely")
        cls = matches[0]

    formulas: dict[str, ast.FunctionDef] = {
        f.name[3:]: f
        for f in cls.body
        if isinstance(f, ast.FunctionDef) and f.name.startswith("_f_")
    }
    if target not in formulas:
        raise FastRecursiveGraphError(f"target Cell {target!r} not found in {cls.name}")
    params = {name: _parameters(fn) for name, fn in formulas.items()}

    raw_calls: dict[str, list[ast.Call]] = {name: [] for name in formulas}
    first_class_refs: list[str] = []
    external_by_cell: dict[str, list[FastRecursiveExternalCall]] = {name: [] for name in formulas}

    for name, fn in formulas.items():
        parents: dict[int, ast.AST] = {
            id(child): parent
            for parent in ast.walk(fn)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self" and node.attr in formulas:
                parent = parents.get(id(node))
                if not (isinstance(parent, ast.Call) and parent.func is node):
                    first_class_refs.append(f"{name}: first-class Cell reference {_source(node)}")
            if not isinstance(node, ast.Call):
                continue
            fun = node.func
            if isinstance(fun, ast.Attribute) and isinstance(fun.value, ast.Name) and fun.value.id == "self":
                if fun.attr in formulas:
                    raw_calls[name].append(node)
                else:
                    external_by_cell[name].append(
                        FastRecursiveExternalCall(
                            caller=name,
                            source=_source(node),
                            root=fun.attr,
                            kind="unknown",
                            lineno=getattr(node, "lineno", 0),
                        )
                    )
            elif isinstance(fun, ast.Attribute):
                root = _nested_root(fun)
                if root is not None:
                    kind = "provider" if root in {"data", "Data"} else ("module" if root in {"np", "pd", "math"} else "unknown")
                    external_by_cell[name].append(
                        FastRecursiveExternalCall(
                            caller=name,
                            source=_source(node),
                            root=root,
                            kind=kind,
                            lineno=getattr(node, "lineno", 0),
                        )
                    )
            elif isinstance(fun, ast.Name) and fun.id not in _BUILTINS and fun.id not in {"ValueError", "RuntimeError", "TypeError"}:
                external_by_cell[name].append(
                    FastRecursiveExternalCall(
                        caller=name,
                        source=_source(node),
                        root=fun.id,
                        kind="unknown",
                        lineno=getattr(node, "lineno", 0),
                    )
                )

    # Static target closure over direct Cell calls.  This is independent of formula
    # syntax/data support and therefore cannot silently shrink because a branch was not
    # exercised by a sampled policy.
    closure: set[str] = set()
    pending = [target]
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.add(name)
        for call in raw_calls[name]:
            assert isinstance(call.func, ast.Attribute)
            pending.append(call.func.attr)

    blockers = [x for x in first_class_refs if x.split(":", 1)[0] in closure]
    call_sites: list[FastRecursiveCallSite] = []
    cell_map: dict[str, FastRecursiveCell] = {}
    formula_notes: set[str] = set()

    for name in sorted(closure):
        fn = formulas[name]
        calls: list[FastRecursiveCallSite] = []
        for call in raw_calls[name]:
            assert isinstance(call.func, ast.Attribute)
            callee = call.func.attr
            try:
                bound = _bind_call(name, params[name], callee, params[callee], call)
            except FastRecursiveGraphError as exc:
                blockers.append(str(exc))
                bound = ()
            site = FastRecursiveCallSite(
                caller=name,
                callee=callee,
                source=_source(call),
                lineno=getattr(call, "lineno", 0),
                col_offset=getattr(call, "col_offset", 0),
                bound_args=bound,
            )
            calls.append(site)
            call_sites.append(site)
        ext = tuple(external_by_cell[name])
        feats = _syntax_features(fn)
        for f in feats:
            formula_notes.add(f)
        cell_map[name] = FastRecursiveCell(
            name=name,
            formula_name=fn.name,
            parameters=params[name],
            source=ast.get_source_segment(path.read_text(encoding="utf-8"), fn) or _source(fn),
            lineno=getattr(fn, "lineno", 0),
            dependency_calls=tuple(calls),
            external_calls=ext,
            syntax_features=feats,
        )

    pairs = {(c.caller, c.callee) for c in call_sites}
    sccs = _tarjan(cell_map, pairs)
    all_external = tuple(x for n in sorted(cell_map) for x in cell_map[n].external_calls)
    return FastRecursiveProgram(
        source_path=str(path),
        class_name=cls.name,
        space_name=space_name,
        module_key=module_key,
        target=target,
        cells=cell_map,
        call_sites=tuple(call_sites),
        external_calls=all_external,
        sccs=sccs,
        graph_blockers=tuple(sorted(set(blockers))),
        formula_notes=tuple(sorted(formula_notes)),
    )


__all__ = [
    "FastRecursiveGraphError",
    "FastRecursiveParameter",
    "FastRecursiveBoundArg",
    "FastRecursiveCallSite",
    "FastRecursiveExternalCall",
    "FastRecursiveCell",
    "FastRecursiveSCC",
    "FastRecursiveProgram",
    "build_fast_recursive_program",
]
