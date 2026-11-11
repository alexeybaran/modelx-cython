from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class CallRef:
    name: str
    offset: int | None  # None for no t argument; 0/current; +/- lag


@dataclass
class Formula:
    name: str
    node: ast.FunctionDef
    dtype: str = "double"
    kind: str = "static"  # static | time | aggregate
    calls: list[CallRef] = field(default_factory=list)
    reduction_expr: ast.AST | None = None
    reduction_var: str | None = None
    stage: int = 0


class CompileError(ValueError):
    pass


def _cell_dtype(fn: ast.FunctionDef) -> str:
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "cell":
            for kw in dec.keywords:
                if kw.arg == "dtype" and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value)
        elif isinstance(dec, ast.Name) and dec.id == "cell":
            return "double"
    return "double"


def _is_cell(fn: ast.FunctionDef) -> bool:
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "cell":
            return True
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "cell":
            return True
    return False


def _offset(node: ast.AST, tname: str = "t") -> int | None:
    if isinstance(node, ast.Name) and node.id == tname:
        return 0
    if isinstance(node, ast.BinOp) and isinstance(node.left, ast.Name) and node.left.id == tname and isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
        if isinstance(node.op, ast.Sub):
            return -node.right.value
        if isinstance(node.op, ast.Add):
            return node.right.value
    return None


def _find_reduction(fn: ast.FunctionDef) -> tuple[ast.AST, str] | None:
    # Supported canonical form: return sum(EXPR for t in range(proj_len()))
    if len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
        return None
    ret = fn.body[0].value
    if not (isinstance(ret, ast.Call) and isinstance(ret.func, ast.Name) and ret.func.id == "sum" and len(ret.args) == 1):
        return None
    gen = ret.args[0]
    if not isinstance(gen, ast.GeneratorExp) or len(gen.generators) != 1:
        return None
    comp = gen.generators[0]
    if not isinstance(comp.target, ast.Name) or not isinstance(comp.iter, ast.Call) or not isinstance(comp.iter.func, ast.Name) or comp.iter.func.id != "range":
        return None
    return gen.elt, comp.target.id


class ModelCompiler:
    """Compile a restricted, function-oriented actuarial model to Cython/OpenMP.

    The supported subset is deliberately small and auditable: pure arithmetic,
    if/elif/else, local scalar assignments, model Cell calls at t/t-1, table
    lookup intrinsics and time reductions. Unsupported constructs fail loudly.
    """

    def __init__(self, source: str | Path | None = None, *, text: str | None = None, source_label: str | None = None):
        if text is None:
            if source is None:
                raise ValueError("source or text is required")
            self.source = Path(source)
            self.text = self.source.read_text()
            filename = str(self.source)
        else:
            self.source = Path(source_label or "<memory-model>")
            self.text = text
            filename = source_label or "<memory-model>"
        self.tree = ast.parse(self.text, filename=filename)
        self.inputs = self._literal_assignment("INPUTS")
        self.outputs = self._literal_assignment("OUTPUTS")
        self.formulas: dict[str, Formula] = {}
        self._parse()
        self._assign_stages()

    def _literal_assignment(self, name):
        for n in self.tree.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                    return ast.literal_eval(n.value)
        raise CompileError(f"Missing literal model metadata {name}")

    def _parse(self):
        funcs = {n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef) and _is_cell(n)}
        for name, fn in funcs.items():
            reduction = _find_reduction(fn)
            kind = "aggregate" if reduction else ("time" if any(a.arg == "t" for a in fn.args.args) else "static")
            f = Formula(name=name, node=fn, dtype=_cell_dtype(fn), kind=kind)
            if reduction:
                f.reduction_expr, f.reduction_var = reduction
            self.formulas[name] = f

        fn_names = set(self.formulas)
        for f in self.formulas.values():
            walk_node = f.reduction_expr if f.kind == "aggregate" else f.node
            for n in ast.walk(walk_node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in fn_names:
                    if n.args:
                        off = _offset(n.args[0], f.reduction_var or "t")
                        if off is None:
                            raise CompileError(f"{f.name}: unsupported dynamic time argument {ast.unparse(n.args[0])} in call to {n.func.id}")
                    else:
                        off = None
                    f.calls.append(CallRef(n.func.id, off))

        # Reject forward dependencies in the first implementation. Backward
        # recurrences can be added as a second scan direction without changing DSL.
        bad = [(f.name, c.name, c.offset) for f in self.formulas.values() for c in f.calls if c.offset is not None and c.offset > 0]
        if bad:
            raise CompileError(f"Forward t+k references require a backward-scan backend: {bad[:5]}")

    def _assign_stages(self):
        # Stage = recurrence pass number. Aggregates are available only after the
        # pass that computes their integrand. A static/time formula depending on an
        # aggregate therefore needs one later stage.
        for _ in range(len(self.formulas) * 4):
            changed = False
            for f in self.formulas.values():
                new = 0
                for c in f.calls:
                    dep = self.formulas[c.name]
                    if dep.kind == "aggregate":
                        new = max(new, dep.stage + 1)
                    else:
                        new = max(new, dep.stage)
                if f.kind == "aggregate" and f.reduction_expr is not None:
                    # Aggregate itself is accumulated during the same stage as its
                    # integrand dependencies, so aggregate calls inside the expr
                    # have already been reflected above if present.
                    pass
                if new != f.stage:
                    f.stage = new
                    changed = True
            if not changed:
                break
        else:
            raise CompileError("Stage analysis did not converge; likely an aggregate dependency cycle")

    def manifest(self):
        return {
            "source": str(self.source),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "formulas": {
                n: {
                    "kind": f.kind,
                    "dtype": f.dtype,
                    "stage": f.stage,
                    "calls": [{"name": c.name, "offset": c.offset} for c in f.calls],
                    "source": ast.get_source_segment(self.text, f.node),
                }
                for n, f in self.formulas.items()
            },
            "passes": self.pass_manifest(),
        }

    def pass_manifest(self):
        max_stage = max((f.stage for f in self.formulas.values() if f.kind == "aggregate"), default=0)
        result = []
        for s in range(max_stage + 1):
            result.append({
                "stage": s,
                "aggregates": sorted(f.name for f in self.formulas.values() if f.kind == "aggregate" and f.stage == s),
                "time_formulas": self._time_order(s),
                "static_available": sorted(f.name for f in self.formulas.values() if f.kind == "static" and f.stage <= s),
            })
        return result

    def _time_order(self, stage: int) -> list[str]:
        names = {f.name for f in self.formulas.values() if f.kind == "time" and f.stage <= stage}
        deps = {n: set() for n in names}
        for n in names:
            f = self.formulas[n]
            for c in f.calls:
                if c.name in names and c.offset == 0:
                    deps[n].add(c.name)
        order = []
        while deps:
            ready = sorted(n for n, ds in deps.items() if not ds)
            if not ready:
                cycle = {n: sorted(ds) for n, ds in deps.items()}
                raise CompileError(f"Same-period algebraic cycle; solver block required: {cycle}")
            for n in ready:
                order.append(n)
                deps.pop(n)
            for ds in deps.values():
                ds.difference_update(ready)
        return order

    def explain(self, name: str):
        f = self.formulas[name]
        return {
            "name": name,
            "kind": f.kind,
            "stage": f.stage,
            "dtype": f.dtype,
            "dependencies": [
                {
                    "name": c.name,
                    "relation": ("static" if c.offset is None else "current" if c.offset == 0 else f"t{c.offset:+d}"),
                }
                for c in f.calls
            ],
            "source": ast.get_source_segment(self.text, f.node),
        }

    def write_manifest(self, path: str | Path):
        Path(path).write_text(json.dumps(self.manifest(), indent=2))

    # --------------------------- Cython generation ---------------------------
    def generate_cython(self, path: str | Path, module_name="basicterm_compiled"):
        code = CythonEmitter(self).emit(module_name)
        Path(path).write_text(code)
        return code


class CythonEmitter:
    def __init__(self, compiler: ModelCompiler):
        self.c = compiler
        self.f = compiler.formulas
        self.local_vars: dict[str, set[str]] = {name: self._locals(form.node) for name, form in self.f.items()}

    def _locals(self, fn: ast.FunctionDef) -> set[str]:
        out = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        out.add(t.id)
        return out

    def ctype(self, dtype):
        return "long" if dtype in ("long", "int", "int64") else "double"

    def input_decl(self, name, spec):
        dt, nd = spec["dtype"], spec["ndim"]
        if dt == "int64" and nd == 1:
            return f"const long[::1] {name}"
        if dt == "float64" and nd == 1:
            return f"const double[::1] {name}"
        if dt == "float64" and nd == 2:
            return f"const double[:, ::1] {name}"
        raise CompileError(f"Unsupported input type {name}: {spec}")

    def np_sig(self, name, spec):
        dt, nd = spec["dtype"], spec["ndim"]
        if dt == "int64" and nd == 1:
            return f"cnp.ndarray[cnp.int64_t, ndim=1] {name}"
        if dt == "float64" and nd == 1:
            return f"cnp.ndarray[cnp.float64_t, ndim=1] {name}"
        if dt == "float64" and nd == 2:
            return f"cnp.ndarray[cnp.float64_t, ndim=2] {name}"
        raise CompileError(f"Unsupported input type {name}: {spec}")

    def expr(self, n: ast.AST, *, tvar="t", locals_map=None):
        locals_map = locals_map or {}
        if isinstance(n, ast.Constant):
            if isinstance(n.value, str):
                return repr(n.value)
            return repr(n.value)
        if isinstance(n, ast.Name):
            if n.id == tvar:
                return "t"
            if n.id in locals_map:
                return locals_map[n.id]
            raise CompileError(f"Unknown name in expression: {n.id}")
        if isinstance(n, ast.UnaryOp):
            op = {ast.USub: "-", ast.UAdd: "+", ast.Not: "not "}.get(type(n.op))
            if op is None: raise CompileError(ast.dump(n))
            return f"({op}{self.expr(n.operand, tvar=tvar, locals_map=locals_map)})"
        if isinstance(n, ast.BinOp):
            a = self.expr(n.left, tvar=tvar, locals_map=locals_map)
            b = self.expr(n.right, tvar=tvar, locals_map=locals_map)
            if isinstance(n.op, ast.Pow):
                return f"pow({a}, {b})"
            if isinstance(n.op, ast.Div):
                # Preserve Python true-division semantics even when both operands
                # are integer C variables/constants. cdivision=True would otherwise
                # silently turn 1/12 into zero.
                return f"((<double>({a})) / (<double>({b})))"
            op = {ast.Add:"+",ast.Sub:"-",ast.Mult:"*",ast.FloorDiv:"//",ast.Mod:"%"}.get(type(n.op))
            if op is None: raise CompileError(ast.dump(n))
            return f"({a} {op} {b})"
        if isinstance(n, ast.BoolOp):
            op = " and " if isinstance(n.op, ast.And) else " or " if isinstance(n.op, ast.Or) else None
            if op is None: raise CompileError(ast.dump(n))
            return "(" + op.join(self.expr(v, tvar=tvar, locals_map=locals_map) for v in n.values) + ")"
        if isinstance(n, ast.Compare):
            if len(n.ops) != 1 or len(n.comparators) != 1: raise CompileError("Chained comparisons unsupported")
            op = {ast.Eq:"==",ast.NotEq:"!=",ast.Lt:"<",ast.LtE:"<=",ast.Gt:">",ast.GtE:">="}.get(type(n.ops[0]))
            return f"({self.expr(n.left,tvar=tvar,locals_map=locals_map)} {op} {self.expr(n.comparators[0],tvar=tvar,locals_map=locals_map)})"
        if isinstance(n, ast.IfExp):
            return f"({self.expr(n.body,tvar=tvar,locals_map=locals_map)} if {self.expr(n.test,tvar=tvar,locals_map=locals_map)} else {self.expr(n.orelse,tvar=tvar,locals_map=locals_map)})"
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            name=n.func.id
            if name in self.f:
                dep=self.f[name]
                if dep.kind == "aggregate":
                    return f"a_{name}"
                if dep.kind == "static":
                    return f"s_{name}"
                if len(n.args)!=1: raise CompileError(f"Time cell {name} requires one t arg")
                off=_offset(n.args[0],tvar)
                if off==0: return f"c_{name}"
                if off==-1: return f"p_{name}"
                raise CompileError(f"Unsupported offset in {name}: {ast.unparse(n.args[0])}")
            if name == "input_scalar":
                key=ast.literal_eval(n.args[0]); return f"{key}[i]"
            if name == "curve1d":
                key=ast.literal_eval(n.args[0]); idx=self.expr(n.args[1],tvar=tvar,locals_map=locals_map); return f"{key}[<Py_ssize_t>({idx})]"
            if name == "table2d":
                key=ast.literal_eval(n.args[0]); r=self.expr(n.args[1],tvar=tvar,locals_map=locals_map); c=self.expr(n.args[2],tvar=tvar,locals_map=locals_map); return f"{key}[<Py_ssize_t>({r}), <Py_ssize_t>({c})]"
            if name in ("max","min"):
                if len(n.args)!=2: raise CompileError(f"{name} requires 2 args")
                a=self.expr(n.args[0],tvar=tvar,locals_map=locals_map); b=self.expr(n.args[1],tvar=tvar,locals_map=locals_map)
                cmp=">=" if name=="max" else "<="
                return f"(({a}) if ({a}) {cmp} ({b}) else ({b}))"
            if name in ("int", "float", "bool"):
                if len(n.args) != 1:
                    raise CompileError(f"{name} requires 1 arg")
                x = self.expr(n.args[0], tvar=tvar, locals_map=locals_map)
                if name == "int":
                    return f"(<long>({x}))"
                if name == "float":
                    return f"(<double>({x}))"
                return f"(({x}) != 0)"
            if name == "round":
                x=self.expr(n.args[0],tvar=tvar,locals_map=locals_map)
                digits=ast.literal_eval(n.args[1]) if len(n.args)>1 else 0
                scale=10.0**digits
                return f"(rint(({x}) * {scale!r}) / {scale!r})"
            raise CompileError(f"Unsupported call {name}")
        raise CompileError(f"Unsupported expression: {ast.dump(n, include_attributes=False)}")

    def stmt_block(self, formula: Formula, target: str, indent: str) -> list[str]:
        locals_map={v:f"l_{formula.name}_{v}" for v in self.local_vars[formula.name]}
        out=[]
        def emit_stmt(st, ind):
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                return
            if isinstance(st, ast.Assign) and len(st.targets)==1 and isinstance(st.targets[0], ast.Name):
                out.append(f"{ind}{locals_map[st.targets[0].id]} = {self.expr(st.value, locals_map=locals_map)}")
            elif isinstance(st, ast.Return):
                out.append(f"{ind}{target} = {self.expr(st.value, locals_map=locals_map)}")
            elif isinstance(st, ast.If):
                out.append(f"{ind}if {self.expr(st.test, locals_map=locals_map)}:")
                for x in st.body: emit_stmt(x, ind+"    ")
                if st.orelse:
                    if len(st.orelse)==1 and isinstance(st.orelse[0], ast.If):
                        # Render elif by emitting nested if under else; equivalent and simpler.
                        out.append(f"{ind}else:")
                        emit_stmt(st.orelse[0], ind+"    ")
                    else:
                        out.append(f"{ind}else:")
                        for x in st.orelse: emit_stmt(x, ind+"    ")
            else:
                raise CompileError(f"{formula.name}: unsupported statement {ast.dump(st, include_attributes=False)}")
        for st in formula.node.body:
            emit_stmt(st, indent)
        return out

    def static_order(self, max_stage: int) -> list[str]:
        names={f.name for f in self.f.values() if f.kind=="static" and f.stage<=max_stage}
        deps={n:set() for n in names}
        for n in names:
            for c in self.f[n].calls:
                if c.name in names:
                    deps[n].add(c.name)
        order=[]
        while deps:
            ready=sorted(n for n,d in deps.items() if not d)
            if not ready:
                raise CompileError(f"Static dependency cycle: {deps}")
            order.extend(ready)
            for n in ready: deps.pop(n)
            for d in deps.values(): d.difference_update(ready)
        return order

    def aggregate_expr(self, f: Formula):
        return self.expr(f.reduction_expr, tvar=f.reduction_var or "t")

    def emit(self, module_name):
        lines=[]
        lines += [
            "# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False",
            "cimport cython",
            "from libc.math cimport pow, rint",
            "import numpy as np",
            "cimport numpy as cnp",
            "",
        ]
        input_decls=",\n    ".join(self.input_decl(n,s) for n,s in self.c.inputs.items())
        lines += ["cdef inline void _run_one(", "    Py_ssize_t i,", f"    {input_decls},", "    double[:, ::1] out", ") noexcept nogil:"]

        # Declarations
        for f in self.f.values():
            ct=self.ctype(f.dtype)
            if f.kind=="static": lines.append(f"    cdef {ct} s_{f.name} = 0")
            elif f.kind=="time":
                lines.append(f"    cdef {ct} c_{f.name} = 0")
                if any(c.name==f.name and c.offset==-1 for x in self.f.values() for c in x.calls) or any(c.name==f.name and c.offset==-1 for x in self.f.values() for c in x.calls):
                    pass
            else: lines.append(f"    cdef double a_{f.name} = 0.0")
        lag_names=sorted({c.name for f in self.f.values() for c in f.calls if c.offset==-1})
        for n in lag_names:
            lines.append(f"    cdef {self.ctype(self.f[n].dtype)} p_{n} = 0")
        for fn, vs in self.local_vars.items():
            if self.f[fn].kind != "aggregate":
                for v in sorted(vs): lines.append(f"    cdef double l_{fn}_{v} = 0.0")
        lines.append("    cdef long t")
        lines.append("")

        static_done=set()
        max_agg_stage=max((f.stage for f in self.f.values() if f.kind=="aggregate"), default=0)
        static_order=self.static_order(max(f.stage for f in self.f.values() if f.kind=="static"))
        for stage in range(max_agg_stage+1):
            lines.append(f"    # ---- recurrence stage {stage} ----")
            for n in static_order:
                f=self.f[n]
                if f.stage<=stage and n not in static_done:
                    # Static formula cannot contain time/reduction bodies.
                    lines += self.stmt_block(f, f"s_{n}", "    ")
                    static_done.add(n)
            for a in sorted(f.name for f in self.f.values() if f.kind=="aggregate" and f.stage==stage):
                lines.append(f"    a_{a} = 0.0")
            for n in lag_names: lines.append(f"    p_{n} = 0")
            lines.append("    for t in range(s_proj_len):")
            time_order=self.c._time_order(stage)
            for n in time_order:
                lines += self.stmt_block(self.f[n], f"c_{n}", "        ")
            for a in sorted(f.name for f in self.f.values() if f.kind=="aggregate" and f.stage==stage):
                lines.append(f"        a_{a} += {self.aggregate_expr(self.f[a])}")
            for n in lag_names:
                if n in time_order:
                    lines.append(f"        p_{n} = c_{n}")
            lines.append("")

        # Final static formulas after final aggregate pass.
        for n in static_order:
            if n not in static_done:
                lines += self.stmt_block(self.f[n], f"s_{n}", "    ")
                static_done.add(n)

        # outputs
        for j,n in enumerate(self.c.outputs):
            f=self.f[n]
            prefix="a_" if f.kind=="aggregate" else "s_"
            lines.append(f"    out[i, {j}] = {prefix}{n}")
        lines.append("")

        # Python wrapper
        np_args=",\n    ".join(self.np_sig(n,s) for n,s in self.c.inputs.items())
        lines += ["cpdef cnp.ndarray run(", f"    {np_args},", "    int threads=1,", "):"]
        first_1d=next(n for n,s in self.c.inputs.items() if s["ndim"]==1)
        lines.append(f"    cdef Py_ssize_t n = {first_1d}.shape[0]")
        lines.append(f"    cdef cnp.ndarray[cnp.float64_t, ndim=2] out_arr = np.empty((n, {len(self.c.outputs)}), dtype=np.float64)")
        for n,s in self.c.inputs.items():
            if s["dtype"]=="int64" and s["ndim"]==1: lines.append(f"    cdef long[::1] m_{n} = {n}")
            elif s["dtype"]=="float64" and s["ndim"]==1: lines.append(f"    cdef double[::1] m_{n} = {n}")
            elif s["dtype"]=="float64" and s["ndim"]==2: lines.append(f"    cdef double[:,::1] m_{n} = {n}")
        lines += ["    cdef double[:,::1] out = out_arr", "    cdef Py_ssize_t i", "    if threads != 1:", "        raise ValueError('Part-1 fast foundation is serial-only; threads must be 1')", "    with nogil:", "        for i in range(n):"]
        callargs=", ".join(f"m_{n}" for n in self.c.inputs)
        lines.append(f"            _run_one(i, {callargs}, out)")
        lines += ["    return out_arr", ""]
        return "\n".join(lines)
