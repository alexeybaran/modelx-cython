"""Vector cells: lists of floats lowered to C double buffers (experiment W0.11, MX_CY_VECTORS=1).

A cell whose every traced value is a list of floats is a *vector cell* ('V'); its cached value
is an F64Vec (``_f64_vectors``). A cell whose value is a tuple of k such lists is a *tuple of
vectors* (('T', k)): a Python tuple of F64Vec. The rule is keyed on the value's type only.

In every formula of such a module, a local is a vector when every assignment to it is a vector
expression; it is *owned* (writable) when every assignment allocates. Lowered forms:

  allocation   [x] * n                         -> vec_full(n, x)
               [e for z in range(a, b)]        -> buffer + C loop  (statement value only)
               [e for a, b in zip(v, w)] / [e for a in v]
  element      v[i]                            -> vec_get (Python index semantics, IndexError)
               v[i] = x  (owned v)             -> vec_set; a write to a vector that is not owned
                                                  is a build error (cached vectors are immutable)
  length       len(v)                          -> v.n
  sums         sum(v), sum(v[a:b])             -> compensated sum, bit-identical to CPython
               sum(e for a, b in zip(v, w) if c) (statement value only) -> C loop, same sum
  tuples       return a, b / a, b = cell(t) / cell(t)[k]

Everything else on a vector runs through the F64Vec sequence protocol and is reported in the
admission report as ``vector_fallbacks``.
"""
import os

import libcst as cst

from .consts import CY_MOD
from .helpers import module_name

PROVIDER = 'modelx_cython._f64_vectors'
ALIAS = module_name(PROVIDER)
VECTOR_TYPE = f'{ALIAS}.F64Vec'
LL_MAX = '9223372036854775807'


def enabled():
    return os.environ.get('MX_CY_VECTORS') == '1'


def module_uses_vectors(module):
    return enabled() and any(cell.vector_shape for cls in module.classes.values()
                             for cell in cls.cells.values())


class VectorWriteError(Exception):
    """A formula writes an element of a vector it did not allocate."""


def _expr(text):
    return cst.parse_expression(text)


def _stmt(text):
    return cst.parse_statement(text)


def _code(node):
    return cst.Module([]).code_for_node(node)


def _is_name(node, value):
    return isinstance(node, cst.Name) and node.value == value


def _index(sub):
    """The index expression of a one-element non-slice subscript, else None."""
    if len(sub.slice) == 1 and isinstance(sub.slice[0].slice, cst.Index) and sub.slice[0].slice.star is None:
        return sub.slice[0].slice.value
    return None


def _slice(sub):
    if len(sub.slice) == 1 and isinstance(sub.slice[0].slice, cst.Slice) and sub.slice[0].slice.step is None:
        return sub.slice[0].slice
    return None


def _int_literal(node):
    if isinstance(node, cst.Integer):
        return int(node.value, 0)
    return None


class _Rename(cst.CSTTransformer):
    def __init__(self, names):
        self.names = names

    def leave_Name(self, original, updated):
        new = self.names.get(updated.value)
        return updated.with_changes(value=new) if new else updated


class VectorLowering(cst.CSTTransformer):
    """Rewrite one ``_f_`` formula (a method taking ``self``) of a module with vector cells."""

    def __init__(self, cls_info, cells, builtins, counter):
        self.cells = cells
        self.shapes = {name: c.vector_shape for name, c in cls_info.cells.items() if c.vector_shape}
        self.builtins = builtins
        self.counter = counter          # shared per module: unique temporaries
        self.env, self.owned = {}, set()
        self.lowered, self.fallbacks = 0, []
        self.skip = set()               # ids of original subscripts that are assignment targets
        self.decls = []

    # ---------------------------------------------------------------- analysis
    def kind(self, node):
        """'V', 'A' (fresh allocation, also a vector), ('T', k) or None for an original node."""
        if isinstance(node, cst.Call):
            f = node.func
            if (isinstance(f, cst.Attribute) and _is_name(f.value, 'self')
                    and f.attr.value in self.shapes):
                return self.shapes[f.attr.value]
            return None
        if isinstance(node, cst.Name):
            return self.env.get(node.value)
        if isinstance(node, cst.Subscript):
            base = self.kind(node.value)
            k = _int_literal(_index(node)) if _index(node) is not None else None
            if isinstance(base, tuple) and k is not None and -base[1] <= k < base[1]:
                return 'V'
            return None
        if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.Multiply):
            for one, other in ((node.left, node.right), (node.right, node.left)):
                if (isinstance(one, cst.List) and len(one.elements) == 1
                        and isinstance(one.elements[0], cst.Element)
                        and not isinstance(other, (cst.List, cst.ListComp))):
                    return 'A'
            return None
        if isinstance(node, cst.ListComp):
            return 'A' if id(node) in self.statement_comps and self.comp_ok(node) else None
        if isinstance(node, cst.IfExp):
            a, b = self.kind(node.body), self.kind(node.orelse)
            if a in ('V', 'A') and b in ('V', 'A'):
                return 'A' if a == b == 'A' else 'V'
        return None

    def is_vec(self, node):
        return self.kind(node) in ('V', 'A')

    def iter_vectors(self, iterator):
        """The vector operands of a comprehension source over vectors, else None."""
        if self.is_vec(iterator):
            return [iterator]
        if (isinstance(iterator, cst.Call) and _is_name(iterator.func, 'zip') and 'zip' in self.builtins
                and iterator.args and all(a.keyword is None and a.star == '' and self.is_vec(a.value)
                                          for a in iterator.args)):
            return [a.value for a in iterator.args]
        return None

    def range_args(self, iterator):
        if (isinstance(iterator, cst.Call) and _is_name(iterator.func, 'range') and 'range' in self.builtins
                and 1 <= len(iterator.args) <= 2 and all(a.keyword is None and a.star == '' for a in iterator.args)):
            return [a.value for a in iterator.args]
        return None

    def targets(self, target, count):
        if count == 1 and isinstance(target, cst.Name):
            return [target.value]
        if isinstance(target, cst.Tuple) and len(target.elements) == count and all(
                isinstance(e.value, cst.Name) for e in target.elements):
            return [e.value.value for e in target.elements]
        return None

    def comp_ok(self, node, ifs_ok=False):
        comp = node.for_in
        if comp.inner_for_in is not None or comp.asynchronous is not None:
            return False
        if comp.ifs and not ifs_ok:
            return False
        vecs = self.iter_vectors(comp.iter)
        if vecs is not None:
            return self.targets(comp.target, len(vecs)) is not None
        return self.range_args(comp.iter) is not None and isinstance(comp.target, cst.Name)

    def analyse(self, body):
        assigns = {}
        self.statement_comps = set()

        class Collect(cst.CSTVisitor):
            def visit_Assign(inner, node):
                if isinstance(node.value, cst.ListComp) and len(node.targets) == 1:
                    self.statement_comps.add(id(node.value))
                for t in node.targets:
                    tgt = t.target
                    if isinstance(tgt, cst.Name):
                        assigns.setdefault(tgt.value, []).append(('value', node.value))
                    elif isinstance(tgt, (cst.Tuple, cst.List)):
                        n = len(tgt.elements)
                        for i, e in enumerate(tgt.elements):
                            if isinstance(e.value, cst.Name):
                                assigns.setdefault(e.value.value, []).append(('unpack', node.value, i, n))
                    elif isinstance(tgt, cst.Subscript):
                        self.skip.add(id(tgt))

            def visit_Return(inner, node):
                if isinstance(node.value, cst.ListComp):
                    self.statement_comps.add(id(node.value))

            def visit_AugAssign(inner, node):
                if isinstance(node.target, cst.Name):
                    assigns.setdefault(node.target.value, []).append(('other',))
                elif isinstance(node.target, cst.Subscript):
                    self.skip.add(id(node.target))

            def visit_AnnAssign(inner, node):
                if isinstance(node.target, cst.Name):
                    assigns.setdefault(node.target.value, []).append(('other',))

            def visit_For(inner, node):
                for n in _names(node.target):
                    assigns.setdefault(n, []).append(('other',))

            def visit_NamedExpr(inner, node):
                assigns.setdefault(node.target.value, []).append(('other',))

            def visit_Param(inner, node):
                assigns.setdefault(node.name.value, []).append(('other',))

            def visit_With(inner, node):
                for item in node.items:
                    if item.asname is not None:
                        for n in _names(item.asname.name):
                            assigns.setdefault(n, []).append(('other',))

            def visit_ExceptHandler(inner, node):
                if node.name is not None:
                    assigns.setdefault(node.name.name.value, []).append(('other',))

            def visit_Global(inner, node):
                for n in node.names:
                    assigns.setdefault(n.name.value, []).append(('other',))

        body.visit(Collect())
        changed = True
        while changed:
            changed = False
            for name, items in assigns.items():
                kinds = []
                for item in items:
                    if item[0] == 'value':
                        kinds.append(self.kind(item[1]))
                    elif item[0] == 'unpack':
                        k = self.kind(item[1])
                        kinds.append('V' if isinstance(k, tuple) and k[1] == item[3] else None)
                    else:
                        kinds.append(None)
                if kinds and all(k in ('V', 'A') for k in kinds):
                    new = 'V'
                elif kinds and all(isinstance(k, tuple) for k in kinds) and len(set(kinds)) == 1:
                    new = kinds[0]
                else:
                    new = None
                if self.env.get(name) != new:
                    self.env[name] = new
                    changed = True
        self.owned = {name for name, items in assigns.items() if self.env.get(name) == 'V'
                      and all(item[0] == 'value' and self.kind(item[1]) == 'A' for item in items)}

    # ---------------------------------------------------------------- helpers
    def temp(self, label):
        self.counter[0] += 1
        return f'_mx_v{self.counter[0]}_{label}'

    def declare(self, name, typ):
        self.decls.append(_stmt(f'{name}: {typ}\n'))

    def fallback(self, node, why):
        self.fallbacks.append(f'{why}: {_code(node)[:80]}')

    # ---------------------------------------------------------------- rewriting
    def visit_FunctionDef(self, node):
        return True

    def leave_Subscript(self, original, updated):
        if id(original) in self.skip:
            return updated
        if self.is_vec(original.value):
            index = _index(updated)
            if index is not None:
                self.lowered += 1
                return cst.Call(_expr(f'{ALIAS}.vec_get'), [cst.Arg(updated.value), cst.Arg(index)])
            self.fallback(original, 'slice of a vector outside sum')
            return updated
        if self.kind(original) == 'V' and isinstance(self.kind(original.value), tuple):
            return cst.Call(_expr(f'{ALIAS}.vec_of'), [cst.Arg(updated)])
        return updated

    def leave_BinaryOperation(self, original, updated):
        if self.kind(original) == 'A':
            if isinstance(updated.left, cst.List):
                item, count = updated.left.elements[0].value, updated.right
            else:
                item, count = updated.right.elements[0].value, updated.left
            self.lowered += 1
            return cst.Call(_expr(f'{ALIAS}.vec_full'), [cst.Arg(count), cst.Arg(item)])
        return updated

    def leave_Call(self, original, updated):
        f = original.func
        if isinstance(f, cst.Name) and f.value in self.builtins and len(original.args) == 1 \
                and original.args[0].keyword is None and original.args[0].star == '':
            arg_o, arg_u = original.args[0].value, updated.args[0].value
            if f.value == 'len' and self.is_vec(arg_o):
                self.lowered += 1
                return cst.Attribute(arg_u, cst.Name('n'))
            if f.value == 'sum':
                if self.is_vec(arg_o):
                    self.lowered += 1
                    return cst.Call(_expr(f'{ALIAS}.vec_sum'), [cst.Arg(arg_u)])
                if isinstance(arg_o, cst.Subscript) and self.is_vec(arg_o.value) and _slice(arg_o):
                    # updated slice subscript was left as is (fallback recorded); drop that record
                    if self.fallbacks and self.fallbacks[-1].startswith('slice of a vector outside sum'):
                        self.fallbacks.pop()
                    s = _slice(arg_u)
                    lo = s.lower if s.lower is not None else _expr('0')
                    hi = s.upper if s.upper is not None else _expr(LL_MAX)
                    self.lowered += 1
                    return cst.Call(_expr(f'{ALIAS}.vec_sum_range'),
                                    [cst.Arg(arg_u.value), cst.Arg(lo), cst.Arg(hi)])
        return updated

    def leave_For(self, original, updated):
        if self.is_vec(original.iter) or self.iter_vectors(original.iter):
            self.fallback(original.iter, 'for statement over a vector')
        return updated

    def leave_ListComp(self, original, updated):
        if self.iter_vectors(original.for_in.iter) and self.kind(original) != 'A':
            self.fallback(original, 'list comprehension over vectors not lowered')
        return updated

    def leave_GeneratorExp(self, original, updated):
        return updated

    # statement level: comprehension buffers, sums over vectors, writes, unpacking, returns
    def leave_SimpleStatementLine(self, original, updated):
        if len(original.body) != 1:
            return updated
        so, su = original.body[0], updated.body[0]
        pre = []
        if isinstance(so, cst.Assign) and len(so.targets) == 1:
            tgt_o = so.targets[0].target
            if isinstance(tgt_o, cst.Subscript) and self.is_vec(tgt_o.value) and _index(tgt_o) is not None:
                base = tgt_o.value
                if not (isinstance(base, cst.Name) and base.value in self.owned):
                    raise VectorWriteError(
                        f'{self.cells.model_path}: assignment to an element of vector '
                        f'{_code(base)!r}, which this formula did not allocate; cached vectors '
                        f'are immutable: allocate a new one ([0.0] * n) and write that')
                self.lowered += 1
                tgt_u = su.targets[0].target
                call = cst.Call(_expr(f'{ALIAS}.vec_set'),
                                [cst.Arg(tgt_u.value), cst.Arg(_index(tgt_u)), cst.Arg(su.value)])
                return updated.with_changes(body=[cst.Expr(call)])
            if isinstance(tgt_o, (cst.Tuple, cst.List)) and isinstance(self.kind(so.value), tuple):
                names = [e.value for e in tgt_o.elements]
                if all(isinstance(n, cst.Name) and self.env.get(n.value) == 'V' for n in names):
                    temps = [self.temp('u') for _ in names]
                    stmts = [_stmt(f'{", ".join(temps)} = {_code(su.value)}\n')]
                    stmts += [_stmt(f'{n.value} = {ALIAS}.vec_of({t})\n') for n, t in zip(names, temps)]
                    self.lowered += 1
                    return cst.FlattenSentinel([updated.with_changes(body=s.body) for s in stmts])
        value_o = getattr(so, 'value', None)
        if isinstance(so, (cst.Assign, cst.Return, cst.AugAssign)) and value_o is not None:
            if isinstance(value_o, cst.ListComp) and self.kind(value_o) == 'A':
                pre, result = self.expand_comp(value_o, su.value)
            elif (isinstance(value_o, cst.Call) and _is_name(value_o.func, 'sum') and 'sum' in self.builtins
                  and len(value_o.args) == 1 and isinstance(value_o.args[0].value, cst.GeneratorExp)
                  and self.iter_vectors(value_o.args[0].value.for_in.iter)
                  and self.comp_ok(value_o.args[0].value, ifs_ok=True)):
                pre, result = self.expand_sum(value_o.args[0].value, su.value.args[0].value)
            if pre:
                su = su.with_changes(value=result)
        if isinstance(so, cst.Return) and so.value is not None:
            su = self.wrap_return(so, su)
        if pre:
            return cst.FlattenSentinel([*pre, updated.with_changes(body=[su])])
        return updated.with_changes(body=[su])

    def wrap_return(self, so, su):
        shape = self.cells.vector_shape
        if shape == 'V' and not self.is_vec(so.value) and not (
                isinstance(su.value, cst.Name) and su.value.value.startswith('_mx_v')):
            return su.with_changes(value=cst.Call(_expr(f'{ALIAS}.vec_of'), [cst.Arg(su.value)]))
        if isinstance(shape, tuple) and isinstance(so.value, cst.Tuple) and isinstance(su.value, cst.Tuple):
            elements = []
            for eo, eu in zip(so.value.elements, su.value.elements):
                if self.is_vec(eo.value):
                    elements.append(eu)
                else:
                    elements.append(eu.with_changes(
                        value=cst.Call(_expr(f'{ALIAS}.vec_of'), [cst.Arg(eu.value)])))
            return su.with_changes(value=su.value.with_changes(elements=elements))
        return su

    def loop_head(self, comp_o, comp_u, tag):
        """Statements before the loop, the loop header text, loop-variable renames, the count."""
        pre, renames = [], {}
        vecs_o = self.iter_vectors(comp_o.iter)
        i = self.temp(f'{tag}i')
        n = self.temp(f'{tag}n')
        self.declare(i, f'{CY_MOD}.Py_ssize_t')
        self.declare(n, f'{CY_MOD}.Py_ssize_t')
        if vecs_o is not None:
            vecs_u = ([comp_u.iter] if len(vecs_o) == 1 and self.is_vec(comp_o.iter)
                      else [a.value for a in comp_u.iter.args])
            names = self.targets(comp_o.target, len(vecs_o))
            xs = []
            for v in vecs_u:
                x = self.temp(f'{tag}x')
                self.declare(x, VECTOR_TYPE)
                pre.append(_stmt(f'{x} = {_code(v)}\n'))
                xs.append(x)
            pre.append(_stmt(f'{n} = {xs[0]}.n\n'))
            for x in xs[1:]:
                pre.append(_stmt(f'if {x}.n < {n}: {n} = {x}.n\n'))
            body = []
            for name, x in zip(names, xs):
                new = self.temp(f'{tag}{name}')
                self.declare(new, f'{CY_MOD}.double')
                renames[name] = new
                body.append(f'{new} = {x}.data[{i}]')
            return pre, f'for {i} in range({n}):', body, renames, n, i
        args = [_code(a) for a in self.range_args(comp_u.iter)]
        lo, hi = (('0', args[0]) if len(args) == 1 else (args[0], args[1]))
        lo_t, hi_t = self.temp(f'{tag}lo'), self.temp(f'{tag}hi')
        self.declare(lo_t, f'{CY_MOD}.longlong')
        self.declare(hi_t, f'{CY_MOD}.longlong')
        z = self.temp(f'{tag}{comp_o.target.value}')
        self.declare(z, f'{CY_MOD}.longlong')
        renames[comp_o.target.value] = z
        pre += [_stmt(f'{lo_t} = {lo}\n'), _stmt(f'{hi_t} = {hi}\n'),
                _stmt(f'{n} = {hi_t} - {lo_t} if {hi_t} > {lo_t} else 0\n')]
        return pre, f'for {i} in range({n}):', [f'{z} = {lo_t} + {i}'], renames, n, i

    def expand_comp(self, comp_o, comp_u):
        pre, head, body, renames, n, i = self.loop_head(comp_o.for_in, comp_u.for_in, 'c')
        r = self.temp('r')
        self.declare(r, VECTOR_TYPE)
        elt = _code(comp_u.elt.visit(_Rename(renames)))
        pre.append(_stmt(f'{r} = {ALIAS}.vec_new({n})\n'))
        loop = '\n'.join([head] + ['    ' + line for line in body] + [f'    {r}.data[{i}] = {elt}']) + '\n'
        pre.append(_stmt(loop))
        self.lowered += 1
        return pre, cst.Name(r)

    def expand_sum(self, gen_o, gen_u):
        pre, head, body, renames, n, i = self.loop_head(gen_o.for_in, gen_u.for_in, 's')
        s = self.temp('sum')
        self.declare(s, f'{ALIAS}.F64Sum')
        elt = _code(gen_u.elt.visit(_Rename(renames)))
        pre.append(_stmt(f'{ALIAS}.f64_init({CY_MOD}.address({s}))\n'))
        add = f'{ALIAS}.f64_add({CY_MOD}.address({s}), {elt})'
        if gen_u.for_in.ifs:
            cond = ' and '.join(f'({_code(c.test.visit(_Rename(renames)))})' for c in gen_u.for_in.ifs)
            lines = body + [f'if {cond}:', f'    {add}']
        else:
            lines = body + [add]
        pre.append(_stmt('\n'.join([head] + ['    ' + line for line in lines]) + '\n'))
        self.lowered += 1
        return pre, _expr(f'{ALIAS}.f64_result({CY_MOD}.address({s}))')


def _names(target):
    if isinstance(target, cst.Name):
        return [target.value]
    if isinstance(target, (cst.Tuple, cst.List)):
        return [n for e in target.elements for n in _names(e.value)]
    return []


def lower(node, cls_info, cells, scope, counter):
    """Lower the vector forms of one ``_f_`` FunctionDef; returns the new node."""
    from libcst.metadata import BuiltinAssignment
    shapes = {c.vector_shape for c in cls_info.cells.values()}
    if not enabled() or shapes == {None}:
        return node
    builtins = {name for name in ('sum', 'len', 'zip', 'range')
                if scope[name] and all(isinstance(b, BuiltinAssignment) for b in scope[name])}
    lowering = VectorLowering(cls_info, cells, builtins, counter)
    lowering.analyse(node.body)
    if not any(v for v in lowering.env.values()) and not cells.vector_shape and not any(
            isinstance(n, cst.Attribute) and n.attr.value in lowering.shapes for n in _walk(node.body)):
        return node
    body = node.body.visit(lowering)
    vector_locals = sorted(name for name, k in lowering.env.items() if k == 'V')
    decls = [_stmt(f'{name}: {VECTOR_TYPE}\n') for name in vector_locals] + lowering.decls
    stmts = list(body.body)
    start = 1 if stmts and isinstance(stmts[0], cst.SimpleStatementLine) and isinstance(
        stmts[0].body[0], cst.Expr) and isinstance(stmts[0].body[0].value, (cst.SimpleString, cst.ConcatenatedString)) else 0
    body = body.with_changes(body=[*stmts[:start], *decls, *stmts[start:]])
    cells.vector_lowered = lowering.lowered
    cells.vector_fallbacks = lowering.fallbacks
    return node.with_changes(body=body)


def _walk(node):
    found = []

    class W(cst.CSTVisitor):
        def on_visit(self, n):
            found.append(n)
            return True
    node.visit(W())
    return found
