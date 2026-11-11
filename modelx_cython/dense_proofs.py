"""Reuse ranges that generated dense entries already check before a miss."""
from collections import Counter

import libcst as cst
from libcst.metadata import ExpressionContext

from .cell_entries import EntryModule, dense_condition, exact_receiver
from .consts import CY_MOD, FORMULA_PREF, MX_SELF
from .scopes import CstFacts, OPAQUE_SCOPES
from .syntax import arguments, dotted_name, same_expression, statements, walk


def _else_body(node):
    if isinstance(node.orelse, cst.If):
        return (node.orelse,)
    return statements(node.orelse.body) if isinstance(node.orelse, cst.Else) else ()


def _checked_guard(entry, layout):
    body = statements(entry.body)
    if len(body) != 1 or not isinstance(body[0], cst.If) or not 0 < len(layout.axes) <= 32:
        return None
    if [param.name.value for param in entry.params.params] != [MX_SELF, *(axis.key for axis in layout.axes)]:
        return None
    expected = cst.parse_expression(dense_condition(layout.axes))
    return body[0] if same_expression(body[0].test, expected) else None


def _argument_range(node, ranges, types, target_type):
    if isinstance(node, cst.Name):
        source_type = types.get(node.value)
        if source_type == target_type or (source_type == 'bint' and target_type == 'long long'):
            return ranges.get(node.value)
    if (target_type == 'long long' and isinstance(node, cst.BinaryOperation)
            and isinstance(node.left, cst.Name) and node.left.value in ranges
            and types[node.left.value] == 'long long' and isinstance(node.right, cst.Integer)
            and isinstance(node.operator, (cst.Add, cst.Subtract))):
        lower, upper = ranges[node.left.value]
        delta = node.right.evaluated_value
        if isinstance(node.operator, cst.Subtract):
            delta = -delta
        if -(1 << 63) < delta < (1 << 63) and -(1 << 63) <= lower + delta < upper + delta <= (1 << 63):
            return lower + delta, upper + delta
    return None


def _attribute_counts(tree):
    return Counter(node.attr.value for node in walk(tree)
                   if isinstance(node, cst.Attribute) and node.attr.value.startswith(FORMULA_PREF))


def _exact_calls(node, receiver, exact=False):
    if isinstance(node, cst.If) and same_expression(node.test, receiver):
        for statement in statements(node.body):
            yield from _exact_calls(statement, receiver, True)
        for statement in _else_body(node):
            yield from _exact_calls(statement, receiver, exact)
        return
    if isinstance(node, cst.IfExp) and same_expression(node.test, receiver):
        yield from _exact_calls(node.body, receiver, True)
        yield from _exact_calls(node.orelse, receiver, exact)
        return
    if exact and isinstance(node, cst.Call) and isinstance(node.func, cst.Name):
        yield node
    if not isinstance(node, OPAQUE_SCOPES):
        for child in node.children:
            yield from _exact_calls(child, receiver, exact)


def _formula_ranges(guard, layout, uses, spill_functions, spill_uses):
    """Bounds common to every generated entry to the same formula/cache."""
    keys = ', '.join(axis.key for axis in layout.axes)
    expected = cst.parse_expression(f'{MX_SELF}.{layout.formula}({keys})')

    def calls_formula(nodes):
        return sum(isinstance(node, cst.Call) and same_expression(node, expected)
                   for statement in nodes for node in walk(statement))

    fallback = _else_body(guard)
    if calls_formula(statements(guard.body)) != 1 or len(fallback) != 1:
        return ()
    fallback, = fallback
    if layout.unbounded_spill:
        # These are bounds of the hot cache branch only. The shared formula
        # also runs through the cold spill; callers must guard these facts.
        expected_spill = cst.parse_expression(f'{layout.spill_entry}({MX_SELF}, {keys})')
        spill = spill_functions.get(layout.spill_entry)
        if (layout.spill_entry and isinstance(fallback, cst.Return)
                and same_expression(fallback.value, expected_spill) and spill
                and calls_formula(statements(spill.body)) == 1):
            return layout.axes
        return ()
    if not layout.exact_spill:
        return layout.axes if uses[layout.formula] == 1 and isinstance(fallback, cst.Raise) else ()
    if not (layout.spill_entry and layout.logical_axes and uses[layout.formula] == 2
            and spill_uses[layout.spill_entry] == 1 and isinstance(fallback, cst.If)
            and len(statements(fallback.body)) == 1 and len(_else_body(fallback)) == 1
            and isinstance(_else_body(fallback)[0], cst.Raise)):
        return ()
    logical = cst.parse_expression(dense_condition(layout.logical_axes))
    spill_call = cst.parse_expression(f'{layout.spill_entry}({MX_SELF}, {keys})')
    branch = statements(fallback.body)[0]
    if (not same_expression(fallback.test, logical) or not isinstance(branch, cst.Return)
            or not same_expression(branch.value, spill_call)):
        return ()
    spill = spill_functions.get(layout.spill_entry)
    if not (spill and [arg.name.value for arg in spill.params.params] == [MX_SELF, *(axis.key for axis in layout.axes)]
            and calls_formula(statements(spill.body)) == 1):
        return ()
    return layout.logical_axes


def prove_entries(state: EntryModule, records, uses, formula_domains_out=None):
    native = {}
    for name, node in state.functions.items():
        if name in records and (guard := _checked_guard(node, records[name])) is not None:
            native[name] = node, guard, records[name]
    formulas = state.methods
    spills = {layout.spill_entry for layout in records.values() if layout.spill_entry}
    spill_functions = {name: node for name, node in state.functions.items() if name in spills}

    def spill_references(nodes):
        return Counter(node.value for node in nodes if isinstance(node, cst.Name)
                       and node.value in spills and state.facts.contexts.get(node) is ExpressionContext.LOAD)

    spill_uses = spill_references(walk(state.syntax)) if spills else Counter()
    ranges_by_entry = {}
    for name, (_, guard, layout) in native.items():
        entry_uses, entry_spills = uses, spill_uses
        public = formulas.get((layout.owner, layout.formula[len(FORMULA_PREF):]))
        public_guard = _checked_guard(public, layout) if public is not None else None
        if public_guard is not None and same_expression(public_guard, guard):
            entry_uses = uses - _attribute_counts(public)
            entry_spills = spill_uses - spill_references(walk(public))
        ranges_by_entry[name] = _formula_ranges(guard, layout, entry_uses, spill_functions, entry_spills)

    masks = {}
    formula_domains = {}
    for name, (_, guard, layout) in native.items():
        source_axes = ranges_by_entry[name]
        formula = formulas.get((layout.owner, layout.formula))
        if not source_axes or formula is None or not any(
                dotted_name(item.decorator) == f'{CY_MOD}.cfunc' for item in formula.decorators):
            continue
        nodes = [node for statement in formula.body.body for node in walk(statement)]
        if any(isinstance(node, getattr(cst, 'Match', ())) for node in nodes):
            continue
        mutable = CstFacts(formula.body.body).bindings(nodes)
        ranges = {axis.key: (axis.lower, axis.stop) for axis in source_axes if axis.key not in mutable}
        types = {axis.key: axis.c_type for axis in source_axes}
        receiver = cst.parse_expression(exact_receiver(layout.owner))
        for call in (call for statement in formula.body.body for call in _exact_calls(statement, receiver)):
            target = call.func.value
            if target not in native or records[target].owner != layout.owner:
                continue
            target_axes = ranges_by_entry[target] or records[target].axes
            args = arguments(call, count=len(target_axes) + 1)
            if args is None or not isinstance(args[0], cst.Name) or args[0].value != MX_SELF:
                continue
            mask = 0
            for index, (arg, axis) in enumerate(zip(args[1:], target_axes)):
                interval = _argument_range(arg, ranges, types, axis.c_type)
                if interval and axis.lower <= interval[0] < interval[1] <= axis.stop:
                    mask |= 1 << index
            if mask:
                if layout.unbounded_spill:
                    if formula not in formula_domains:
                        domain = state.names.reserve('_mx_dense_domain', family=True)
                        formula_domains[formula] = domain, source_axes
                        if formula_domains_out is not None:
                            formula_domains_out[(layout.owner, layout.formula)] = domain, source_axes, name
                    masks[call] = mask, formula_domains[formula][0]
                else:
                    masks[call] = mask, None
    if not masks:
        return state
    flag = state.names.reserve('_mx_proven_axes', family=True)
    targets = {call.func.value for call in masks}

    class Proofs(cst.CSTTransformer):
        def leave_Call(self, before, after):
            if isinstance(before.func, cst.Name) and before.func.value in targets:
                mask, domain = masks.get(before, (0, None))
                value = cst.Integer(str(mask))
                if domain:
                    value = cst.IfExp(cst.Name(domain), value, cst.Integer('0'))
                return after.with_changes(args=(*after.args, cst.Arg(value)))
            return after

        def leave_FunctionDef(self, before, after):
            if before in formula_domains:
                domain, axes = formula_domains[before]
                declaration = cst.parse_statement(f'{domain}: {CY_MOD}.bint = {dense_condition(axes)}\n')
                body = list(after.body.body)
                docstring = (body and isinstance(body[0], cst.SimpleStatementLine)
                             and len(body[0].body) == 1 and isinstance(body[0].body[0], cst.Expr)
                             and isinstance(body[0].body[0].value, (cst.SimpleString, cst.ConcatenatedString)))
                body.insert(int(bool(docstring)), declaration)
                after = after.with_changes(body=after.body.with_changes(body=tuple(body)))
            name = before.name.value
            if name not in targets:
                return after
            layout, proved = records[name], ranges_by_entry[name]
            guard = after.body.body[0]
            fallback = guard.orelse
            if proved and layout.logical_axes and not layout.unbounded_spill:
                test = cst.parse_expression(dense_condition(proved, flag))
                if isinstance(fallback, cst.If):
                    fallback = fallback.with_changes(test=test)
                else:
                    nested = fallback.body.body[0].with_changes(test=test)
                    fallback = fallback.with_changes(body=fallback.body.with_changes(body=(nested,)))
            guard = guard.with_changes(test=cst.parse_expression(dense_condition(layout.axes, flag, proved)),
                                       orelse=fallback)
            parameter = cst.Param(cst.Name(flag), annotation=cst.Annotation(cst.parse_expression(f'{CY_MOD}.uint')))
            return after.with_changes(params=after.params.with_changes(params=(*after.params.params, parameter)),
                                      body=after.body.with_changes(body=(guard,)))

    return state.replace(state.syntax.visit(Proofs()))
