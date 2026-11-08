"""Reuse ranges that generated dense entries already check before a miss."""

import ast
from collections import Counter, defaultdict

from .cell_entries import dense_condition, exact_receiver
from .consts import CY_MOD, FORMULA_PREF, MX_SELF
from .scopes import binding_names as _binding_names


def _checked_guard(entry, layout):
    if len(entry.body) != 1 or not isinstance(entry.body[0], ast.If):
        return None
    guard = entry.body[0]
    axes = layout.axes
    if not 0 < len(axes) <= 32:
        return None
    if [param.arg for param in entry.args.args] != [MX_SELF, *(axis.key for axis in axes)]:
        return None
    expected = ast.parse(dense_condition(axes), mode="eval").body
    return guard if ast.dump(guard.test) == ast.dump(expected) else None


def _argument_range(node, ranges, types, target_type):
    if isinstance(node, ast.Name):
        source_type = types.get(node.id)
        if source_type == target_type or (source_type == "bint" and target_type == "long long"):
            return ranges.get(node.id)
    if (target_type == "long long" and isinstance(node, ast.BinOp) and isinstance(node.left, ast.Name)
            and node.left.id in ranges and types[node.left.id] == "long long"
            and isinstance(node.right, ast.Constant) and type(node.right.value) is int
            and isinstance(node.op, (ast.Add, ast.Sub))):
        lower, upper = ranges[node.left.id]
        delta = node.right.value if isinstance(node.op, ast.Add) else -node.right.value
        if -(1 << 63) < delta < (1 << 63) and -(1 << 63) <= lower + delta < upper + delta <= (1 << 63):
            return lower + delta, upper + delta
    return None


def _attribute_counts(tree):
    return Counter(node.attr for node in ast.walk(tree)
                   if isinstance(node, ast.Attribute) and node.attr.startswith(FORMULA_PREF))


def _exact_calls(node, receiver, exact=False):
    if isinstance(node, ast.If) and ast.dump(node.test) == receiver:
        for statement in node.body:
            yield from _exact_calls(statement, receiver, True)
        for statement in node.orelse:
            yield from _exact_calls(statement, receiver, exact)
        return
    if isinstance(node, ast.IfExp) and ast.dump(node.test) == receiver:
        yield from _exact_calls(node.body, receiver, True)
        yield from _exact_calls(node.orelse, receiver, exact)
        return
    if exact and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        yield node
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                             ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
        for child in ast.iter_child_nodes(node):
            yield from _exact_calls(child, receiver, exact)


def _formula_ranges(guard, layout, uses, spill_functions, spill_uses):
    """Return the bounds common to every generated entry to the formula."""
    keys = ', '.join(axis.key for axis in layout.axes)
    expected = ast.dump(ast.parse(f'{MX_SELF}.{layout.formula}({keys})', mode='eval').body)

    def calls_formula(nodes):
        return sum(isinstance(node, ast.Call) and ast.dump(node) == expected
                   for statement in nodes for node in ast.walk(statement))

    if calls_formula(guard.body) != 1 or len(guard.orelse) != 1:
        return ()
    fallback, = guard.orelse
    if not layout.exact_spill:
        return layout.axes if uses[layout.formula] == 1 and isinstance(fallback, ast.Raise) else ()
    if not (layout.spill_entry and layout.logical_axes and uses[layout.formula] == 2
            and spill_uses[layout.spill_entry] == 1
            and isinstance(fallback, ast.If) and len(fallback.body) == 1
            and len(fallback.orelse) == 1 and isinstance(fallback.orelse[0], ast.Raise)):
        return ()
    logical = ast.parse(dense_condition(layout.logical_axes), mode='eval').body
    spill_call = ast.parse(f'return {layout.spill_entry}({MX_SELF}, {keys})').body[0]
    if ast.dump(fallback.test) != ast.dump(logical) or ast.dump(fallback.body[0]) != ast.dump(spill_call):
        return ()
    spill = spill_functions.get(layout.spill_entry)
    if not (spill and [arg.arg for arg in spill.args.args] == [MX_SELF, *(axis.key for axis in layout.axes)]
            and calls_formula(spill.body) == 1):
        return ()
    return layout.logical_axes


def _rewrite(source, tree, records, uses):
    native = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in records:
            layout = records[node.name]
            guard = _checked_guard(node, layout)
            if guard is not None:
                native[node.name] = node, guard, layout
    formulas = {(owner.name, fn.name): fn for owner in tree.body if isinstance(owner, ast.ClassDef)
                for fn in owner.body if isinstance(fn, ast.FunctionDef)}
    spills = {layout.spill_entry for layout in records.values() if layout.spill_entry}
    spill_functions = {node.name: node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name in spills}
    spill_uses = Counter(node.id for node in ast.walk(tree)
                         if isinstance(node, ast.Name) and node.id in spills) if spills else Counter()
    calls = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in native:
            calls[node.func.id].append(node)

    ranges_by_entry = {}
    for name, (_, guard, layout) in native.items():
        entry_uses, entry_spills = uses, spill_uses
        public = formulas.get((layout.owner, layout.formula[len(FORMULA_PREF):]))
        public_guard = _checked_guard(public, layout) if public is not None else None
        if public_guard is not None and ast.dump(public_guard) == ast.dump(guard):
            # The public cache entry repeats the same checked path to the same
            # formula/cache. Account for it without admitting unchecked uses.
            entry_uses = uses - _attribute_counts(public)
            entry_spills = spill_uses - Counter(node.id for node in ast.walk(public)
                if isinstance(node, ast.Name) and node.id in spills)
        ranges_by_entry[name] = _formula_ranges(
            guard, layout, entry_uses, spill_functions, entry_spills)
    masks = {}
    for name, (_, guard, layout) in native.items():
        source_axes = ranges_by_entry[name]
        if not source_axes:
            continue
        formula = formulas.get((layout.owner, layout.formula))
        if formula is None or not any(
                isinstance(decorator, ast.Attribute) and decorator.attr == "cfunc"
                and isinstance(decorator.value, ast.Name) and decorator.value.id == CY_MOD
                for decorator in formula.decorator_list):
            continue
        nodes = list(ast.walk(formula))
        if any(isinstance(node, getattr(ast, "Match", ())) for node in nodes):
            continue
        mutable = _binding_names(nodes)
        ranges = {axis.key: (axis.lower, axis.stop) for axis in source_axes if axis.key not in mutable}
        types = {axis.key: axis.c_type for axis in source_axes}
        receiver = ast.dump(ast.parse(exact_receiver(layout.owner), mode="eval").body)
        # Private guarded calls are emitted only in the current scope by inline_entries.
        for call in (call for statement in formula.body for call in _exact_calls(statement, receiver)):
            target = call.func.id
            if target not in native or records[target].owner != layout.owner:
                continue
            target_axes = ranges_by_entry[target] or records[target].axes
            if len(call.args) != len(target_axes) + 1 or call.keywords:
                continue
            receiver_arg = call.args[0]
            if not isinstance(receiver_arg, ast.Name) or receiver_arg.id != MX_SELF:
                continue
            mask = 0
            for i, (arg, axis) in enumerate(zip(call.args[1:], target_axes)):
                interval = _argument_range(arg, ranges, types, axis.c_type)
                if interval and axis.lower <= interval[0] < interval[1] <= axis.stop:
                    mask |= 1 << i
            if mask:
                masks[call] = mask

    if not masks:
        return source
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    changes = []
    flag = "_mx_proven_axes"
    while flag.encode() in source:
        flag += "_"

    def span(node):
        # Python AST columns are UTF-8 byte offsets.
        return (offsets[node.lineno - 1] + node.col_offset,
                offsets[node.end_lineno - 1] + node.end_col_offset)

    for name in {call.func.id for call in masks}:
        fn, guard, layout = native[name]
        _, end = span(fn.args.args[-1])
        changes.append((end, end, f", {flag}: {CY_MOD}.uint".encode()))
        proved_axes = ranges_by_entry[name]
        condition = dense_condition(layout.axes, flag, proved_axes)
        changes.append((*span(guard.test), condition.encode()))
        if proved_axes and layout.logical_axes:
            logical_guard = guard.orelse[0]
            logical = dense_condition(proved_axes, flag)
            changes.append((*span(logical_guard.test), logical.encode()))
        for call in calls[name]:
            _, end = span(call)
            changes.append((end - 1, end - 1, f", {masks.get(call, 0)}".encode()))
    for start, stop, replacement in sorted(changes, reverse=True):
        source = source[:start] + replacement + source[stop:]
    return source


def reuse_dense_bounds(modules, records):
    """Optimize private entries across the translated model, with an external-use audit."""
    if not records:
        return
    uses = Counter()
    inputs = {}
    for module, path in modules.items():
        source = path.read_bytes()
        tree = ast.parse(source)
        uses.update(_attribute_counts(tree))
        if module in records:
            inputs[module] = path, source, tree
    for module, (path, source, tree) in inputs.items():
        rewritten = _rewrite(source, tree, records[module], uses)
        if rewritten != source:
            path.write_bytes(rewritten)
