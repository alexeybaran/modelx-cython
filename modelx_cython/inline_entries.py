"""Emit private cache entries for resolved, same-Space numeric Cell calls."""

import ast
import libcst as cst

from modelx_cython.cell_entries import DenseEntry, FormulaEntry, exact_receiver
from modelx_cython.consts import CY_MOD, FORMULA_PREF, HAS_PREF, MX_POW, MX_SELF, MX_SYS_MOD
from .scopes import OPAQUE_SCOPES, binding_names, lexical_nodes


_NUMERIC_TYPES = {"bint", "double", "long long"}


class _Names(cst.CSTVisitor):
    def __init__(self, node):
        self.names = set()
        node.visit(self)

    def visit_Name(self, node):
        self.names.add(node.value)


def allows_exact_receiver(node, owner):
    """Check guard names against the authored formula before emitting guards."""
    return {'type', owner}.isdisjoint(_Names(node).names)


def _eligible(cell, params):
    return (
        cell.has_formula_def
        and not cell.is_locked
        and not cell.formula_is_generator
        and not cell.formula_has_captured_default
        and not cell.body_has_closure
        and not cell.called_with_kwargs
        and cell.get_rettype_expr(c_style=True) in _NUMERIC_TYPES
        and not params.posonly_params
        and not params.kwonly_params
        and not isinstance(params.star_arg, cst.Param)
        and not params.star_kwarg
        and tuple(p.name.value for p in params.params) == (MX_SELF, *cell.params)
        and params.params[0].default is None
        and all(cell.get_argtype_expr(p, c_style=True) in _NUMERIC_TYPES for p in cell.params)
    )


def _literal_defaults(cell, params):
    defaults = tuple(param for param in params.params[1:] if param.default is not None)
    if defaults and not (cell.has_args() and cell.is_arrayable()):
        return None
    values = []
    for param in defaults:
        if cell.get_argtype_expr(param.name.value, c_style=True) != 'long long':
            return None
        default = param.default
        if isinstance(default, cst.Integer):
            value = default.evaluated_value
        elif (isinstance(default, cst.UnaryOperation) and isinstance(default.operator, cst.Minus)
              and isinstance(default.expression, cst.Integer)):
            value = -default.expression.evaluated_value
        else:
            return None
        if not -(1 << 63) <= value < (1 << 63):
            return None
        values.append(default)
    return tuple(values)


class _Calls(cst.CSTTransformer):
    def __init__(self, owner, entries, exact_guards=()):
        self.entries = entries
        self.exact_receiver = cst.parse_expression(exact_receiver(owner))
        self.exact_guards = exact_guards
        self.exact_depth = 0

    def has_exact_guard(self, node):
        return isinstance(node.test, cst.Name) and node.test.value in self.exact_guards

    def proves_exact_receiver(self, node):
        test = node.test
        return (self.has_exact_guard(node) or test.deep_equals(self.exact_receiver)
                or (isinstance(test, cst.BooleanOperation) and isinstance(test.operator, cst.And)
                    and test.left.deep_equals(self.exact_receiver)))

    def visit_If_body(self, node):
        if self.proves_exact_receiver(node):
            self.exact_depth += 1

    def leave_If_body(self, node):
        if self.proves_exact_receiver(node):
            self.exact_depth -= 1

    def leave_If(self, original, updated):
        if self.has_exact_guard(original):
            return updated.with_changes(test=cst.BooleanOperation(
                self.exact_receiver, cst.And(), updated.test))
        return updated

    # These scopes may bind a different receiver or evaluate later.
    def visit_FunctionDef(self, node):
        return False

    visit_ClassDef = visit_Lambda = visit_GeneratorExp = visit_FunctionDef
    visit_ListComp = visit_SetComp = visit_DictComp = visit_FunctionDef

    def visit_BinaryOperation(self, node):
        # Leave operands at their established Cython power boundary.
        return not isinstance(node.operator, cst.Power)

    def visit_Call(self, node):
        func = node.func
        return not (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                    and func.value.value == MX_SYS_MOD and func.attr.value == MX_POW)

    def leave_Call(self, original, updated):
        func = updated.func
        if not (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                and func.value.value == MX_SELF and func.attr.value in self.entries):
            return updated
        entry, arity, defaults = self.entries[func.attr.value]
        missing = arity - len(updated.args)
        if not 0 <= missing <= len(defaults) or any(a.keyword or a.star for a in updated.args):
            return updated
        supplied = tuple(cst.Arg(value) for value in defaults[len(defaults) - missing:])
        direct = updated.with_changes(
            func=cst.Name(entry), args=(cst.Arg(func.value), *updated.args, *supplied))
        if self.exact_depth:
            return direct
        return cst.IfExp(
            test=self.exact_receiver,
            body=direct,
            orelse=updated,
            lpar=(cst.LeftParen(),),
            rpar=(cst.RightParen(),),
        )

    def return_branches(self, expr):
        """Split arithmetic over admitted Cells into exact and virtual expressions."""
        if isinstance(expr, cst.IfExp) and expr.test.deep_equals(self.exact_receiver):
            return expr.body, expr.orelse, 1
        if isinstance(expr, (cst.Integer, cst.Float)):
            return expr, expr, 0
        if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, (cst.Add, cst.Subtract, cst.Multiply)):
            left, right = self.return_branches(expr.left), self.return_branches(expr.right)
            if left is not None and right is not None:
                return (expr.with_changes(left=left[0], right=right[0]),
                        expr.with_changes(left=left[1], right=right[1]),
                        left[2] + right[2])
        return None

    def leave_SimpleStatementLine(self, original, updated):
        if len(updated.body) != 1 or not isinstance(updated.body[0], cst.Return):
            return updated
        statement = updated.body[0]
        branches = self.return_branches(statement.value)
        if branches is None or branches[2] < 2:
            return updated
        fast, fallback, _ = branches
        guard = cst.If(self.exact_receiver, cst.IndentedBlock(body=[
            updated.with_changes(body=[statement.with_changes(value=fast)], leading_lines=[]),
        ]), leading_lines=updated.leading_lines)
        return cst.FlattenSentinel([
            guard, updated.with_changes(body=[statement.with_changes(value=fallback)], leading_lines=[]),
        ])


class _ExactBranches(cst.CSTTransformer):
    """Carry an existing generated type guard into its expression branches."""
    def __init__(self, owner):
        self.test = cst.parse_expression(exact_receiver(owner))
        self.facts = [None]

    def on_visit(self, node):
        return False if isinstance(node, OPAQUE_SCOPES) else super().on_visit(node)

    def enter(self, node, value):
        self.facts.append(value if node.test.deep_equals(self.test) else self.facts[-1])

    def visit_If_body(self, node):
        self.enter(node, True)

    def leave_If_body(self, node):
        self.facts.pop()

    def visit_IfExp_body(self, node):
        self.enter(node, True)

    def leave_IfExp_body(self, node):
        self.facts.pop()

    def visit_IfExp_orelse(self, node):
        self.enter(node, False)

    def leave_IfExp_orelse(self, node):
        self.facts.pop()

    def leave_IfExp(self, before, after):
        if self.facts[-1] is None or not before.test.deep_equals(self.test):
            return after
        value = after.body if self.facts[-1] else after.orelse
        return value.with_changes(lpar=(*after.lpar, *value.lpar),
                                  rpar=(*value.rpar, *after.rpar))


def prune_exact_branches(method, owner):
    # A local receiver reassignment invalidates an enclosing receiver fact.
    body = ast.parse(cst.Module([]).code_for_node(method)).body[0].body
    if MX_SELF in binding_names(lexical_nodes(body)):
        return method
    return method.with_changes(body=method.body.visit(_ExactBranches(owner)))


def add_inline_entries(module, classes):
    """Move each admitted cache body and report its dense layout to the build."""
    used_names = _Names(module).names
    optional_receiver = '_mx_private_receiver'
    while optional_receiver in used_names:
        optional_receiver += '_'
    functions = []
    dense_entries = {}
    formula_entries = {}
    next_entry = 0
    body = []
    for statement in module.body:
        if not isinstance(statement, cst.ClassDef) or statement.name.value not in classes:
            body.append(statement)
            continue

        owner = statement.name.value
        cells = classes[owner].cells
        entries = {}
        methods = []
        for method in statement.body.body:
            cell = cells.get(method.name.value) if isinstance(method, cst.FunctionDef) else None
            if cell is None or cell.is_special() or not _eligible(cell, method.params):
                methods.append(method)
                continue
            defaults = _literal_defaults(cell, method.params)
            if defaults is None:
                methods.append(method)
                continue

            # Reserve a module-level identifier outside all emitted scopes.
            while (entry := f"_mx_cache_entry_{next_entry}") in used_names:
                next_entry += 1
            next_entry += 1
            cell.inline_entry = entry
            entries[cell.name] = entry, len(cell.params), defaults
            formula_entries[entry] = FormulaEntry(owner, FORMULA_PREF + cell.name,
                tuple(cell.params), getattr(cell, 'formula_allows_exact_receiver', False))
            params = method.params
            # The public method always has a receiver. Internal calls prove
            # the exact receiver before entering this private C function.
            receiver = params.params[0].with_changes(annotation=cst.Annotation(
                cst.parse_expression(f'{optional_receiver}[{owner}]')))
            keys = tuple(param.with_changes(default=None, equal=cst.MaybeSentinel.DEFAULT)
                         if param.default is not None else param
                         for param in params.params[1:])
            functions.append(method.with_changes(
                name=cst.Name(entry),
                params=params.with_changes(params=(receiver, *keys)),
                decorators=tuple(cst.Decorator(cst.parse_expression(f"{CY_MOD}.{kind}"))
                                 for kind in ("cfunc", "inline")),
            ))
            # Public methods retain virtual formula dispatch. Private entries
            # are called only behind the exact-receiver proof emitted below.
            # Both entry paths use the same Cell cache fields.
            methods.append(method)

        if entries:
            dense = {}
            for name in entries:
                cell = cells[name]
                if cell.has_args() and cell.is_arrayable():
                    entry = entries[name][0]
                    spill = getattr(cell, 'dense_spill_entry', None)
                    dense_entries[entry] = DenseEntry(owner, FORMULA_PREF + name,
                                                      cell.dense_axes, cell.has_dense_spill,
                                                      cell.dense_logical_axes if spill else (), spill)
                    dense[HAS_PREF + name] = cell
            for index, method in enumerate(methods):
                if isinstance(method, cst.SimpleStatementLine) and len(method.body) == 1:
                    field = method.body[0]
                    if isinstance(field, cst.AnnAssign) and isinstance(field.target, cst.Name):
                        cell = dense.get(field.target.value)
                        if cell is not None:
                            annotation = cst.parse_expression(cell.get_readiness_array_decl_expr())
                            methods[index] = method.with_changes(body=(field.with_changes(
                                annotation=cst.Annotation(annotation)),))
                if not isinstance(method, cst.FunctionDef):
                    continue
                name = method.name.value
                if name.startswith(FORMULA_PREF):
                    cell = cells.get(name[len(FORMULA_PREF):])
                    compiled = (cell is not None and not cell.formula_is_generator
                                and not cell.formula_has_captured_default)
                else:
                    cell = cells.get(name)
                    compiled = (cell is not None and not cell.has_formula_def
                                and not cell.body_has_closure and not cell.called_with_kwargs)
                if not compiled or cell.is_special():
                    continue
                # Use authored names when a formula pass has emitted its own guard.
                guard_allowed = getattr(cell, 'formula_allows_exact_receiver', None)
                if guard_allowed is None:
                    guard_allowed = allows_exact_receiver(method, owner)
                if guard_allowed:
                    calls = _Calls(owner, entries, getattr(cell, 'exact_reduction_guards', ()))
                    rewritten = method.with_changes(body=method.body.visit(calls))
                    methods[index] = prune_exact_branches(rewritten, owner)
        body.append(statement.with_changes(body=statement.body.with_changes(body=methods)))
    if functions:
        index = 0
        for index, statement in enumerate(body):
            if not (isinstance(statement, cst.SimpleStatementLine)
                    and len(statement.body) == 1):
                break
            item = statement.body[0]
            if (index == 0 and isinstance(item, cst.Expr)
                    and isinstance(item.value, (cst.SimpleString, cst.ConcatenatedString))):
                continue
            if (isinstance(item, cst.ImportFrom) and isinstance(item.module, cst.Name)
                    and item.module.value == '__future__'):
                continue
            break
        else:
            index = len(body)
        body.insert(index, cst.parse_statement(
            f'from typing import Optional as {optional_receiver}\n'))
    return module.with_changes(body=(*body, *functions)), dense_entries, formula_entries
