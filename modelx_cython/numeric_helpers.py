"""Native helper values at proven numeric and explicit conversion boundaries."""
import ast
import math
import re

import libcst as cst

from .cell_entries import exact_receiver
from .helpers import module_name
from .scopes import FormulaTransformer, has_scope_effects
from .helper_members import helper_reference


class NumericReturns(cst.CSTTransformer):
    def __init__(self, cls, cell, providers, builtin_keys=()):
        self.cls, self.cell, self.providers = cls, cell, providers
        self.contexts = {}
        self.operations = set()
        self.builtin_keys = builtin_keys
        self.function_depth = 0
        self.scope_safe = True

    def visit_FunctionDef(self, node):
        if not self.function_depth:
            syntax = ast.parse(cst.Module([]).code_for_node(node)).body[0]
            self.scope_safe = not has_scope_effects(syntax.body)
        self.function_depth += 1

    @staticmethod
    def self_call(node):
        return (isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute)
                and isinstance(node.func.value, cst.Name) and node.func.value.value == 'self'
                and not any(arg.keyword or arg.star for arg in node.args))

    def typed_cell_call(self, node, rettype):
        if not self.self_call(node):
            return False
        cell = self.cls.cells.get(node.func.attr.value)
        return (cell is not None and not cell.is_special() and not cell.called_with_kwargs
                and not cell.body_has_closure and cell.get_rettype_expr(c_style=True) == rettype)

    @staticmethod
    def single_statement(line):
        if isinstance(line, cst.SimpleStatementLine) and len(line.body) == 1:
            return line.body[0]
        return None

    @staticmethod
    def assigns(node, name):
        return (isinstance(node, cst.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0].target, cst.Name)
                and node.targets[0].target.value == name)

    def grouped(self, body, key, column, cap, lookup_ref, contains, lookup, prefix, declarations):
        operations = getattr(lookup_ref, 'grouped_helper', None)
        if not (getattr(self.cell, 'formula_allows_exact_receiver', False)
                and operations
                and type(column.evaluated_value) is str
                and len(body) >= 6):
            return None
        group_guard, key_line, table_line = body[-5:-2]
        subtable = self.single_statement(table_line).value
        if not (self.self_call(subtable) and len(subtable.args) == 2
                and isinstance(group_guard, cst.If) and group_guard.orelse is None
                and isinstance(group_guard.test, cst.UnaryOperation)
                and isinstance(group_guard.test.operator, cst.Not)
                and isinstance(group_guard.body, cst.IndentedBlock)
                and len(group_guard.body.body) == 1):
            return None
        has = group_guard.test.expression
        missing = self.single_statement(group_guard.body.body[0])
        if not (self.self_call(has) and len(has.args) == 2
                and isinstance(missing, cst.Return) and isinstance(missing.value, cst.Float)
                and missing.value.value == '0.0'):
            return None
        record, group = (arg.value for arg in subtable.args)
        if not (isinstance(record, cst.Attribute) and isinstance(record.value, cst.Name)
                and record.value.value == 'self' and record.attr.value in self.cls.refs
                and isinstance(group, cst.Name)
                and all(arg.value.deep_equals(other.value) for arg, other in zip(has.args, subtable.args))):
            return None
        provider, _ = lookup_ref.helper
        for call, operation in ((has, 'has_group'), (subtable, 'subtable')):
            ref = self.cls.refs.get(call.func.attr.value)
            if ref is None or ref.helper != (provider, operation):
                return None
        bindings = [self.single_statement(line) for line in body[:-5]]
        if sum(self.assigns(binding, group.value) and self.typed_cell_call(binding.value, 'long long')
               for binding in bindings) != 1:
            return None
        module = cst.Module([])
        prior = ast.parse(''.join(module.code_for_node(line) for line in body[:-5]))
        if sum(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == group.value
               for node in ast.walk(prior)) != 1:
            return None

        code = module.code_for_node
        alias = module_name(provider)
        probe, get = (f'{alias}.{name}' for name in operations[:2])
        context_arg = ''
        if len(operations) == 3 and not self.cell.parent.is_locked:
            context = self.lookup_context(provider)
            if context is not None:
                probe = f'{alias}.{operations[2]}'
                context_arg = f'{code(context)}, '
        slot, view, saved = (prefix + '_' + name for name in ('slot', 'view', 'record'))
        value, status = prefix + '_value', prefix + '_status'
        probe_code = (f'{saved} = {code(record)}\n'
                      f'{view} = {probe}({context_arg}{code(has.func)}, {saved}, {code(group)}, '
                      f'{code(column)}, _mx_cy.address({slot}))\n')
        value_args = (f'{code(subtable.func)}, {code(contains.func)}, {code(lookup.func)}, '
                      f'{view}, {slot}, {code(key)}, _mx_cy.address({value})')
        value_code = f'{status} = {get}({value_args})\n'
        resource_guard = f'{code(record)} is {saved}'
        admission = ''
        bound_operations = getattr(lookup_ref, 'bound_grouped_helper', None)
        if context_arg and bound_operations and len(bound_operations) == 3:
            bound_probe, bound_value, admit_value = (f'{alias}.{name}' for name in bound_operations)
            bound = prefix + '_bound'
            admission = f'{bound}: _mx_cy.bint = False\n'
            probe_code = (f'{bound} = {bound_probe}({code(context)}, {code(group)}, _mx_cy.address({slot}))\n'
                          f'if not {bound}:\n'+''.join('    '+line for line in probe_code.splitlines(True)))
            value_code = (f'if {bound}:\n'
                          f'    {status} = {bound_value}({code(context)}, {slot}, {code(key)}, _mx_cy.address({value}))\n'
                          f'else:\n    {status} = {admit_value}({code(context)}, {value_args})\n')
            resource_guard = f'({bound} or {resource_guard})'
        native = cst.parse_module(
            f'{slot}: _mx_cy.Py_ssize_t = -2\n{view}: object = None\n{saved}: object = None\n'
            f'{admission}'
            f'if {exact_receiver(self.cls.name)}:\n'
            + ''.join('    '+line for line in probe_code.splitlines(True)) +
            f'if {slot} == -1 or ({slot} == -2 and not {code(has)}):\n    return 0.0\n'
            f'{code(key_line)}'
            f'if {slot} >= 0 and {resource_guard}:\n'
            + ''.join('    '+line for line in value_code.splitlines(True)) +
            f'    if {status} == 0:\n        return 0.0\n'
            f'    if {status} == 1:\n        return min({value}, {code(cap)})\n'
        ).body
        return (*body[:-5], *declarations, *native, table_line)

    def bounded(self, body):
        """Recognize a typed key, lower-bound default and upper clamp."""
        if len(body) < 5 or not self.cell.formula_allows_exact_receiver:
            return None
        key_line, lower_line, upper_line, guard, result_line = body[-5:]
        key, lower, upper, result = map(
            self.single_statement, (key_line, lower_line, upper_line, result_line))
        if not (isinstance(result, cst.Return) and self.self_call(result.value)
                and len(result.value.args) == 3
                and all(isinstance(stmt, cst.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0].target, cst.Name)
                        for stmt in (key, lower, upper))
                and self.typed_cell_call(key.value, 'long long')
                and isinstance(guard, cst.If) and guard.orelse is None
                and isinstance(guard.body, cst.IndentedBlock) and len(guard.body.body) == 1):
            return None
        lookup = result.value
        ref = self.cls.refs.get(lookup.func.attr.value)
        operation = getattr(ref, 'bounded_helper', None)
        if not (operation and ref.helper):
            return None
        table, clamp, column = (arg.value for arg in lookup.args)
        key_name, lo, hi = (stmt.targets[0].target for stmt in (key, lower, upper))
        if not (len({key_name.value, lo.value, hi.value}) == 3
                and isinstance(table, cst.Attribute) and isinstance(table.value, cst.Name)
                and table.value.value == 'self' and table.attr.value in self.cls.refs
                and isinstance(column, cst.SimpleString) and type(column.evaluated_value) is str
                and isinstance(clamp, cst.Call) and isinstance(clamp.func, cst.Name)
                and clamp.func.value == 'min' and len(clamp.args) == 2
                and all(not arg.keyword and not arg.star for arg in clamp.args)
                and clamp.args[0].value.deep_equals(key_name)
                and clamp.args[1].value.deep_equals(hi)
                and isinstance(guard.test, cst.Comparison)
                and guard.test.left.deep_equals(key_name) and len(guard.test.comparisons) == 1
                and isinstance(guard.test.comparisons[0].operator, cst.LessThan)
                and guard.test.comparisons[0].comparator.deep_equals(lo)):
            return None
        below = self.single_statement(guard.body.body[0])
        if not (isinstance(below, cst.Return) and isinstance(below.value, cst.Float)):
            return None
        provider, _ = ref.helper
        for call, name in ((lower.value, 'first_key'), (upper.value, 'last_key')):
            if not (self.self_call(call) and len(call.args) == 1
                    and call.args[0].value.deep_equals(table)
                    and getattr(self.cls.refs.get(call.func.attr.value), 'helper', None)
                    == (provider, name)):
                return None
        context = self.lookup_context(provider)
        if context is None:
            return None
        module = cst.Module([])
        used = set(re.findall(r'\b[A-Za-z_]\w*\b',
                              ''.join(module.code_for_node(line) for line in body)))
        value = '_mx_bounded_value'
        while value in used or value in self.cell.params:
            value += '_'
        args = ', '.join(module.code_for_node(arg) for arg in (
            lookup.func, lower.value.func, upper.value.func, context,
            table, key_name, column, below.value))
        admission = f'{module_name(provider)}.{operation}({args}, _mx_cy.address({value}))'
        bound = getattr(ref, 'bound_bounded_helper', None)
        if bound:
            bound_args = ', '.join(module.code_for_node(arg) for arg in (context, key_name, below.value))
            admission = (f'({module_name(provider)}.{bound}({bound_args}, _mx_cy.address({value})) '
                         f'or {admission})')
        fast = cst.parse_module(
            f'{value}: _mx_cy.double\n'
            f'if {exact_receiver(self.cls.name)} and {admission}:\n'
            f'    return {value}\n'
        ).body
        self.providers.add(provider)
        return (*body[:-4], *fast, *body[-4:])

    def leave_FunctionDef(self, before, after):
        """Fuse a terminal membership check with its guarded numeric consumer."""
        self.function_depth -= 1
        if (self.function_depth or not self.scope_safe or 'min' not in self.builtin_keys or self.cell.is_locked
                or self.cell.cls in self.cls.visitor.dynamic_ref_classes
                or self.cell.get_rettype_expr(c_style=True) != 'double'
                or not isinstance(after.body, cst.IndentedBlock) or len(after.body.body) < 4):
            return after
        body = after.body.body
        bounded = self.bounded(body)
        if bounded is not None:
            return after.with_changes(body=after.body.with_changes(body=bounded))
        key_binding, table_binding, guard, result = body[-4:]
        result = self.single_statement(result)
        if not (isinstance(guard, cst.If) and guard.orelse is None
                and isinstance(guard.body, cst.IndentedBlock) and len(guard.body.body) == 1
                and isinstance(guard.test, cst.UnaryOperation) and isinstance(guard.test.operator, cst.Not)
                and isinstance(result, cst.Return) and isinstance(result.value, cst.Call)
                and isinstance(result.value.func, cst.Name) and result.value.func.value == 'min'
                and len(result.value.args) == 2
                and all(not arg.keyword and not arg.star for arg in result.value.args)):
            return after
        absent = self.single_statement(guard.body.body[0])
        contains = guard.test.expression
        lookup, cap = (arg.value for arg in result.value.args)
        if not (isinstance(absent, cst.Return) and isinstance(absent.value, cst.Float)
                and absent.value.value == '0.0' and self.self_call(contains)
                and len(contains.args) == 2 and self.self_call(lookup) and len(lookup.args) == 3
                and self.typed_cell_call(cap, 'double')):
            return after
        table, key, column = (arg.value for arg in lookup.args)
        checked_table, checked_key = (arg.value for arg in contains.args)
        if not (all(isinstance(arg, cst.Name) for arg in (table, key, checked_table, checked_key))
                and table.value == checked_table.value and key.value == checked_key.value
                and isinstance(column, cst.SimpleString) and key.value != table.value):
            return after
        key_binding = self.single_statement(key_binding)
        if (not self.assigns(key_binding, key.value) or not self.typed_cell_call(key_binding.value, 'long long')
                or not self.assigns(self.single_statement(table_binding), table.value)):
            return after
        lookup_ref = self.cls.refs.get(lookup.func.attr.value)
        contains_ref = self.cls.refs.get(contains.func.attr.value)
        operation = getattr(lookup_ref, 'optional_helper', None)
        if not operation or not lookup_ref.helper or contains_ref is None:
            return after
        provider, name = lookup_ref.helper
        if contains_ref.helper != (provider, 'contains'):
            return after

        module = cst.Module([])
        used_names = set(re.findall(r'\b[A-Za-z_]\w*\b', module.code_for_node(after)))
        prefix = '_mx_optional'
        while any(prefix + suffix in used_names for suffix in
                  ('_value', '_status', '_fallback', '_slot', '_view', '_record', '_bound')):
            prefix += '_'
        value, status = prefix + '_value', prefix + '_status'
        fallback = prefix + '_fallback'
        args = ', '.join(module.code_for_node(arg) for arg in
                         (contains.func, lookup.func, table, key, column))
        native = cst.parse_module(
            f'{value}: _mx_cy.double\n{status}: _mx_cy.int\n'
            f'{status} = {module_name(provider)}.{operation}({args}, _mx_cy.address({value}))\n'
            f'if {status} == 0:\n    return 0.0\n'
            f'if {status} == 1:\n    return min({value}, {module.code_for_node(cap)})\n'
        ).body
        # Keep lookup ahead of the typed cap even when Cython lowers min.
        fallback_lookup = cst.parse_statement(
            f'{fallback} = {module.code_for_node(lookup)}\n')
        fallback_result = body[-1].with_changes(body=(result.with_changes(
            value=result.value.with_changes(args=(
                result.value.args[0].with_changes(value=cst.Name(fallback)),
                result.value.args[1],
            )),
        ),))
        self.providers.add(provider)
        self.operations.update((f'{provider}.{name}', f'{provider}.contains'))
        grouped = self.grouped(body, key, column, cap, lookup_ref, contains, lookup, prefix, native[:2])
        if grouped is not None:
            return after.with_changes(body=after.body.with_changes(
                body=(*grouped, *native[2:], guard, fallback_lookup, fallback_result)))
        return after.with_changes(body=after.body.with_changes(
            body=(*body[:-2], *native, guard, fallback_lookup, fallback_result)))

    def lookup_context(self, provider):
        context = f'_v_mx_context_{self.cell.name}'
        if context[len('_v_'):] in self.cls.cells:
            return None
        if context in self.contexts and self.contexts[context] != provider:
            return None
        self.contexts[context] = provider
        return cst.parse_expression('self.' + context)

    def clamped_lookup(self, call, ref):
        if not ('min' in self.builtin_keys
                and getattr(ref, 'clamped_helper', None) and len(call.args) == 3):
            return None
        table, key, column = (arg.value for arg in call.args)
        if (not isinstance(table, cst.Name) or not isinstance(column, cst.SimpleString)
                or not isinstance(key, cst.Call) or not isinstance(key.func, cst.Name)
                or key.func.value != 'min' or len(key.args) != 2
                or any(arg.keyword or arg.star for arg in key.args)):
            return None
        native, last = (arg.value for arg in key.args)
        if not self.self_call(native) or not self.self_call(last) or len(last.args) != 1:
            return None
        cell = self.cls.cells.get(native.func.attr.value)
        last_ref = self.cls.refs.get(last.func.attr.value)
        last_table = last.args[0].value
        provider, lookup = ref.helper
        if (cell is None or cell.is_special() or cell.called_with_kwargs or cell.body_has_closure
                or cell.get_rettype_expr(c_style=True) != 'long long'
                or last_ref is None or last_ref.helper != (provider, 'last_key')
                or not isinstance(last_table, cst.Name) or last_table.value != table.value):
            return None
        self.providers.add(provider)
        self.operations.update((f'{provider}.{lookup}', f'{provider}.last_key'))
        operation = ref.clamped_helper
        args = (call.func, table, native, last.func, column)
        context_operation = getattr(ref, 'clamped_context_helper', None)
        if context_operation:
            context = self.lookup_context(provider)
            if context is not None:
                operation = context_operation
                args = (call.func, context, table, native, last.func, column)
        return call.with_changes(
            func=cst.Attribute(cst.Name(module_name(provider)), cst.Name(operation)),
            args=tuple(cst.Arg(arg) for arg in args),
        )

    def native_key(self, key):
        if isinstance(key, cst.Integer):
            return -(1 << 63) <= key.evaluated_value < (1 << 63)
        if isinstance(key, cst.Name) and key.value in self.cell.params:
            return self.cell.get_argtype_expr(key.value, c_style=True) in ('long long', 'bint')
        if (isinstance(key, cst.Attribute) and isinstance(key.value, cst.Name)
                and key.value.value == 'self'):
            ref = self.cls.refs.get(key.attr.value)
            return ref is not None and ref.get_type_expr(c_style=True) in ('long long','bint')
        if (isinstance(key, cst.Call) and isinstance(key.func, cst.Name)
                and key.func.value in self.builtin_keys and len(key.args) >= 2
                and not any(a.keyword or a.star for a in key.args)):
            return all(self.native_key(a.value) for a in key.args)
        if (isinstance(key, cst.Call) and isinstance(key.func, cst.Attribute)
                and isinstance(key.func.value, cst.Name) and key.func.value.value == 'self'):
            cell = self.cls.cells.get(key.func.attr.value)
            return (cell is not None and not cell.is_special() and not cell.called_with_kwargs
                    and not cell.body_has_closure
                    and cell.get_rettype_expr(c_style=True) in ('long long', 'bint'))
        return False

    def bound_numeric(self, call, native, ref, provider, context):
        operations = getattr(ref, 'bound_numeric_helper', None)
        if not (operations and len(operations) == 3 and self.cell.formula_allows_exact_receiver
                and not self.cell.is_locked and self.cell.cls not in self.cls.visitor.dynamic_ref_classes):
            return native
        table, key, column = (arg.value for arg in call.args[:3])
        module = cst.Module([])
        code = module.code_for_node
        ready = ''
        if (isinstance(table, cst.Attribute) and isinstance(table.value, cst.Name)
                and table.value.value == 'self' and table.attr.value in self.cls.refs):
            source = code(table)
        elif self.self_call(table) and not table.args:
            cell = self.cls.cells.get(table.func.attr.value)
            if not (cell is not None and cell.has_formula_def and not cell.has_args()
                    and not cell.is_locked and not cell.body_has_closure
                    and cell.get_rettype_expr(c_style=True) == 'object'):
                return native
            source = f'self._v_{cell.name}'
            ready = f'self._has_{cell.name} and '
        else:
            return native
        named = isinstance(column, cst.SimpleString) and type(column.evaluated_value) is str
        if not named and not self.native_key(column):
            return native
        default = 0.0
        if len(call.args) == 4:
            try:
                literal = ast.literal_eval(code(call.args[3].value))
                default = float(literal)
                if type(literal) not in (int, float) or not math.isfinite(default) or default != literal:
                    return native
            except (ValueError, TypeError, OverflowError, SyntaxError):
                return native
        alias = module_name(provider)
        column_pointer = f'_mx_cy.cast(_mx_cy.p_void, {code(column)})' if named else '_mx_cy.NULL'
        test = cst.parse_expression(
            f'{exact_receiver(self.cls.name)} and ({ready}{alias}.{operations[0]}('
            f'{code(context)}, _mx_cy.cast(_mx_cy.p_void, {source}), {column_pointer}))')
        result = cst.parse_expression(
            f'{alias}.{operations[1]}({code(context)}, {code(key)}, {0 if named else code(column)}, '
            f'{named}, {len(call.args) == 4}, {default!r})')
        prepare = native.with_changes(func=cst.Attribute(cst.Name(alias), cst.Name(operations[2])))
        return cst.IfExp(test=test, body=result, orelse=prepare,
                         lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))

    def leave_Return(self, before, after):
        if (self.function_depth != 1 or not self.scope_safe or self.cell.get_rettype_expr(c_style=True) != 'double'
                or (not self.cell.has_formula_def and self.cell.called_with_kwargs)):
            return after
        call = after.value
        if not (isinstance(call, cst.Call) and isinstance(call.func, cst.Attribute)
                and isinstance(call.func.value, cst.Name) and call.func.value.value == 'self'
                and 3 <= len(call.args) <= 4 and not any(a.keyword or a.star for a in call.args)):
            return after
        ref = self.cls.refs.get(call.func.attr.value)
        operation = getattr(ref, 'numeric_helper', None)
        if not operation or not ref.helper:
            return after
        clamped = self.clamped_lookup(call, ref)
        if clamped is not None:
            return after.with_changes(value=clamped)
        if not self.native_key(call.args[1].value):
            return after
        provider, _ = ref.helper
        self.operations.add(provider + '.' + ref.helper[1])
        alias = module_name(provider)
        context = self.lookup_context(provider)
        if context is None:
            return after
        self.providers.add(provider)
        native = call.with_changes(
            func=cst.Attribute(cst.Name(alias), cst.Name(operation)),
            args=(cst.Arg(call.func), cst.Arg(context), *call.args))
        return after.with_changes(value=self.bound_numeric(call, native, ref, provider, context))


class ContextualFloat(FormulaTransformer):
    """Lower explicit float(helper(...)) throughout one admitted formula scope.

    The provider receives both current callable expressions. Its fallback must
    perform conversion(helper(*args)), not an implicit C-double conversion.
    Run after generator lowering, so admitted payloads use this same matcher;
    authored child scopes keep their ordinary evaluation semantics.
    """

    def __init__(self, cls, cell, providers, builtin_keys=(), contexts=None, scope_safe=None):
        super().__init__()
        self.cls, self.cell, self.providers = cls, cell, providers
        self.builtin_keys = builtin_keys
        self.contexts = {} if contexts is None else contexts
        self.operations = set()
        self.scope_safe = False
        self.original_scope_safe = scope_safe

    self_call = staticmethod(NumericReturns.self_call)
    native_key = NumericReturns.native_key
    lookup_context = NumericReturns.lookup_context

    def visit_FunctionDef(self, node):
        syntax = ast.parse(cst.Module([]).code_for_node(node)).body[0]
        self.scope_safe = (not has_scope_effects(syntax.body) if self.original_scope_safe is None
                           else self.original_scope_safe)
        return self.scope_safe

    def lower_call(self, call):
        """Return the contextual call, or None when admission is unproven."""
        if not (self.scope_safe and 'float' in self.builtin_keys
                and isinstance(call.func, cst.Name) and call.func.value == 'float'
                and len(call.args) == 1 and not call.args[0].keyword
                and not call.args[0].star):
            return None
        lookup = call.args[0].value
        if not (isinstance(lookup, cst.Call) and 3 <= len(lookup.args) <= 4
                and not any(arg.keyword or arg.star for arg in lookup.args)):
            return None
        ref = helper_reference(lookup.func, self.cls.refs)
        operation = getattr(ref, 'float_numeric_helper', None)
        if not getattr(ref, 'helper', None):
            return None
        key = lookup.args[1].value
        args = lookup.args
        if not (operation and self.component_kind(key) == 'long long'):
            operations = getattr(ref, 'keyed_float_numeric_helper', None)
            tuple_key = isinstance(key, cst.Tuple)
            if tuple_key:
                if any(not isinstance(part, cst.Element) for part in key.elements):
                    return None
                parts = tuple(part.value for part in key.elements)
            else:
                parts = (key,)
            if not (operations and len(parts) < len(operations)
                    and all(self.native_component(part) for part in parts)):
                return None
            operation = operations[len(parts)]
            args = (lookup.args[0], *(cst.Arg(part) for part in parts),
                    cst.Arg(cst.Name('True' if tuple_key else 'False')), *lookup.args[2:])
        provider, original = ref.helper
        context = self.lookup_context(provider)
        if context is None:
            return None
        self.providers.add(provider)
        self.operations.add(provider + '.' + original)
        return call.with_changes(
            func=cst.Attribute(cst.Name(module_name(provider)), cst.Name(operation)),
            args=(cst.Arg(call.func), cst.Arg(lookup.func), cst.Arg(context), *args))

    def native_component(self, key):
        return self.component_kind(key) is not None

    def component_kind(self, key):
        kinds = ('long long', 'bint', 'str', 'unicode')
        if isinstance(key, cst.Integer):
            return 'long long' if -(1 << 63) <= key.evaluated_value < (1 << 63) else None
        if isinstance(key, cst.SimpleString):
            return 'str' if type(key.evaluated_value) is str else None
        if isinstance(key, cst.Name):
            if key.value in ('True', 'False'):
                return 'bint'
            if key.value in self.cell.params:
                kind = self.cell.get_argtype_expr(key.value, c_style=True)
                return kind if kind in kinds else None
        if (isinstance(key, cst.UnaryOperation) and isinstance(key.expression, cst.Integer)
                and isinstance(key.operator, (cst.Minus, cst.Plus))):
            value = key.expression.evaluated_value * (-1 if isinstance(key.operator, cst.Minus) else 1)
            return 'long long' if -(1 << 63) <= value < (1 << 63) else None
        if isinstance(key, cst.Attribute) and isinstance(key.value, cst.Name) and key.value.value == 'self':
            ref = self.cls.refs.get(key.attr.value)
            kind = ref.get_type_expr(c_style=True) if ref is not None else None
            return kind if kind in kinds else None
        if (isinstance(key, cst.Call) and isinstance(key.func, cst.Name)
                and key.func.value in self.builtin_keys and key.func.value in ('min', 'max')
                and len(key.args) >= 2 and not any(arg.keyword or arg.star for arg in key.args)):
            result = {self.component_kind(arg.value) for arg in key.args}
            return result.pop() if len(result) == 1 else None
        if self.self_call(key):
            cell = self.cls.cells.get(key.func.attr.value)
            if cell is not None and not cell.is_special() and not cell.called_with_kwargs and not cell.body_has_closure:
                kind = cell.get_rettype_expr(c_style=True)
                return kind if kind in kinds else None
        return None

    def leave_Call(self, before, after):
        return self.lower_call(after) or after
