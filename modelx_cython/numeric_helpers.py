"""Native helper returns only at an already-double formula return boundary."""
import math

import libcst as cst

from .cell_entries import exact_receiver
from .helper_members import helper_reference
from .helpers import module_name
from .scopes import CstFacts, Names, has_scope_effects
from .syntax import arguments, literal_value, walk
from .typedefs import primitive_kind


class _ReplaceExpression(cst.CSTTransformer):
    def __init__(self, source, replacement):
        self.source, self.replacement = source, replacement

    def on_leave(self, before, after):
        return self.replacement if before is self.source else after


class NumericReturns(cst.CSTTransformer):
    def __init__(self, cls, cell, providers, builtin_keys=(), resolver=None):
        self.cls, self.cell, self.providers = cls, cell, providers
        self.contexts = {}
        self.operations = set()
        self.builtin_keys = builtin_keys
        self.resolver = resolver
        self.function_depth = 0
        self.scope_safe = True

    def visit_FunctionDef(self, node):
        if not self.function_depth:
            self.scope_safe = not has_scope_effects(node.body.body)
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

    def direct_ref(self, node):
        return (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name)
                and node.value.value == 'self' and node.attr.value in self.cls.refs)

    def locked_result_type(self, node):
        if self.resolver is None or not isinstance(node, cst.Call) or node.args:
            return None
        from .usage import RESOLVED, _extract_self_chain
        chain = _extract_self_chain(node.func)
        if not isinstance(chain, list):
            return None
        status, name = self.resolver.resolve_chain(self.cls.fqname, chain)
        if status == RESOLVED:
            return self.resolver.locked_scalar_results.get(name)
        return None

    def stable_input_guard(self, node, body):
        """A cached shared Cell can use the collect-owned binding on an exact Space."""
        if self.direct_ref(node):
            return ''
        if self.locked_result_type(node) != 'object':
            return None
        from .usage import _extract_self_chain
        chain = _extract_self_chain(node.func)
        if len(chain) != 2:
            return None
        owner = self.cls.refs.get(chain[0])
        kind = getattr(owner, 'decl_type_expr', None)
        if not kind or not kind.isidentifier() or kind in Names(cst.Module(body)).names:
            return None
        return f'type(self.{chain[0]}) is {kind} and '

    def grouped(self, body, key, column, cap, lookup_ref, contains, lookup, prefix, declarations,
                *, methods=False):
        operations = (('group_slice_context_probe', 'group_slice_context_value') if methods
                      else getattr(lookup_ref, 'grouped_helper', None))
        if not (getattr(self.cell, 'formula_allows_exact_receiver', False)
                and operations
                and type(column.evaluated_value) is str
                and len(body) >= 6):
            return None
        group_guard, key_line, table_line = body[-5:-2]
        subtable = self.single_statement(table_line).value
        if not (isinstance(group_guard, cst.If) and group_guard.orelse is None
                and isinstance(group_guard.body, cst.IndentedBlock)
                and len(group_guard.body.body) == 1):
            return None
        missing = self.single_statement(group_guard.body.body[0])
        if not (isinstance(missing, cst.Return) and isinstance(missing.value, cst.Float)
                and missing.value.value == '0.0'):
            return None
        if methods:
            from .builder import NATIVE_TABLE_PROVIDER
            comparison = self.not_in(group_guard.test)
            args = arguments(subtable, count=1)
            if not (comparison and args and isinstance(subtable.func, cst.Attribute)
                    and subtable.func.attr.value == 'slice'):
                return None
            group, record = comparison
            if not (subtable.func.value.deep_equals(record) and args[0].deep_equals(group)):
                return None
            provider = NATIVE_TABLE_PROVIDER
            probe_functions = value_functions = ''
        else:
            if not (isinstance(group_guard.test, cst.UnaryOperation)
                    and isinstance(group_guard.test.operator, cst.Not)
                    and self.self_call(subtable) and len(subtable.args) == 2):
                return None
            has = group_guard.test.expression
            if not (self.self_call(has) and len(has.args) == 2
                    and all(arg.value.deep_equals(other.value) for arg, other in zip(has.args, subtable.args))):
                return None
            record, group = (arg.value for arg in subtable.args)
            provider, _ = lookup_ref.helper
            for call, operation in ((has, 'has_group'), (subtable, 'subtable')):
                ref = self.cls.refs.get(call.func.attr.value)
                if ref is None or ref.helper != (provider, operation):
                    return None
            code = cst.Module([]).code_for_node
            probe_functions = f'{code(has.func)}, '
            value_functions = ', '.join(code(call.func) for call in (subtable, contains, lookup)) + ', '
        direct = self.direct_ref(record)
        if not ((direct or self.locked_result_type(record) == 'object')
                and isinstance(group, cst.Name)):
            return None
        bindings = [self.single_statement(line) for line in body[:-5]]
        if sum(self.assigns(binding, group.value) and self.typed_cell_call(binding.value, 'long long')
               for binding in bindings) != 1:
            return None
        module = cst.Module([])
        if CstFacts(body[:-5]).stores[group.value] != 1:
            return None

        code = module.code_for_node
        alias = module_name(provider)
        probe, get = (f'{alias}.{name}' for name in operations[:2])
        context_arg = ''
        if methods or len(operations) == 3 and not self.cell.parent.is_locked:
            context = self.lookup_context(provider)
            if context is not None:
                if not methods:
                    probe = f'{alias}.{operations[2]}'
                context_arg = f'{code(context)}, '
        slot, view, saved = (prefix + '_' + name for name in ('slot', 'view', 'record'))
        value, status = prefix + '_value', prefix + '_status'
        probe_code = (f'{saved} = {code(record)}\n'
                      f'{view} = {probe}({context_arg}{probe_functions}{saved}, {code(group)}, '
                      f'{code(column)}, _mx_cy.address({slot}))\n')
        value_args = (f'{value_functions}'
                      f'{view}, {slot}, {code(key)}, _mx_cy.address({value})')
        value_code = f'{status} = {get}({context_arg if methods else ""}{value_args})\n'
        resource_guard = f'{code(record)} is {saved}'
        admission = ''
        bound_operations = (('bound_group_probe', 'bound_group_value', operations[1]) if methods
                            else getattr(lookup_ref, 'bound_grouped_helper', None))
        stable_guard = self.stable_input_guard(record, body)
        if stable_guard is not None and context_arg and bound_operations and len(bound_operations) == 3:
            bound_probe, bound_value, admit_value = (f'{alias}.{name}' for name in bound_operations)
            bound = prefix + '_bound'
            admission = f'{bound}: _mx_cy.bint = False\n'
            probe_code = (f'{bound} = {stable_guard}{bound_probe}({code(context)}, {code(group)}, _mx_cy.address({slot}))\n'
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
            f'if {slot} == -1 or ({slot} == -2 and ({code(group_guard.test)})):\n    return 0.0\n'
            f'{code(key_line)}'
            f'if {slot} >= 0 and {resource_guard}:\n'
            + ''.join('    '+line for line in value_code.splitlines(True)) +
            f'    if {status} == 0:\n        return 0.0\n'
            f'    if {status} == 1:\n        return min({value}, {code(cap)})\n'
        ).body
        return (*body[:-5], *declarations, *native, table_line)

    @staticmethod
    def not_in(expression):
        if (isinstance(expression, cst.Comparison) and len(expression.comparisons) == 1
                and isinstance(expression.comparisons[0].operator, cst.NotIn)):
            return expression.left, expression.comparisons[0].comparator

    def grouped_table_method(self, body):
        """Recognize the existing grouped read through slice and index methods."""
        key_line, table_line, guard, result_line = body[-4:]
        result = self.single_statement(result_line)
        if not (isinstance(result, cst.Return) and isinstance(guard, cst.If)
                and guard.orelse is None and isinstance(guard.body, cst.IndentedBlock)
                and len(guard.body.body) == 1):
            return None
        args = arguments(result.value, 'min', 2)
        absence = self.not_in(guard.test)
        absent = self.single_statement(guard.body.body[0])
        if not (args and absence and isinstance(absent, cst.Return)
                and isinstance(absent.value, cst.Float) and absent.value.evaluated_value == 0.0):
            return None
        lookup, cap = args
        key, index = absence
        lookup_args = arguments(lookup, count=2)
        if not (lookup_args and isinstance(lookup.func, cst.Attribute) and lookup.func.attr.value == 'lookup_float'
                and isinstance(index, cst.Attribute) and index.attr.value == 'index'
                and isinstance(index.value, cst.Name) and index.value.deep_equals(lookup.func.value)
                and isinstance(key, cst.Name) and key.deep_equals(lookup_args[0])
                and isinstance(lookup_args[1], cst.SimpleString)
                and self.typed_cell_call(cap, 'double')):
            return None
        key_binding = self.single_statement(key_line)
        if not (self.assigns(key_binding, key.value) and self.typed_cell_call(key_binding.value, 'long long')
                and self.assigns(self.single_statement(table_line), index.value.value)):
            return None
        prefix = Names(cst.Module(body)).reserve('_mx_optional', family=True)
        declarations = cst.parse_module(
            f'{prefix}_value: _mx_cy.double\n{prefix}_status: _mx_cy.int\n').body
        grouped = self.grouped(body, key, lookup_args[1], cap, None, None, lookup, prefix, declarations,
                               methods=True)
        if grouped is not None:
            from .builder import NATIVE_TABLE_PROVIDER
            self.providers.add(NATIVE_TABLE_PROVIDER)
            self.operations.add(f'{NATIVE_TABLE_PROVIDER}.group_slice_context_value')
            return (*grouped, guard, result_line)
        return None

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
        direct = self.direct_ref(table)
        if not (len({key_name.value, lo.value, hi.value}) == 3
                and (direct or self.locked_result_type(table) == 'object')
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
        value = Names(cst.Module(body), used=self.cell.params).reserve('_mx_bounded_value')
        args = ', '.join(module.code_for_node(arg) for arg in (
            lookup.func, lower.value.func, upper.value.func, context,
            table, key_name, column, below.value))
        admission = f'{module_name(provider)}.{operation}({args}, _mx_cy.address({value}))'
        bound = getattr(ref, 'bound_bounded_helper', None)
        stable_guard = self.stable_input_guard(table, body)
        if stable_guard is not None and bound:
            bound_args = ', '.join(module.code_for_node(arg) for arg in (context, key_name, below.value))
            admission = (f'(({stable_guard}{module_name(provider)}.{bound}({bound_args}, _mx_cy.address({value}))) '
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
        native_clamp = self.clamped_table_method(body)
        if native_clamp is not None:
            return after.with_changes(body=after.body.with_changes(body=native_clamp))
        fields = self.table_fields(body)
        if fields is not None:
            return after.with_changes(body=after.body.with_changes(body=fields))
        rounded = self.rounded_inputs(body)
        if rounded is not None:
            return after.with_changes(body=after.body.with_changes(body=rounded))
        bounded = self.bounded(body)
        if bounded is not None:
            return after.with_changes(body=after.body.with_changes(body=bounded))
        bounded = self.bounded_table_method(body)
        if bounded is not None:
            return after.with_changes(body=after.body.with_changes(body=bounded))
        grouped = self.grouped_table_method(body)
        if grouped is not None:
            return after.with_changes(body=after.body.with_changes(body=grouped))
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
        prefix = Names(after).reserve('_mx_optional', family=True)
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
        for name, owner in self.contexts.items():
            if owner == provider:
                return cst.parse_expression('self.' + name)
        context = f'_v_mx_context_{self.cell.name}'
        # Keep the existing single-provider layout; a second provider owns a
        # distinct extension type and must never receive the first's context.
        if context in self.contexts:
            context += module_name(provider)
        cache_fields = (prefix + name for name in self.cls.cells for prefix in ('_v_', '_has_'))
        previous = getattr(self.cls, 'native_contexts', {})
        context = Names(used=(*previous, *self.contexts, *self.cls.refs, *cache_fields)).reserve(context)
        self.contexts[context] = provider
        return cst.parse_expression('self.' + context)

    def table_source(self, value):
        from .builder import NATIVE_TABLE_PROVIDER
        cell = self.cls.cells.get(value.func.attr.value) if self.self_call(value) else None
        return ((cell is not None and cell.native_table_result and len(value.args) == len(cell.params))
                or self.locked_result_type(value) == module_name(NATIVE_TABLE_PROVIDER) + '.NativeTable')

    def rounded_inputs(self, body):
        if 'round' not in self.builtin_keys or self.resolver is None:
            return None
        result = self.single_statement(body[-1])
        if not isinstance(result, cst.Return):
            return None
        expression = result.value
        choices = (expression.body, expression.orelse) if isinstance(expression, cst.IfExp) else (expression,)
        calls = [node for node in choices if arguments(node, 'round', 2) is not None]
        if len(calls) != 1:
            return None
        call, = calls
        from .builder import NATIVE_TABLE_PROVIDER
        from .powers import OperandKind, KIND_FLOAT, KIND_INT, KIND_UNKNOWN, kind_of_type_expr
        facts = CstFacts(body)
        cells = {cell.fqname: kind_of_type_expr(cell.get_rettype_expr(c_style=True))
                 for cls in self.cls.module.classes.values() for cell in cls.cells.values()
                 if cell.has_typeinfo()}
        refs = {self.cls.fqname: {
            name: kind_of_type_expr(ref.get_type_expr(c_style=True))
            for name, ref in self.cls.refs.items()}}
        kinds = {name: kind_of_type_expr(self.cell.get_argtype_expr(name, c_style=True, formula=True))
                 for name in self.cell.params}
        classifier = OperandKind(self.resolver, cells, refs, self.cls.fqname, kinds)
        integer_locals = set()
        for line in body[:-1]:
            statement = self.single_statement(line)
            if (isinstance(statement, cst.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0].target, cst.Name)):
                name = statement.targets[0].target.value
                kind = classifier.kind(statement.value)
                if (facts.stores[name] == 1 and kind != KIND_UNKNOWN
                        and not any(isinstance(node, cst.Power) for node in walk(statement.value))):
                    kinds[name] = kind
                    if kind == KIND_INT and self.native_key(statement.value, allow_bool=False):
                        integer_locals.add(name)
        number, digits = (arg.value for arg in call.args)
        if (classifier.kind(number) != KIND_FLOAT
                or any(isinstance(node, cst.Power) for node in walk(number))
                or not (self.native_key(digits, allow_bool=False)
                        or isinstance(digits, cst.Name) and digits.value in integer_locals)):
            return None
        operation = 'repeated_round_float'
        alias = module_name(NATIVE_TABLE_PROVIDER)
        replacement = call.with_changes(
            func=cst.Attribute(cst.Name(alias), cst.Name(operation)),
            args=(cst.Arg(self.lookup_context(NATIVE_TABLE_PROVIDER)), cst.Arg(call.func), *call.args))
        self.providers.add(NATIVE_TABLE_PROVIDER)
        self.operations.add(f'{NATIVE_TABLE_PROVIDER}.{operation}')
        return (*body[:-1], body[-1].visit(_ReplaceExpression(call, replacement)))

    def table_fields(self, body):
        """Bind literal-column reads on a locally assigned native Table."""
        from .builder import NATIVE_TABLE_PROVIDER
        facts = CstFacts(body)
        sources, text_keys = set(), set()
        replaced = []
        changed = False
        for line in body:
            statement = self.single_statement(line)
            if isinstance(statement, cst.Assign) and len(statement.targets) == 1:
                target, value = statement.targets[0].target, statement.value
                if (isinstance(target, cst.Name) and target.value not in self.cell.params
                        and facts.stores[target.value] == 1 and self.table_source(value)):
                    sources.add(target.value)
                elif (isinstance(target, cst.Name) and target.value not in self.cell.params
                        and facts.stores[target.value] == 1 and self.typed_cell_call(value, 'str')):
                    text_keys.add(target.value)
                elif (isinstance(value, cst.Call) and isinstance(value.func, cst.Attribute)
                        and value.func.attr.value == 'lookup_float'
                        and isinstance(value.func.value, cst.Name) and value.func.value.value in sources
                        and len(value.args) == 2 and not any(arg.keyword or arg.star for arg in value.args)
                        and isinstance(value.args[0].value, cst.Name) and value.args[0].value.value in text_keys
                        and isinstance(value.args[1].value, cst.SimpleString)
                        and type(value.args[1].value.evaluated_value) is str):
                    operation = 'read_table_field_float'
                    alias = module_name(NATIVE_TABLE_PROVIDER)
                    self.providers.add(NATIVE_TABLE_PROVIDER)
                    self.operations.add(f'{NATIVE_TABLE_PROVIDER}.{operation}')
                    line = line.with_changes(body=(statement.with_changes(value=value.with_changes(
                        func=cst.Attribute(cst.Name(alias), cst.Name(operation)),
                        args=(cst.Arg(self.lookup_context(NATIVE_TABLE_PROVIDER)),
                              cst.Arg(value.func.value), *value.args))),))
                    changed = True
            replaced.append(line)
        return tuple(replaced) if changed else None

    def first_lookup(self, expression):
        """Find an initial table read without moving an effectful operand."""
        if isinstance(expression, cst.BinaryOperation) and isinstance(
                expression.operator, (cst.Add, cst.Subtract, cst.Multiply, cst.Divide)):
            operand = (expression.right if isinstance(expression.left, (cst.Integer, cst.Float))
                       else expression.left)
            return self.first_lookup(operand)
        if not isinstance(expression, cst.Call):
            return None
        function = expression.func
        if (isinstance(function, cst.Name) and function.value in {'min', 'max'}
                and function.value in self.builtin_keys and len(expression.args) == 2
                and not any(arg.keyword or arg.star for arg in expression.args)):
            left, right = (arg.value for arg in expression.args)
            return self.first_lookup(right if isinstance(left, (cst.Integer, cst.Float)) else left)
        if (isinstance(function, cst.Attribute) and function.attr.value == 'lookup_float'
                and isinstance(function.value, cst.Name) and len(expression.args) == 2
                and not any(arg.keyword or arg.star for arg in expression.args)):
            return expression
        return None

    @staticmethod
    def index_max_int(node, table):
        return (isinstance(node, cst.Call) and not node.args and isinstance(node.func, cst.Attribute)
                and node.func.attr.value == 'max_int' and isinstance(node.func.value, cst.Attribute)
                and node.func.value.attr.value == 'index' and node.func.value.value.deep_equals(table))

    def bounded_table_method(self, body):
        """Bind the existing below-range default and upper clamp on an owned Table."""
        if len(body) < 6 or 'int' not in self.builtin_keys:
            return None
        source, key, low, high = map(self.single_statement, body[-6:-2])
        guard, result_line = body[-2:]
        result = self.single_statement(result_line)
        if not (isinstance(result, cst.Return) and isinstance(guard, cst.If) and guard.orelse is None
                and isinstance(guard.body, cst.IndentedBlock) and len(guard.body.body) == 1
                and all(isinstance(item, cst.Assign) and len(item.targets) == 1
                        and isinstance(item.targets[0].target, cst.Name) for item in (source, key, low, high))):
            return None
        table, key_name, low_name, high_name = (item.targets[0].target for item in (source, key, low, high))
        if (len({name.value for name in (table, key_name, low_name, high_name)}) != 4
                or not self.table_source(source.value) or not self.typed_cell_call(key.value, 'long long')):
            return None
        first = arguments(low.value, 'int', 1)
        lookup = self.first_lookup(result.value)
        below = self.single_statement(guard.body.body[0])
        if not (first and lookup is result.value and lookup.func.value.deep_equals(table)
                and isinstance(below, cst.Return) and isinstance(below.value, cst.Float)
                and isinstance(guard.test, cst.Comparison) and len(guard.test.comparisons) == 1
                and isinstance(guard.test.comparisons[0].operator, cst.LessThan)
                and guard.test.left.deep_equals(key_name)
                and guard.test.comparisons[0].comparator.deep_equals(low_name)
                and self.index_max_int(high.value, table)):
            return None
        clamp = arguments(lookup.args[0].value, 'min', 2)
        column = lookup.args[1].value
        first_arg = arguments(first[0], count=1)
        index = first_arg[0] if first_arg else None
        owned_index = (arguments(first[0], 'min', 1) is not None
                       and isinstance(index, cst.Attribute) and index.attr.value == 'index'
                       and index.value.deep_equals(table))
        from .builder import NATIVE_TABLE_PROVIDER
        if not (clamp and clamp[0].deep_equals(key_name) and clamp[1].deep_equals(high_name)
                and isinstance(column, cst.SimpleString) and type(column.evaluated_value) is str
                and (owned_index or (first_arg and first_arg[0].deep_equals(table)
                and getattr(helper_reference(first[0].func, self.cls.refs), 'helper', None)
                == (NATIVE_TABLE_PROVIDER, 'first_key')))):
            return None
        operation = 'read_bounded_table_float'
        self.providers.add(NATIVE_TABLE_PROVIDER)
        self.operations.add(f'{NATIVE_TABLE_PROVIDER}.{operation}')
        code = cst.Module([]).code_for_node
        value = Names(cst.Module(body), used=self.cell.params).reserve('_mx_bounded_value')
        args = (table, self.lookup_context(NATIVE_TABLE_PROVIDER),
                cst.Name('None') if owned_index else first[0].func, key_name, column, below.value)
        index_arg = ', True' if owned_index else ''
        fast = cst.parse_module(
            f'{value}: _mx_cy.double\n'
            f'if {module_name(NATIVE_TABLE_PROVIDER)}.{operation}('
            f'{", ".join(map(code, args))}, _mx_cy.address({value}){index_arg}):\n    return {value}\n').body
        return (*body[:-4], *fast, *body[-4:])

    def clamped_table_method(self, body):
        """Use the owned table's native bounds for a typed clamp and scalar read."""
        from .builder import NATIVE_TABLE_PROVIDER
        if 'min' not in self.builtin_keys:
            return None
        source, clamp, result = map(self.single_statement, body[-3:])
        if not (isinstance(result, cst.Return) and isinstance(clamp, cst.Assign)
                and isinstance(source, cst.Assign) and len(source.targets) == len(clamp.targets) == 1
                and isinstance(source.targets[0].target, cst.Name)
                and isinstance(clamp.targets[0].target, cst.Name)):
            return None
        table, key = source.targets[0].target, clamp.targets[0].target
        if table.value == key.value or not self.table_source(source.value):
            return None
        lookup = self.first_lookup(result.value)
        if not (lookup and lookup.func.value.deep_equals(table)
                and lookup.args[0].value.deep_equals(key)
                and isinstance(lookup.args[1].value, cst.SimpleString)
                and type(lookup.args[1].value.evaluated_value) is str
                and key.value not in CstFacts((body[-1].visit(
                    _ReplaceExpression(lookup, cst.Integer('0'))),)).reads):
            return None

        def call_args(node, name, size):
            if (isinstance(node, cst.Call) and isinstance(node.func, cst.Name)
                    and node.func.value == name and name in self.builtin_keys
                    and len(node.args) == size and not any(arg.keyword or arg.star for arg in node.args)):
                return tuple(arg.value for arg in node.args)
            return ()

        def bound(node, operation):
            if operation == 'last_key' and self.index_max_int(node, table):
                return cst.Name('None')
            args = call_args(node, 'int', 1)
            if not args or not isinstance(args[0], cst.Call):
                return None
            call = args[0]
            ref = helper_reference(call.func, self.cls.refs)
            if (getattr(ref, 'helper', None) == (NATIVE_TABLE_PROVIDER, operation)
                    and isinstance(call.func, cst.Attribute)
                    and isinstance(call.func.value, cst.Attribute)
                    and len(call.args) == 1 and not (call.args[0].keyword or call.args[0].star)
                    and call.args[0].value.deep_equals(table)):
                return call.func.value
            return None

        upper = call_args(clamp.value, 'min', 2)
        if not upper or (last := bound(upper[1], 'last_key')) is None:
            return None
        lower = call_args(upper[0], 'max', 2)
        first = bound(lower[1], 'first_key') if lower else None
        base = lower[0] if lower else upper[0]
        index_bound = isinstance(last, cst.Name) and last.value == 'None'
        if ((lower and (first is None or not index_bound and not first.deep_equals(last)))
                or not self.native_key(base, allow_bool=False)):
            return None
        names = Names(cst.Module(body), used=self.cell.params)
        number, value = (names.reserve('_mx_clamp_' + name) for name in ('key', 'value'))
        module = cst.Module([])
        code = module.code_for_node
        original_clamp = body[-2].visit(_ReplaceExpression(base, cst.Name(number)))
        alias = module_name(NATIVE_TABLE_PROVIDER)
        operation = 'read_clamped_table_float'
        namespace = first if lower and index_bound else last
        args = [code(table), code(self.lookup_context(NATIVE_TABLE_PROVIDER)), str(bool(lower)), code(namespace), number,
                code(lookup.args[1].value), f'_mx_cy.address({value})']
        self.providers.add(NATIVE_TABLE_PROVIDER)
        self.operations.add(f'{NATIVE_TABLE_PROVIDER}.{operation}')
        prefix = cst.parse_module(
            f'{number}: _mx_cy.longlong = {code(base)}\n'
            f'{value}: _mx_cy.double\n'
            f'if not {alias}.{operation}({", ".join(args)}):\n'
            f'    {code(original_clamp).strip()}\n'
            f'    {value} = {code(lookup)}\n').body
        ret = body[-1].visit(_ReplaceExpression(lookup, cst.Name(value)))
        return (*body[:-2], *prefix, ret)

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

    def native_key(self, key, *, allow_bool=True):
        integer_types = ('long long', 'bint') if allow_bool else ('long long',)
        if isinstance(key, cst.Integer):
            return primitive_kind(literal_value(key)) == 'longlong'
        if isinstance(key, cst.Name) and key.value in self.cell.params:
            return self.cell.get_argtype_expr(key.value, c_style=True, formula=True) in integer_types
        if (isinstance(key, cst.Attribute) and isinstance(key.value, cst.Name)
                and key.value.value == 'self'):
            ref = self.cls.refs.get(key.attr.value)
            return ref is not None and ref.get_type_expr(c_style=True) in integer_types
        if (isinstance(key, cst.Call) and isinstance(key.func, cst.Name)
                and key.func.value in self.builtin_keys and len(key.args) >= 2
                and not any(a.keyword or a.star for a in key.args)):
            return all(self.native_key(a.value, allow_bool=allow_bool) for a in key.args)
        if (isinstance(key, cst.Call) and isinstance(key.func, cst.Attribute)
                and isinstance(key.func.value, cst.Name) and key.func.value.value == 'self'):
            cell = self.cls.cells.get(key.func.attr.value)
            return (cell is not None and not cell.is_special() and not cell.called_with_kwargs
                    and not cell.body_has_closure
                    and cell.get_rettype_expr(c_style=True) in integer_types)
        return self.locked_result_type(key) in integer_types

    def bound_numeric(self, args, native, operations, provider, context=None, member=None):
        if not (operations and len(operations) == 3 and self.cell.formula_allows_exact_receiver
                and not self.cell.is_locked and self.cell.cls not in self.cls.visitor.dynamic_ref_classes):
            return native
        table, key, column = (arg.value for arg in args[:3])
        table_method = context is None
        if member is not None:
            from .builder import NATIVE_TABLE_PROVIDER
            if (provider, operations[0]) != (NATIVE_TABLE_PROVIDER, 'numeric_resource_ready'):
                return native
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
                    and (cell.native_table_result if table_method else
                         (member is not None and cell.native_table_result)
                         or cell.get_rettype_expr(c_style=True) == 'object')):
                return native
            source = f'self._v_{cell.name}' + ('._owner' if cell.native_table_result else '')
            ready = f'self._has_{cell.name} and '
        else:
            return native
        named = isinstance(column, cst.SimpleString) and 'b' not in column.prefix.lower()
        if not named and not self.native_key(column):
            return native
        default = 0.0
        if len(args) == 4:
            node = args[3].value
            token = node.expression if isinstance(node, cst.UnaryOperation) else node
            if not isinstance(token, (cst.Integer, cst.Float)):
                return native
            try:
                literal = literal_value(node)
                default = float(literal)
                if type(literal) not in (int, float) or not math.isfinite(default) or default != literal:
                    return native
            except (ValueError, TypeError, OverflowError, SyntaxError):
                return native
        alias = module_name(provider)
        operation = cst.Attribute(cst.Name(alias), cst.Name(operations[2]))
        if table_method:
            context = self.lookup_context(provider)
            self.providers.add(provider)
            self.operations.add(f'{provider}.{operations[2]}')
            prepare = cst.Call(operation, args=(cst.Arg(context), *args))
        else:
            prepare = native.with_changes(func=operation)
        column_pointer = f'_mx_cy.cast(_mx_cy.p_void, {code(column)})' if named else '_mx_cy.NULL'
        ready_operation = operations[0] if member is None else 'numeric_binding_ready'
        member_arg = '' if member is None else f', {code(member)}'
        test = cst.parse_expression(
            f'{exact_receiver(self.cls.name)} and ({ready}{alias}.{ready_operation}('
            f'{code(context)}, _mx_cy.cast(_mx_cy.p_void, {source}), {column_pointer}{member_arg}))')
        result = cst.parse_expression(
            f'{alias}.{operations[1]}({code(context)}, {code(key)}, {0 if named else code(column)}, '
            f'{named}, {len(args) == 4}, {default!r})')
        bound = cst.IfExp(test=test, body=result, orelse=prepare,
                          lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))
        if table_method:
            return cst.IfExp(test=cst.Attribute(context,cst.Name('_private_owner')),
                             body=bound, orelse=native,
                             lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))
        return bound

    def bound_table_method(self, call):
        table = call.func.value
        method = call.func.attr.value
        if (method not in ('lookup_float', 'previous') or not self.self_call(table)
                or not 2 <= len(call.args) <= (3 if method == 'lookup_float' else 2)
                or any(arg.keyword or arg.star for arg in call.args)
                or not self.native_key(call.args[0].value, allow_bool=False)):
            return call
        from .builder import NATIVE_TABLE_PROVIDER
        provider = NATIVE_TABLE_PROVIDER
        args = (cst.Arg(table), *call.args)
        if method == 'lookup_float':
            operations = ('numeric_resource_ready', 'bound_lookup_f64', 'prepare_table_method')
            return call if table.args else self.bound_numeric(args, call, operations, provider)
        if not self.table_source(table):
            return call
        if self.native_key(call.args[1].value, allow_bool=False):
            operation = 'read_previous_table_int'
        elif table.args:
            operation = 'read_previous_table_float'
        else:
            operation = 'prepare_previous_table_method'
        native_args = ((cst.Arg(self.lookup_context(provider)), *args)
                       if operation == 'prepare_previous_table_method' else args)
        self.providers.add(provider)
        self.operations.add(f'{provider}.{operation}')
        native = cst.Call(cst.Attribute(cst.Name(module_name(provider)), cst.Name(operation)),
                          args=native_args)
        operations = ('numeric_resource_ready', 'bound_previous_f64', 'prepare_previous_table_method')
        return native if table.args else self.bound_numeric(args, native, operations, provider)

    def record_subscript(self, node):
        if not (isinstance(node, cst.Subscript) and len(node.slice) == 1
                and isinstance(node.slice[0].slice, cst.Index)
                and self.self_call(node.value) and not node.value.args):
            return None
        key = node.slice[0].slice.value
        if not isinstance(key, cst.SimpleString) or type(key.evaluated_value) is not str:
            return None
        source = self.cls.cells.get(node.value.func.attr.value)
        if source is not None and source.observed_record_result:
            return node.value, key

    def leave_Assign(self, before, after):
        if self.function_depth != 1 or not self.scope_safe:
            return after
        read = self.record_subscript(after.value)
        if not read:
            return after
        from .builder import NATIVE_TABLE_PROVIDER
        operation = 'read_record_item'
        self.providers.add(NATIVE_TABLE_PROVIDER)
        self.operations.add(f'{NATIVE_TABLE_PROVIDER}.{operation}')
        alias = module_name(NATIVE_TABLE_PROVIDER)
        return after.with_changes(value=cst.Call(cst.Attribute(cst.Name(alias), cst.Name(operation)),
                                                 args=tuple(cst.Arg(arg) for arg in read)))

    def record_float(self, call, receiver, key, *, cast=False):
        from .builder import NATIVE_TABLE_PROVIDER
        provider = NATIVE_TABLE_PROVIDER
        operation = 'read_cast_record_field_float' if cast else 'read_record_field_float'
        alias = module_name(provider)
        context = self.lookup_context(provider)
        self.providers.add(provider)
        self.operations.add(f'{provider}.{operation}')
        args = (cst.Arg(call.func), cst.Arg(context), cst.Arg(receiver), cst.Arg(key))
        source = self.cls.cells.get(receiver.func.attr.value)
        namespace = None if cast else call.func.value
        module_member = (isinstance(namespace, cst.Attribute) and isinstance(namespace.value, cst.Name)
                         and namespace.value.value == 'self')
        bound = ((cast or module_member) and source is not None and source.has_formula_def
                 and not source.is_locked and source.get_rettype_expr(c_style=True) == 'object'
                 and self.cell.formula_allows_exact_receiver)
        if bound and not cast:
            args = (*args, cst.Arg(namespace))
        native = call.with_changes(func=cst.Attribute(cst.Name(alias), cst.Name(operation)), args=args)
        if not bound:
            return native
        code = cst.Module([]).code_for_node
        ready, guard = ('record_cast_ready', call.func) if cast else ('record_field_ready', namespace)
        condition = cst.parse_expression(
            f'{exact_receiver(self.cls.name)} and self._has_{source.name} and '
            f'{alias}.{ready}({code(context)}, _mx_cy.cast(_mx_cy.p_void, self._v_{source.name}), '
            f'_mx_cy.cast(_mx_cy.p_void, {code(key)}), {code(guard)})')
        result = cst.parse_expression(f'{alias}.bound_record_field({code(context)})')
        return cst.IfExp(test=condition, body=result, orelse=native,
                         lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))

    def leave_Return(self, before, after):
        if (self.function_depth != 1 or not self.scope_safe or self.cell.get_rettype_expr(c_style=True) != 'double'
                or (not self.cell.has_formula_def and self.cell.called_with_kwargs)):
            return after
        call = after.value
        cast = arguments(call, 'float', 1) if 'float' in self.builtin_keys else None
        record = self.record_subscript(cast[0]) if cast else None
        if record and self.cell.has_args() and not self.cell.is_locked:
            return after.with_changes(value=self.record_float(call, *record, cast=True))
        if (isinstance(call, cst.Call) and len(call.args) == 2
                and not any(arg.keyword or arg.star for arg in call.args)
                and self.self_call(call.args[0].value) and not call.args[0].value.args
                and isinstance(call.args[1].value, cst.SimpleString)
                and type(call.args[1].value.evaluated_value) is str
                and self.cell.has_args() and not self.cell.is_locked
                and getattr(helper_reference(call.func, self.cls.refs), 'helper', None)
                == ('modelx_cython.runtime._native_inputs', 'lookup_float')):
            return after.with_changes(value=self.record_float(call, *(arg.value for arg in call.args)))
        if isinstance(call, cst.Call) and isinstance(call.func, cst.Attribute) and self.self_call(call.func.value):
            return after.with_changes(value=self.bound_table_method(call))
        if not (isinstance(call, cst.Call) and 3 <= len(call.args) <= 4
                and not any(a.keyword or a.star for a in call.args)):
            return after
        ref = helper_reference(call.func, self.cls.refs)
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
        member = None if isinstance(call.func.value, cst.Name) else call.func
        operations = getattr(ref, 'bound_numeric_helper', None)
        return after.with_changes(value=self.bound_numeric(call.args, native, operations, provider, context, member))
