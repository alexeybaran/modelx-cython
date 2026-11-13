"""Lower typed reductions using existing CST nodes and binding metadata."""
from functools import cached_property

import libcst as cst

from .cell_entries import exact_receiver
from .helpers import REDUCTIONS, module_name
from .helper_members import helper_reference
from .scopes import CstFacts, FormulaTransformer, Names, has_scope_effects, lexical_nodes
from .syntax import arguments, code, literal_value
from .typedefs import primitive_kind


def formula_lines(formula):
    """Map a node of ``formula`` to its line in it (1 = the ``def`` line; 0 if not found)."""
    from libcst.metadata import MetadataWrapper, PositionProvider
    try:
        positions = MetadataWrapper(cst.Module(body=[formula]), unsafe_skip_copy=True).resolve(PositionProvider)
        first = positions[formula.name].start.line
    except Exception:        # a formula assembled by an earlier pass may not round-trip
        return lambda node: 0
    return lambda node: positions[node].start.line - first + 1 if node in positions else 0


def untyped_entry(lines, node, form, reason):
    """One admission entry of an expression form left to the interpreter (W1.4)."""
    import re
    text = re.sub(r'\s*#[^\'"\n]*$', '', cst.Module(body=[]).code_for_node(node), flags=re.M)
    text = ' '.join(text.split()).replace('( ', '(').replace('[ ', '[')
    return {'expr': text if len(text) <= 160 else text[:157] + '...', 'form': form,
            'line': lines(node) if lines is not None else 0, 'reason': reason}


BUILTIN_PROVIDER = 'modelx_cython._native_iterables'
STREAMS_PROVIDER = 'modelx_cython.runtime._native_streams'
BUILTIN_REDUCTIONS = frozenset(('all', 'any', 'max', 'min', 'sum'))


def has_reduction_contract(formula, refs, builtin_reductions=frozenset()):
    """Find declared consumers and supported built-in generator reductions."""
    return any(isinstance(node, cst.Call) and (
               getattr(helper_reference(node.func, refs), 'reduction', None) in REDUCTIONS
               or isinstance(node.func, cst.Name) and node.func.value in builtin_reductions
               and len(node.args) in (1, 2) and isinstance(node.args[0].value, cst.GeneratorExp))
               for node in lexical_nodes(formula.body.body))


def _call(node, name, count):
    return arguments(node, name, count) is not None


def _parts(generator):
    part = generator.for_in
    while part is not None:
        yield part
        part = part.inner_for_in


def _literal_int(node, expected):
    try:
        value = literal_value(node)
        return type(value) is int and value == expected
    except (ValueError, TypeError):
        return False


def safe_range_increment(iterator):
    """Prove the final C loop increment, including an overshoot of stop."""
    args = arguments(iterator)
    if args is None:
        return False
    if len(args) < 3:
        return True
    try:
        step = literal_value(args[2])
        if type(step) not in (int, bool):
            return False
        if step in (-1, 1):
            return True
        bounds = [literal_value(arg) for arg in args]
        if not all(primitive_kind(value) in ('longlong', 'bint') for value in bounds):
            return False
        values = range(*bounds)
        return not values or primitive_kind(values[-1] + step) == 'longlong'
    except (ValueError, TypeError):
        return False


class NativeReductions(FormulaTransformer):
    def __init__(self, cls_info, cells, providers, formula, reserve_fallback, builtin_bounds,
                 builtin_reductions=frozenset(), enum_symbols=None):
        super().__init__()
        self.cls_info, self.cells, self.providers = cls_info, cells, providers
        self.formula, self.reserve_fallback = formula, reserve_fallback
        self.builtin_bounds = builtin_bounds
        self.builtin_reductions = builtin_reductions
        self.enum_symbols = enum_symbols or {}
        self.fallbacks, self.exact_receiver_guards, self.count = [], set(), 0
        self.captures, self.integer_captures = {}, {}
        self.scope_safe = (not has_scope_effects(formula.body.body)
                           and 'self' not in self.body_facts.stores
                           and not {'locals', 'vars', 'eval', 'exec'} & self.body_facts.reads
                           and not {'StopIteration', 'RuntimeError'} & (
                               self.body_facts.stores.keys() | set(cells.params)))
        self.names = Names(formula)
        if isinstance(formula.body, cst.IndentedBlock):
            lines = formula.body.body
            for assignment, consumer in zip(lines, lines[1:]):
                if (not isinstance(assignment, cst.SimpleStatementLine) or len(assignment.body) != 1
                        or not isinstance(assignment.body[0], cst.Assign)
                        or not isinstance(consumer, cst.SimpleStatementLine) or len(consumer.body) != 1
                        or not isinstance(consumer.body[0], (cst.Return, cst.Assign))):
                    continue
                expression = assignment.body[0]
                if len(expression.targets) != 1:
                    continue
                target, rhs = expression.targets[0].target, expression.value
                if isinstance(target, cst.Name):
                    if self.double_expression(rhs):
                        self.captures[id(consumer)] = target.value
                    elif isinstance(rhs, cst.Call) and self.kind(rhs, {}) == 'longlong':
                        self.integer_captures[id(consumer)] = target.value

    @cached_property
    def body_facts(self):
        return CstFacts(self.formula.body.body)

    def enum_range(self, tree):
        """Use a native range for a declared enum tuple feeding a typed Cell argument."""
        part = tree.for_in
        if (not self.enum_symbols or not self.cells.formula_allows_exact_receiver
                or part.inner_for_in or part.ifs or part.asynchronous
                or not isinstance(part.target, cst.Name) or not isinstance(part.iter, cst.Tuple)
                or not part.iter.elements):
            return None
        operands = arguments(tree.elt)
        if (operands is None or not isinstance(tree.elt.func, cst.Attribute)
                or not isinstance(tree.elt.func.value, cst.Name) or tree.elt.func.value.value != 'self'):
            return None
        cell = self.cls_info.cells.get(tree.elt.func.attr.value)
        if cell is None or cell.called_with_kwargs or cell.body_has_closure or len(operands) != len(cell.params):
            return None
        name = part.target.value
        positions = [index for index, operand in enumerate(operands) if name in CstFacts(operand).reads]
        if len(positions) != 1:
            return None
        index, = positions
        if (not isinstance(operands[index], cst.Name) or operands[index].value != name
                or cell.get_argtype_expr(cell.params[index], c_style=True) != 'long long'):
            return None
        values = []
        owner = None
        for element in part.iter.elements:
            if not isinstance(element, cst.Element):
                return None
            member = element.value
            if (not isinstance(member, cst.Attribute) or not isinstance(member.value, cst.Attribute)
                    or not isinstance(member.value.value, cst.Name) or member.value.value.value != 'self'):
                return None
            ref = member.value.attr.value
            entry = self.enum_symbols.get((self.cls_info.name, ref, member.attr.value))
            if entry is None or owner is not None and ref != owner:
                return None
            owner = ref
            values.append(entry[1])
        start = values[0]
        if all(value == start + index for index, value in enumerate(values)):
            bounds = str(len(values)) if start == 0 else f'{start}, {start + len(values)}'
            return cst.parse_expression(f'range({bounds})')
        return None

    def double_expression(self, node):
        if isinstance(node, cst.Float):
            return True
        if (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name)
                and node.value.value == 'self'):
            ref = self.cls_info.refs.get(node.attr.value)
            return ref is not None and ref.get_type_expr(c_style=True) == 'double'
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, (cst.Plus, cst.Minus)):
            return self.double_expression(node.expression)
        if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, (cst.Add, cst.Subtract, cst.Multiply, cst.Divide)):
            return self.double_expression(node.left) and self.double_expression(node.right)
        return False

    def reserve_prefix(self):
        return self.names.reserve(f'_mx_reduce_{self.count}', family=True)

    def visit_FunctionDef(self, node):
        return self.scope_safe

    def native_truth(self, node, locals_):
        if self.kind(node, locals_) in ('double', 'longlong', 'bint'):
            return True
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Not):
            return self.native_truth(node.expression, locals_)
        if isinstance(node, cst.BooleanOperation):
            return self.native_truth(node.left, locals_) and self.native_truth(node.right, locals_)
        if isinstance(node, cst.Comparison) and all(isinstance(part.operator, (
                cst.Equal, cst.NotEqual, cst.LessThan, cst.LessThanEqual, cst.GreaterThan, cst.GreaterThanEqual))
                for part in node.comparisons):
            operands = [node.left, *(part.comparator for part in node.comparisons)]
            kinds = [self.kind(value, locals_) for value in operands]
            if not all(kind in ('double', 'longlong', 'bint') for kind in kinds):
                return False
            return 'double' not in kinds or all(kind != 'longlong' or (
                isinstance(value, cst.Integer) and abs(value.evaluated_value) <= (1 << 53))
                for value, kind in zip(operands, kinds))
        return False

    def kind(self, node, locals_):
        if isinstance(node, (cst.Integer, cst.Float, cst.Imaginary, cst.SimpleString)):
            return primitive_kind(literal_value(node))
        if isinstance(node, cst.Name):
            if node.value in ('True', 'False', 'None'):
                return primitive_kind(literal_value(node))
            if node.value in locals_:
                return locals_[node.value]
            if node.value in self.cells.params and self.cells.has_typeinfo():
                return {'double': 'double', 'long long': 'longlong', 'bint': 'bint'}.get(
                    self.cells.get_argtype_expr(node.value, c_style=True, formula=True))
            return None
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, (cst.Plus, cst.Minus)):
            try:
                return primitive_kind(literal_value(node))
            except (ValueError, TypeError):
                kind = self.kind(node.expression, locals_)
                if isinstance(node.operator, cst.Plus) or kind == 'double':
                    return kind
                return 'longlong' if kind == 'bint' else None
        if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name) and node.value.value == 'self':
            ref = self.cls_info.refs.get(node.attr.value)
            if ref is not None:
                return {float: 'double', int: 'longlong', bool: 'bint'}.get(ref.type_)
        absolute = arguments(node, 'abs', 1)
        if (absolute and self.builtin_bounds and 'abs' not in locals_
                and 'abs' not in self.cells.params and 'abs' not in self.body_facts.stores
                and self.kind(absolute[0], locals_) == 'double'):
            return 'double'
        if (arguments(node) is not None and isinstance(node.func, cst.Attribute)
                and isinstance(node.func.value, cst.Name) and node.func.value.value == 'self'):
            cell = self.cls_info.cells.get(node.func.attr.value)
            if cell is not None and cell.has_typeinfo() and not cell.called_with_kwargs and not cell.body_has_closure:
                return {'double': 'double', 'long long': 'longlong', 'bint': 'bint'}.get(cell.get_rettype_expr(c_style=True))
        if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, (cst.Add, cst.Subtract, cst.Multiply, cst.Divide)):
            left, right = self.kind(node.left, locals_), self.kind(node.right, locals_)
            if (left in ('double', 'longlong', 'bint')
                    and right in ('double', 'longlong', 'bint') and 'double' in (left, right)):
                return 'double'
        return None

    def cold_fallback(self, source, generator, capture, function, capture_type='_mx_cy.double',
                      boxed_param=None, trailing_args=''):
        original = f'{function}({source}{trailing_args})'
        if not self.cells.has_formula_def or not self.scope_safe:
            return original
        names = CstFacts(generator).reads
        targets = {part.target.value for part in _parts(generator)}
        stores = self.body_facts.stores
        if ((names - targets) & stores.keys()) - {capture}:
            return original
        if capture is not None and stores[capture] != 1:
            return original
        params = self.formula.params
        parameters = [*params.posonly_params, *params.params, *params.kwonly_params]
        parameters += [param for param in (params.star_arg, params.star_kwarg) if isinstance(param, cst.Param)]
        param_names = {param.name.value for param in parameters}
        if targets & param_names or not (names & param_names) <= {'self', *self.cells.params}:
            return original
        args = []
        for name in self.cells.params:
            if name in names:
                kind = self.cells.get_argtype_expr(name, c_style=True, formula=True)
                if kind not in ('double', 'long long', 'bint', 'object'):
                    return original
                args.append((name, 'object' if name == boxed_param else
                             self.cells.get_argtype_expr(name, formula=True)))
        if capture is not None:
            args.append((capture, capture_type))
        helper = self.reserve_fallback()
        formals = ', '.join([f'self: {self.cls_info.name}', f'{function}: object']
                            + [f'{name}: {kind}' for name, kind in args])
        self.fallbacks.append(cst.parse_statement(
            f'\n\n@_mx_cy.cfunc\ndef {helper}({formals}) -> object:\n    return {original}\n'))
        actuals = ', '.join(['self', function] + [name for name, _ in args])
        return f'{helper}({actuals})'

    def bounded_any(self, before, after, func, provider, operation, source, tree):
        capture = self.integer_captures.get(id(before))
        parts = tuple(_parts(tree))
        if not (self.builtin_bounds and capture and capture not in self.cells.params
                and capture not in {'self', 'min', 'max', 'range'}
                and self.cells.get_rettype_expr(c_style=True) == 'bint' and len(parts) == 1):
            return after
        part, = parts

        def named(node, name):
            return isinstance(node, cst.Name) and node.value == name

        if not (isinstance(part.target, cst.Name) and not part.ifs and part.asynchronous is None
                and _call(part.iter, 'range', 2)):
            return after
        index = part.target.value
        predicate_args = arguments(tree.elt, count=1)
        if (index in {capture, 'self', 'range', 'min', 'max', *self.cells.params}
                or predicate_args is None or not named(predicate_args[0], index)
                or self.kind(tree.elt, {index: 'longlong'}) != 'bint'):
            return after
        low, high = arguments(part.iter)
        if not (_call(low, 'max', 2) and _call(high, 'min', 2)
                and _literal_int(low.args[0].value, 0) and named(low.args[1].value, capture)):
            return after
        upper, limit = arguments(high)
        if not (isinstance(upper, cst.BinaryOperation) and isinstance(upper.operator, cst.Add)
                and named(upper.left, capture) and isinstance(upper.right, cst.Integer)
                and 0 < upper.right.evaluated_value <= (1 << 31) - 1
                and arguments(limit, count=0) is not None and self.kind(limit, {}) == 'longlong'):
            return after
        offset = upper.right.evaluated_value
        alias, prefix = module_name(provider), self.reserve_prefix()
        function = f'{prefix}_function'
        count = len(self.fallbacks)
        fallback = self.cold_fallback(source, tree, capture, function, 'object')
        if len(self.fallbacks) == count:
            return after
        self.providers.add(provider)
        self.count += 1
        guard = f'{prefix}_native'
        self.exact_receiver_guards.add(guard)
        predicate = f'self.{tree.elt.func.attr.value}({prefix}_index)'
        return cst.FlattenSentinel(cst.parse_module(f"""
{function}: object = {code(func)}
{prefix}_first: _mx_cy.longlong = {capture}
{guard}: _mx_cy.bint = {alias}.{operation}_accepts({function}) and {prefix}_first <= {(1 << 63) - 1 - offset}
if {guard}:
    {prefix}_low: _mx_cy.longlong = {prefix}_first if {prefix}_first > 0 else 0
    {prefix}_high: _mx_cy.longlong = {prefix}_first + {offset}
    {prefix}_limit: _mx_cy.longlong = {code(limit)}
    if {prefix}_high > {prefix}_limit:
        {prefix}_high = {prefix}_limit
    {prefix}_index: _mx_cy.longlong
    try:
        for {prefix}_index in range({prefix}_low, {prefix}_high):
            if {predicate}:
                return True
    except StopIteration as {prefix}_error:
        raise RuntimeError("generator raised StopIteration") from {prefix}_error
    return False
return {fallback}
""").body)

    def single_range(self, after, func, provider, operation, reduction, source, tree):
        parts = tuple(_parts(tree))
        if len(parts) != 1 or not self.cells.formula_allows_exact_receiver:
            return after
        part, = parts
        if not isinstance(part.target, cst.Name) or part.ifs or part.asynchronous is not None:
            return after
        index = part.target.value
        if index in {'self', 'range', 'abs', 'max', *self.cells.params}:
            return after
        alias, prefix = module_name(provider), self.reserve_prefix()
        counter, guard, function = f'{prefix}_index', f'{prefix}_native', f'{prefix}_function'
        predicate, boxed_param = tree.elt, None
        if reduction == 'bool_all':
            args = arguments(part.iter, 'range', 1)
            if not (args is not None and arguments(args[0], count=0) is not None
                    and self.kind(args[0], {}) == 'longlong'
                    and self.cells.get_rettype_expr(c_style=True) == 'bint'):
                return after
            iterator, extra_guard = f'range({prefix}_limit)', ''
            if isinstance(predicate, cst.Comparison):
                if len(predicate.comparisons) != 1:
                    return after
                comparison = predicate.comparisons[0]
                absolute = arguments(predicate.left, 'abs', 1)
                if not (isinstance(comparison.operator, (cst.LessThan, cst.LessThanEqual, cst.GreaterThan, cst.GreaterThanEqual))
                        and absolute is not None and self.kind(comparison.comparator, {}) == 'double'
                        and isinstance(comparison.comparator, (cst.Float, cst.Attribute))):
                    return after
                value = self.indexed_cell(absolute[0], index, 'double', counter)
                if value is None:
                    return after
                operator = {cst.LessThan: '<', cst.LessThanEqual: '<=', cst.GreaterThan: '>', cst.GreaterThanEqual: '>='}[type(comparison.operator)]
                test = f'{alias}.f64_abs({value}) {operator} {code(comparison.comparator)}'
            else:
                test = self.indexed_cell(predicate, index, 'bint', counter)
                if test is None:
                    return after
            loop = f"""    {prefix}_limit: _mx_cy.longlong = {code(args[0])}
    try:
        for {counter} in {iterator}:
            if not ({test}):
                return False
    except StopIteration as {prefix}_error:
        raise RuntimeError("generator raised StopIteration") from {prefix}_error
    return True"""
        else:
            value, bounds = self.indexed_cell(predicate, index, 'double', counter), self.descending_window(part.iter)
            if value is None or bounds is None or self.cells.get_rettype_expr(c_style=True) != 'double':
                return after
            start, offset, step = bounds
            boxed_param, extra_guard = start, f' and {start} >= 0'
            stop, best, item = f'{prefix}_stop', f'{prefix}_best', f'{prefix}_item'
            first = self.indexed_cell(predicate, index, 'double', start)
            loop = f"""    {stop}: _mx_cy.longlong = {start} - {offset}
    if {stop} < -1:
        {stop} = -1
    {best}: _mx_cy.double
    {item}: _mx_cy.double
    try:
        {best} = {first}
        for {counter} in range({start} - {step}, {stop}, -{step}):
            {item} = {value}
            if {item} > {best}:
                {best} = {item}
    except StopIteration as {prefix}_error:
        raise RuntimeError("generator raised StopIteration") from {prefix}_error
    return {best}"""
        count = len(self.fallbacks)
        fallback = self.cold_fallback(source, tree, None, function, boxed_param=boxed_param)
        if len(self.fallbacks) == count:
            return after
        self.providers.add(provider)
        self.count += 1
        return cst.FlattenSentinel(cst.parse_module(f"""
{function}: object = {code(func)}
{guard}: _mx_cy.bint = {alias}.{operation}_accepts({function}){extra_guard}
if {exact_receiver(self.cls_info.name)} and {guard}:
    {counter}: _mx_cy.longlong
{loop}
return {fallback}
""").body)

    def indexed_cell(self, node, index, kind, counter):
        args = arguments(node, count=1)
        if not (args is not None and isinstance(args[0], cst.Name) and args[0].value == index
                and self.kind(node, {index: 'longlong'}) == kind):
            return None
        cell = self.cls_info.cells[node.func.attr.value]
        if len(cell.params) != 1 or cell.get_argtype_expr(cell.params[0], c_style=True) != 'long long':
            return None
        return f'self.{node.func.attr.value}({counter})'

    def descending_window(self, node):
        args = arguments(node, 'range', 3)
        if args is None:
            return None
        start, limit, step = args
        if not (isinstance(start, cst.Name) and start.value in self.cells.params
                and self.cells.get_argtype_expr(start.value, c_style=True, formula=True) == 'long long'
                and _call(limit, 'max', 2) and _literal_int(limit.args[0].value, -1)
                and isinstance(step, cst.UnaryOperation) and isinstance(step.operator, cst.Minus)):
            return None
        lower = limit.args[1].value
        if not (isinstance(lower, cst.BinaryOperation) and isinstance(lower.operator, cst.Subtract)
                and isinstance(lower.left, cst.Name) and lower.left.value == start.value
                and isinstance(lower.right, cst.Integer) and isinstance(step.expression, cst.Integer)
                and 0 < step.expression.evaluated_value <= lower.right.evaluated_value <= (1 << 31) - 1):
            return None
        return start.value, lower.right.evaluated_value, step.expression.evaluated_value

    def minimum(self, after, tree):
        """Keep the first minimum in a typed range, with the ordinary generator as fallback."""
        part = tree.for_in
        bounds = arguments(part.iter, 'range')
        if (self.cells.get_rettype_expr(c_style=True) != 'double'
                or not isinstance(part.target, cst.Name)
                or part.inner_for_in is not None or part.ifs or part.asynchronous is not None
                or not bounds or len(bounds) > 3 or not safe_range_increment(part.iter)
                or not all(self.kind(arg, {}) in ('longlong', 'bint') for arg in bounds)):
            return after
        index = part.target.value
        if self.kind(tree.elt, {index: 'longlong'}) != 'double':
            return after
        prefix = self.reserve_prefix()
        function = f'{prefix}_function'
        count = len(self.fallbacks)
        fallback = self.cold_fallback(code(tree), tree, None, function)
        if len(self.fallbacks) == count:
            return after
        self.providers.add(BUILTIN_PROVIDER)
        self.count += 1
        counter = f'{prefix}_index'
        contexts = CstFacts(tree.elt).contexts

        class RenameIndex(cst.CSTTransformer):
            def leave_Name(self, before, after):
                return after.with_changes(value=counter) if before.value == index and before in contexts else after

        element = code(tree.elt.visit(RenameIndex()))
        bound_names = [f'{prefix}_bound_{i}' for i in range(len(bounds))]
        bound_assignments = '\n'.join(f'    {name}: _mx_cy.longlong = {code(value)}'
                                      for name, value in zip(bound_names, bounds))
        alias = module_name(BUILTIN_PROVIDER)
        return cst.FlattenSentinel(cst.parse_module(f"""
{function}: object = min
if {exact_receiver(self.cls_info.name)} and {alias}.min_accepts({function}):
{bound_assignments}
    {counter}: _mx_cy.longlong
    {prefix}_seen: _mx_cy.bint = False
    {prefix}_best: _mx_cy.double = 0.0
    {prefix}_value: _mx_cy.double
    try:
        for {counter} in range({', '.join(bound_names)}):
            {prefix}_value = {element}
            if not {prefix}_seen or {prefix}_value < {prefix}_best:
                {prefix}_best = {prefix}_value
                {prefix}_seen = True
    except StopIteration as {prefix}_error:
        raise RuntimeError("generator raised StopIteration") from {prefix}_error
    if not {prefix}_seen:
        return {function}(())
    return {prefix}_best
return {fallback}
""").body)

    def leave_SimpleStatementLine(self, before, after):
        if len(before.body) != 1:
            return after
        statement = before.body[0]
        assignment = isinstance(statement, cst.Assign)
        if not (isinstance(statement, cst.Return) or assignment and len(statement.targets) == 1
                and isinstance(statement.targets[0].target, cst.Name)):
            return after
        value = statement.value
        args = arguments(value)
        if not args or not isinstance(args[0], cst.GeneratorExp) or not self.builtin_bounds:
            return after
        func, tree = value.func, args[0]
        source = code(tree)
        builtin = func.value if isinstance(func, cst.Name) and func.value in self.builtin_reductions else None
        float_start = (builtin == 'sum' and len(args) == 2 and isinstance(args[1], cst.Float)
                       and args[1].evaluated_value == 0.0)
        if len(args) != 1 and not float_start:
            return after
        if builtin == 'min':
            return after if assignment else self.minimum(after, tree)
        if builtin == 'all':
            return after if assignment else self.single_range(
                after, func, BUILTIN_PROVIDER, 'all', 'bool_all', source, tree)
        if builtin == 'any':
            return after if assignment else self.bounded_any(
                before, after, func, BUILTIN_PROVIDER, 'any', source, tree)
        if builtin == 'max':
            return after if assignment else self.single_range(
                after, func, BUILTIN_PROVIDER, 'max', 'f64_max', source, tree)
        ref = helper_reference(func, self.cls_info.refs)
        if builtin == 'sum':
            provider, reduction = STREAMS_PROVIDER, 'f64_sum' if float_start else 'ordinary_sum'
        elif getattr(ref, 'reduction', None) in REDUCTIONS and ref.helper:
            provider, operation = ref.helper
            reduction = ref.reduction
        else:
            return after
        if assignment and reduction != 'f64_sum':
            return after
        if (reduction == 'ordinary_sum'
                and self.cells.get_rettype_expr(c_style=True) != 'double'):
            # Ordinary sum keeps integer/mixed/object semantics. Only an existing
            # double return boundary may normalize empty integer zero to double.
            return after
        if reduction == 'bool_any':
            return self.bounded_any(before, after, func, provider, operation, source, tree)
        if reduction in ('bool_all', 'f64_max'):
            return self.single_range(after, func, provider, operation, reduction, source, tree)
        original_tree = tree
        enum_range = self.enum_range(tree)
        if enum_range is not None:
            tree = tree.with_changes(for_in=tree.for_in.with_changes(iter=enum_range))
        capture = self.captures.get(id(before))
        if capture == 'range':
            return after
        generator_facts = CstFacts(tree)
        if capture is not None and capture not in generator_facts.reads:
            capture = None
        local_types = {capture: 'double'} if capture is not None else {}
        for part in _parts(tree):
            if part.asynchronous is not None or not isinstance(part.target, cst.Name):
                return after
            name, iterator = part.target.value, part.iter
            if name == capture:
                return after
            if name not in local_types and name in CstFacts(iterator).reads:
                return after
            bounds = arguments(iterator, 'range')
            if (bounds is not None and 'range' not in local_types and 1 <= len(bounds) <= 3
                    and all(self.kind(arg, local_types) in ('longlong', 'bint') for arg in bounds)
                    and safe_range_increment(iterator)):
                local_types[name] = 'longlong'
            elif (isinstance(iterator, cst.Tuple)
                    and all(isinstance(element, cst.Element) for element in iterator.elements)):
                kind = (self.kind(iterator.elements[0].value, local_types)
                        if len(iterator.elements) == 1 else None)
                # Literal domains need no sampled category schema or size limit.
                # Keep literal objects intact: even a homogeneous numeric domain
                # can flow into an object-argument Cell. No new loop coercion.
                if kind is None:
                    try:
                        values = [literal_value(element.value) for element in iterator.elements]
                    except (ValueError, TypeError):
                        return after
                    if not all(type(value) in (int, float, bool, str, type(None)) for value in values):
                        return after
                    kind = 'object'
                local_types[name] = kind
            else:
                return after
            if not all(self.native_truth(condition.test, local_types) for condition in part.ifs):
                return after
        if self.kind(tree.elt, local_types) != 'double':
            return after
        alias, prefix = module_name(provider), self.reserve_prefix()
        count = len(self.fallbacks)
        fallback = self.cold_fallback(source, original_tree, capture, f'{prefix}_function',
                                      trailing_args=', 0.0' if float_start else '')
        if len(self.fallbacks) == count:
            return after
        # Iterator/capture names must not alias the accumulator or call-layout
        # temporaries (e.g. an authored iterator named `state`).
        for role in ('function', 'state', 'iterable', 'error', 'bound_0', 'bound_1', 'bound_2'):
            self.names.reserve(f'{prefix}_{role}')
        renamed = {name: self.names.reserve(f'{prefix}_{name}') for name in local_types}
        contexts = generator_facts.contexts

        class Rename(cst.CSTTransformer):
            def leave_Name(self, before, after):
                if before.value in renamed and before in contexts:
                    return after.with_changes(value=renamed[before.value])
                return after

        tree = tree.visit(Rename())
        declarations = [f'{prefix}_function: object', f'{prefix}_state: {alias}.F64Sum']
        declarations += [f'{renamed[name]}: _mx_cy.{kind}' for name, kind in local_types.items()]
        guard = f'{alias}.f64_accepts({prefix}_function)'
        if enum_range is not None:
            guard += f' and {exact_receiver(self.cls_info.name)}'
        lines = declarations + [f'{prefix}_function = {code(func)}',
                               f'if {guard}:',
                               f'    {alias}.f64_init(_mx_cy.address({prefix}_state))']
        if capture is not None:
            lines.append(f'    {renamed[capture]} = {capture}')
        parts = tuple(_parts(tree))
        outer = parts[0]
        bounds = arguments(outer.iter, 'range')
        if bounds is not None:
            bound_names = [f'{prefix}_bound_{index}' for index in range(len(bounds))]
            lines.extend(f'    {name}: _mx_cy.longlong = {code(value)}'
                         for name, value in zip(bound_names, bounds))
            outer_iterator = f'range({", ".join(bound_names)})'
        elif len(outer.iter.elements) == 1:
            # A generator acquires its first iterable before entering its body.
            # Keep that evaluation outside the PEP479 boundary as well.
            lines.append(f'    {outer.target.value} = ({code(outer.iter.elements[0].value)})')
            outer_iterator = None
        else:
            lines.append(f'    {prefix}_iterable: object = {code(outer.iter)}')
            outer_iterator = f'{prefix}_iterable'
        lines.append('    try:')
        indent = '        '
        for index, part in enumerate(parts):
            if index == 0 and outer_iterator is None:
                pass
            elif index == 0:
                lines.append(indent + f'for {part.target.value} in {outer_iterator}:')
                indent += '    '
            elif isinstance(part.iter, cst.Tuple) and len(part.iter.elements) == 1:
                lines.append(indent + f'{part.target.value} = ({code(part.iter.elements[0].value)})')
            else:
                lines.append(indent + f'for {part.target.value} in {code(part.iter)}:')
                indent += '    '
            for condition in part.ifs:
                lines.append(indent + f'if ({code(condition.test)}):')
                indent += '    '
        result = code(statement.targets[0].target) + ' = ' if assignment else 'return '
        lines += [indent + f'{alias}.f64_add(_mx_cy.address({prefix}_state), {code(tree.elt)})',
                  f'    except StopIteration as {prefix}_error:',
                  f'        raise RuntimeError("generator raised StopIteration") from {prefix}_error',
                  f'    {result}{alias}.f64_result(_mx_cy.address({prefix}_state))',
                  'else:', f'    {result}{fallback}']
        self.providers.add(provider)
        self.count += 1
        return cst.FlattenSentinel(cst.parse_module('\n'.join(lines) + '\n').body)
