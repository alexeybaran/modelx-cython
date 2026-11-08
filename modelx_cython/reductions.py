"""Lower registered reductions at proved return and assignment boundaries.

Only range loops and singleton bindings with proven numeric expressions qualify.
The existing formula body remains authoritative; general generators stay intact.
"""
import ast

import libcst as cst

from .cell_entries import exact_receiver
from .helpers import module_name
from .scopes import FormulaTransformer, Names, expression_ast, has_scope_effects
from .typedefs import primitive_kind


def _call(node, name, argc):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == name and len(node.args) == argc and not node.keywords
            and not any(isinstance(arg, ast.Starred) for arg in node.args))


def safe_range_increment(iterator):
    """Prove the final C loop increment, including an overshoot of stop."""
    if len(iterator.args) < 3:
        return True
    try:
        step = ast.literal_eval(iterator.args[2])
        if type(step) not in (int, bool):
            return False
        if step in (-1, 1):
            return True
        bounds = [ast.literal_eval(arg) for arg in iterator.args]
        if not all(primitive_kind(value) in ('longlong', 'bint') for value in bounds):
            return False
        values = range(*bounds)
        return not values or primitive_kind(values[-1] + step) == 'longlong'
    except (ValueError, TypeError):
        return False


class NativeReductions(FormulaTransformer):
    def __init__(self, cls_info, cells, providers, formula, reserve_fallback, builtin_bounds):
        super().__init__()
        self.cls_info = cls_info
        self.cells = cells
        self.providers = providers
        self.formula = formula
        self.reserve_fallback = reserve_fallback
        self.builtin_bounds = builtin_bounds
        self.fallbacks = []
        self.exact_receiver_guards = set()
        self.count = 0
        self.module = cst.Module([])
        self.captures = {}
        self.integer_captures = {}
        syntax = ast.parse(self.module.code_for_node(formula))
        self.scope_safe = not has_scope_effects(syntax.body[0].body)
        self.names = Names(formula)
        if isinstance(formula.body, cst.IndentedBlock):
            lines = formula.body.body
            for assignment, consumer in zip(lines, lines[1:]):
                if (not isinstance(assignment, cst.SimpleStatementLine)
                        or len(assignment.body) != 1
                        or not isinstance(assignment.body[0], cst.Assign)
                        or not isinstance(consumer, cst.SimpleStatementLine)
                        or len(consumer.body) != 1
                        or not isinstance(consumer.body[0], (cst.Return, cst.Assign))):
                    continue
                expression = assignment.body[0]
                if len(expression.targets) != 1:
                    continue
                target = expression.targets[0].target
                if isinstance(target, cst.Name):
                    rhs = expression_ast(expression.value)
                    if self.double_expression(rhs):
                        self.captures[id(consumer)] = target.value
                    elif isinstance(rhs, ast.Call) and self.kind(rhs, {}) == 'longlong':
                        self.integer_captures[id(consumer)] = target.value

    def double_expression(self, node):
        """Check an existing binding formed solely from C doubles."""
        if isinstance(node, ast.Constant):
            return type(node.value) is float
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == 'self'):
            ref = self.cls_info.refs.get(node.attr)
            return ref is not None and ref.get_type_expr(c_style=True) == 'double'
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return self.double_expression(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return self.double_expression(node.left) and self.double_expression(node.right)
        return False

    def reserve_prefix(self):
        return self.names.reserve(f'_mx_reduce_{self.count}', family=True)

    def visit_FunctionDef(self, node):
        return self.scope_safe

    def native_truth(self, node, locals_):
        """Prove a filter's primitive truth/comparison semantics, not its size."""
        if self.kind(node, locals_) in ('double', 'longlong', 'bint'):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.native_truth(node.operand, locals_)
        if isinstance(node, ast.BoolOp):
            return all(self.native_truth(value, locals_) for value in node.values)
        if isinstance(node, ast.Compare) and all(isinstance(op, (
                ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
            operands = [node.left, *node.comparators]
            kinds = [self.kind(value, locals_) for value in operands]
            if not all(kind in ('double', 'longlong', 'bint') for kind in kinds):
                return False
            # Unknown int64 values must not be rounded to double for Python
            # mixed int/float comparisons. Booleans and small literals are exact.
            return 'double' not in kinds or all(kind != 'longlong' or (
                isinstance(value, ast.Constant) and abs(value.value) <= (1 << 53))
                for value, kind in zip(operands, kinds))
        return False

    def kind(self, node, locals_):
        if isinstance(node, ast.Constant):
            return primitive_kind(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            # Classify the signed literal as a whole: -2**63 fits, +2**63
            # does not. A native negation of an unknown int64 can overflow.
            try:
                return primitive_kind(ast.literal_eval(node))
            except (ValueError, TypeError):
                kind = self.kind(node.operand, locals_)
                if isinstance(node.op, ast.UAdd) or kind == 'double':
                    return kind
                return 'longlong' if kind == 'bint' else None
        if isinstance(node, ast.Name):
            return locals_.get(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self':
            ref = self.cls_info.refs.get(node.attr)
            if ref is not None:
                return {float: 'double', int: 'longlong', bool: 'bint'}.get(ref.type_)
        if (isinstance(node, ast.Call) and not node.keywords
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'):
            cell = self.cls_info.cells.get(node.func.attr)
            if (cell is not None and cell.has_typeinfo() and not cell.called_with_kwargs
                    and not cell.body_has_closure):
                result = cell.get_rettype_expr(c_style=True)
                return {'double': 'double', 'long long': 'longlong', 'bint': 'bint'}.get(result)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = self.kind(node.left, locals_), self.kind(node.right, locals_)
            # Do not introduce native integer arithmetic merely from observations.
            if left and right and 'double' in (left, right):
                return 'double'
        return None

    def cold_fallback(self, source, generator, capture, function, capture_type='_mx_cy.double',
                      boxed_param=None):
        original = f'{function}({source})'
        if not self.cells.has_formula_def:
            return original
        formula, = ast.parse(self.module.code_for_node(self.formula)).body
        nodes = [node for statement in formula.body for node in ast.walk(statement)]
        # A moved generator cannot share enclosing bindings with another closure.
        if has_scope_effects(formula.body):
            return original
        names = {node.id for node in ast.walk(generator) if isinstance(node, ast.Name)
                 and isinstance(node.ctx, ast.Load)}
        targets = {part.target.id for part in generator.generators}
        stores = [node.id for node in nodes if isinstance(node, ast.Name)
                  and isinstance(node.ctx, ast.Store)]
        if ((names - targets) & set(stores)) - {capture}:
            return original
        if capture is not None and stores.count(capture) != 1:
            return original
        params = {param.arg for param in ast.walk(formula.args) if isinstance(param, ast.arg)}
        if targets & params or not (names & params) <= {'self', *self.cells.params}:
            return original
        args = []
        for name in self.cells.params:
            if name in names:
                kind = self.cells.get_argtype_expr(name, c_style=True)
                if kind not in ('double', 'long long', 'bint', 'object'):
                    return original
                args.append((name, 'object' if name == boxed_param else self.cells.get_argtype_expr(name)))
        if capture is not None:
            args.append((capture, capture_type))

        helper = self.reserve_fallback()
        formals = ', '.join([f'self: {self.cls_info.name}', f'{function}: object'] +
                            [f'{name}: {kind}' for name, kind in args])
        self.fallbacks.append(cst.parse_statement(
            f'\n\n@_mx_cy.cfunc\ndef {helper}({formals}) -> object:\n    return {original}\n'))
        actuals = ', '.join(['self', function] + [name for name, _ in args])
        return f'{helper}({actuals})'

    def bounded_any(self, before, after, func, ref, source, tree):
        capture = self.integer_captures.get(id(before))
        if not (self.builtin_bounds and capture and capture not in self.cells.params
                and capture not in {'self', 'min', 'max', 'range'}
                and self.cells.get_rettype_expr(c_style=True) == 'bint'
                and len(tree.generators) == 1):
            return after
        part, = tree.generators

        def named(node, name):
            return isinstance(node, ast.Name) and node.id == name

        if not (isinstance(part.target, ast.Name) and not part.ifs and not part.is_async
                and _call(part.iter, 'range', 2)):
            return after
        index = part.target.id
        if (index in {capture, 'self', 'range', 'min', 'max', *self.cells.params}
                or not isinstance(tree.elt, ast.Call) or len(tree.elt.args) != 1
                or not named(tree.elt.args[0], index)
                or self.kind(tree.elt, {index: 'longlong'}) != 'bint'):
            return after
        low, high = part.iter.args
        if not (_call(low, 'max', 2) and _call(high, 'min', 2)
                and isinstance(low.args[0], ast.Constant) and type(low.args[0].value) is int
                and low.args[0].value == 0 and named(low.args[1], capture)):
            return after
        upper, limit = high.args
        if not (isinstance(upper, ast.BinOp) and isinstance(upper.op, ast.Add)
                and named(upper.left, capture) and isinstance(upper.right, ast.Constant)
                and type(upper.right.value) is int and 0 < upper.right.value <= (1 << 31) - 1
                and isinstance(limit, ast.Call) and not limit.args
                and self.kind(limit, {}) == 'longlong'):
            return after
        offset = upper.right.value
        provider, operation = ref.helper
        alias = module_name(provider)
        prefix = self.reserve_prefix()
        function = f'{prefix}_function'
        before_fallbacks = len(self.fallbacks)
        fallback = self.cold_fallback(source, tree, capture, function, 'object')
        if len(self.fallbacks) == before_fallbacks:
            return after
        self.providers.add(provider)
        self.count += 1
        guard = f'{prefix}_native'
        self.exact_receiver_guards.add(guard)
        predicate = f'self.{tree.elt.func.attr}({prefix}_index)'
        return cst.FlattenSentinel(cst.parse_module(f"""
{function}: object = self.{func.attr.value}
{prefix}_first: _mx_cy.longlong = {capture}
{guard}: _mx_cy.bint = {alias}.{operation}_accepts({function}) and {prefix}_first <= {(1 << 63) - 1 - offset}
if {guard}:
    {prefix}_low: _mx_cy.longlong = {prefix}_first if {prefix}_first > 0 else 0
    {prefix}_high: _mx_cy.longlong = {prefix}_first + {offset}
    {prefix}_limit: _mx_cy.longlong = {ast.unparse(limit)}
    if {prefix}_high > {prefix}_limit:
        {prefix}_high = {prefix}_limit
    {prefix}_index: _mx_cy.longlong
    for {prefix}_index in range({prefix}_low, {prefix}_high):
        if {predicate}:
            return True
    return False
return {fallback}
""").body)

    def single_range(self, after, func, ref, source, tree):
        """Lower the proven typed predicate or bounded descending maximum."""
        if len(tree.generators) != 1 or not self.cells.formula_allows_exact_receiver:
            return after
        part, = tree.generators
        if not isinstance(part.target, ast.Name) or part.ifs or part.is_async:
            return after
        index = part.target.id
        if index in {'self', 'range', 'abs', 'max', *self.cells.params}:
            return after

        provider, operation = ref.helper
        alias = module_name(provider)
        prefix = self.reserve_prefix()
        counter = f'{prefix}_index'
        guard = f'{prefix}_native'
        function = f'{prefix}_function'
        predicate = tree.elt
        boxed_param = None
        if ref.reduction == 'bool_all':
            if not (_call(part.iter, 'range', 1)
                    and isinstance(part.iter.args[0], ast.Call)
                    and not part.iter.args[0].args
                    and self.kind(part.iter.args[0], {}) == 'longlong'
                    and self.cells.get_rettype_expr(c_style=True) == 'bint'):
                return after
            bound = ast.unparse(part.iter.args[0])
            extra_guard = ''
            iterator = f'range({bound})'
            if isinstance(predicate, ast.Compare):
                if not (len(predicate.ops) == 1
                        and isinstance(predicate.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                        and _call(predicate.left, 'abs', 1)
                        and self.kind(predicate.comparators[0], {}) == 'double'
                        and isinstance(predicate.comparators[0], (ast.Constant, ast.Attribute))):
                    return after
                value = self.indexed_cell(predicate.left.args[0], index, 'double', counter)
                if value is None:
                    return after
                op = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>='}[type(predicate.ops[0])]
                test = f'{alias}.f64_abs({value}) {op} {ast.unparse(predicate.comparators[0])}'
            else:
                test = self.indexed_cell(predicate, index, 'bint', counter)
                if test is None:
                    return after
            loop = f"""    for {counter} in {iterator}:
        if not ({test}):
            return False
    return True"""
        else:
            value = self.indexed_cell(predicate, index, 'double', counter)
            bounds = self.descending_window(part.iter)
            if value is None or bounds is None or self.cells.get_rettype_expr(c_style=True) != 'double':
                return after
            start, offset, step = bounds
            boxed_param = start
            extra_guard = f' and {start} >= 0'
            stop = f'{prefix}_stop'
            best = f'{prefix}_best'
            item = f'{prefix}_item'
            # A nonnegative start and negative stop floor guarantee a first value.
            # Later values replace it only on strict greater-than, as in max().
            first = self.indexed_cell(predicate, index, 'double', start)
            loop = f"""    {stop}: _mx_cy.longlong = {start} - {offset}
    if {stop} < -1:
        {stop} = -1
    {best}: _mx_cy.double = {first}
    {item}: _mx_cy.double
    for {counter} in range({start} - {step}, {stop}, -{step}):
        {item} = {value}
        if {item} > {best}:
            {best} = {item}
    return {best}"""

        before_fallbacks = len(self.fallbacks)
        fallback = self.cold_fallback(source, tree, None, function, boxed_param=boxed_param)
        if len(self.fallbacks) == before_fallbacks:
            return after
        self.providers.add(provider)
        self.count += 1
        return cst.FlattenSentinel(cst.parse_module(f"""
{function}: object = self.{func.attr.value}
{guard}: _mx_cy.bint = {alias}.{operation}_accepts({function}){extra_guard}
if {exact_receiver(self.cls_info.name)} and {guard}:
    {counter}: _mx_cy.longlong
{loop}
return {fallback}
""").body)

    def indexed_cell(self, node, index, kind, counter):
        if not (isinstance(node, ast.Call) and len(node.args) == 1
                and isinstance(node.args[0], ast.Name) and node.args[0].id == index
                and self.kind(node, {index: 'longlong'}) == kind):
            return None
        cell = self.cls_info.cells[node.func.attr]
        if len(cell.params) != 1 or cell.get_argtype_expr(cell.params[0], c_style=True) != 'long long':
            return None
        return f'self.{node.func.attr}({counter})'

    def descending_window(self, node):
        """Prove range(start, max(-1, start - offset), -step) for start >= 0."""
        if not _call(node, 'range', 3):
            return None
        start, limit, step = node.args
        if not (isinstance(start, ast.Name) and start.id in self.cells.params
                and self.cells.get_argtype_expr(start.id, c_style=True) == 'long long'
                and _call(limit, 'max', 2) and ast.unparse(limit.args[0]) == '-1'
                and isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub)):
            return None
        lower = limit.args[1]
        if not (isinstance(lower, ast.BinOp) and isinstance(lower.op, ast.Sub)
                and isinstance(lower.left, ast.Name) and lower.left.id == start.id
                and isinstance(lower.right, ast.Constant) and type(lower.right.value) is int
                and isinstance(step.operand, ast.Constant) and type(step.operand.value) is int
                and 0 < step.operand.value <= lower.right.value <= (1 << 31) - 1):
            return None
        return start.id, lower.right.value, step.operand.value

    def leave_SimpleStatementLine(self, before, after):
        if not hasattr(ast, 'unparse'):
            return after
        if len(before.body) != 1:
            return after
        statement = before.body[0]
        assignment = isinstance(statement, cst.Assign)
        if not (isinstance(statement, cst.Return) or assignment and len(statement.targets) == 1
                and isinstance(statement.targets[0].target, cst.Name)):
            return after
        value = statement.value
        if not isinstance(value, cst.Call) or len(value.args) != 1:
            return after
        func = value.func
        if not (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                and func.value.value == 'self'):
            return after
        ref = self.cls_info.refs.get(func.attr.value)
        reduction = getattr(ref, 'reduction', None)
        if (reduction not in ('f64_sum', 'bool_any', 'bool_all', 'f64_max')
                or not ref.helper or value.args[0].keyword or value.args[0].star):
            return after
        source = self.module.code_for_node(value.args[0].value)
        tree = expression_ast(source)
        if not isinstance(tree, ast.GeneratorExp) or not self.builtin_bounds:
            return after
        # A named sum result may remain an object when a replacement helper
        # returns one. Do not force an intermediate through the Cell's return
        # type or turn a predicate's early result into a return from the Cell.
        if assignment and reduction != 'f64_sum':
            return after
        if reduction == 'bool_any':
            return self.bounded_any(before, after, func, ref, source, tree)
        if reduction in ('bool_all', 'f64_max'):
            return self.single_range(after, func, ref, source, tree)
        capture = self.captures.get(id(before))
        if capture == 'range':
            return after
        if capture is not None and not any(isinstance(node, ast.Name) and node.id == capture
                                           for node in ast.walk(tree)):
            capture = None
        local_types = {capture: 'double'} if capture is not None else {}
        for part in tree.generators:
            if part.is_async or not isinstance(part.target, ast.Name):
                return after
            name = part.target.id
            # Reserved generated names must never capture authored names.
            if name == capture:
                return after
            iterator = part.iter
            if name not in local_types and any(isinstance(n, ast.Name) and n.id == name
                                               for n in ast.walk(iterator)):
                return after
            if (isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name)
                    and iterator.func.id == 'range' and not iterator.keywords
                    and 1 <= len(iterator.args) <= 3
                    and all(self.kind(arg, local_types) in ('longlong', 'bint')
                            for arg in iterator.args)
                    and safe_range_increment(iterator)):
                local_types[name] = 'longlong'
            elif isinstance(iterator, ast.Tuple) and len(iterator.elts) == 1:
                kind = self.kind(iterator.elts[0], local_types)
                if not kind:
                    return after
                local_types[name] = kind
            else:
                return after
            if not all(self.native_truth(condition, local_types) for condition in part.ifs):
                return after
        if self.kind(tree.elt, local_types) != 'double':
            return after
        provider, operation = ref.helper
        alias = module_name(provider)
        prefix = self.reserve_prefix()
        previous_fallbacks = len(self.fallbacks)
        fallback = self.cold_fallback(source, tree, capture, f'{prefix}_function')
        if len(self.fallbacks) == previous_fallbacks:
            return after
        # Keep generator locals isolated from the enclosing authored scope.
        renamed = {name: f'{prefix}_{name}' for name in local_types}
        class Rename(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in renamed:
                    return ast.copy_location(ast.Name(renamed[node.id], node.ctx), node)
                return node
        rename = Rename()
        for part in tree.generators:
            rename.visit(part)
        rename.visit(tree.elt)
        declarations = [f'{prefix}_function: object', f'{prefix}_state: {alias}.F64Sum']
        declarations += [f'{renamed[name]}: _mx_cy.{kind}' for name,kind in local_types.items()]
        lines = declarations + [f'{prefix}_function = self.{func.attr.value}',
                               f'if {alias}.f64_accepts({prefix}_function):',
                               f'    {alias}.f64_init(_mx_cy.address({prefix}_state))']
        if capture is not None:
            lines.append(f'    {renamed[capture]} = {capture}')
        indent = '    '
        for part in tree.generators:
            if isinstance(part.iter, ast.Tuple):
                lines.append(indent + f'{part.target.id} = {ast.unparse(part.iter.elts[0])}')
            else:
                lines.append(indent + f'for {part.target.id} in {ast.unparse(part.iter)}:')
                indent += '    '
            for condition in part.ifs:
                lines.append(indent + f'if {ast.unparse(condition)}:')
                indent += '    '
        result = (self.module.code_for_node(statement.targets[0].target) + ' = '
                  if assignment else 'return ')
        lines += [indent + f'{alias}.f64_add(_mx_cy.address({prefix}_state), {ast.unparse(tree.elt)})',
                  f'    {result}{alias}.f64_result(_mx_cy.address({prefix}_state))',
                  'else:', f'    {result}{fallback}']
        self.providers.add(provider)
        self.count += 1
        return cst.FlattenSentinel(cst.parse_module('\n'.join(lines)+'\n').body)
