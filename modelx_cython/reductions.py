"""Temporary lowering of explicit registered numeric sums in return statements.

Only range loops and singleton bindings with proven numeric expressions qualify.
The existing formula body remains authoritative; general generators stay intact.
"""
import ast

import libcst as cst

from .helpers import module_name


class NativeSums(cst.CSTTransformer):
    def __init__(self, cls_info, providers):
        self.cls_info = cls_info
        self.providers = providers
        self.count = 0
        self.module = cst.Module([])

    def visit_FunctionDef(self, node):
        return '_mx_reduce_' not in self.module.code_for_node(node)

    def kind(self, node, locals_):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return 'double' if type(node.value) is float else 'longlong'
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
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return self.kind(node.operand, locals_)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = self.kind(node.left, locals_), self.kind(node.right, locals_)
            # Do not introduce native integer arithmetic merely from observations.
            if left and right and 'double' in (left, right):
                return 'double'
        return None

    def leave_SimpleStatementLine(self, before, after):
        if not hasattr(ast, 'unparse'):
            return after
        if len(before.body) != 1 or not isinstance(before.body[0], cst.Return):
            return after
        value = before.body[0].value
        if not isinstance(value, cst.Call) or len(value.args) != 1:
            return after
        func = value.func
        if not (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                and func.value.value == 'self'):
            return after
        ref = self.cls_info.refs.get(func.attr.value)
        if (not ref or getattr(ref, 'reduction', None) != 'f64_sum'
                or not ref.helper or value.args[0].keyword or value.args[0].star):
            return after
        source = self.module.code_for_node(value.args[0].value)
        tree = ast.parse('(' + source + ')', mode='eval').body
        if not isinstance(tree, ast.GeneratorExp):
            return after
        local_types = {}
        for part in tree.generators:
            if part.is_async or part.ifs or not isinstance(part.target, ast.Name):
                return after
            name = part.target.id
            # Reserved generated names must never capture authored names.
            if name.startswith('_mx_'):
                return after
            iterator = part.iter
            if name not in local_types and any(isinstance(n, ast.Name) and n.id == name
                                               for n in ast.walk(iterator)):
                return after
            if (isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name)
                    and iterator.func.id == 'range' and not iterator.keywords
                    and 1 <= len(iterator.args) <= 3
                    and all(self.kind(arg, local_types) in ('longlong', 'bint')
                            for arg in iterator.args)):
                local_types[name] = 'longlong'
            elif isinstance(iterator, ast.Tuple) and len(iterator.elts) == 1:
                kind = self.kind(iterator.elts[0], local_types)
                if not kind:
                    return after
                local_types[name] = kind
            else:
                return after
        if self.kind(tree.elt, local_types) != 'double':
            return after
        provider, operation = ref.helper
        alias = module_name(provider)
        prefix = f'_mx_reduce_{self.count}'
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
        indent = '    '
        for part in tree.generators:
            if isinstance(part.iter, ast.Tuple):
                lines.append(indent + f'{part.target.id} = {ast.unparse(part.iter.elts[0])}')
            else:
                lines.append(indent + f'for {part.target.id} in {ast.unparse(part.iter)}:')
                indent += '    '
        lines += [indent + f'{alias}.f64_add(_mx_cy.address({prefix}_state), {ast.unparse(tree.elt)})',
                  f'    return {alias}.f64_result(_mx_cy.address({prefix}_state))',
                  'else:', f'    return {prefix}_function({source})']
        self.providers.add(provider)
        self.count += 1
        return cst.FlattenSentinel(cst.parse_module('\n'.join(lines)+'\n').body)
