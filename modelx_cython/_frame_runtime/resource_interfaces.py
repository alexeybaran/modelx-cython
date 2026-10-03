"""Declared resource signatures/helper domains; no alternate frontend."""
import ast
import hashlib
import keyword
from .contracts import ContractError


def _cell_interface(name, cell):
    """Describe the author-visible signature without interpreting its body.

    Cython type choices are supplied separately by tracing/specification. This
    record preserves the Python calling contract for public adapter emission.
    """
    statements = ast.parse(cell['source']).body
    if len(statements) != 1:
        raise ContractError('A Cell source must contain one formula: ' + name)
    formula = statements[0]
    if isinstance(formula, ast.FunctionDef) and formula.name == name:
        args, result, kind = formula.args, formula.returns, 'function'
    elif (isinstance(formula, ast.Expr)
          and isinstance(formula.value, ast.Lambda)):
        args, result, kind = formula.value.args, None, 'lambda'
    else:
        raise ContractError('Invalid Cell formula signature: ' + name)
    positional = [*args.posonlyargs, *args.args]
    default_start = len(positional) - len(args.defaults)
    params = []
    for index, arg in enumerate(positional):
        parameter = {'name': arg.arg, 'kind': ('POSITIONAL_ONLY' if index < len(args.posonlyargs)
                                                else 'POSITIONAL_OR_KEYWORD')}
        if arg.annotation is not None:
            parameter['annotation'] = ast.unparse(arg.annotation)
        if index >= default_start:
            parameter['default'] = ast.unparse(args.defaults[index - default_start])
        params.append(parameter)
    if args.vararg is not None:
        params.append({'name': args.vararg.arg, 'kind': 'VAR_POSITIONAL'})
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parameter = {'name': arg.arg, 'kind': 'KEYWORD_ONLY'}
        if arg.annotation is not None:
            parameter['annotation'] = ast.unparse(arg.annotation)
        if default is not None:
            parameter['default'] = ast.unparse(default)
        params.append(parameter)
    if args.kwarg is not None:
        params.append({'name': args.kwarg.arg, 'kind': 'VAR_KEYWORD'})
    interface = {'formula_kind': kind, 'parameters': params,
                 'cached': bool(cell.get('cached', True)),
                 'allow_none': bool(cell.get('allow_none', False)),
                 'source_sha256': cell['sha256']}
    if result is not None:
        interface['return_annotation'] = ast.unparse(result)
    return interface



def is_empty_space_formula(source):
    """Recognize a declaration-only Space formula without running it."""
    if source is None:
        return True
    try:
        statements = ast.parse(source).body
    except SyntaxError:
        return False
    if len(statements) != 1:
        return False
    expression = statements[0]
    if isinstance(expression, ast.Expr) and isinstance(expression.value, ast.Lambda):
        body = expression.value.body
    elif (isinstance(expression, ast.FunctionDef) and len(expression.body) == 1
          and isinstance(expression.body[0], ast.Return)):
        body = expression.body[0].value
    else:
        return False
    if isinstance(body, ast.Constant) and body.value is None:
        return True
    # Older exports encode a no-op item formula as an empty reference map.
    return (isinstance(body, ast.Dict) and len(body.keys) == 1
            and isinstance(body.keys[0], ast.Constant)
            and body.keys[0].value == 'refs'
            and isinstance(body.values[0], ast.Dict)
            and not body.values[0].keys)



def generated_space_name(relative):
    """Give a nested export a stable Cython identifier without losing its path."""
    if (not isinstance(relative, str) or not relative
            or any(not part.isidentifier() or keyword.iskeyword(part)
                   for part in relative.split('.'))):
        raise ContractError('Invalid exported Space path: ' + repr(relative))
    if '.' not in relative:
        return relative
    return 'MxcfNested' + hashlib.sha256(relative.encode('utf-8')).hexdigest()[:20]



def _direct_lookup_key_table(source):
    """Recognize an explicit helper result, without typing other expressions.

    ``lookup.key(data.table(), value)`` already declares the table whose
    labels this Cell returns.  The ordinary trace only sees a Python string;
    the frame additionally needs that table identity to encode the label in
    its native key domain and to return the original label at public entry.
    """
    try:
        node = ast.parse(source).body[0]
    except (SyntaxError, IndexError):
        return None
    if isinstance(node, ast.FunctionDef):
        body = [part for part in node.body if not (isinstance(part, ast.Expr)
                and isinstance(part.value, ast.Constant)
                and isinstance(part.value.value, str))]
        if (not body or not isinstance(body[-1], ast.Return)
                or any(isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.Lambda, ast.ClassDef, ast.Yield, ast.YieldFrom))
                       for statement in body for part in ast.walk(statement))):
            return None
        values = [part.value for statement in body for part in ast.walk(statement)
                  if isinstance(part, ast.Return)]
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Lambda):
        values = [node.value.body]
    else:
        return None
    tables = []
    for value in values:
        if not (isinstance(value, ast.Call) and len(value.args) == 2
                and not value.keywords and isinstance(value.func, ast.Attribute)
                and value.func.attr == 'key' and isinstance(value.func.value, ast.Name)
                and value.func.value.id == 'lookup'):
            return None
        table = value.args[0]
        if not (isinstance(table, ast.Call) and not table.args and not table.keywords
                and isinstance(table.func, ast.Attribute)
                and isinstance(table.func.value, ast.Name)
                and table.func.value.id == 'data'):
            return None
        tables.append(table.func.attr)
    return tables[0] if tables and len(set(tables)) == 1 else None

