"""Pass the standard mx2cy trace's observed Cell types to the native frame.

This supplies reference context. It does not rewrite formula expressions or
claim that a small trace proves every possible input type.
"""

import ast
import numbers
from enum import IntEnum

from .contracts import ContractError, TableKey


def _scalar_type(observed):
    if observed is bool:
        return bool
    if isinstance(observed, type) and issubclass(observed, IntEnum):
        return observed
    if isinstance(observed, type) and issubclass(observed, numbers.Integral):
        return int
    if isinstance(observed, type) and issubclass(observed, numbers.Real):
        return float
    if isinstance(observed, type) and issubclass(observed, str):
        return str
    return object


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


def from_mx2cy_trace(logger, source_manifest, *, space='Projection'):
    """Return optional annotations for exactly the source-preserved Cells.

    Unobserved Cells keep the frontend's object fallback. The caller remains
    responsible for validating compiled results beyond the traced sample.
    """
    key = source_manifest['model'] + '.' + space
    try:
        source_cells = source_manifest['spaces'][key]['cells']
    except KeyError as exc:
        raise ContractError('Source manifest has no Space: ' + key) from exc
    parts = space.split('.')
    if any(not part.isidentifier() for part in parts):
        raise ContractError('Invalid traced Space identity: ' + space)
    module_path = ''.join('._m_' + part for part in parts[:-1])
    suffix = module_path + '._mx_classes._c_' + parts[-1] + '._f_'
    result = {}
    for fullname, observed in logger.cells_info.items():
        if suffix not in fullname:
            continue
        before, name = fullname.rsplit(suffix, 1)
        if not before or name not in source_cells:
            continue
        if name in result:
            raise ContractError('Ambiguous traced Cell: ' + name)
        row = {'return': (object if observed.ret_type.is_array else
                          _scalar_type(observed.ret_type.value_type))}
        if row['return'] in (str, int):
            table = _direct_lookup_key_table(source_cells[name].get('source', ''))
            if table is not None:
                row['return'] = TableKey(table)
        row.update({arg: _scalar_type(typ)
                    for arg, typ in observed.arg_types.items()})
        result[name] = row
    return result
