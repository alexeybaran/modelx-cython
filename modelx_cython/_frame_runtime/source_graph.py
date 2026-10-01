"""Source-verified Space graph for the exported native-frame handoff.

The graph describes names and declared interfaces only. Formula bodies remain
the exporter's original sources and are compiled by Cython downstream.
"""

import ast
import hashlib
import keyword

from .contracts import ContractError


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


def inherited_parameters(graph, model, name):
    """Return visible ancestor parameter names, shadowed by the local Space."""
    parts = name.split('.')
    inherited = {}
    for depth in range(1, len(parts)):
        parent = graph['spaces'][model + '.' + '.'.join(parts[:depth])]
        for parameter in parent.get('parameters', ()):
            inherited[parameter] = None
    own = graph['spaces'][model + '.' + name].get('parameters', ())
    return tuple(parameter for parameter in inherited if parameter not in own)


def generated_space_name(relative):
    """Give a nested export a stable Cython identifier without losing its path."""
    if (not isinstance(relative, str) or not relative
            or any(not part.isidentifier() or keyword.iskeyword(part)
                   for part in relative.split('.'))):
        raise ContractError('Invalid exported Space path: ' + repr(relative))
    if '.' not in relative:
        return relative
    return 'MxcfNested' + hashlib.sha256(relative.encode('utf-8')).hexdigest()[:20]


def describe_source_graph(manifest):
    """Return the complete, deterministic Space/reference graph in an export.

    A reference is a Space edge only when it names a Space in the same lexical
    scope or an ancestor scope. Other references remain external bindings; the
    compiler must not silently treat them as disconnected model objects.
    """
    if manifest.get('schema') != 'modelx.formula_sources.v2':
        raise ContractError('Unsupported modelx formula-source manifest')
    model = manifest.get('model')
    spaces = manifest.get('spaces')
    if not isinstance(model, str) or not model or not isinstance(spaces, dict):
        raise ContractError('Incomplete modelx formula-source manifest')
    paths = {}
    for fullname, info in spaces.items():
        parts = fullname.split('.')
        if parts[0] != model or len(parts) < 2 or any(not p.isidentifier() for p in parts):
            raise ContractError('Formula source Space belongs to another model')
        if not isinstance(info, dict) or not isinstance(info.get('cells'), dict):
            raise ContractError('Formula source Space has no Cell map: ' + fullname)
        paths[fullname] = parts
        for name, cell in info['cells'].items():
            if not isinstance(cell, dict) or not isinstance(cell.get('source'), str):
                raise ContractError('Missing formula source: ' + fullname + '.' + name)
            digest = hashlib.sha256(cell['source'].encode('utf-8')).hexdigest()
            if digest != cell.get('sha256'):
                raise ContractError('Formula source hash mismatch: ' + fullname + '.' + name)

    nodes = {}
    edges = []
    for fullname in sorted(paths):
        parts = paths[fullname]
        info = spaces[fullname]
        refs = info.get('references', ())
        explicit_targets = info.get('reference_targets', {})
        parameters = info.get('parameters', ())
        if (not isinstance(refs, list) or any(not isinstance(r, str) for r in refs)
                or not isinstance(explicit_targets, dict)
                or any(ref not in refs or not isinstance(target, str)
                       for ref, target in explicit_targets.items())
                or not isinstance(parameters, list)
                or any(not isinstance(p, str) for p in parameters)):
            raise ContractError('Invalid Space interface: ' + fullname)
        internal = {}
        external = []
        for ref in refs:
            target = explicit_targets.get(ref)
            if target is None:
                for i in range(len(parts) - 1, 0, -1):
                    candidate = '.'.join([*parts[:i], ref])
                    if candidate in paths:
                        target = candidate
                        break
            if target is not None and target.startswith(model + '.') and target not in paths:
                raise ContractError('Space reference target is absent from export: '
                                    + fullname + '.' + ref)
            if target is None:
                external.append(ref)
            elif target not in paths:
                external.append(ref)
            else:
                internal[ref] = target
                edges.append({'from': fullname, 'reference': ref, 'to': target})
        nodes[fullname] = {
            'parameters': list(parameters),
            'cells': {name: info['cells'][name]['sha256'] for name in sorted(info['cells'])},
            'formula': info.get('formula'),
            'collect': info.get('collect'),
            'space_references': internal,
            'external_references': external,
        }
    return {'schema': 'mxcf.exported_space_graph.v1', 'model': model,
            'source_provenance': manifest.get('source_provenance', 'author_sources'),
            'spaces': nodes, 'edges': edges}
