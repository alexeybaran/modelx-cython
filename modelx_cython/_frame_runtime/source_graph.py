"""Source-verified Space graph for the exported native-frame handoff.

The graph describes names and declared interfaces only. Formula bodies remain
the exporter's original sources and are compiled by Cython downstream.
"""

import hashlib

from .contracts import ContractError


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
        parameters = info.get('parameters', ())
        if (not isinstance(refs, list) or any(not isinstance(r, str) for r in refs)
                or not isinstance(parameters, list)
                or any(not isinstance(p, str) for p in parameters)):
            raise ContractError('Invalid Space interface: ' + fullname)
        internal = {}
        external = []
        for ref in refs:
            target = None
            for i in range(len(parts) - 1, 0, -1):
                candidate = '.'.join([*parts[:i], ref])
                if candidate in paths:
                    target = candidate
                    break
            if target is None:
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
            'spaces': nodes, 'edges': edges}
