"""Build a connected collect package from one verified modelx export.

The source graph determines collect regions and their consumers. Per-Space
preparation options describe input representation; they do not choose which
Spaces belong to the connected calculation.
"""
from __future__ import annotations

import json
import keyword
from pathlib import Path

from .contracts import ContractError
from .source_graph import describe_source_graph


def plan_collect_regions(graph, *, external_names=()):
    """Select collect Spaces and every Space consuming their results.

    Unselected source Spaces remain explicit external dependencies. The
    returned order puts compiled dependencies before their consumers.
    """
    if graph.get('schema') != 'mxcf.exported_space_graph.v1':
        raise ContractError('Expected a verified exported Space graph')
    model = graph.get('model')
    nodes = graph.get('spaces')
    if not isinstance(model, str) or not isinstance(nodes, dict):
        raise ContractError('Invalid exported Space graph')
    prefix = model + '.'
    names = {}
    for fullname, node in nodes.items():
        if (not fullname.startswith(prefix) or len(fullname.split('.')) != 2
                or not isinstance(node, dict)):
            raise ContractError('Connected collect build supports flat Spaces')
        names[fullname[len(prefix):]] = node
    roots = sorted(name for name, node in names.items() if node.get('collect') is not None)
    if not roots:
        raise ContractError('Export has no declared collect Space')
    external = set(external_names)
    if any(not isinstance(name, str) or not name.isidentifier() for name in external):
        raise ContractError('External Space names must be identifiers')
    selected = set(roots)
    # Follow reverse Space edges: consumers of a collect result also belong to
    # the compiled region. This handles multiple independent collect regions.
    changed = True
    while changed:
        changed = False
        for name, node in names.items():
            if name in selected:
                continue
            dependencies = node.get('space_references', {})
            if not isinstance(dependencies, dict):
                raise ContractError('Invalid Space dependency map: ' + name)
            if any(target in {prefix + member for member in selected}
                   for target in dependencies.values()):
                selected.add(name)
                changed = True
    order = []
    visiting = set()
    visited = set()

    def visit(name):
        if name in visited:
            return
        if name in visiting:
            raise ContractError('Cyclic collect Space dependencies')
        visiting.add(name)
        for ref, target in names[name].get('space_references', {}).items():
            if target not in nodes or not target.startswith(prefix):
                raise ContractError('Invalid source Space edge: ' + target)
            dependency = target[len(prefix):]
            if dependency in selected:
                visit(dependency)
            elif dependency not in external:
                raise ContractError('Unbound external Space: ' + target)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for name in sorted(selected):
        visit(name)
    return {'model': model, 'collect_roots': roots, 'compiled_spaces': order,
            'external_spaces': sorted({target[len(prefix):]
                for name in selected
                for target in names[name].get('space_references', {}).values()
                if target[len(prefix):] not in selected})}


def build_collect_package(exported, destination, *, external_imports,
                          space_options=None, package_name=None, eager=False):
    """Compile all discovered collect regions and assemble one package.

    ``space_options`` contains only representation and tracing contracts
    accepted by ``from_exported``. The graph, not that mapping, decides which
    Spaces compile. Build artifacts stay outside the exported source tree.
    """
    from .compiler import build
    from .connected_package import assemble_connected_package
    from .modelx_frontend_exported import open_exported_source

    exported = Path(exported)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Connected build destination must be fresh')
    manifest = json.loads((exported / '_mx_formula_sources.json').read_text(encoding='utf-8'))
    graph = describe_source_graph(manifest)
    external_imports = dict(external_imports)
    plan = plan_collect_regions(graph, external_names=external_imports)
    options = {str(name): dict(value) for name, value in (space_options or {}).items()}
    if set(options) - set(plan['compiled_spaces']):
        raise ContractError('Options supplied for an unselected Space')
    package_name = package_name or (plan['model'] + 'Compiled')
    if not package_name.isidentifier() or keyword.iskeyword(package_name):
        raise ContractError('Connected package name must be an identifier')
    destination.mkdir(parents=True)
    (destination / 'build_plan.json').write_text(json.dumps(plan, indent=2) + '\n',
                                                 encoding='utf-8')
    artifacts = {}
    with open_exported_source(exported) as derive:
        for name in plan['compiled_spaces']:
            arguments = dict(options.get(name, {}))
            if 'projection' in arguments:
                raise ContractError('Space options cannot override discovered projection')
            spec = derive(destination / '_sources' / name,
                          projection=name, **arguments)
            artifact = destination / '_artifacts' / name
            build(spec, artifact)
            artifacts[name] = artifact
    package = destination / package_name
    assembled = assemble_connected_package(package, graph, artifacts,
        external_imports=external_imports, eager=eager)
    return {'plan': plan, 'package': package, 'manifest': assembled,
            'artifacts': artifacts}
