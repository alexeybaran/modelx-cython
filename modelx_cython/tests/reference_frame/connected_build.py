"""Build a connected collect package from one verified modelx export.

The source graph determines collect regions and their consumers. Per-Space
preparation options describe input representation; they do not choose which
Spaces belong to the connected calculation.
"""
from __future__ import annotations

import json
import keyword
from pathlib import Path

from modelx_cython._frame_runtime.contracts import ContractError
from .source_graph import (generated_space_name, inherited_parameters,
                           is_empty_space_formula)


def plan_package(graph, *, external_names=(), collect_only=False):
    """Plan one compiled package from the verified exported Space graph.

    A complete package includes every Space. A collect-only package includes
    declared collect Spaces and their consumers, leaving other Spaces as
    explicit external dependencies. Both use the same dependency ordering.
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
        relative = fullname[len(prefix):] if fullname.startswith(prefix) else ''
        if (not relative or any(not part.isidentifier()
                                for part in relative.split('.'))
                or not isinstance(node, dict)):
            raise ContractError('Invalid exported Space identity: ' + str(fullname))
        names[relative] = node
    generated = {name: generated_space_name(name) for name in names}
    if len(set(generated.values())) != len(generated):
        raise ContractError('Exported Space paths collide in generated names')
    roots = sorted(name for name, node in names.items() if node.get('collect') is not None)
    if collect_only and not roots:
        raise ContractError('Export has no declared collect Space')
    external = set(external_names)
    if any(not isinstance(name, str) or not name
           or any(not part.isidentifier() for part in name.split('.'))
           for name in external):
        raise ContractError('External Space names must be qualified identifiers')
    selected = set(roots) if collect_only else set(names)
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
            # The loader constructs all Space shells before wiring references.
            # A reference cycle is therefore distinct from recursive Cell work.
            return
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
            'generated_spaces': {name: generated[name] for name in order},
            'external_spaces': sorted({target[len(prefix):]
                for name in selected
                for target in names[name].get('space_references', {}).values()
                if target[len(prefix):] not in selected})}


def plan_collect_regions(graph, *, external_names=()):
    """Compatibility spelling for the collect-only package plan."""
    return plan_package(graph, external_names=external_names, collect_only=True)


def build_package(exported, destination, *, external_imports,
                  space_options=None, package_name=None, eager=False,
                  collect_only=False, trace_logger=None):
    """Compile a complete or collect-only connected package.

    ``space_options`` contains only representation and tracing contracts
    accepted by ``from_exported``. The graph, not that mapping, decides which
    Spaces compile. A standard mx2cy ``trace_logger`` supplies observed Cell
    types across the selected Spaces before generation; explicit contract
    options cannot silently override that trace. Build artifacts stay outside
    the exported source tree.
    """
    from modelx_cython.tests.reference_frame.compiler import build
    from .connected_package import assemble_connected_package
    from .modelx_frontend_exported import open_exported_source
    from .trace_contracts import from_mx2cy_trace

    exported = Path(exported)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Connected build destination must be fresh')
    external_imports = dict(external_imports)
    with open_exported_source(exported) as derive:
        graph = derive.source_graph
        plan = plan_package(graph, external_names=external_imports,
                            collect_only=collect_only)
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
        shells = []
        for name in plan['compiled_spaces']:
            node = graph['spaces'][plan['model'] + '.' + name]
            if not node['cells']:
                # A static, Cell-free Space is a hierarchy shell, not a
                # formula kernel. Its children are assembled below from the
                # same source graph. Parameterized shells need item-scoped
                # inherited bindings and cannot be flattened safely.
                if node.get('collect') is not None or not is_empty_space_formula(node.get('formula')):
                    raise ContractError('Cell-free Space with formula or collect '
                                        'requires compiled execution: ' + name)
                if options.get(name):
                    raise ContractError('A Cell-free Space has no kernel options: ' + name)
                shells.append(name)
                continue
            arguments = dict(options.get(name, {}))
            if 'projection' in arguments or 'inherited_parameters' in arguments:
                raise ContractError('Space options cannot override discovered hierarchy')
            if trace_logger is not None:
                if 'contracts' in arguments:
                    raise ContractError('Trace contracts and Space contract options conflict')
                arguments['contracts'] = from_mx2cy_trace(
                    trace_logger, derive.source_manifest, space=name)
            declared = graph['spaces'][plan['model'] + '.' + name]['space_references']
            inherited = inherited_parameters(graph, plan['model'], name)
            if inherited:
                parts = name.split('.')
                for depth in range(1, len(parts)):
                    ancestor = '.'.join(parts[:depth])
                    ancestor_node = graph['spaces'][plan['model'] + '.' + ancestor]
                    if not is_empty_space_formula(ancestor_node.get('formula')):
                        raise ContractError('Inherited parameter binding requires '
                                            'a no-op ancestor Space formula: ' + ancestor)
                arguments['inherited_parameters'] = inherited
            arguments['object_bindings'] = tuple(dict.fromkeys(
                (*declared, *arguments.get('object_bindings', ()))))
            spec = derive(destination / '_sources' / name,
                          projection=name, **arguments)
            artifact = destination / '_artifacts' / plan['generated_spaces'][name]
            build(spec, artifact)
            artifacts[name] = artifact
    package = destination / package_name
    assembled = assemble_connected_package(package, graph, artifacts,
        shells=shells, external_imports=external_imports, eager=eager)
    return {'plan': plan, 'package': package, 'manifest': assembled,
            'artifacts': artifacts}


def build_collect_package(exported, destination, *, external_imports,
                          space_options=None, package_name=None, eager=False,
                          trace_logger=None):
    """Compatibility entry for a package containing collect regions."""
    return build_package(exported, destination,
                         external_imports=external_imports,
                         space_options=space_options,
                         package_name=package_name, eager=eager,
                         collect_only=True, trace_logger=trace_logger)
