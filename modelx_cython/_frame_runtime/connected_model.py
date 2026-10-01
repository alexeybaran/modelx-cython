"""Wire separately compiled Spaces from their source-verified reference graph.

The frame frontend currently builds one flat Space per artifact. This loader
keeps those artifacts independent while giving their original formulas the
compiled Space references declared by the exported model's source graph.
External helpers and uncompiled Spaces are explicit inputs; no formula is
rewritten or interpreted here.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from .contracts import ContractError
from .model import CompiledModel
from .source_graph import (generated_space_name, inherited_parameters,
                           is_empty_space_formula)


class _SpaceView:
    """Expose nested children and base parameter references on a compiled Space."""

    def __init__(self, space, children, base_values=None, references=None,
                 parameters=(), inherited=(), bindings=None):
        object.__setattr__(self, '_space', space)
        object.__setattr__(self, '_children', children)
        object.__setattr__(self, '_base_values', dict(base_values or {}))
        object.__setattr__(self, '_references', dict(references or {}))
        object.__setattr__(self, '_parameters', tuple(parameters))
        object.__setattr__(self, '_inherited', tuple(inherited))
        object.__setattr__(self, '_bindings', dict(bindings or {}))
        object.__setattr__(self, '_items', {})
        object.__setattr__(self, '_child_views', {})

    def _with_bindings(self, bindings):
        if not bindings:
            return self
        return _SpaceView(self._space, self._children, self._base_values,
                          self._references, self._parameters, self._inherited,
                          bindings)

    def _select(self, args, kwargs):
        if not self._parameters:
            raise TypeError('Space has no item parameters')
        if len(args) > len(self._parameters):
            raise TypeError('Too many Space parameters')
        values = dict(zip(self._parameters, args))
        for name, value in kwargs.items():
            if name not in self._parameters or name in values:
                raise TypeError('Unknown or repeated Space parameter: ' + name)
            values[name] = value
        if set(values) != set(self._parameters):
            raise TypeError('Missing Space parameters')
        key = tuple(values[name] for name in self._parameters)
        hash(key)
        try:
            return self._items[key]
        except KeyError:
            pass
        bound = {**self._bindings, **values}
        if self._space is None:
            item = self._with_bindings(bound)
            object.__setattr__(item, '_parameters', ())
        else:
            missing = set(self._inherited) - set(bound)
            if missing:
                raise TypeError('Missing inherited Space parameters: ' + ', '.join(sorted(missing)))
            selected = self._space(**bound)
            item = (_SpaceView(selected, self._children, self._base_values,
                               self._references, bindings=bound)
                    if self._children else selected)
        self._items[key] = item
        return item

    def __getattr__(self, name):
        if name in self._children:
            if name not in self._child_views:
                child = self._children[name]
                self._child_views[name] = (child._with_bindings(self._bindings)
                                           if isinstance(child, _SpaceView) else child)
            return self._child_views[name]
        if name in self._base_values:
            return self._base_values[name]
        if name in self._bindings:
            return self._bindings[name]
        if name in self._references:
            return self._references[name]
        if self._space is None:
            raise AttributeError(name)
        if self._inherited:
            missing = set(self._inherited) - set(self._bindings)
            if missing:
                raise AttributeError('Missing inherited Space parameters: '
                                     + ', '.join(sorted(missing)))
            return getattr(self._space(**self._bindings), name)
        if (self._base_values and name in self._space._model.manifest['cells']
                and set(self._space.parameters) == set(self._base_values)):
            return getattr(self._space(**self._base_values), name)
        return getattr(self._space, name)

    def __setattr__(self, name, value):
        if self._space is None:
            self._references[name] = value
            return
        setattr(self._space, name, value)

    def __call__(self, *args, **kwargs):
        if self._parameters:
            return self._select(args, kwargs)
        if self._space is None:
            raise TypeError('Cell-free Space has no item formula')
        return self._space(*args, **kwargs)

    def __getitem__(self, key):
        if self._parameters:
            args = key if isinstance(key, tuple) else (key,)
            return self._select(args, {})
        if self._space is None:
            raise TypeError('Cell-free Space has no item formula')
        return self._space[key]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ConnectedModel:
    """Own compiled Space artifacts and expose their ordinary Space entries."""

    def __init__(self, name, models, spaces, order):
        self.name = name
        self._models = models
        self._spaces = spaces
        self._order = order

    def __getattr__(self, name):
        try:
            return self._spaces[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def close(self):
        for name in reversed(self._order):
            if name in self._models:
                self._models[name].close()


def load_connected(graph, artifacts, *, external_bindings, tables_by_space=None,
                   shells=()):
    """Load compiled Space artifacts and bind their declared dependencies.

    ``artifacts`` and ``tables_by_space`` use public Space names. An omitted
    Space in the source graph must be supplied through ``external_bindings``.
    Each artifact's original Cell hashes must match the graph before loading.
    The return value exposes the same Space names and Cell entry points as the
    exported model. It owns the compiled artifacts and must be closed.
    """
    if isinstance(graph, (str, Path)):
        graph = json.loads(Path(graph).read_text(encoding='utf-8'))
    if graph.get('schema') != 'mxcf.exported_space_graph.v1':
        raise ContractError('Expected a verified exported Space graph')
    model_name = graph.get('model')
    if not isinstance(model_name, str) or not model_name:
        raise ContractError('Missing graph model identity')
    artifacts = {str(name): Path(path) for name, path in artifacts.items()}
    shells = set(shells)
    external_bindings = dict(external_bindings)
    tables_by_space = dict(tables_by_space or {})
    if (not artifacts or set(artifacts) & set(external_bindings)
            or shells & (set(artifacts) | set(external_bindings))):
        raise ContractError('Compiled and external Space names must be distinct')
    if set(tables_by_space) - set(artifacts):
        raise ContractError('Tables supplied for an uncompiled Space')
    graph_spaces = graph.get('spaces', {})
    if not isinstance(graph_spaces, dict):
        raise ContractError('Invalid exported Space graph')
    nodes = {}
    for name in (*artifacts, *sorted(shells)):
        generated_space_name(name)
        fullname = model_name + '.' + name
        if fullname not in graph_spaces:
            raise ContractError('Compiled Space is absent from source graph: ' + name)
        nodes[name] = graph_spaces[fullname]
        if name in shells and (nodes[name]['cells']
                               or not is_empty_space_formula(nodes[name].get('formula'))):
            raise ContractError('Invalid Cell-free Space shell: ' + name)
        base_values = nodes[name].get('base_parameter_values', {})
        if (not isinstance(base_values, dict)
                or set(base_values) - set(nodes[name].get('parameters', ()))
                or any(type(value) not in (type(None), bool, int, float, str)
                       for value in base_values.values())):
            raise ContractError('Invalid base parameter references: ' + name)

    def target_name(ref, target):
        # The exported identity, not the local alias, names the target Space.
        prefix = model_name + '.'
        if (not isinstance(target, str) or not target.startswith(prefix)
                or target not in graph_spaces):
            raise ContractError('Invalid source Space edge: ' + str(target))
        return target[len(prefix):]

    order = []
    visiting = set()
    visited = set()

    def visit(name):
        if name in visited:
            return
        if name in visiting:
            # Only reference wiring cycles are allowed here. Recursive Cell
            # evaluation remains guarded by each compiled cache entry.
            return
        visiting.add(name)
        for ref, target in nodes[name]['space_references'].items():
            dependency = target_name(ref, target)
            if dependency in nodes:
                visit(dependency)
            elif dependency not in external_bindings:
                raise ContractError('Unbound source Space: ' + target)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for name in sorted(nodes):
        visit(name)

    manifests = {}
    for name, artifact in artifacts.items():
        manifest = json.loads((artifact / 'manifest.json').read_text(encoding='utf-8'))
        plan = manifest.get('model', {})
        if (plan.get('projection') != generated_space_name(name)
                or plan.get('source_space', model_name + '.' + name)
                != model_name + '.' + name
                or manifest.get('public_api', {}).get('model_name') != model_name):
            raise ContractError('Compiled Space identity differs from source graph: ' + name)
        source_cells = nodes[name]['cells']
        cells = manifest.get('cells', {})
        if set(cells) != set(source_cells):
            raise ContractError('Compiled Cell inventory differs from source graph: ' + name)
        for cell_name, digest in source_cells.items():
            if cells[cell_name].get('coverage', {}).get('original_source_sha256') != digest:
                raise ContractError('Compiled Cell source differs from graph: '
                                    + name + '.' + cell_name)
        manifests[name] = manifest

    models = {}
    spaces = dict(external_bindings)
    try:
        # Construct every real Space before connecting any references. This
        # preserves Space identity even when A refers to B and B refers to A.
        for name in order:
            if name in shells:
                spaces[name] = _SpaceView(None, {}, parameters=nodes[name]['parameters'])
                continue
            artifact = artifacts[name]
            manifest = manifests[name]
            plan = manifest['model']
            if name in tables_by_space:
                tables = tables_by_space[name]
            else:
                public = manifest.get('public_api', {})
                if public.get('schema') != 'mxcf.package.v1':
                    raise ContractError('Compiled Space has no embedded Data: ' + name)
                path = artifact / public['tables_file']
                if _sha(path) != public['tables_sha256']:
                    raise ContractError('Embedded Data hash mismatch: ' + name)
                tables = pickle.loads(path.read_bytes())
            model = CompiledModel(artifact, tables, _defer_bindings=True)
            models[name] = model
            spaces[name] = getattr(model, plan['projection'])
        children = {}
        for name in nodes:
            if '.' in name:
                parent, leaf = name.rsplit('.', 1)
                if parent not in nodes:
                    raise ContractError('Nested Space parent is absent: ' + parent)
                children.setdefault(parent, {})[leaf] = name
        for name in sorted(nodes, key=lambda value: value.count('.'), reverse=True):
            base_values = nodes[name].get('base_parameter_values', {})
            if name in shells:
                spaces[name]._children.update({
                    leaf: spaces[child] for leaf, child in children.get(name, {}).items()})
                spaces[name]._base_values.update(base_values)
            elif name in children or base_values or inherited_parameters(graph, model_name, name):
                inherited = inherited_parameters(graph, model_name, name)
                spaces[name] = _SpaceView(spaces[name], {
                    leaf: spaces[child] for leaf, child in children.get(name, {}).items()},
                    base_values, parameters=nodes[name]['parameters'] if (inherited or name in children) else (),
                    inherited=inherited)
        for name in order:
            refs = {ref: spaces[target_name(ref, target)]
                    for ref, target in nodes[name]['space_references'].items()}
            if name in shells:
                spaces[name]._references.update(refs)
                continue
            plan = manifests[name]['model']
            def resolve(ref):
                if ref in refs:
                    return refs[ref]
                if ref in external_bindings:
                    return external_bindings[ref]
                default = plan.get('reference_defaults', {}).get(ref)
                if default is not None:
                    return default
                raise ContractError('Unbound connected reference: '
                                    + name + '.' + ref)
            formula = plan.get('space_formula')
            formula_refs = (formula.get('references', ()) if formula is not None else ())
            space_bindings = {ref: resolve(ref) for ref in formula_refs}
            overrides = {ref: resolve(ref)
                         for ref in plan.get('required_object_bindings', ())}
            models[name]._finish_bindings(overrides=overrides,
                                          space_bindings=space_bindings or None)
        return ConnectedModel(model_name, models, spaces, order)
    except BaseException:
        for name in reversed(order):
            if name in models:
                models[name].close()
        raise
