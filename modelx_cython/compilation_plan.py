"""Final original-compiler metadata shared by every code-generation choice.

This is a description of compilation, not another type or formula evaluator.
Build it after usage/specification/helper resolution and before source emission.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType
from typing import Mapping


def _digest(source):
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def _identity(fqname):
    parts = fqname.split('.')
    return '.'.join([parts[0], *(p[3:] for p in parts[1:]
                                if p.startswith(('_m_', '_c_')))])


def _signature(node):
    """Retain syntax, including defaults; never execute a default expression."""
    return ast.unparse(node.args)


@dataclass(frozen=True)
class CellEmissionPlan:
    identity: str
    name: str
    formula_source: str
    signature: str
    public_result: str
    public_arguments: tuple[tuple[str, str], ...]
    cached: bool
    traced: bool
    cache_kind: str
    cache_domains: tuple[tuple[str, int, int], ...]
    dependencies: tuple[str, ...]
    reference_reads: tuple[str, ...]
    locked: bool
    source_provenance: str
    original_info: object = field(repr=False, compare=False)
    formula_node: ast.FunctionDef = field(repr=False, compare=False)

    def inventory(self):
        return dict(identity=self.identity, name=self.name,
                    signature=self.signature, formula_sha256=_digest(self.formula_source),
                    public_result=self.public_result,
                    public_arguments=dict(self.public_arguments), cached=self.cached,
                    traced=self.traced, cache_kind=self.cache_kind,
                    cache_domains=[dict(argument=n, lower=lo, upper=hi)
                                   for n, lo, hi in self.cache_domains],
                    dependencies=list(self.dependencies),
                    reference_reads=list(self.reference_reads), locked=self.locked,
                    source_provenance=self.source_provenance)


@dataclass(frozen=True)
class SpaceEmissionPlan:
    identity: str
    class_identity: str
    class_name: str
    parent: str
    children: tuple[str, ...]
    parameters: tuple[str, ...]
    cells: Mapping[str, CellEmissionPlan]
    reference_types: tuple[tuple[str, str], ...]
    reference_targets: tuple[tuple[str, str], ...]
    helper_bindings: tuple[dict, ...]
    locked: bool
    space_formula_source: str | None
    collect_source: str | None
    collection: Mapping | None
    original_info: object = field(repr=False, compare=False)

    def inventory(self):
        return dict(identity=self.identity, class_identity=self.class_identity,
                    parent=self.parent, children=list(self.children),
                    parameters=list(self.parameters),
                    references=dict(self.reference_types),
                    reference_targets=dict(self.reference_targets),
                    helper_bindings=list(self.helper_bindings), locked=self.locked,
                    space_formula_source=self.space_formula_source,
                    collect_source=self.collect_source,
                    collection=dict(self.collection) if self.collection is not None else None,
                    cells={n: c.inventory() for n, c in self.cells.items()})


@dataclass(frozen=True)
class PackageEmissionPlan:
    model: str
    spaces: Mapping[str, SpaceEmissionPlan]
    by_class: Mapping[str, SpaceEmissionPlan]
    collection_consumers: tuple[dict, ...]

    def inventory(self):
        return dict(schema='modelx_cython.compilation_plan.v1', model=self.model,
                    type_authority='ModuleInfo after specifications and usage analysis',
                    collection_consumers=list(self.collection_consumers),
                    spaces={n: s.inventory() for n, s in self.spaces.items()})


def build_package_plan(units):
    """Snapshot resolved ModuleInfo; do not trace, import or rediscover a model."""
    units = tuple(units)
    spaces, classes = {}, {}
    for unit in units:
        tree = ast.parse(unit.source)
        nodes = {c.name: c for c in tree.body if isinstance(c, ast.ClassDef)}
        for name, info in unit.module_info.classes.items():
            identity = _identity(info.fqname)
            if identity in spaces:
                raise ValueError('Ambiguous exported Space identity: ' + identity)
            methods = {n.name: n for n in nodes[name].body
                       if isinstance(n, ast.FunctionDef)}
            # __call__ and exporter plumbing are not economic Cells.
            names = {n for n, c in info.cells.items() if not c.is_special()}
            collector = methods.get('collect')
            collection = _collection(collector)
            if collection is not None:
                # Exported traversal is an interface, not an authored formula.
                names.discard('collect')
            cells = {}
            for cell_name in sorted(names):
                cell = info.cells[cell_name]
                fn = methods.get('_f_' + cell_name, methods.get(cell_name))
                if fn is None:
                    raise ValueError('Missing exported Cell body: ' + cell.fqname)
                reads = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
                         and isinstance(n.value, ast.Name) and n.value.id == 'self'}
                cached = cell.has_formula_def
                dense = (cached and cell.has_args() and cell.has_typeinfo()
                         and cell.is_arrayable())
                domains = (tuple((p, 0, int(size)) for p, size in zip(
                    cell.params, info.cells_arg_sizes[tuple(cell.params)])) if dense else ())
                kind = ('uncached' if not cached else 'dense' if dense
                        else 'dictionary' if cell.has_args() else 'scalar')
                cells[cell_name] = CellEmissionPlan(
                    identity + '.' + cell_name, cell_name, ast.unparse(fn) + '\n',
                    _signature(fn), cell.get_rettype_expr(c_style=True),
                    tuple((p, cell.get_argtype_expr(p, c_style=True)) for p in cell.params),
                    cached, cell.has_typeinfo(), kind, domains,
                    tuple(sorted(reads & names)), tuple(sorted(reads - names)),
                    cell.is_locked, 'original exported method', cell, fn)
            helper = getattr(info, 'helper_info', None)
            bindings = tuple(dict(vars(b)) for b in helper.bindings.values()) if helper else ()
            formula = methods.get('_mx_space_formula')
            space = SpaceEmissionPlan(
                identity, info.fqname, name, identity.rpartition('.')[0],
                tuple(identity + '.' + child for child in info.spaces),
                tuple(info.visitor.space_params.get(name, ())), MappingProxyType(cells),
                tuple((n, r.get_type_expr(c_style=True)) for n, r in info.refs.items()),
                tuple((n, _identity(r.mx_class)) for n, r in info.refs.items() if r.mx_class),
                bindings, info.is_locked, ast.unparse(formula) if formula else None,
                ast.unparse(collector) if collector else None,
                MappingProxyType(collection) if collection else None, info)
            spaces[identity] = space
            classes[info.fqname] = space
    models = {u.fqname.split('.')[0] for u in units}
    if len(models) != 1:
        raise ValueError('Compilation units must belong to one exported package')
    consumers = []
    for space in spaces.values():
        for cell in space.cells.values():
            for node in ast.walk(cell.formula_node):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'collect'):
                    continue
                path, value = [], node.func.value
                while isinstance(value, ast.Attribute):
                    path.append(value.attr)
                    value = value.value
                target = space if isinstance(value, ast.Name) and value.id == 'self' else None
                for part in reversed(path):
                    if target is None:
                        break
                    ref = dict(target.reference_targets).get(part)
                    target = spaces.get(ref or (target.identity + '.' + part))
                consumers.append(dict(cell=cell.identity, expression=ast.unparse(node),
                                      target=target.identity if target else None,
                                      declared=bool(target and target.collection)))
    return PackageEmissionPlan(models.pop(), MappingProxyType(spaces),
                               MappingProxyType(classes), tuple(consumers))


def _collection(method):
    """Read the exporter's declared traversal arguments, never its data domain."""
    if method is None:
        return None
    calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'collect_from_space'
             and isinstance(n.func.value, ast.Name) and n.func.value.id == '_mx_sys']
    if len(calls) != 1 or len(calls[0].args) != 5:
        return dict(status='unresolved', source=ast.unparse(method))
    call = calls[0]
    assignments = {n.targets[0].id: n.value for n in ast.walk(method)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name)}
    domain = call.args[1]
    if isinstance(domain, ast.Name):
        domain = assignments.get(domain.id, domain)
    try:
        over = ast.literal_eval(call.args[2])
        parameters = ast.literal_eval(call.args[3])
    except (ValueError, TypeError):
        return dict(status='unresolved', source=ast.unparse(method))
    return dict(status='declared', over=over, parameters=list(parameters),
                domain_source=ast.unparse(domain), outputs_cell='collect_outputs')
