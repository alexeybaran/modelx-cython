"""Physical frame bindings consume the original finalized compilation plan.

No tracing, lexical discovery or sample-type reconstruction lives here. Table
packing remains the existing helper/model-plan responsibility. Formula metadata
cannot execute; the generated Cython owns calculation semantics.
"""
import ast
from enum import IntEnum
from types import ModuleType
from types import SimpleNamespace
import inspect

from modelx_cython.formula_emitter import FormulaEmitter, SourceFormula
from .contracts import TableKey
from .trace_contracts import _direct_lookup_key_table


CTYPES = {'double': float, 'long long': int, 'bint': bool, 'str': str,
          'unicode': str, 'object': object}


def cell_bindings(cell):
    """Public types win; native category/key encodings are explicit adapters."""
    result = CTYPES.get(cell.public_result, object)
    observed = cell.original_info._rt
    if observed is not None and result is int:
        actual = observed.ret_type.value_type
        if isinstance(actual, type) and issubclass(actual, IntEnum):
            result = actual
    body = FormulaEmitter.native(cell, dict(cell.public_arguments), object)
    if result in (str, int):
        table = _direct_lookup_key_table(ast.unparse(body))
        if table is not None:
            result = TableKey(table)
    return {'return': result,
            **{n: CTYPES.get(t, object) for n, t in cell.public_arguments}}


def formula_module(space, *, globals_=None, filename='<compiled model>'):
    """One canonical body with original signature/cache status, no exec import."""
    module = ModuleType(space.class_identity)
    module.__file__ = filename
    module.__dict__.update(globals_ or {})
    for name, cell in space.cells.items():
        bindings = cell_bindings(cell)
        arguments = {n: t for n, t in bindings.items() if n != 'return'}
        node = FormulaEmitter.native(cell, arguments, bindings['return'])
        formula = SourceFormula(node, arguments, bindings['return'],
                                cached=cell.cached, filename=filename,
                                module_name=module.__name__, globals_=module.__dict__)
        setattr(module, name, formula)
    return module


def cache_domains(space):
    return {name: {argument: (lower, upper - lower)
                   for argument, lower, upper in cell.cache_domains}
            for name, cell in space.cells.items() if cell.cache_domains}


class _ResolvedCell:
    """Metadata comes from the plan; only Data preparation invokes the export."""
    def __init__(self, cell, instance):
        bindings = cell_bindings(cell)
        arguments = {n: t for n, t in bindings.items() if n != 'return'}
        self.formula = SimpleNamespace(source=ast.unparse(
            FormulaEmitter.native(cell, arguments, bindings['return'])) + '\n')
        self.parameters = tuple(n for n, _ in cell.public_arguments)
        self.is_cached = cell.cached
        # The original generated body, rather than a new boundary validator,
        # governs None. Do not add a rejection absent from original execution.
        self.allow_none = True
        self._entry = getattr(instance, cell.name)

    def __call__(self):
        if self.parameters:
            raise TypeError('Resource preparation requires a parameterless Data Cell')
        return self._entry()


class _ResolvedDomain:
    def __init__(self, instance, source):
        node = ast.parse(source, mode='eval').body
        self._call = isinstance(node, ast.Call) and not node.args and not node.keywords
        target = node.func if self._call else node
        if not (isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name) and target.value.id == 'self'):
            from .contracts import ContractError
            raise ContractError('Resolved resource domain needs a direct declared reference or Cell')
        self._instance, self._name = instance, target.attr
        self.ref_name = self._name if self._call else None

    def value(self, _impl):
        value = getattr(self._instance, self._name)
        return value() if self._call else value


def _space_source(space, instance):
    if not space.parameters:
        return None
    if space.space_formula_source is not None:
        from .exported_fallback import _UnbindFormula
        node = ast.parse(space.space_formula_source).body[0]
        node.args.args.pop(0)
        node.decorator_list = []
        node.name = '_formula'
        return ast.unparse(ast.fix_missing_locations(_UnbindFormula().visit(node))) + '\n'
    # The original exporter represents a declaration-only dynamic Space by
    # __call__ alone. Its actual public signature owns the parameter defaults.
    return 'def _formula' + str(inspect.signature(instance.__call__)) + ":\n    return {'refs': {}}\n"


def resolved_resource_model(package_plan, exported):
    """Bind already resolved identities to export values, without rediscovery.

    This thin adapter supplies values to existing pack/prepare routines. It does
    not parse an author manifest, walk runtime Spaces to discover structure,
    trace types, or evaluate Projection formulas. Public shells still come from
    the original compiler. This is not the final package runtime.
    """
    spaces, graph_spaces = {}, {}
    for identity, space in package_plan.spaces.items():
        path = tuple(identity.split('.')[1:])
        instance = exported
        for part in path:
            instance = getattr(instance, part)
        references = {name: getattr(instance, name) for name, _ in space.reference_types
                      if name not in space.parameters}
        cells = {name: _ResolvedCell(cell, instance) for name, cell in space.cells.items()}
        source = _space_source(space, instance)
        domains = None
        if space.collection is not None:
            if space.collection['status'] != 'declared':
                from .contracts import ContractError
                raise ContractError('Unresolved collection declaration: ' + identity)
            domains = {space.collection['over']: _ResolvedDomain(
                instance, space.collection['domain_source'])}
        spaces[path] = SimpleNamespace(parameters=space.parameters, spaces={},
            refs=references, cells=cells,
            formula=(SimpleNamespace(func=instance.__call__, source=source)
                     if source is not None else None),
            _impl=SimpleNamespace(collect_over_spec=domains))
        from .source_graph import _cell_interface
        interfaces = {}
        for name, cell in space.cells.items():
            interfaces[name] = _cell_interface(name, dict(
                source=cells[name].formula.source, cached=cell.cached,
                allow_none=True, sha256=cell.inventory()['formula_sha256']))
        graph_spaces[identity] = dict(formula=source, cell_interfaces=interfaces,
                                     parameters=list(space.parameters))
    graph = dict(spaces=graph_spaces,
                 source_provenance='resolved_original_compilation_plan')
    model = SimpleNamespace(name=package_plan.model, spaces=spaces, source_graph=graph)
    for path, space in spaces.items():
        if len(path) == 1:
            setattr(model, path[0], space)
    return model


def from_compilation_plan(package_plan, exported, destination, *, projection, **options):
    """Reuse physical resource preparation with original discovery as authority."""
    from dataclasses import replace
    from .modelx_frontend_native_mode import from_modelx
    space = package_plan.spaces[package_plan.model + '.' + projection]
    spec = from_modelx(resolved_resource_model(package_plan, exported), destination,
                      projection=projection, emission_space=space, **options)
    return replace(spec, cache_domains=cache_domains(space))
