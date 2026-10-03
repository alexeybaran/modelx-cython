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
import numbers

from modelx_cython.formula_emitter import FormulaEmitter, SourceFormula
from .contracts import TableKey
from .trace_contracts import _direct_lookup_key_table


CTYPES = {'double': float, 'long long': int, 'bint': bool, 'str': str,
          'unicode': str, 'object': object}


def cell_bindings(cell):
    """Consume the shared physical snapshot; older records use its resolver."""
    if cell.physical_bindings is not None:
        return dict(cell.physical_bindings)
    return _resolve_cell_bindings(cell)


def _resolve_cell_bindings(cell):
    """Keep the public ABI separate from native category/key representations.

    A public object ABI is required for a mix of IntEnum and int observations.
    It must not erase a concrete category binding within the prepared context.
    Public wrappers still need explicit checked encoding/decoding; this function
    does not change public signatures or authorize dropping general object paths.
    """
    result = CTYPES.get(cell.public_result, object)
    observed = cell.original_info._rt
    explicit_result = getattr(cell.original_info, '_spec_ret_t', '')
    if observed is not None and result in (int, object) and not explicit_result:
        actual = observed.ret_type.value_type
        if not observed.ret_type.is_array and _native_category(actual) is not None:
            result = actual
    if result in (str, int) and not cell.receiver_requirements:
        body = FormulaEmitter.native(cell, dict(cell.public_arguments), object)
        table = _direct_lookup_key_table(ast.unparse(body))
        if table is not None:
            result = TableKey(table)
    arguments = {n: CTYPES.get(t, object) for n, t in cell.public_arguments}
    if observed is not None:
        choices = getattr(observed, 'observed_arg_types', {})
        for name, value in tuple(arguments.items()):
            if value is not object:
                continue
            kinds = choices.get(name, (observed.arg_types.get(name, object),))
            categories = {t for t in kinds if _native_category(t) is not None}
            if len(categories) == 1 and all(isinstance(t, type) and
                    issubclass(t, numbers.Integral) for t in kinds):
                arguments[name] = next(iter(categories))
    return {'return': result, **arguments}


def _native_category(typ):
    """Use the existing emitter's category domain contract, not new inference."""
    if not isinstance(typ, type) or not issubclass(typ, IntEnum):
        return None
    from .compiler import type_info
    from .contracts import ContractError
    try:
        type_info(typ)
    except ContractError:
        return None
    return typ


def formula_module(space, *, globals_=None, filename='<compiled model>'):
    """One canonical body with original signature/cache status, no exec import."""
    module = ModuleType(space.class_identity)
    module.__file__ = filename
    module.__dict__.update(globals_ or {})
    for name, cell in space.cells.items():
        bindings = cell_bindings(cell)
        # Public reference aliases may differ from the class name carried by
        # physical result metadata. Checked decoding needs that same canonical
        # class, not a reconstructed Enum with equivalent integer members.
        for typ in bindings.values():
            if _native_category(typ) is not None:
                existing = module.__dict__.get(typ.__name__, typ)
                if existing is not typ:
                    from .contracts import ContractError
                    raise ContractError('Physical category name conflicts with a formula binding: ' + typ.__name__)
                module.__dict__[typ.__name__] = typ
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
        if cell.receiver_requirements:
            # Preserve the public interface for graph metadata without trying
            # to detach operations that require the original receiver. These
            # bodies stay on the generated class, never the context module.
            node = ast.parse(cell.formula_source).body[0]
            node.name = cell.name
            node.args.args.pop(0)
        else:
            node = FormulaEmitter.native(cell, arguments, bindings['return'])
        self.formula = SimpleNamespace(source=ast.unparse(node) + '\n')
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
        # The standard exporter freezes its policy domain with tuple(...).
        # Resolve the enclosed declared field/Cell, retaining that snapshot
        # operation; do not evaluate arbitrary domain expressions here.
        self._tuple = (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id == 'tuple' and len(node.args) == 1
                       and not node.keywords)
        if self._tuple:
            node = node.args[0]
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
        result = value() if self._call else value
        return tuple(result) if self._tuple else result


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
        references = {name: getattr(instance, name, None) for name, _ in space.reference_types
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
    from types import MappingProxyType
    from .modelx_frontend_native_mode import from_modelx
    space = package_plan.spaces[package_plan.model + '.' + projection]
    declared_rows = options.get('row_references') or {}
    # Original Space formulas can assign references that do not exist on the
    # template. An explicit resolved row declaration supplies that receiver
    # context without changing formula semantics or inventing a table reader.
    # Other receiver operations remain excluded.
    contextual = {name: replace(cell, receiver_requirements=tuple(reason
        for reason in cell.receiver_requirements
        if reason not in {'original receiver member: ' + reference for reference in declared_rows}))
        for name, cell in space.cells.items()}
    # Portfolio traversal and other explicit receiver operations stay compiled
    # on the original public class. They are not free functions in a per-policy
    # calculation context, nor another model facade to reconstruct here.
    retained = {name: cell for name, cell in contextual.items()
                if not cell.receiver_requirements}
    excluded = {name: list(cell.receiver_requirements) for name, cell in contextual.items()
                if cell.receiver_requirements}
    if any(set(cell.dependencies) & excluded.keys() for cell in retained.values()):
        from .contracts import ContractError
        raise ContractError('Native context needs an original-receiver dependency bridge')
    model = resolved_resource_model(package_plan, exported)
    if declared_rows:
        records = options['policy_records']
        model.spaces[tuple(projection.split('.'))].refs.update(
            {name: records.iloc[0][column] for name, column in declared_rows.items()})
    instance = exported
    for part in projection.split('.'):
        instance = getattr(instance, part)
    # Resource-name discovery must inspect the same receiver-resolved bodies
    # that emission consumes. Filtering the old descriptions left ``self``
    # qualification on row-bound bodies and hid their scalar/library references.
    model.spaces[tuple(projection.split('.'))].cells = {
        name: _ResolvedCell(cell, instance) for name, cell in retained.items()}
    context_space = replace(space, cells=MappingProxyType(retained))
    if options.get('native_typed_inputs') and not options.get('signatures'):
        # Typed input expansion consumes the same finalized public signatures.
        # This is not a second trace or inference pass in the resource adapter.
        options['signatures'] = {
            name: {'return': CTYPES[cell.public_result].__name__,
                   'parameters': {arg: CTYPES[typ].__name__
                                  for arg, typ in cell.public_arguments}}
            for name, cell in retained.items()}
    spec = from_modelx(model, destination, projection=projection,
                      emission_space=context_space, **options)
    spec.model['original_receiver_cells'] = excluded
    return replace(spec, cache_domains=cache_domains(context_space),
                   original_cache_kinds={name: cell.cache_kind
                                         for name, cell in retained.items()})
