"""Bind explicitly registered helpers without freezing exported References.

Providers supply a .pyx/.pxd pair. Their Cython functions receive the current
Python callable first and must preserve its behavior when it is replaced.
"""
import hashlib
from importlib.resources import files
import keyword
import inspect

import libcst as cst

from .scopes import FormulaTransformer

# One registry for tracing, merged References and validation. Tuple lengths
# describe provider ABI operations, not model or formula source patterns.
CAPABILITIES = {
    'reduction': ('reduction', None),
    'iteration': ('iteration_helper', (4,)),
    'iteration_f64': ('iteration_f64_helper', (4,)),
    'iteration_str': ('iteration_str_helper', (4,)),
    'scalar_result': ('scalar_result', None),
    'call_layout': ('call_layout', None),
    'numeric': ('numeric_helper', None),
    'float_numeric': ('float_numeric_helper', None),
    'keyed_float_numeric': ('keyed_float_numeric_helper', (5,)),
    'clamped': ('clamped_helper', None),
    'clamped_context': ('clamped_context_helper', None),
    'optional': ('optional_helper', None),
    'grouped': ('grouped_helper', (2, 3)),
    'bounded': ('bounded_helper', None),
    'bound_bounded': ('bound_bounded_helper', None),
    'bound_grouped': ('bound_grouped_helper', (3,)),
    'bound_numeric': ('bound_numeric_helper', (3,)),
}
REDUCTIONS = frozenset(('f64_sum', 'bool_any', 'bool_all', 'f64_max'))
SCALAR_RESULTS = frozenset(('float64', 'int64', 'bool', 'unicode', 'object'))


def module_name(provider):
    return '_mx_helper_' + hashlib.sha256(provider.encode()).hexdigest()[:12]


def valid_identifier(name):
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


def registered_helper(value):
    contract = value.__dict__.get('__modelx_cython_helper__')
    if contract is None:
        return None
    if not (isinstance(contract, tuple) and len(contract) == 2
            and all(isinstance(item, str) for item in contract)):
        raise ValueError('helper registration must be (provider, operation)')
    provider, operation = contract
    if '.' not in provider or not all(valid_identifier(part) for part in (*provider.split('.'), operation)):
        raise ValueError('helper registration must name a packaged provider and an operation')
    return contract


def helper_capability(value, name):
    capability = value.__dict__.get(f'__modelx_cython_{name}__')
    if capability is None:
        return None
    lengths = CAPABILITIES[name][1]
    if lengths:
        valid = (isinstance(capability, tuple) and len(capability) in lengths
                 and all(valid_identifier(part) for part in capability))
    else:
        valid = valid_identifier(capability)
    if (not valid
            or name == 'reduction' and capability not in REDUCTIONS
            or name == 'scalar_result' and capability not in SCALAR_RESULTS):
        raise ValueError(f'invalid helper capability {name!r} on {value.__qualname__}')
    if name == 'call_layout' and capability != 'keywords':
        raise ValueError(f'invalid helper capability {name!r} on {value.__qualname__}')
    return capability


def helper_parameters(value):
    """Read declared call parameters, never formula bodies or sampled values."""
    if helper_capability(value, 'call_layout') != 'keywords':
        return None
    parameters = inspect.signature(value).parameters.values()
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters):
        return None
    return tuple((p.name, p.default is not p.empty, p.kind is p.KEYWORD_ONLY)
                 for p in parameters)


class HelperCalls(FormulaTransformer):
    def __init__(self, refs, used):
        super().__init__()
        self.refs = refs
        self.used = used
        self.operations = set()

    def leave_Call(self, original, updated):
        func = updated.func
        ref = _declared_reference(func, self.refs)
        contract = ref.helper if ref is not None else None
        layout = getattr(ref, 'call_layout', None)
        if not contract or any(arg.star for arg in updated.args):
            return updated
        if layout != 'keywords' and any(arg.keyword for arg in updated.args):
            return updated
        parameters = getattr(ref, 'helper_parameters', None)
        if layout == 'keywords' and parameters is not None:
            positional = sum(arg.keyword is None for arg in updated.args)
            slots = [name for name, _, keyword_only in parameters if not keyword_only]
            named = [arg.keyword.value for arg in updated.args if arg.keyword]
            supplied = set(slots[:positional]) | set(named)
            if (positional > len(slots)
                    or len(set(named)) != len(named)
                    or set(named) & set(slots[:positional])
                    or set(named) - {name for name, _, _ in parameters}
                    or any(not optional and name not in supplied for name, optional, _ in parameters)):
                return updated
        provider, operation = contract
        self.used.add(provider)
        self.operations.add(provider + '.' + operation)
        args = tuple(arg.with_changes(value=arg.value.with_changes(
                     lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),)))
                     if isinstance(arg.value, cst.GeneratorExp) and not arg.value.lpar else arg
                     for arg in updated.args)
        prefix = (cst.Arg(func),)
        if layout == 'keywords':
            names = [arg.keyword.value for arg in updated.args if arg.keyword]
            prefix += (cst.Arg(cst.Tuple(tuple(cst.Element(cst.SimpleString(repr(name)))
                                               for name in names))),)
            if parameters is not None:
                positional = sum(arg.keyword is None for arg in args)
                slots = [name for name, _, keyword_only in parameters if not keyword_only]
                supplied = set(slots[:positional]) | set(names)
                # Native optional C arguments cannot skip a preceding slot.
                # A provider-owned sentinel carries omission without evaluating
                # a Python default or changing replacement call layout.
                args = (*args, *(cst.Arg(
                    cst.Attribute(cst.Name(module_name(provider)), cst.Name('omitted_argument')),
                    keyword=cst.Name(name)) for name, optional, _ in parameters
                    if optional and name not in supplied))
        return updated.with_changes(
            func=cst.Attribute(cst.Name(module_name(provider)), cst.Name(operation)),
            args=(*prefix, *args))

    def rewrite(self, formula):
        """Bind registered symbols across lexical scopes, without body analysis.

        ScopeProvider identifies the actual receiver parameter. A lambda that
        shadows it, or a formula that rebinds it, retains its ordinary call.
        """
        binder = _ScopedHelperCalls(self.refs, self.used)
        wrapper = cst.MetadataWrapper(cst.Module(body=(formula,)))
        result = wrapper.visit(binder).body[0]
        self.operations.update(binder.operations)
        return result


class _ScopedHelperCalls(HelperCalls):
    METADATA_DEPENDENCIES = (cst.metadata.ScopeProvider,)

    def __init__(self, refs, used):
        super().__init__(refs, used)
        self.receiver = None

    def on_visit(self, node):
        if self.receiver is None and isinstance(node, cst.FunctionDef):
            parameters = (*node.params.posonly_params, *node.params.params)
            if parameters and parameters[0].name.value == 'self':
                self.receiver = parameters[0]
        return cst.CSTTransformer.on_visit(self, node)

    def on_leave(self, original_node, updated_node):
        return cst.CSTTransformer.on_leave(self, original_node, updated_node)

    def leave_Call(self, original, updated):
        function = original.func
        if _declared_reference(function, self.refs) is None:
            return updated
        while isinstance(function, cst.Attribute):
            function = function.value
        scope = self.get_metadata(cst.metadata.ScopeProvider, function, None)
        assignments = tuple(scope['self']) if scope is not None else ()
        if (len(assignments) != 1 or self.receiver is None
                or assignments[0].node is not self.receiver):
            return updated
        return super().leave_Call(original, updated)


def _declared_reference(function, refs):
    # Kept lazy to avoid the shared registry/resolver import cycle.
    from .helper_members import helper_reference
    return helper_reference(function, refs)


def install(providers, model_path):
    modules = []
    for provider in sorted(providers):
        package, _, name = provider.rpartition('.')
        if not package:
            raise ValueError('helper provider must belong to a package')
        for suffix in ('.pyx', '.pxd'):
            source = files(package).joinpath(name + suffix)
            target = model_path / (module_name(provider) + suffix)
            target.write_bytes(source.read_bytes())
        modules.append(model_path.relative_to(model_path.parent)
                       / (module_name(provider) + '.pyx'))
    return modules
