"""Bind explicitly registered helpers without freezing exported References.

Providers supply a .pyx/.pxd pair. Their Cython functions receive the current
Python callable first and must preserve its behavior when it is replaced.
"""
import hashlib
from importlib import resources
import keyword
import inspect

import libcst as cst

from .scopes import FormulaTransformer
from .consts import CY_MOD

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
    'clamped': ('clamped_helper', None),
    'clamped_context': ('clamped_context_helper', None),
    'optional': ('optional_helper', None),
    'grouped': ('grouped_helper', (2, 3)),
    'bounded': ('bounded_helper', None),
    'bound_bounded': ('bound_bounded_helper', None),
    'bound_grouped': ('bound_grouped_helper', (3,)),
    'bound_numeric': ('bound_numeric_helper', (3,)),
    'bound_numeric_namespace': ('bound_numeric_namespace_helper', None),
}
REDUCTIONS = frozenset(('f64_sum', 'ordinary_sum', 'bool_any', 'bool_all', 'f64_max'))
SCALAR_RESULTS = frozenset(('float64', 'int64', 'bool', 'unicode', 'object'))


def module_name(provider):
    return '_mx_helper_' + hashlib.sha256(provider.encode()).hexdigest()[:12]


def valid_identifier(name):
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


def read_resource(package, name):
    return (resources.files(package).joinpath(name).read_bytes()
            if hasattr(resources, 'files') else resources.read_binary(package, name))


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


def helper_result_type(value):
    """Read an owner type and optional parameter that must retain its empty default."""
    identity = vars(value).get('__modelx_cython_result_type__')
    if identity is None:
        return None
    conditional = type(identity) is tuple
    if conditional and len(identity) == 2:
        owner, parameter = identity
        declared = inspect.signature(value).parameters.get(parameter) if valid_identifier(parameter) else None
    else:
        owner, parameter, declared = identity, None, None
    if (not isinstance(owner, str) or '.' not in owner
            or not all(valid_identifier(part) for part in owner.split('.'))
            or conditional and (declared is None or type(declared.default) is not tuple
                                or declared.default or helper_parameters(value) is None)):
        raise ValueError(f'invalid native helper result type on {value.__qualname__}')
    return owner, parameter


class HelperCalls(FormulaTransformer):
    def __init__(self, refs, used, cells=(), current=None):
        super().__init__()
        self.refs = refs
        self.used = used
        self.operations = set()
        self.cells = cells
        self.current = current          # the formula's cells (argument kinds), when known
        self.math_native = 0            # W1.4: module math members bound at build time
        self.math_untyped = []

    # ---------------------------------------------------------------- math branch (W1.4)
    def module_math(self, original, updated):
        """``self.<ref>.<fn>(x)`` on a module Reference: the provider gets the Reference.

        Returns the new call, ``updated`` itself to keep the Python call (listed in the
        admission report), or None when the call is no module math member.
        """
        from .helper_members import MATH_PROVIDER, argument_kind
        func = updated.func
        if not (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Attribute)
                and isinstance(func.value.value, cst.Name) and func.value.value.value == 'self'):
            return None
        members = getattr(self.refs.get(func.value.attr.value), 'helper_members', {})
        if not any(getattr(member, 'module_math', None) or hasattr(member, 'module_constant')
                   for member in members.values()):
            return None
        name = getattr(members.get(func.attr.value), 'module_math', None)
        if not name:
            self.decline(original, f'math.{func.attr.value} has no native form; runs as a Python call')
            return updated
        if len(updated.args) != 1 or updated.args[0].keyword or updated.args[0].star:
            self.decline(original, f'math.{name} with {len(updated.args)} arguments or keywords; '
                         'runs as a Python call')
            return updated
        operation = 'm_' + name
        if name in ('floor', 'ceil'):
            kind = argument_kind(updated.args[0].value, self.current, self.cells, self.refs)
            if kind in ('double', 'longlong'):
                operation += '_ll'
        self.used.add(MATH_PROVIDER)
        self.operations.add(MATH_PROVIDER + '.' + operation)
        self.math_native += 1
        return updated.with_changes(
            func=cst.Attribute(cst.Name(module_name(MATH_PROVIDER)), cst.Name(operation)),
            args=(cst.Arg(func.value), cst.Arg(updated.args[0].value)))

    def module_constant(self, updated):
        """``self.<ref>.inf`` (``nan``, ``pi``, ``e``, ``tau``) of the math module."""
        from .helper_members import MATH_PROVIDER
        if not (isinstance(updated.value, cst.Attribute) and isinstance(updated.value.value, cst.Name)
                and updated.value.value.value == 'self'):
            return None
        members = getattr(self.refs.get(updated.value.attr.value), 'helper_members', {})
        index = getattr(members.get(updated.attr.value), 'module_constant', None)
        if index is None:
            return None
        self.used.add(MATH_PROVIDER)
        self.operations.add(MATH_PROVIDER + '.m_constant')
        self.math_native += 1
        return cst.Call(cst.Attribute(cst.Name(module_name(MATH_PROVIDER)), cst.Name('m_constant')),
                        (cst.Arg(updated.value), cst.Arg(cst.Integer(str(index)))),
                        lpar=updated.lpar, rpar=updated.rpar)

    def decline(self, node, reason):
        from .reductions import untyped_entry
        self.math_untyped.append(untyped_entry(getattr(self, 'line_of', None), node, 'math', reason))

    def leave_Call(self, original, updated):
        math = self.module_math(original, updated)
        if math is not None:
            return math
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
        binder = _ScopedHelperCalls(self.refs, self.used, self.cells, self.current)
        wrapper = cst.MetadataWrapper(cst.Module(body=(formula,)))
        result = wrapper.visit(binder).body[0]
        self.operations.update(binder.operations)
        self.math_native += binder.math_native
        self.math_untyped.extend(binder.math_untyped)
        return result


class _ScopedHelperCalls(HelperCalls):
    METADATA_DEPENDENCIES = (cst.metadata.ScopeProvider, cst.metadata.ParentNodeProvider,
                             cst.metadata.ExpressionContextProvider, cst.metadata.PositionProvider)

    def __init__(self, refs, used, cells, current=None):
        super().__init__(refs, used, cells, current)
        self.receiver = None
        self.first_line = None

    def line_of(self, node):
        position = self.get_metadata(cst.metadata.PositionProvider, node, None)
        if position is None or self.first_line is None:
            return 0
        return position.start.line - self.first_line + 1

    def on_visit(self, node):
        if self.receiver is None and isinstance(node, cst.FunctionDef):
            parameters = (*node.params.posonly_params, *node.params.params)
            if parameters and parameters[0].name.value == 'self':
                self.receiver = parameters[0]
            position = self.get_metadata(cst.metadata.PositionProvider, node.name, None)
            self.first_line = position.start.line if position is not None else None
        return cst.CSTTransformer.on_visit(self, node)

    def on_leave(self, original_node, updated_node):
        return cst.CSTTransformer.on_leave(self, original_node, updated_node)

    def _owns_receiver(self, expression):
        while isinstance(expression, cst.Attribute):
            expression = expression.value
        if not isinstance(expression, cst.Name) or expression.value != 'self':
            return False
        scope = self.get_metadata(cst.metadata.ScopeProvider, expression, None)
        assignments = tuple(scope['self']) if scope is not None else ()
        return (len(assignments) == 1 and self.receiver is not None
                and assignments[0].node is self.receiver)

    def leave_Attribute(self, original, updated):
        if (isinstance(original.value, cst.Attribute) and self._owns_receiver(original)
                and self.get_metadata(cst.metadata.ExpressionContextProvider, original, None)
                is cst.metadata.ExpressionContext.LOAD):
            parent = self.get_metadata(cst.metadata.ParentNodeProvider, original)
            if not (isinstance(parent, cst.Call) and parent.func is original):
                constant = self.module_constant(updated)
                if constant is not None:
                    return constant
        if (original.attr.value not in self.cells
                or not isinstance(original.value, cst.Name)
                or not self._owns_receiver(original)
                or self.get_metadata(cst.metadata.ExpressionContextProvider, original, None)
                   is not cst.metadata.ExpressionContext.LOAD):
            return updated
        parent = self.get_metadata(cst.metadata.ParentNodeProvider, original)
        if isinstance(parent, cst.Call) and parent.func is original:
            return updated
        # A Cell used as a value needs its ordinary bound Python callable.
        # Cython must not coerce a native method pointer into an object tuple.
        # Direct calls retain the existing typed/override-aware Cell path.
        receiver = cst.Call(cst.Attribute(cst.Name(CY_MOD), cst.Name('cast')),
                            (cst.Arg(cst.Name('object')), cst.Arg(updated.value)))
        return updated.with_changes(value=receiver)

    def leave_Call(self, original, updated):
        function = original.func
        if not self._owns_receiver(function):
            return updated
        if _declared_reference(function, self.refs) is None:
            self.module_math(original, updated)       # lists math members with no native form
            return updated
        return super().leave_Call(original, updated)


def _declared_reference(function, refs):
    # Kept lazy to avoid the shared registry/resolver import cycle.
    from .helper_members import helper_reference
    return helper_reference(function, refs)


def install(providers, model_path, object_providers=None):
    modules = []
    expected = {}
    object_providers = object_providers or {}
    for provider in sorted(providers):
        package, _, name = provider.rpartition('.')
        if not package:
            raise ValueError('helper provider must belong to a package')
        alias = module_name(provider)
        sources = {suffix: read_resource(package, name + suffix) for suffix in ('.pyx', '.pxd')}
        manifest = object_providers.get(provider)
        if manifest:
            digest = hashlib.sha256()
            for payload in (read_resource(package, name + '.provider.json'), *sources.values()):
                digest.update(len(payload).to_bytes(8, 'little'))
                digest.update(payload)
            fingerprint = digest.hexdigest()
            expected[alias] = ((manifest.protocol, manifest.provider, manifest.abi), fingerprint)
            sources['.pyx'] += f'\n\n__modelx_cython_provider_build__ = {fingerprint!r}\n'.encode()
        for suffix, payload in sources.items():
            (model_path / (alias + suffix)).write_bytes(payload)
        modules.append(model_path.relative_to(model_path.parent)
                       / (alias + '.pyx'))
    if expected:
        guard = model_path / '_mx_provider_guard.py'
        guard.write_text(f'''from importlib import import_module

for _alias, (_contract, _fingerprint) in {expected!r}.items():
    _message = (f'Native provider {{_contract[1]}} requires ABI {{_contract[2]}} '
                f'and build fingerprint {{_fingerprint}}')
    try:
        _provider = import_module('.' + _alias, __package__)
    except ImportError as _error:
        raise ImportError(_message) from _error
    if (getattr(_provider, '__modelx_cython_provider__', None) != _contract
            or getattr(_provider, '__modelx_cython_provider_build__', None) != _fingerprint):
        raise ImportError(_message)
''', encoding='utf-8')
        init = model_path / '__init__.py'
        init.write_text('from . import _mx_provider_guard\n' + init.read_text(encoding='utf-8'),
                        encoding='utf-8')
    return modules
