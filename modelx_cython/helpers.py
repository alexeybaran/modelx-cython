"""Bind explicitly registered helpers without freezing exported References.

Providers supply a .pyx/.pxd pair. Their Cython functions receive the current
Python callable first and must preserve its behavior when it is replaced.
"""
import hashlib
from importlib.resources import files
import keyword

import libcst as cst


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


def helper_capability(value, name, grouped=False):
    capability = value.__dict__.get(f'__modelx_cython_{name}__')
    if capability is None:
        return None
    if grouped:
        valid = (isinstance(capability, tuple) and len(capability) in (2, 3)
                 and all(valid_identifier(part) for part in capability))
    else:
        valid = valid_identifier(capability)
    if not valid:
        raise ValueError(f'invalid helper capability {name!r} on {value.__qualname__}')
    return capability


class HelperCalls(cst.CSTTransformer):
    def __init__(self, refs, used):
        self.refs = refs
        self.used = used
        self.operations = set()

    def leave_Call(self, original, updated):
        func = updated.func
        if not (isinstance(func, cst.Attribute)
                and isinstance(func.value, cst.Name) and func.value.value == 'self'):
            return updated
        ref = self.refs.get(func.attr.value)
        contract = ref.helper if ref is not None else None
        if not contract or any(arg.keyword or arg.star for arg in updated.args):
            return updated
        provider, operation = contract
        self.used.add(provider)
        self.operations.add(provider + '.' + operation)
        args = tuple(arg.with_changes(value=arg.value.with_changes(
                     lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),)))
                     if isinstance(arg.value, cst.GeneratorExp) and not arg.value.lpar else arg
                     for arg in updated.args)
        return updated.with_changes(
            func=cst.Attribute(cst.Name(module_name(provider)), cst.Name(operation)),
            args=(cst.Arg(func), *args))


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
