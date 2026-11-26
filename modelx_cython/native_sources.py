"""Native support sources (a .pyx/.pxd pair of this package) copied into a compiled model package.

The numeric caches are such a pair: a Space class whose Cells use one cimports it under a private
alias, and the build compiles the copy beside the model's modules. A provider without a .pyx is a header of
inline code only (the reduction state): its .pxd is copied and nothing is compiled or imported at run time.
"""
import hashlib
from importlib import resources


def module_name(provider):
    """The private alias of ``provider`` (a dotted module name) inside a compiled model package."""
    return '_mx_helper_' + hashlib.sha256(provider.encode()).hexdigest()[:12]


def read_resource(package, name):
    return (resources.files(package).joinpath(name).read_bytes()
            if hasattr(resources, 'files') else resources.read_binary(package, name))


def install(providers, model_path):
    """Copy each provider's .pyx and .pxd into ``model_path`` under its alias; the .pyx paths to compile."""
    modules = []
    for provider in sorted(providers):
        package, _, name = provider.rpartition('.')
        if not package:
            raise ValueError('a native source must belong to a package')
        alias = module_name(provider)
        header_only = not resources.files(package).joinpath(name + '.pyx').is_file()
        for suffix in ('.pxd',) if header_only else ('.pyx', '.pxd'):
            (model_path / (alias + suffix)).write_bytes(read_resource(package, name + suffix))
        if not header_only:
            modules.append(model_path.relative_to(model_path.parent) / (alias + '.pyx'))
    return modules
