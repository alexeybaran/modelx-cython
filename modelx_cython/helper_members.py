"""Registered functions behind module References, using the existing registry.

These records contain capabilities, not frozen callable bindings. Emission must
still evaluate the current member callable and let the provider check identity.
The shared tracer/merged-Reference wiring is a separately owned integration.
"""
from types import FunctionType, ModuleType, SimpleNamespace

import libcst as cst

from .helpers import CAPABILITIES, helper_capability, registered_helper


def module_helpers(value):
    """Snapshot declared member metadata without executing module descriptors."""
    if type(value) is not ModuleType:
        return {}
    result = {}
    for name, function in vars(value).copy().items():
        if type(function) is not FunctionType:
            continue
        helper = registered_helper(function)
        if helper is None:
            continue
        capabilities = {field: helper_capability(function, suffix)
                        for suffix, (field, _) in CAPABILITIES.items()}
        result[name] = SimpleNamespace(helper=helper, **capabilities)
    return result


def helper_reference(function, refs):
    """Resolve direct References or one declared module member in this scope."""
    if not isinstance(function, cst.Attribute):
        return None
    receiver = function.value
    if isinstance(receiver, cst.Name) and receiver.value == 'self':
        return refs.get(function.attr.value)
    if (isinstance(receiver, cst.Attribute) and isinstance(receiver.value, cst.Name)
            and receiver.value.value == 'self'):
        namespace = refs.get(receiver.attr.value)
        return getattr(namespace, 'helper_members', {}).get(function.attr.value)
    return None
