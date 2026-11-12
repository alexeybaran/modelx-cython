"""Registered functions behind module References, using the existing registry.

These records contain capabilities, not frozen callable bindings. Emission must
still evaluate the current member callable and let the provider check identity.
The shared tracer/merged-Reference wiring is a separately owned integration.
"""
from math import erf, exp, isinf, log, sqrt
from types import BuiltinFunctionType, FunctionType, ModuleType, SimpleNamespace

import libcst as cst

from .helpers import CAPABILITIES, helper_capability, helper_parameters, registered_helper


_SCALAR_MATH = {erf: 'float64', exp: 'float64', isinf: 'bool', log: 'float64', sqrt: 'float64'}


def module_helpers(value):
    """Snapshot declared member metadata without executing module descriptors."""
    if type(value) is not ModuleType:
        return {}
    result = {}
    for name, function in vars(value).copy().items():
        if type(function) is BuiltinFunctionType and function in _SCALAR_MATH:
            result[name] = SimpleNamespace(
                helper=('modelx_cython.runtime._native_math', function.__name__),
                helper_parameters=None, scalar_result=_SCALAR_MATH[function], reduction=None,
            )
            continue
        if type(function) is not FunctionType:
            continue
        helper = registered_helper(function)
        if helper is None:
            continue
        capabilities = {field: helper_capability(function, suffix)
                        for suffix, (field, _) in CAPABILITIES.items()}
        result[name] = SimpleNamespace(helper=helper, helper_parameters=helper_parameters(function),
                                       **capabilities)
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
