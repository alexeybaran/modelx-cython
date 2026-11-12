"""Registered functions behind module References, using the existing registry.

These records contain capabilities, not frozen callable bindings. Emission must
still evaluate the current member callable and let the provider check identity.
The shared tracer/merged-Reference wiring is a separately owned integration.
"""
import math
import sys
from types import FunctionType, ModuleType, SimpleNamespace

import libcst as cst

from .helpers import CAPABILITIES, helper_capability, helper_parameters, registered_helper


_NATIVE_MATH = {'erf': 'float64', 'exp': 'float64', 'isinf': 'bool', 'log': 'float64', 'sqrt': 'float64'}


def module_helpers(value):
    """Snapshot declared member metadata without executing module descriptors."""
    if type(value) is not ModuleType:
        return {}
    result = {}
    if value is sys.modules.get('pandas'):
        return {'isna': SimpleNamespace(helper=('modelx_cython.runtime._native_math', 'isna'), helper_parameters=None,
                                        helper_arity=1,
                                        **{field: ('object' if suffix == 'scalar_result' else None)
                                           for suffix, (field, _) in CAPABILITIES.items()})}
    if value is math:
        # The standard math module: the compiler's own native math provider implements these
        # with identity checks against math.<name>, so a replaced function keeps its behaviour.
        return {name: SimpleNamespace(helper=('modelx_cython.runtime._native_math', name), helper_parameters=None,
                                      helper_arity=1,
                                      **{field: (result if suffix == 'scalar_result' else None)
                                         for suffix, (field, _) in CAPABILITIES.items()})
                for name, result in _NATIVE_MATH.items()}
    for name, function in vars(value).copy().items():
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
