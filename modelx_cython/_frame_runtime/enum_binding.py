"""Bind public Enum results to their original importable Python classes."""
import importlib
import inspect
from enum import IntEnum

from .contracts import ContractError


def enum_origin(typ):
    """Return an import path only when it resolves to this exact class."""
    module = getattr(typ, '__module__', None)
    qualname = getattr(typ, '__qualname__', '')
    if not module or not qualname or '<locals>' in qualname:
        return None
    try:
        result = importlib.import_module(module)
        for part in qualname.split('.'):
            result = getattr(result, part)
    except (ImportError, AttributeError):
        return None
    return {'module': module, 'qualname': qualname} if result is typ else None


def resolve_enum(info):
    """Load the canonical class and reject a changed public category domain."""
    if 'module' not in info:
        return IntEnum(info['name'], info['members'])
    try:
        result = importlib.import_module(info['module'])
        for part in info['qualname'].split('.'):
            result = getattr(result, part)
    except (ImportError, AttributeError) as exc:
        raise ContractError('Cannot load public Enum ' + info['name']) from exc
    if (not inspect.isclass(result) or not issubclass(result, IntEnum)
            or {k: int(v) for k, v in result.__members__.items()} != info['members']):
        raise ContractError('Public Enum contract changed: ' + info['name'])
    return result
