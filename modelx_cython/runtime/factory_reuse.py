"""Runtime binding guard for the optional scalar factory evaluator."""
import builtins
from types import BuiltinFunctionType

_TYPES = {'int': (0).__class__, 'bool': True.__class__, 'float': (0.).__class__,
          'str': ''.__class__, 'dict': {}.__class__, 'type': (0).__class__.__class__}
try:
    _TYPES['int']('')
except Exception as error:
    _TYPES['ValueError'] = error.__class__


def canonical_factory_builtins():
    namespace = builtins.__dict__
    for name, cls in _TYPES.items():
        if namespace.get(name) is not cls:
            return False
    for name in ('abs', 'max', 'min', 'round', 'isinstance', 'setattr'):
        fn = namespace.get(name)
        if _TYPES['type'](fn) is not BuiltinFunctionType or fn.__self__ is not builtins or fn.__name__ != name:
            return False
    return True
