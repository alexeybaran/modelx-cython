"""Detached Python miss bodies; all Cell calls return to the compiled memo.

No formula interpreter, second model, or Python-side result cache. Source is
bundled with the artifact. Each runtime gets its own globals/receiver bindings.
Object values retain Python identity; compiled entries own their result caches.
"""
import ast
import builtins
import dis
from enum import IntEnum
import hashlib
import inspect
import math
import numbers
import textwrap
import types
from functools import lru_cache
from pathlib import Path
from .contracts import ContractError


@lru_cache(maxsize=2048)
def _compiled_cell_entry(source):
    """Reuse immutable wrapper bytecode; each Cell still gets its own globals."""
    return compile(source, '<compiled Cell entry>', 'exec')


def _global_names(code):
    names = {i.argval for i in dis.get_instructions(code) if i.opname in ('LOAD_GLOBAL', 'LOAD_NAME')}
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names.update(_global_names(constant))
    return names


def referenced_names(fn, seen=None):
    """Lexical globals used by a formula and its ordinary same-module helpers."""
    seen = set() if seen is None else seen
    if id(fn) in seen: return set()
    seen.add(id(fn))
    names = _global_names(fn.__code__)
    for name in tuple(names):
        helper = fn.__globals__.get(name)
        if inspect.isfunction(helper) and helper.__module__ == fn.__module__:
            names.update(referenced_names(helper, seen))
    return names


def cell_dependencies(fn, cells, seen=None):
    """Lexical Cell references, including local library helpers/closures.

    This is name binding, not value/control-flow interpretation. Reflective Cell
    discovery is outside M1; only source-resolved Cell names enter the receiver.
    """
    seen = set() if seen is None else seen
    if id(fn) in seen:
        return set()
    seen.add(id(fn))
    names = _global_names(fn.__code__)
    result = names & cells.keys()
    for name in names - cells.keys():
        helper = fn.__globals__.get(name)
        if inspect.isfunction(helper) and helper.__module__ == fn.__module__:
            result.update(cell_dependencies(helper, cells, seen))
    return result


def bundle_source(cells, python_cells, inputs, enums, *, compiled_numeric=()):
    """Bundle executable source and imports, not live modelx/exported objects."""
    from modelx_cython.formula_emitter import formula_source, is_formula
    compiled_numeric = set(compiled_numeric)
    from .enum_binding import enum_origin
    chunks = ['# Detached raw formulas. Cell names are rebound by PythonScope.', 'from enum import IntEnum']
    for name, enum in sorted(enums.items()):
        origin = enum_origin(enum)
        if origin is None:
            chunks.append('class ' + name + '(IntEnum):\n' + '\n'.join(
                '    ' + key + ' = ' + repr(int(value)) for key, value in enum.__members__.items()))
        else:
            expression = "__import__(" + repr(origin['module']) + ", fromlist=['*'])"
            for part in origin['qualname'].split('.'):
                expression += '.' + part
            chunks.append(name + ' = ' + expression)
    done = set(enums) | set(inputs) | (set(cells) - set(python_cells))
    required = set()
    used_inputs = set()

    def add_function(name, fn):
        if fn.__closure__:
            raise ContractError(name + ': captured Python closures need an explicit detached binding')
        node = ast.parse(textwrap.dedent(formula_source(fn))).body[0]
        if not isinstance(node, ast.FunctionDef):
            raise ContractError(name + ': a plain Python function is required')
        if any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(node)):
            raise ContractError(name + ': mutation of formula globals is not supported')
        if name not in compiled_numeric:
            node.name = name
            node.decorator_list = []
            node.returns = None
            if name in cells:
                node.args.defaults = []  # The shared Cell entry always binds every argument.
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                arg.annotation = None
            chunks.append(ast.unparse(node))
        for used in sorted(_global_names(fn.__code__)):
            if used in inputs:
                used_inputs.add(used)
            if used in done or used in vars(builtins):
                continue
            if used in cells:
                if used in python_cells:
                    add(used, cells[used])
                continue
            if used in fn.__globals__:
                add(used, fn.__globals__[used], owner=fn.__module__)
            else:
                required.add(used)

    def add(name, value, owner=None):
        if name in done:
            return
        done.add(name)
        if isinstance(value, types.ModuleType):
            chunks.append(f'import {value.__name__} as {name}')
        elif is_formula(value) and (name in python_cells or value.__module__ == owner):
            add_function(name, value)
        elif (inspect.isfunction(value) or inspect.isclass(value)) and '<locals>' not in value.__qualname__:
            module = value.__module__
            # Import ordinary libraries, never the source model as a second execution graph.
            if module in {f.__module__ for f in cells.values()}:
                raise ContractError(name + ': non-enum model classes need an explicit detached binding')
            chunks.append(f'from {module} import {value.__name__} as {name}')
        else:
            try:
                literal = ast.literal_eval(repr(value))
            except (ValueError, SyntaxError):
                raise ContractError(name + ': unsupported captured global; bind it explicitly') from None
            chunks.append(name + ' = ' + repr(literal))

    for name in sorted(python_cells):
        add(name, cells[name])
    return '\n\n'.join(chunks) + '\n', sorted(required), sorted(used_inputs)


class CellCodec:
    """Shared Python/native boundary; owns callable wrappers, never Cell results."""
    def __init__(self, manifest, kernel, record):
        from .enum_binding import resolve_enum
        self.manifest, self.kernel = manifest, kernel
        self._original_receiver = None
        self._record, self._raw_record = None, record
        infos = list(manifest['inputs'].values())
        for row in manifest['cells'].values():
            infos += [row['result']] + [p['type'] for p in row['params']]
        self.namespace = {i['name']: resolve_enum(i) for i in infos if i['kind'] == 'enum'}
        self.entries = {}

    def entry(self, name):
        if name not in self.entries:
            self.entries[name] = self._proxy(name)
        return self.entries[name]

    def _proxy(self, name):
        if name in self.manifest.get("direct_object_entries", ()):
            return getattr(self.kernel, "_entry_" + name)
        """Let Python bind the real signature once, not interpret it on each call."""
        row = self.manifest['cells'][name]
        parameters, arguments = [], []
        for i, p in enumerate(row['params']):
            default = p['default']
            if default is not None and p['type']['kind'] == 'bool':
                default = repr(bool(ast.literal_eval(default)))
            parameters.append(p['name'] + ('=' + default if default is not None else ''))
            if p.get('positional_only') and (i + 1 == len(row['params']) or
                                             not row['params'][i + 1].get('positional_only')):
                parameters.append('/')
            arguments.append(f"_mx_scope.encode(_mx_row['params'][{i}]['type'], {p['name']})")
        args = '(' + ', '.join(arguments) + (',' if arguments else '') + ')'
        call = f'_mx_scope.kernel.call(_mx_name, {args})'
        if row['result']['kind'] in ('enum', 'table_key'):
            call = f"_mx_scope.decode(_mx_row['result'], {call})"
        source = 'def ' + name + '(' + ', '.join(parameters) + '):\n    return ' + call
        namespace = dict(_mx_scope=self, _mx_row=row, _mx_name=name)
        exec(_compiled_cell_entry(source), namespace)
        return namespace[name]

    def encode(self, info, value):
        kind = info['kind']
        if kind == 'object':
            expected = info.get('python_type', 'object')
            if expected == 'object':
                return value
            typ = type(None) if expected == 'NoneType' else getattr(builtins, expected)
            if not isinstance(value, typ):
                raise TypeError('Python object boundary requires ' + expected)
            return value
        if kind == 'table_key':
            if value == info['default'] and info['default'] is not None:
                return value
            return self.record[info['table']].code(value)
        if kind == 'float':
            if type(value) not in (float, int) and (isinstance(value, (bool,)) or not isinstance(value, numbers.Real)):
                raise TypeError('Python formula must return/pass a real scalar')
            value = float(value)
            if not math.isfinite(value):
                raise ValueError('Non-finite scalar at Python/native boundary')
            return value
        if kind == 'bool':
            if type(value) is not bool:
                raise TypeError('Python formula must return/pass a bool')
            return value
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError('Python formula must return/pass an integral scalar')
        value = int(value)
        if not -(1 << 63) <= value < (1 << 63):
            raise OverflowError('Python integer outside signed int64')
        if kind == 'enum' and value not in info['members'].values():
            raise ValueError('Python category outside declared domain')
        return value

    def decode(self, info, value):
        kind = info['kind']
        if kind == 'enum':
            return self.namespace[info['name']](value)
        if kind == 'bool':
            return bool(value)
        if kind == 'table_key':
            if value == info['default'] and info['default'] is not None:
                return value
            table = self.record[info['table']]
            # Decode labels without copying the table's numeric buffers.
            for label in table.labels:
                if table.code(label) == value:
                    return label
            raise KeyError(value)
        return value

    @property
    def record(self):
        # Packed records are immutable tuples. Materialize a name map only when
        # Python actually needs input lookup or a table-key decoder.
        if self._record is None:
            self._record = dict(zip(self.manifest['inputs'], self._raw_record))
        return self._record



class PythonScope(CellCodec):
    """One receiver/globals namespace per native context, no cache of its own."""
    def __init__(self, artifact, manifest, kernel, bindings=None, *, program=None):
        self.manifest, self.kernel = manifest, kernel
        self._original_receiver = None
        self.bindings = tuple(dict(row) for row in bindings) if bindings is not None else ()
        if program is None:
            path = Path(artifact) / manifest['package'] / '_python.py'
            source = path.read_text(encoding='utf-8')
            if hashlib.sha256(source.encode()).hexdigest() != manifest['python_source_sha256']:
                raise ContractError('Python calculator source hash mismatch')
            program = compile(source, str(path), 'exec')
        self.program = program
        required = set(manifest['python_references'])
        if required and not self.bindings:
            raise ContractError('Missing detached Python receiver bindings')
        for binding in self.bindings:
            if required - binding.keys():
                raise ContractError('Missing detached Python references: ' + repr(sorted(required - binding.keys())))
            if binding.keys() & (manifest['inputs'].keys() | manifest['cells'].keys()):
                raise ContractError('Python bindings cannot override inputs or Cells')
        self.namespace = {'__name__': manifest['package'] + '._python', '__builtins__': vars(builtins)}
        exec(self.program, self.namespace)
        from .enum_binding import resolve_enum
        for name, info in manifest.get('python_enums', {}).items():
            actual = self.namespace[name]
            if (not isinstance(actual, type) or not issubclass(actual, IntEnum)
                    or {k: int(v) for k, v in actual.__members__.items()} != info['members']
                    or ('module' in info and actual is not resolve_enum(info))):
                raise ContractError('Public Enum contract changed: ' + name)
        compiled_numeric = set(manifest.get('cython_numeric_fallback_cells', ()))
        compiled_objects = set(manifest.get('cython_object_cells', ()))
        self.calculators = {name: (getattr(kernel, '_gil_entry_' + name)
                                   if name in compiled_numeric else None
                                   if name in compiled_objects else self.namespace[name])
                            for name in manifest['python_cells']}
        self.calls = {name: 0 for name in self.calculators}
        self._record, self._raw_record = {}, ()
        self._plain_arguments = {name: all(p['type']['kind'] in ('float', 'int', 'object')
                                          for p in manifest['cells'][name]['params'])
                                 for name in self.calculators}
        self.diagnostics = manifest['diagnostics']
        # A direct ItemSpace may use only a few Cells, yet its Python receiver
        # used to build a callable for every Cell in the compiled model. Keep
        # the formulas and globals local to this receiver; create each callable
        # only when a Python formula or an external call actually needs it.
        self.entries = {}
        for row in manifest['cells'].values():
            if row.get('calculator') == 'python':
                for name in row['dependencies']:
                    self.namespace[name] = self.entry(name)
        for name in self.calculators:
            self.namespace[name] = self.entry(name)

    def select(self, record, policy):
        self._record, self._raw_record = None, record
        binding = self.bindings[policy] if self.bindings else {}
        self.namespace.update(binding)
        for name in self.manifest['python_inputs']:
            self.namespace[name] = self.decode(self.manifest['inputs'][name], self.record[name])

    def original_reference(self, name):
        """Ordinary public fields remain live across checked callback boundaries."""
        receiver = getattr(self, '_original_receiver', None)
        return getattr(receiver, name) if receiver is not None else self.namespace[name]

    def calculate(self, name, arguments):
        row = self.manifest['cells'][name]
        args = arguments if self._plain_arguments[name] else tuple(self.decode(p['type'], value) for p, value in zip(row['params'], arguments))
        if self.diagnostics:
            self.calls[name] += 1
        value = self.calculators[name](*args)
        if getattr(self, '_original_receiver', None) is not None:
            self.kernel._mx_synchronize_inputs()
        if value is None and row.get('allow_none') is False:
            raise ValueError('None is not allowed for Cell ' + name)
        return self.encode(row['result'], value)
