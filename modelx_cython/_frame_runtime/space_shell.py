"""Attach the native frame to the original compiler's public Space shell.

Discovery, specification precedence, public declarations, construction and
virtual dispatch belong to the original compiler. This adapter consumes its
ModuleInfo and emits a private kernel. Exact and override-aware entries share
its cache owner; derived formulas retain original virtual dispatch.
Unsupported graphs retain generated Cython, not an interpreted replacement.
"""
import ast
import copy
import hashlib
import json
import shutil
from types import ModuleType
from pathlib import Path

from .contracts import KernelSpec
from modelx_cython.formula_emitter import FormulaEmitter, SourceFormula

TYPES = {'double': float, 'long long': int, 'bint': bool, 'object': object, 'str': str}
BUILTINS = {'float', 'int', 'bool', 'abs', 'min', 'max', 'range',
            'ValueError', 'IndexError', 'ArithmeticError', 'RuntimeError', 'round'}


def _input_type(reference):
    """Preserve the public field type; carry Space references as owned objects.

    Calls through such a reference stay on a checked Cython edge. Its target's
    original public wrapper remains responsible for dispatch and cache ownership.
    This does not reconstruct a second Space or snapshot its current results.
    """
    if reference.mx_class:
        return object
    return TYPES.get(reference.get_type_expr(c_style=True))


def _reason(fn, info):
    if info.is_locked:
        return 'locked Space'
    a = fn.args
    if a.kw_defaults or a.kwonlyargs or a.posonlyargs or a.vararg or a.kwarg:
        return 'non-simple signature'
    try:
        for default in a.defaults:
            ast.literal_eval(default)
    except (ValueError, TypeError):
        return 'non-literal signature default'
    if any(arg.arg == 'state' or arg.arg.startswith('_mx_') for arg in a.args[1:]):
        return 'name overlaps the current native receiver namespace'
    # Object ownership and checked Python operations are supported Cython
    # representations, not reasons to reject the dependency graph. Deferred
    # closures and global state mutation still need a separate receiver bridge.
    forbidden = (ast.Lambda, ast.Yield, ast.YieldFrom, ast.GeneratorExp,
                 ast.Global, ast.Nonlocal)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and (node.id == 'state' or node.id.startswith('_mx_')):
            return 'name overlaps the current native receiver namespace'
        if isinstance(node, forbidden) or isinstance(node, ast.FunctionDef) and node is not fn:
            return 'effectful or object formula'
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self' and (
                node.attr.startswith('_') or not isinstance(node.ctx, ast.Load)):
            return 'receiver mutation or private receiver reference'
        if isinstance(node, ast.Raise) and (node.cause is not None or
                not isinstance(node.exc, ast.Call) or
                not isinstance(node.exc.func, ast.Name) or
                any(not isinstance(a, ast.Constant) for a in node.exc.args)):
            return 'effectful exception construction'
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in BUILTINS and node.func.id not in {'dict', 'list', 'tuple', 'set', 'str', 'len'}:
            return 'unresolved or effectful call'
    return None


def plan(unit, *, package_plan=None):
    """Use final original types/specifications; do not trace or rediscover Spaces."""
    if package_plan is None:
        from modelx_cython.compilation_plan import build_package_plan
        package_plan = build_package_plan((unit,))
    plans, inventory = {}, []
    for cls_name, info in unit.module_info.classes.items():
        shared = package_plan.by_class[info.fqname]
        methods = {'_f_' + n: c.formula_node for n, c in shared.cells.items()}
        candidates, reasons, dependencies, reads = {}, {}, {}, {}
        for name, cell in info.cells.items():
            if name not in shared.cells:
                continue
            cell_plan = shared.cells[name]
            fn = methods.get('_f_' + name)
            reason = None
            if fn is None:
                reason = 'missing original formula body'
            elif cell_plan.public_result not in TYPES:
                reason = 'unsupported original public result type'
            else:
                reason = _reason(fn, info)
            if not reason and set(info.cells) & BUILTINS:
                reason = 'Cell/builtin name collision in free-function namespace'
            if not reason and cell_plan.cache_kind == 'dense':
                sizes = info.cells_arg_sizes[tuple(cell.params)]
                slots = 1
                for size in sizes:
                    slots *= size
                if any(size <= 0 or size > 1000000 for size in sizes) or slots > 20000000:
                    reason = 'dense layout outside current frame allocation contract'
            if not reason:
                dependencies[name] = set(cell_plan.dependencies)
                reads[name] = set(cell_plan.reference_reads)
                if any(k not in info.refs or _input_type(info.refs[k]) is None
                       for k in reads[name]):
                    reason = 'unresolved input representation'
                else:
                    candidates[name] = fn
            if reason:
                reasons[name] = reason
        # An unsupported dependency does not exclude its compiled callers.
        # Bind the actual original public Cell on a checked Cython edge. Its
        # original receiver/cache remains authoritative; no second body or memo
        # is emitted for that Cell. Live lookup observes callbacks/rebinding.
        bridges = {name: dependencies[name] - candidates.keys() for name in candidates}
        for name, missing in tuple(bridges.items()):
            if any(not dep.isidentifier() or dep.startswith('_') for dep in missing):
                reasons[name] = 'unsupported checked Cell binding name'
                del candidates[name]
        bridges = {name: dependencies[name] - candidates.keys() for name in candidates}
        for name in shared.cells:
            cell = info.cells[name]
            inventory.append({'space': info.fqname, 'cell': name,
                              'execution': 'frame_exact' if name in candidates else 'original_cython',
                              # "original_cython" is the retained receipt label,
                              # not a choice of compiler versus native execution.
                              'formula_compiler': 'Cython',
                              'derived_cache': 'frame_virtual' if name in candidates else 'original_class',
                              'receiver_representation': ('native_frame' if name in candidates
                                                          else 'original_space'),
                              'reason': reasons.get(name),
                              'checked_cell_dependencies': sorted(bridges.get(name, ())),
                              'original_module_sha256': hashlib.sha256(unit.source.encode()).hexdigest(),
                              'formula_sha256': (hashlib.sha256(ast.unparse(methods['_f_' + name]).encode()).hexdigest()
                                                 if '_f_' + name in methods else None),
                              'public_return_type': cell.get_rettype_expr(c_style=True),
                              'public_arguments': {p: cell.get_argtype_expr(p, c_style=True) for p in cell.params}})
        if candidates:
            refs = set().union(*(reads[n] for n in candidates))
            external = set().union(*bridges.values())
            inputs = {n: _input_type(info.refs[n]) for n in sorted(refs)}
            inputs.update((n, object) for n in sorted(external))
            plans[cls_name] = {'methods': candidates, 'info': info, 'emission_plan': shared,
                               'checked_cell_dependencies': bridges, 'inputs': inputs}
    return plans, inventory


def emit_private_kernels(plans, package_path, package_name, *, formula_transform):
    """One build graph, with private kernel modules beneath the exported package."""
    from .compiler import build
    modules = []
    for cls_name, p in plans.items():
        domains = {}
        info = p['info']
        name = '_mx_frame_' + cls_name.removeprefix('_c_')
        folder = package_path / name
        folder.mkdir()
        (folder / '__init__.py').write_text('', encoding='utf-8')
        module_name = package_name + '.' + name
        module = ModuleType(module_name)
        module.__file__ = str(package_path / '_mx_classes.py')
        # The original helper pass owns discovery, preparation and guards. Reuse
        # that transformation and its existing private fields; bind them into
        # this context rather than preparing another table/provider runtime.
        helper = getattr(info, 'helper_info', None)
        prepared, aliases, nodes = {}, {}, {}
        input_expressions = {n: 'self.' + n for n in p['inputs']}
        helper_modules = {}
        import libcst as cst
        from modelx_cython.experimental_helpers import code
        types = dict(helper.declarations()) if helper else {}
        enum_bindings = getattr(info, 'enum_info', None)
        if enum_bindings:
            types.update(enum_bindings.declarations())
        native_calls = set()
        for cell_name in p['methods']:
            cell = p['emission_plan'].cells[cell_name]
            transformed = formula_transform(cst.parse_statement(cell.formula_source),
                                            info, cell.original_info)
            node = ast.parse(code(transformed)).body[0]
            nodes[cell_name] = node
            for part in ast.walk(node):
                if isinstance(part, ast.Attribute) and isinstance(part.value, ast.Name) and part.value.id == 'self' and part.attr in types:
                    private = part.attr
                    if private not in aliases:
                        alias = 'mxprepared' + str(len(aliases))
                        while alias in p['inputs']: alias += 'x'
                        aliases[private] = alias
                        p['inputs'][alias] = object
                        input_expressions[alias] = 'self.' + private
                        tp = types[private]
                        prepared[alias] = 'object' if tp == 'object' else package_name.split('.')[0] + '.' + tp
                if isinstance(part, ast.Name) and part.id in {'_mx_curves', '_mx_sys', '_mx_cy'}:
                    helper_modules[part.id] = ('cython' if part.id == '_mx_cy' else
                                              package_name.split('.')[0] + '.' + part.id)
                if (isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute)
                        and isinstance(part.func.value, ast.Name)
                        and (part.func.value.id, part.func.attr) in
                        {('_mx_sys', '_mx_pow'), ('_mx_curves', 'power')}):
                    native_calls.add(part.func.value.id + '.' + part.func.attr)
        p['input_expressions'] = input_expressions
        p['prepared_input_types'] = prepared
        globals_ = {n: ModuleType(path) for n, path in helper_modules.items()}
        for name, original in sorted(p['methods'].items()):
            cell = p['emission_plan'].cells[name]
            arguments = {n: TYPES[t] for n, t in cell.public_arguments}
            result = TYPES[cell.public_result]
            node = FormulaEmitter.native(cell, arguments, result, source_node=nodes.get(name),
                                         reference_aliases=aliases)
            setattr(module, name, SourceFormula(node, arguments, result,
                    cached=cell.cached, filename=module.__file__, module_name=module_name, globals_=globals_))
            if cell.cache_domains:
                domains[name] = {arg: (lo, hi - lo) for arg, lo, hi in cell.cache_domains}
        name = folder.name
        capacity = max([1] + [s for params in domains.values() for _, s in params.values()])
        signatures = {name: {'parameters': {n: TYPES[t].__name__ for n, t in cell.public_arguments},
                             'return': TYPES[cell.public_result].__name__}
                      for name, cell in p['emission_plan'].cells.items() if name in p['methods']}
        spec = KernelSpec(module, p['inputs'], {}, max(1, capacity), '0',
                          cell_outputs=tuple(sorted(p['methods'])),
                          cache_domains=domains, original_space_shell=True, cython_objects=True,
                          cython_signatures=signatures, original_prepared_inputs=prepared,
                          original_helper_modules=helper_modules,
                          original_native_calls=tuple(sorted(native_calls)),
                          original_cache_kinds={n: p['emission_plan'].cells[n].cache_kind
                                                for n in p['methods']})
        artifact = folder / 'artifact'
        build(spec, artifact, compile_native=False)
        manifest_file = artifact / 'manifest.json'
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        if any(row['coverage']['detached_python'] for row in manifest['cells'].values()):
            raise ValueError('Original-shell admission requires compiled Cython bodies')
        private = manifest['package']
        target = folder / private
        shutil.move(str(artifact / private), str(target))
        # Let original create_setup consume this extension in the same build.
        kernel = target / '_kernel.pyx'
        kernel.write_text('# distutils: language=c++\n' + kernel.read_text(encoding='utf-8'), encoding='utf-8')
        (target / '_kernel.pxd').write_text(owner_entry_declarations(manifest), encoding='utf-8')
        manifest['source_sha256'] = hashlib.sha256(kernel.read_bytes()).hexdigest()
        manifest['build_driver'] = 'original mx2cy package setup'
        manifest['original_space_shell'] = True
        manifest['package'] = package_name + '.' + name + '.' + private
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        p['kernel_module'] = manifest['package'] + '._kernel'
        p['manifest'] = manifest
        modules.extend([Path(name) / private / '_lookups.py', Path(name) / private / '_kernel.pyx'])
        if (target / '_table_lookup.py').exists():
            modules.append(Path(name) / private / '_table_lookup.py')
    return modules


def transform(source, declarations, plans, original_methods):
    """Shared frame caches with exact or original virtual formula dispatch."""
    if not plans:
        return source, declarations
    tree = ast.parse(source)
    tree.body.insert(0, ast.parse(
        'from cython.cimports.cpython.pycapsule import PyCapsule_GetPointer as _mx_frame_capsule_ptr').body[0])
    owner_module = next(iter(plans.values()))['kernel_module'].split('.')[0] + '._mx_frame_owner'
    tree.body.insert(0, ast.parse(f'from {owner_module} import SpaceFrameOwner as _mx_SpaceFrameOwner').body[0])
    for cls in tuple(tree.body):
        if not isinstance(cls, ast.ClassDef) or cls.name not in plans:
            continue
        p = plans[cls.name]
        for cell_name in p['methods']:
            cell = p['emission_plan'].cells[cell_name]
            alias = '_mx_frame_' + cls.name + '_' + cell_name
            ctypes = ['object', *(t for _, t in cell.public_arguments),
                      *(t.__name__ for t in p['inputs'].values())]
            if p['manifest'].get('python_cells'):
                ctypes.insert(1, 'object')
            ctypes = [dict(float='double', int='long long', bool='bint', str='object').get(t, t) for t in ctypes]
            result_type = p['manifest']['cells'][cell_name]['result']['ctype']
            exception = '' if result_type == 'object' else ' except? -1'
            declarations += (f'\nctypedef {result_type} (*{alias}_t)('
                             + ', '.join(ctypes) + ')' + exception + '\n'
                             + f'cdef {alias}_t {alias}\n')
        for cell_name in p['methods']:
            row = p['manifest']['cells'][cell_name]
            alias = '_mx_virtual_' + cls.name + '_' + cell_name
            ctypes = ['object', 'object'] + [arg['type']['ctype'] for arg in row['params']]
            result = row['result']['ctype']
            exception = '' if result == 'object' else ' except? -1'
            declarations += (f'\nctypedef {result} (*{alias}_t)('
                             + ', '.join(ctypes) + ')' + exception + '\n'
                             + f'cdef {alias}_t {alias}\n')
        cls.body.insert(0, ast.AnnAssign(ast.Name('_mx_frame_expired', ast.Store()),
                                       ast.Attribute(ast.Name('_mx_cy', ast.Load()), 'bint', ast.Load()), None, 1))
        cls.body.insert(0, ast.AnnAssign(ast.Name('_mx_frame_owner', ast.Store()),
                                       ast.Name('object', ast.Load()), None, 1))
        marker = next(l for l in declarations.splitlines() if l.startswith('cdef class ' + cls.name + '('))
        declarations = declarations.replace(marker, marker + '\n    cdef object _mx_frame_owner\n    cdef bint _mx_frame_expired', 1)
        expressions = p.get('input_expressions', {n: 'self.' + n for n in p['inputs']})
        inputs = '(' + ', '.join(expressions[k] for k in p['inputs']) + (',' if p['inputs'] else '') + ')'
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            if fn.name in p['methods']:
                expired = ast.parse('if self._mx_frame_expired:\n    raise RuntimeError("Collection policy cache has expired")').body
                args = ', '.join([*(a.arg for a in fn.args.args[1:]),
                                  *(expressions[k] for k in p['inputs'])])
                receiver = ', self' if p['manifest'].get('python_cells') else ''
                alias = '_mx_frame_' + cls.name + '_' + fn.name
                cell = p['emission_plan'].cells[fn.name]
                c_api = {'object': 'PyObject *', 'str': 'PyObject *', 'long long': 'PY_LONG_LONG',
                         'double': 'double', 'bint': 'int'}
                ctypes = ['object', *(t for _, t in cell.public_arguments),
                          *({float: 'double', int: 'long long', bool: 'bint', object: 'object', str: 'object'}[t]
                            for t in p['inputs'].values())]
                if p['manifest'].get('python_cells'):
                    ctypes.insert(1, 'object')
                result_type = p['manifest']['cells'][fn.name]['result']['ctype']
                checked = not p['manifest']['cells'][fn.name]['coverage']['python_free_proven']
                capsule_signature = (c_api[result_type] + ('(' if result_type == 'object' else ' (')
                                     + ', '.join(c_api[t] for t in ctypes) + ')').encode()
                head = ast.parse(f'''if type(self) is {cls.name}:
    global {alias}
    if self._mx_frame_owner is None:
        self._mx_frame_owner = _mx_SpaceFrameOwner({p['kernel_module']!r}, {inputs})
    {"self._mx_frame_owner.prepare_checked()" if checked else "pass"}
    if {alias} == _mx_cy.NULL:
        {alias} = _mx_cy.cast({alias}_t, _mx_frame_capsule_ptr(self._mx_frame_owner.entry_capsule({fn.name!r}), {capsule_signature!r}))
    return {alias}(self._mx_frame_owner.kernel{receiver}{', ' + args if args else ''})''').body
                if not cell.cached:
                    fn.body = expired + head + copy.deepcopy(original_methods[cls.name][fn.name].body)
                    continue
                virtual = '_mx_virtual_' + cls.name + '_' + fn.name
                row = p['manifest']['cells'][fn.name]
                callback = '_mx_frame_formula_' + fn.name
                body_args = ', '.join(a.arg for a in fn.args.args[1:])
                virtual_types = ['object', 'object'] + [arg['type']['ctype'] for arg in row['params']]
                virtual_signature = (c_api[result_type] + ('(' if result_type == 'object' else ' (')
                                     + ', '.join(c_api[t] for t in virtual_types) + ')').encode()
                derived = ast.parse(f'''global {virtual}
if self._mx_frame_owner is None:
    self._mx_frame_owner = _mx_SpaceFrameOwner({p['kernel_module']!r}, {inputs})
{"self._mx_frame_owner.prepare_checked()" if p['manifest'].get('python_cells') else "pass"}
if {virtual} == _mx_cy.NULL:
    {virtual} = _mx_cy.cast({virtual}_t, _mx_frame_capsule_ptr(self._mx_frame_owner.virtual_capsule({fn.name!r}), {virtual_signature!r}))
return {virtual}(self._mx_frame_owner.kernel, self.{callback}{', ' + body_args if body_args else ''})''').body
                fn.body = expired + head + derived

            elif fn.name.startswith('_f_') and fn.name[3:] in p['methods']:
                # Keep decorators and the original virtual declaration, but
                # restore its uncached body. Dependencies enter public wrappers.
                fn.body = copy.deepcopy(original_methods[cls.name][fn.name].body)
        for cell_name in p['methods']:
            if not p['emission_plan'].cells[cell_name].cached:
                continue
            original = original_methods[cls.name]['_f_' + cell_name]
            callback = ast.FunctionDef(
                name='_mx_frame_formula_' + cell_name,
                args=copy.deepcopy(original.args),
                body=ast.parse('return self._f_' + cell_name + '(' +
                               ', '.join(a.arg for a in original.args.args[1:]) + ')').body,
                decorator_list=[], returns=copy.deepcopy(original.returns),
                type_comment=None)
            cls.body.append(callback)
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n', declarations


def add_collect_hooks(source, plans):
    """Delegate lifetime to the existing exported collector and detachment rules."""
    tree = ast.parse(source)
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in plans:
            continue
        methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
        snippets = {
            '_mx_collect_attach': '''_mx_frame_scratch = context.get((template, "frame"))
self._mx_frame_expired = False
if _mx_frame_scratch is not None:
    _mx_frame_scratch.reset()
    self._mx_frame_owner = _mx_frame_scratch''',
            '_mx_collect_release': '''if self._mx_frame_owner is not None:
    context[(template, "frame")] = self._mx_frame_owner
self._mx_frame_owner = None
self._mx_frame_expired = True''',
        }
        for name, code in snippets.items():
            if name not in methods:
                fn = ast.parse(f'def {name}(self, context, template):\n    pass').body[0]
                fn.body = ast.parse(code).body
                cls.body.append(fn)
            else:
                # Existing attach may return when its legacy owner is absent.
                methods[name].body = ast.parse(code).body + methods[name].body
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def add_owner_entries(source, manifest):
    """Retained call, non-invalidating input update, completed-cache pickle state."""
    from .model_codegen import add_model_entries
    source = add_model_entries(source, manifest)
    if not manifest.get('python_cells'):
        marker = 'cdef class Kernel:\n'
        assert source.count(marker) == 1
        source = source.replace(marker, marker + '    cdef unsigned int _mx_virtual_depth\n', 1)
        for signature in ('    def reset(self):', '    def replace_records(self, records):',
                          '    def _reset_item_cache(self):'):
            source = source.replace(signature, signature +
                '\n        if self._mx_virtual_depth:\n'
                '            raise RuntimeError("Cannot reset cache during a Cell evaluation")', 1)
    lines = ['    def bind_inputs(self, tuple record):']
    lines += [f'        self.{"" if row["kind"] == "object" else "_frame."}{name} = record[{i}]'
              for i, (name, row) in enumerate(manifest['inputs'].items())]
    lines += [f'        self._frame.{name} = <PyObject*>self.{name}'
              for name, row in manifest['inputs'].items() if row['kind'] == 'object']
    if manifest.get('python_cells'):
        lines += ['        self._records = (record,)',
                  '        self._python_scope.select(record, 0)']
        # Rebind only input globals. Library globals keep their initialized
        # bindings. This does not reset or invalidate any completed Cell.
        # gil_codegen may bind input names from compiled bodies omitted from
        # the Python source; its declared fields are the emission authority.
        for name in manifest['inputs']:
            if ' _gil_global_' + name + '\n' in source:
                lines += [f'        self._gil_global_{name} = self._python_scope.namespace[{name!r}]']
    if not manifest['inputs']:
        lines.append('        pass')
    bound = ', '.join(f'self.{"" if row["kind"] == "object" else "_frame."}{name}'
                      for name, row in manifest['inputs'].items())
    lines += ['    def bound_inputs(self):',
              '        return (' + bound + (',' if bound else '') + ')']
    if manifest.get('python_cells'):
        prepared = manifest.get('original_prepared_inputs', {})
        refs = ', '.join(('self.' if name in prepared else 'receiver.') + name
                         for name in manifest['inputs'])
        lines += ['    def _mx_synchronize_inputs(self):',
                  '        receiver = self._python_scope._original_receiver',
                  '        if receiver is not None:',
                  '            self.bind_inputs((' + refs + (',' if refs else '') + '))']
        # The checked object formula can call back into numeric execution after
        # changing public inputs. Update the held native fields before that edge.
        marker = '    def call(self, str cell, tuple args):\n        cdef _Command command'
        source = source.replace(marker, marker + '\n        self._mx_synchronize_inputs()', 1)
    from .typed_memo_codegen import completed_cache_methods
    lines.extend(completed_cache_methods(manifest['cells']))
    from .virtual_cache_codegen import entries as virtual_entries
    virtual_methods, virtual_exports = virtual_entries(manifest)
    lines.extend(virtual_methods)
    typed = {n: r for n, r in manifest['cells'].items()
             if r.get('storage') in {'typed_scalar', 'typed_epoch_slabs', 'typed_python_dict'}}
    lines += ['    def snapshot(self):', '        result = []', '        cdef Py_ssize_t i']
    objects = set(manifest.get('object_cells', ()))
    growing = {n: r for n, r in manifest['cells'].items() if r.get('storage') == 'growing_numeric'}
    numeric_rows = {n: r for n, r in manifest['cells'].items()
                    if n not in objects and n not in growing and n not in typed and r['cached']}
    for name in growing:
        lines += [f'        result.append((({name!r}, "numeric"), self._numeric_snapshot_{name}()))']
    for name in typed:
        lines += [f'        result.append((({name!r}, "typed"), self._typed_snapshot_{name}()))']
    for name, row in numeric_rows.items():
        value = '_floats' if row['result']['kind'] == 'float' else '_ints'
        lines += [f"        for i in range({row['offset']}, {row['offset'] + row['slots']}):",
                  '            if self._frame._seen[i] == self._frame._epoch:',
                  f'                result.append((i, self._frame.{value}[i]))']
    for name in sorted(objects):
        row = manifest['cells'][name]
        if not row['cached'] or name in typed:
            continue
        if row['storage'] == 'object_singleton':
            lines += [f'        if self._object_initialized and self._object_singleton_{name} is not _DICT_EMPTY and self._object_singleton_{name} is not _DICT_BUSY:',
                      f'            result.append((({name!r}, "object"), self._object_singleton_{name}))']
        elif row['storage'] == 'python_dict':
            lines += ['        if self._object_initialized:',
                      f'            for key, value in self._object_{name}.items():',
                      '                if value is not _DICT_EMPTY and value is not _DICT_BUSY:',
                      f'                    result.append((({name!r}, key), value))']

        elif row['storage'] == 'object_slots':
            lines += ['        if self._object_initialized:',
                      f'            for i in range({row["slots"]}):',
                      f'                entry = self._object_{name}[i]',
                      '                if entry is not None and (<_ObjectEntry>entry).state == 2:',
                      f'                    result.append((({name!r}, i), (<_ObjectEntry>entry).value))']
    lines += ['        return result', '    def restore(self, entries):', '        cdef Py_ssize_t i']
    if objects:
        lines.append('        cdef _ObjectEntry entry')
    lines.append('        for key, value in entries:')
    if objects or growing:
        lines += ['            if isinstance(key, tuple):',
                  '                name, index = key']
        cases = 0
        for name in growing:
            lines += [f'                {"if" if cases == 0 else "elif"} name == {name!r}:',
                      '                    if index != "numeric": raise ValueError("Invalid numeric cache snapshot")',
                      f'                    self._numeric_restore_{name}(value)']
            cases += 1
        for name in typed:
            lines += [f'                {"if" if cases == 0 else "elif"} name == {name!r}:',
                      '                    if index != "typed": raise ValueError("Invalid typed cache snapshot")',
                      f'                    self._typed_restore_{name}(value)']
            cases += 1
        for name in sorted(objects):
            row = manifest['cells'][name]
            if not row['cached'] or name in typed:
                continue
            lines += [f'                {"if" if cases == 0 else "elif"} name == {name!r}:']
            cases += 1
            if row['storage'] == 'object_singleton':
                lines += ['                    if index != "object": raise ValueError("Invalid completed object slot")',
                          f'                    self._object_singleton_{name} = value']
            elif row['storage'] == 'python_dict':
                lines += [f'                    self._object_{name}[index] = value']

            elif row['storage'] == 'object_slots':
                lines += [f'                    if not 0 <= index < {row["slots"]}: raise ValueError("Invalid completed object slot")',
                          '                    entry = _ObjectEntry()',
                          '                    entry.state = 2', '                    entry.value = value',
                          '                    entry.touched = True',
                          '                    self._object_dirty.append(entry)',
                          f'                    self._object_{name}[index] = entry']
        if cases:
            lines += ['                else: raise ValueError("Unknown completed object Cell")']
        else:
            lines += ['                raise ValueError("No cached object Cell")']
        lines += ['                continue']
    lines += ['            i = key']
    for j, row in enumerate(numeric_rows.values()):
        value = '_floats' if row['result']['kind'] == 'float' else '_ints'
        lines += [f"            {'if' if j == 0 else 'elif'} {row['offset']} <= i < {row['offset'] + row['slots']}:",
                  f'                self._frame.{value}[i] = value']
    if numeric_rows:
        lines += ['            else: raise ValueError("Invalid completed cache slot")',
                  '            self._frame._seen[i] = self._frame._epoch']
    else:
        lines.append('            raise ValueError("No numeric cache slots")')
    source = source.rstrip() + '\n\n' + '\n'.join(lines) + '\n'
    numeric_names = [n for n in manifest['cells'] if n not in objects]
    for name, row in manifest['cells'].items():
        signature = _owner_signature(name, row, manifest)
        entry = [signature + ':', '    cdef Kernel owner = kernel',
                 '    cdef _Command command']
        checked = bool(manifest.get('python_cells')) and not row['coverage']['python_free_proven']
        if checked:
            entry += ['    saved_receiver = getattr(owner._python_scope, "_original_receiver", None)',
                      '    owner._python_scope._original_receiver = receiver', '    try:']
        prefix_size = len(entry)
        if checked:
            inputs = ', '.join('_mx_input_' + n for n in manifest['inputs'])
            entry.append('    owner.bind_inputs((' + inputs + (',' if inputs else '') + '))')
        else:
            for input_name in manifest['inputs']:
                info = manifest['inputs'][input_name]
                if info['kind'] == 'object':
                    entry += [f'    owner.{input_name} = _mx_input_{input_name}',
                              f'    owner._frame.{input_name} = <PyObject*>owner.{input_name}']
                else:
                    entry.append(f'    owner._frame.{input_name} = _mx_input_{input_name}')
        if name in objects:
            args = ', '.join(p['name'] for p in row['params'])
            entry += ['    owner._frame._entry_depth += 1', '    try:',
                      f'        return owner._object_cell_{name}({args})',
                      '    finally:', '        owner._frame._entry_depth -= 1']
            if checked:
                entry[prefix_size:] = ['    ' + line for line in entry[prefix_size:]]
                entry += ['    finally:', '        owner._python_scope._original_receiver = saved_receiver']
            source += '\n' + '\n'.join(entry) + '\n'
            continue
        entry += ['    command.mode = 0', f'    command.identity = {numeric_names.index(name)}']
        for index, param in enumerate(row['params']):
            field = 'f' if param['type']['kind'] == 'float' else 'i'
            entry.append(f'    command.args[{index}].{field} = {param["name"]}')
        result = 'f' if row['result']['kind'] == 'float' else 'i'
        entry += ['    owner._invoke(&command)', f'    return command.value.{result}']
        if checked:
            entry[prefix_size:] = ['    ' + line for line in entry[prefix_size:]]
            entry += ['    finally:', '        owner._python_scope._original_receiver = saved_receiver']
        source += '\n' + '\n'.join(entry) + '\n'
    source += '\n' + '\n'.join(virtual_exports) + '\n'
    return source


def _owner_signature(name, row, manifest):
    """A private native ABI; the original public .pxd stays authoritative."""
    args = ['object kernel']
    if manifest.get('python_cells'):
        args.append('object receiver')
    args += [p['type']['ctype'] + ' ' + p['name'] for p in row['params']]
    args += [info['ctype'] + ' _mx_input_' + n for n, info in manifest['inputs'].items()]
    exception = '' if row['result']['ctype'] == 'object' else ' except? -1'
    return (f'cdef {row["result"]["ctype"]} _mx_shell_{name}('
            + ', '.join(args) + ')' + exception)


def owner_entry_declarations(manifest):
    from .virtual_cache_codegen import signature
    virtual = '\n'.join(signature(n, r) for n, r in manifest['cells'].items()) + '\n'
    return virtual + '\n'.join(_owner_signature(n, row, manifest)
                     for n, row in manifest['cells'].items()) + '\n'
