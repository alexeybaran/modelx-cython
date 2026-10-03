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
    parameters = {arg.arg for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == 'self' and node.attr in parameters):
            # Preserve the original receiver when removing it would capture a
            # public argument. Compiled callers keep their checked Cell edge.
            return 'qualified receiver name shadows public argument; original checked entry'
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
    direct_calls = {id(n.func): n for n in ast.walk(fn) if isinstance(n, ast.Call)}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == 'self' and node.attr in info.cells):
            call = direct_calls.get(id(node))
            if call is None:
                return 'first-class Cell reference requires the original checked wrapper'
            if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
                return 'expanded Cell arguments require the original checked wrapper'
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


def plan(unit, *, package_plan=None, prepared_contexts=None):
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
            if (shared.collection is not None and
                    name == shared.collection.get('outputs_cell')):
                # The export declaration is orchestration around Cell entries,
                # not part of their native numeric dependency closure. Keep its
                # one public cache/body on the original Cython receiver so a
                # numeric-only policy does not allocate object memo fields just
                # to describe collection. Object-valued calculation Cells still
                # use the shared context; callers of this declaration use the
                # usual checked bridge. Prepared regions own their full plan.
                reason = 'collection declaration retains original orchestration cache'
            elif fn is None:
                reason = 'missing original formula body'
            elif cell_plan.public_result not in TYPES:
                reason = 'unsupported original public result type'
            elif cell_plan.receiver_requirements:
                reason = '; '.join(cell_plan.receiver_requirements)
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
        # Keep the original typed receiver for closures that read an external
        # Space through an object Cell ABI. Wrapping those existing checked
        # calls in a second numeric context adds ownership/guard work without
        # producing a connected native input path. Prepared graphs and typed
        # array/MemoryView edges retain their resolved native representation.
        original_receiver = set()
        for name, fn in candidates.items():
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Attribute)
                        and isinstance(node.func.value.value, ast.Name)
                        and node.func.value.value.id == 'self'):
                    continue
                ref = info.refs.get(node.func.value.attr)
                target = package_plan.by_class.get(ref.mx_class) if ref and ref.mx_class else None
                if target is None or target.native_context is not None:
                    continue
                called = target.cells.get(node.func.attr)
                if called is not None and called.public_result == 'object':
                    original_receiver.add(name)
        changed = True
        while changed:
            previous = set(original_receiver)
            original_receiver.update(n for n in candidates
                                     if dependencies[n] & original_receiver)
            changed = original_receiver != previous
        for name in original_receiver:
            reasons[name] = 'external object Cell ABI retains the connected original receiver'
            del candidates[name]
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
    for shared in package_plan.spaces.values():
        context = (prepared_contexts or {}).get(shared.class_identity)
        if (context is None or shared.class_name not in unit.module_info.classes
                or unit.module_info.classes[shared.class_name].fqname != shared.class_identity):
            continue
        if context.public_emission_plan is not shared:
            raise ValueError('Prepared representation must use the authoritative Space plan')
        plans[shared.class_name] = dict(info=shared.original_info, emission_plan=shared,
            prepared_spec=context,
            methods={n: shared.cells[n].formula_node for n in context.cells})
        for item in inventory:
            if item['space'] == shared.class_identity and item['cell'] in context.cells:
                item.update(execution='prepared_public_graph',
                            receiver_representation='original_public_space_with_prepared_context',
                            derived_cache='original_class', reason=None)
    return plans, inventory


def emit_private_kernels(plans, package_path, package_name, *, formula_transform):
    """One build graph, with private kernel modules beneath the exported package."""
    from .compiler import build
    modules = []
    for cls_name, p in plans.items():
        if p.get('prepared_spec') is not None:
            spec = p['prepared_spec']
            if spec.public_emission_plan is not p['emission_plan']:
                raise ValueError('Prepared graph must use the same resolved public Space plan')
            p['methods'] = {n: p['emission_plan'].cells[n].formula_node for n in spec.cells}
            modules.extend(_assemble_private_kernel(spec, p, package_path, package_name, cls_name))
            continue
        domains = {}
        info = p['info']
        name = '_mx_frame_' + cls_name.removeprefix('_c_')
        folder = package_path / name
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
                          original_receiver_type=info.fqname,
                          original_live_fields=tuple(n for n in p["inputs"] if n in info.refs),
                          original_reference_types={n: info.refs[n].mx_class for n in p["inputs"]
                                                    if n in info.refs and info.refs[n].mx_class},
                          cython_signatures=signatures, original_prepared_inputs=prepared,
                          original_helper_modules=helper_modules,
                          original_native_calls=tuple(sorted(native_calls)),
                          original_cache_kinds={n: p['emission_plan'].cells[n].cache_kind
                                                for n in p['methods']})
        modules.extend(_assemble_private_kernel(spec, p, package_path, package_name, cls_name))
    return modules


def _assemble_private_kernel(spec, plan, package_path, package_name, cls_name):
    """Both receiver representations use the original package build planner."""
    from .compiler import build
    name = '_mx_frame_' + cls_name.removeprefix('_c_')
    folder = package_path / name
    folder.mkdir(exist_ok=True)
    build(spec, folder, compile_native=False, package_build=True)
    (folder / '__init__.py').write_text('', encoding='utf-8')
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if any(row['coverage']['detached_python'] for row in manifest['cells'].values()):
        raise ValueError('Original-shell admission requires compiled Cython bodies')
    private = manifest['package']
    target = folder / private
    kernel = target / '_kernel.pyx'
    kernel.write_text('# distutils: language=c++\n' + kernel.read_text(encoding='utf-8'), encoding='utf-8')
    if spec.public_emission_plan is None:
        (target / '_kernel.pxd').write_text(owner_entry_declarations(manifest), encoding='utf-8')
    manifest['source_sha256'] = hashlib.sha256(kernel.read_bytes()).hexdigest()
    manifest['build_driver'] = 'original mx2cy package setup'
    manifest['original_space_shell'] = spec.original_space_shell
    manifest['public_space_classes'] = True
    manifest['package'] = package_name + '.' + name + '.' + private
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    plan['kernel_module'], plan['manifest'] = manifest['package'] + '._kernel', manifest
    modules = [Path(name) / private / '_lookups.py', Path(name) / private / '_kernel.pyx']
    for child in ('_table_lookup.py', '_space_formula.pyx'):
        if (target / child).exists():
            modules.append(Path(name) / private / child)
    return modules


def transform(source, declarations, plans, original_methods):
    """Shared frame caches with exact or original virtual formula dispatch."""
    if not plans:
        return source, declarations
    prepared = {name: p for name, p in plans.items() if p['manifest'].get('public_entry_abi')}
    if prepared:
        source, declarations = transform_prepared(source, declarations, prepared)
        plans = {name: p for name, p in plans.items() if name not in prepared}
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
        if p['manifest'].get('python_cells'):
            # Let Cython retain its authoritative state reducer. Synchronize
            # current public references only at pickle entry, before the owner
            # captures its input record; completed caches are not invalidated.
            reducer = '_mx_pickle_' + cls.name
            tree.body.extend(ast.parse(f"""
@_mx_cy.locals(self={cls.name})
def {reducer}(self):
    if self._mx_frame_owner is not None:
        self._mx_frame_owner.prepare_checked()
        self._mx_frame_owner.kernel.bind_inputs({inputs})
    return self.__reduce_cython__()
import copyreg as _mx_pickle_registry
_mx_pickle_registry.pickle({cls.name}, {reducer})
""").body)
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n', declarations


def transform_prepared(source, declarations, plans):
    """Original public classes enter the complete prepared dependency graph.

    This adapter emits public wrappers only. Formula specialization for derived
    receivers stays on the original class; exact receivers share the native
    graph's cache. Input synchronization and full-domain compatibility remain
    qualification gates before enabling this representation by default.
    """
    tree = ast.parse(source)
    first = next(iter(plans.values()))
    owner_module = first.get('owner_module', first['kernel_module'].split('.')[0] + '._mx_frame_owner')
    tree.body[:0] = ast.parse(
        'from cython.cimports.cpython.pycapsule import PyCapsule_GetPointer as _mx_frame_capsule_ptr\n'
        f'from {owner_module} import prepared_owner as _mx_prepared_owner\n').body
    capi = {'object': 'PyObject *', 'str': 'PyObject *', 'unicode': 'PyObject *',
            'long long': 'PY_LONG_LONG', 'double': 'double', 'bint': 'int'}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in plans:
            continue
        p = plans[cls.name]
        fields = ('_mx_frame_owner', '_mx_prepared_resources', '_mx_frame_expired', '_mx_frame_rebind')
        for field in fields:
            annotation = (ast.Attribute(ast.Name('_mx_cy', ast.Load()), 'bint', ast.Load())
                          if field in ('_mx_frame_expired', '_mx_frame_rebind') else ast.Name('object', ast.Load()))
            cls.body.insert(0, ast.AnnAssign(ast.Name(field, ast.Store()), annotation, None, 1))
        marker = next(l for l in declarations.splitlines() if l.startswith('cdef class ' + cls.name + '('))
        declarations = declarations.replace(marker, marker +
            '\n    cdef object _mx_frame_owner\n    cdef object _mx_prepared_resources\n    cdef bint _mx_frame_expired\n    cdef bint _mx_frame_rebind', 1)
        binding = p['manifest'].get('public_space_binding', {})
        if 'domain_formula_entry' in binding:
            # Original private formulas are cdef entries, not Python methods.
            # Call through the typed original receiver, keeping formula
            # execution in Cython and its public visibility unchanged.
            cls.body.append(ast.parse('def ' + binding['domain_entry'] + '(self):\n'
                '    return self.' + binding['domain_formula_entry'] + '()').body[0])
        selected_receiver = (' and type(self._parent) is ' + cls.name
                             if p['emission_plan'].parameters else '')
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name not in p['methods']:
                continue
            cell = p['emission_plan'].cells[fn.name]
            alias = '_mx_frame_' + cls.name + '_' + fn.name
            types = ['object', *(t for _, t in cell.public_arguments)]
            result = cell.public_result
            ctype = lambda t: 'object' if t == 'unicode' else t
            exception = '' if result in ('object', 'str', 'unicode') else ' except? -1'
            declarations += (f'\nctypedef {ctype(result)} (*{alias}_t)('
                             + ', '.join(ctype(t) for t in types) + ')' + exception + '\n'
                             + f'cdef {alias}_t {alias}\n')
            capsule = (capi[result] + ('(' if result in ('object', 'str', 'unicode') else ' (')
                       + ', '.join(capi[t] for t in types) + ')').encode()
            args = ', '.join(a.arg for a in fn.args.args[1:])
            checked = p['manifest']['public_entry_abi'][fn.name]['checked_scope_required']
            entry = ast.parse(f'''if self._mx_frame_expired:
    raise RuntimeError("Collection policy cache has expired")
if type(self) is {cls.name}{selected_receiver}:
    global {alias}
    if self._mx_frame_owner is None or self._mx_frame_rebind:
        _mx_template = self
        while type(_mx_template._parent) is {cls.name}:
            _mx_template = _mx_cy.cast({cls.name}, _mx_template._parent)
        if _mx_template._mx_prepared_resources is None:
            _mx_template._mx_prepared_resources = {{}}
        self._mx_frame_owner = _mx_prepared_owner(self, {p['kernel_module']!r}, _mx_template._mx_prepared_resources, self._mx_frame_owner)
        self._mx_frame_rebind = False
    {"self._mx_frame_owner.prepare_checked()" if checked else "pass"}
    if {alias} == _mx_cy.NULL:
        {alias} = _mx_cy.cast({alias}_t, _mx_frame_capsule_ptr(self._mx_frame_owner.entry_capsule({fn.name!r}), {capsule!r}))
    return {alias}(self._mx_frame_owner.kernel{', ' + args if args else ''})''').body
            template_type = ast.AnnAssign(ast.Name('_mx_template', ast.Store()), ast.Name(cls.name, ast.Load()), None, 1)
            fn.body = [template_type] + entry + fn.body
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
        if plans[cls.name]['manifest'].get('public_entry_abi'):
            snippets['_mx_collect_attach'] = '''_mx_frame_scratch = context.get((template, "frame"))
self._mx_frame_expired = False
self._mx_frame_rebind = True
if _mx_frame_scratch is not None:
    self._mx_frame_owner = _mx_frame_scratch'''
            plan = plans[cls.name]
            if plan['manifest'].get('streaming_collect_map'):
                # Reuse the typed fold behind the original transient receiver.
                # Its Space formula still executes before input selection. A
                # changed/unsupported map domain falls back before evaluating
                # this policy, retaining previously accumulated policy results.
                module = plan['kernel_module']
                owner_module = plan.get('owner_module', module.split('.')[0]+'._mx_frame_owner')
                sources = plan['manifest']['compiled_collect_map']['sources']
                keys = '{' + ', '.join(repr(n)+': self.'+n for n in sources) + '}'
                hook = ast.parse(f'''def _mx_collect_map(self, accumulator):
    from {owner_module} import prepared_owner
    keys = {keys}
    if accumulator is None or not accumulator.identical(keys):
        for values in keys.values():
            if type(values) is not tuple:
                return None
            for key in values:
                if type(key) is not int or not (-(1 << 63) <= key < (1 << 63)):
                    return None
            if len(set(values)) != len(values):
                return None
        if accumulator is not None and not accumulator.matches(keys):
            return None
    _mx_template: {cls.name} = self
    while type(_mx_template._parent) is {cls.name}:
        _mx_template = _mx_cy.cast({cls.name}, _mx_template._parent)
    if _mx_template._mx_prepared_resources is None:
        _mx_template._mx_prepared_resources = {{}}
    self._mx_frame_owner = prepared_owner(self, {module!r}, _mx_template._mx_prepared_resources, self._mx_frame_owner)
    self._mx_frame_rebind = False
    if accumulator is None:
        accumulator = self._mx_frame_owner.kernel.new_collect_accumulator(keys)
    self._mx_frame_owner.kernel.append_collect_accumulator(accumulator)
    return accumulator
''').body[0]
                cls.body.append(hook)
            declaration = plan['manifest']['cells'].get('collect_outputs', {})
            if (declaration.get('checked_namespace_required') is False
                    and declaration.get('storage') == 'object_singleton'
                    and not plan['manifest']['public_space_binding'].get('receiver_rows')):
                # The scalar traversal already validates exact type, domain
                # and declaration shape. Preserve ordinary binding of the first
                # transient receiver; a real Space formula disables this hook.
                module = plan['kernel_module']
                owner_module = plan.get('owner_module', module.split('.')[0]+'._mx_frame_owner')
                hook = ast.parse(f'''def _mx_collect_scalar(self, coordinates, arguments, output):
    from {owner_module} import prepared_collection
    root = {cls.name}(self)
    root._mx_copy_refs(self, self)
    self._mx_assign_params(root, coordinates[0], *arguments)
    root._mx_roots.append(root)
    if self._mx_prepared_resources is None:
        self._mx_prepared_resources = {{}}
    try:
        return prepared_collection(root, {module!r}, self._mx_prepared_resources, coordinates, output)
    finally:
        root._mx_frame_expired = True
''').body[0]
                cls.body.append(hook)
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
    lines = ['    cpdef void bind_inputs(self, tuple record) except *:']
    object_bindings = (bool(manifest.get('python_cells')) and bool(manifest['inputs'])
                       and all(row['kind'] == 'object' for row in manifest['inputs'].values()))
    if object_bindings:
        # Identity checks observe replacement without invoking user equality.
        # Mutated objects remain live through their existing references. Native
        # scalar binders retain their conversion/error order and bitwise guards.
        marker = 'cdef class Kernel:\n'
        assert source.count(marker) == 1
        source = source.replace(marker, marker + '    cdef object _mx_bound_input_scope\n', 1)
        unchanged = ['self._python_scope is not None',
                     'self._mx_bound_input_scope is self._python_scope',
                     f'len(record) == {len(manifest["inputs"])}']
        unchanged += [f'self.{name} is record[{i}]'
                      for i, name in enumerate(manifest['inputs'])]
        lines += ['        if ' + ' and '.join(unchanged) + ':', '            return']
    lines += [f'        self.{"" if row["kind"] == "object" else "_frame."}{name} = record[{i}]'
              for i, (name, row) in enumerate(manifest['inputs'].items())]
    lines += [f'        self._frame.{name} = <PyObject*>self.{name}'
              for name, row in manifest['inputs'].items() if row['kind'] == 'object']
    if manifest.get('python_cells'):
        lines += ['        self._records = (record,)',
                  '        self._python_scope.bind_original_inputs(record)']
        # Rebind only input globals. Library globals keep their initialized
        # bindings. This does not reset or invalidate any completed Cell.
        # gil_codegen may bind input names from compiled bodies omitted from
        # the Python source; its declared fields are the emission authority.
        for name in manifest['inputs']:
            if ' _gil_global_' + name + '\n' in source:
                lines += [f'        self._gil_global_{name} = self._python_scope.namespace[{name!r}]']
        if object_bindings:
            # Publish only after namespace/global synchronization succeeds. A
            # replacement scope must bind even when every input is identical.
            lines.append('        self._mx_bound_input_scope = self._python_scope')
    if not manifest['inputs']:
        lines.append('        pass')
    bound = ', '.join(f'self.{"" if row["kind"] == "object" else "_frame."}{name}'
                      for name, row in manifest['inputs'].items())
    lines += ['    def bound_inputs(self):',
              '        return (' + bound + (',' if bound else '') + ')']
    if manifest.get('python_cells'):
        prepared = manifest.get('original_prepared_inputs', {})
        live_fields = set(manifest.get('original_live_fields', ()))
        def read_input(name):
            if name in prepared:
                return 'self.' + name
            if name in live_fields:
                return 'receiver.' + name
            # Checked Cell dependencies are bound methods, not reference
            # fields. Preserve Python method binding and virtual dispatch;
            # converting a raw cpdef C function loses its hidden dispatch ABI.
            return f'getattr(receiver, {name!r})'
        refs = ', '.join(read_input(name) for name in manifest['inputs'])
        receiver_type = manifest.get('original_receiver_type')
        lines += ['    cpdef void _mx_synchronize_inputs(self) except *:',
                  ('        cdef _mx_original_space receiver = self._mx_active_receiver'
                   if receiver_type else '        receiver = self._python_scope._original_receiver'),
                  '        if receiver is not None:']
        if receiver_type:
            # Live fields are compared without Python equality or namespace
            # reconstruction. Float bits preserve signed zero and NaN payloads.
            source = 'from libc.string cimport memcmp as _mx_input_memcmp\n' + source
            changed = []
            for name, row in manifest['inputs'].items():
                if name in prepared:
                    continue
                if row['kind'] == 'object':
                    changed.append(f'self.{name} is not {read_input(name)}')
                elif row['kind'] == 'float':
                    changed.append(f'_mx_input_memcmp(&self._frame.{name}, &receiver.{name}, sizeof(double)) != 0')
                else:
                    changed.append(f'self._frame.{name} != receiver.{name}')
            lines += ['            if not (' + (' or '.join(changed) or 'False') + '):',
                      '                return']
        lines += ['            self.bind_inputs((' + refs + (',' if refs else '') + '))']
        # The checked object formula can call back into numeric execution after
        # changing public inputs. Update the held native fields before that edge.
        marker = '    def call(self, str cell, tuple args):\n        cdef _Command command'
        source = source.replace(marker, marker + '\n        self._mx_synchronize_inputs()', 1)
    from .typed_memo_codegen import completed_cache_methods
    lines.extend(completed_cache_methods(manifest['cells']))
    from .virtual_cache_codegen import entries as virtual_entries
    virtual_methods, virtual_exports = virtual_entries(manifest)
    lines.extend(virtual_methods)
    from .cache_state_codegen import methods as cache_state_methods
    lines.extend(cache_state_methods(manifest))
    objects = set(manifest.get('object_cells', ()))
    source = source.rstrip() + '\n\n' + '\n'.join(lines) + '\n'
    numeric_names = [n for n in manifest['cells'] if n not in objects]
    for name, row in manifest['cells'].items():
        signature = _owner_signature(name, row, manifest)
        entry = [signature + ':', '    cdef Kernel owner = kernel',
                 '    cdef _Command command']
        checked = bool(manifest.get('python_cells')) and not row['coverage']['python_free_proven']
        from .completed_cache_codegen import completed_dense_hit
        entry.extend(completed_dense_hit(row, manifest, owner='owner',
                     arguments=['_mx_arg_' + str(i) for i in range(len(row['params']))], indent=4))
        # Original cached entries return a completed value before preparing
        # uncached work. Keep that behavior at the checked owner boundary:
        # rebinding cannot invalidate a completed Cell, and a later miss still
        # runs the full live-input synchronization and rollback protocol.
        # Reuse the existing object entry and its authoritative singleton slot;
        # guards and diagnostic accounting retain their established paths.
        if (row['cached'] and row.get('storage') == 'object_singleton'
                and not row['params'] and not manifest.get('diagnostics')):
            entry += ['    if owner._frame._scope_valid and owner._object_initialized and not owner._object_key_guard and not owner._clearing_objects:',
                      f'        if owner._object_singleton_{name} is not _DICT_EMPTY and owner._object_singleton_{name} is not _DICT_BUSY:',
                      f'            return owner._object_cell_{name}()']
        receiver_field = ('owner._mx_active_receiver' if manifest.get('original_typed_receiver')
                          else 'owner._python_scope._original_receiver')
        if checked:
            if manifest.get('original_typed_receiver'):
                entry += ['    cdef _mx_original_space saved_receiver = owner._mx_active_receiver',
                          '    owner._mx_active_receiver = receiver', '    try:']
            else:
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
            args = ', '.join('_mx_arg_' + str(i) for i, _ in enumerate(row['params']))
            entry += ['    owner._frame._entry_depth += 1', '    try:',
                      f'        return owner._object_cell_{name}({args})',
                      '    finally:', '        owner._frame._entry_depth -= 1']
            if checked:
                entry[prefix_size:] = ['    ' + line for line in entry[prefix_size:]]
                entry += ['    finally:', f'        {receiver_field} = saved_receiver']
            source += '\n' + '\n'.join(entry) + '\n'
            continue
        entry += ['    command.mode = 0', f'    command.identity = {numeric_names.index(name)}']
        for index, param in enumerate(row['params']):
            field = 'f' if param['type']['kind'] == 'float' else 'i'
            entry.append(f'    command.args[{index}].{field} = _mx_arg_{index}')
        result = 'f' if row['result']['kind'] == 'float' else 'i'
        entry += ['    owner._invoke(&command)', f'    return command.value.{result}']
        if checked:
            entry[prefix_size:] = ['    ' + line for line in entry[prefix_size:]]
            entry += ['    finally:', f'        {receiver_field} = saved_receiver']
        source += '\n' + '\n'.join(entry) + '\n'
    source += '\n' + '\n'.join(virtual_exports) + '\n'
    return source


def _owner_signature(name, row, manifest):
    """A private native ABI; the original public .pxd stays authoritative."""
    args = ['object kernel']
    if manifest.get('python_cells'):
        args.append('object receiver')
    args += [p['type']['ctype'] + ' _mx_arg_' + str(i)
             for i, p in enumerate(row['params'])]
    args += [info['ctype'] + ' _mx_input_' + n for n, info in manifest['inputs'].items()]
    exception = '' if row['result']['ctype'] == 'object' else ' except? -1'
    return (f'cdef {row["result"]["ctype"]} _mx_shell_{name}('
            + ', '.join(args) + ')' + exception)


def owner_entry_declarations(manifest):
    from .virtual_cache_codegen import signature
    virtual = '\n'.join(signature(n, r) for n, r in manifest['cells'].items()) + '\n'
    return virtual + '\n'.join(_owner_signature(n, row, manifest)
                     for n, row in manifest['cells'].items()) + '\n'
