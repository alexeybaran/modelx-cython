"""Cython-compiled Python-semantic calculators in the existing owned-object memo.

No pandas interpretation and no second cache: supported ordinary bodies become
checked GIL-held Cython methods. Generator expressions are passed to Cython;
unsupported nested functions/lambdas retain detached Python bodies.
Cell calls from Python bind directly to the same compiled memo methods.
"""
import ast
import builtins
import copy
import inspect
import textwrap
from .compiler import function_ast
from .contracts import ContractError
from .python_calculator import _global_names


def augment(source, manifest, spec, forced_python=()):
    objects = set(manifest.get('object_cells', ()))
    rows = manifest['cells']
    from .typed_memo_codegen import checked_signatures, CTYPES
    signatures = checked_signatures(spec, rows)
    typed_fields = spec.typed_mapping_inputs or {}
    numeric = {n for n,h in signatures.items() if h['return'] in ('int','float','bool')}
    native_numeric = {n for n in numeric
                      if rows[n].get('calculator') == 'native'}
    native_called = set()
    curve_cells={n for n,h in signatures.items() if h['return']=='curve'}
    bodies, globals_, tables_ = {}, set(), set()
    from .coverage import cython_fallback_reasons
    # Numeric Cells that need Python-object inputs still have a Cython formula
    # body. Only their cache entry must cross the checked GIL boundary. This
    # keeps ordinary exported formulas in the same compiler pipeline instead
    # of treating object input access as a reason to interpret the whole body.
    candidates = objects | {name for name, row in rows.items()
                            if row.get('calculator') == 'python'
                            and row['result']['kind'] in ('float', 'int', 'bool')}
    for name in sorted(candidates - set(forced_python)):
        node = function_ast(spec.cells[name])
        if cython_fallback_reasons(spec.cells[name], rows):
            continue
        bodies[name] = node
        globals_.update(_global_names(spec.cells[name].__code__) - rows.keys() - vars(builtins).keys())
    numeric_bodies = set(bodies) - objects
    if numeric_bodies != set(manifest.get('cython_numeric_fallback_cells', ())):
        raise ContractError('Cython numeric body selection differs from bundled source')
    if set(typed_fields) - globals_:
        raise ContractError('Typed mapping input is not used by a Cython Cell formula')
    declared = {(name, key) for name, fields in typed_fields.items() for key in fields}
    used = {(node.value.id, node.slice.value)
            for body in bodies.values() for node in ast.walk(body)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant) and type(node.slice.value) is str}
    if declared - used:
        raise ContractError('Typed mapping field is not used by a Cython Cell formula')

    cython_aliases = set()
    python_power_sites = []
    raw_tables = set()
    bound_table_names = set(spec.initial_tables or {}) if spec.model is not None else set()

    class Bind(ast.NodeTransformer):
        def __init__(self, fn):
            self.cython_names = {a.asname or a.name for n in ast.walk(function_ast(fn))
                                 if isinstance(n, ast.Import) for a in n.names if a.name == 'cython'}
            self.cython_names.update(n for n, v in fn.__globals__.items()
                                     if inspect.ismodule(v) and v.__name__ == 'cython')
            cython_aliases.update(self.cython_names)
            self.locals = set(fn.__code__.co_varnames) | set(fn.__code__.co_cellvars) | set(inspect.signature(fn).parameters) | {
                n.id for n in ast.walk(function_ast(fn)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        def visit_Import(self, node):
            # cython.declare/locals are Cython's own typing language. In .pyx
            # cimport must be at module scope; Python keeps the original import.
            aliases = [a for a in node.names if a.name != 'cython']
            return ast.copy_location(ast.Import(aliases), node) if aliases else None

        def visit_Subscript(self, node):
            # A build-declared numeric mapping field is bound once at policy
            # selection. All other subscripts keep their Python semantics.
            if (isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name)
                    and node.value.id not in self.locals
                    and isinstance(node.slice, ast.Constant)
                    and type(node.slice.value) is str
                    and node.slice.value in typed_fields.get(node.value.id, {})):
                field = '_gil_field_' + node.value.id + '_' + node.slice.value
                return ast.copy_location(ast.Attribute(
                    ast.Name('self', ast.Load()), field, ast.Load()), node)
            return self.generic_visit(node)

        def visit_BinOp(self, node):
            # Cython infers a complex C value for a floating/nonliteral power.
            # An explicit object base delegates ** to Python's numeric protocol,
            # preserving negative-base, zero-base and user-defined semantics.
            # Do not use cpow=True: that silently changes those edge cases.
            if isinstance(node.op, ast.Pow) and not (
                    isinstance(node.right, ast.Constant) and type(node.right.value) is int
                    and node.right.value >= 0):
                python_power_sites.append(True)
                return ast.copy_location(ast.BinOp(
                    ast.Call(ast.Attribute(ast.Name('_mxcf_cy', ast.Load()), 'cast', ast.Load()),
                             [ast.Name('object', ast.Load()), self.visit(node.left)], []),
                    ast.Pow(), self.visit(node.right)), node)
            return self.generic_visit(node)

        def visit_Call(self, node):
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                    and node.func.value.id == 'lookup' and node.func.attr in
                    {'at','get','previous','previous_group','max_key','max_level','keys','group_curve'} and node.args):
                table = node.args[0]
                if (isinstance(table,ast.Call) and not table.args and not table.keywords and
                        isinstance(table.func,ast.Attribute) and isinstance(table.func.value,ast.Name)
                        and table.func.value.id=='data'):
                    tables_.add(table.func.attr)
                    return ast.copy_location(ast.Call(ast.Attribute(ast.Attribute(ast.Name('self',ast.Load()),
                        '_gil_table_'+table.func.attr,ast.Load()),node.func.attr,ast.Load()),
                        [self.visit(a) for a in node.args[1:]], [self.visit(k) for k in node.keywords]),node)
            # Data readers in the exported table contract are immutable frame
            # bindings. Bind the whole DataFrame once per scope; leave every
            # pandas operation and every scenario/policy selection untouched.
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'data' and 'data' not in self.locals
                    and node.func.attr in bound_table_names and not node.args and not node.keywords):
                raw_tables.add(node.func.attr)
                return ast.copy_location(ast.Attribute(ast.Name('self', ast.Load()),
                                         '_gil_raw_table_' + node.func.attr, ast.Load()), node)
            if isinstance(node.func, ast.Name) and node.func.id in rows and node.func.id not in self.locals:
                name = node.func.id
                # Use Python signature binding at build time for keyword/default calls.
                signature = inspect.signature(spec.cells[name])
                bound = signature.bind(*node.args, **{k.arg:k.value for k in node.keywords})
                bound.apply_defaults()
                args = [self.visit(v) if isinstance(v, ast.AST) else ast.Constant(v) for v in bound.arguments.values()]
                if name in curve_cells:
                    return ast.copy_location(ast.Call(ast.Attribute(ast.Name('self',ast.Load()), '_object_cell_'+name, ast.Load()), args, []), node)
                if name in native_numeric:
                    native_called.add(name)
                    return ast.copy_location(ast.Call(ast.Attribute(ast.Name('self',ast.Load()), '_native_cell_'+name, ast.Load()), args, []), node)
                if name in numeric:
                    if name in objects:
                        return ast.copy_location(ast.Call(ast.Attribute(ast.Name('self',ast.Load()), '_typed_cell_'+name, ast.Load()), args, []), node)
                    return ast.copy_location(ast.Call(ast.Attribute(ast.Name('self',ast.Load()), 'call',ast.Load()),
                        [ast.Constant(name),ast.Tuple(args,ast.Load())],[]),node)
                if name in objects:
                    return ast.copy_location(ast.Call(ast.Attribute(ast.Name('self',ast.Load()), '_object_cell_'+name, ast.Load()), args, []), node)
                return ast.copy_location(ast.Call(ast.Attribute(ast.Name('self',ast.Load()), 'call',ast.Load()),
                    [ast.Constant(name),ast.Tuple(args,ast.Load())],[]),node)
            return self.generic_visit(node)
        def visit_Name(self, node):
            if node.id in self.cython_names:
                return node
            if isinstance(node.ctx,ast.Load) and node.id in globals_ and node.id not in self.locals:
                return ast.copy_location(ast.Attribute(ast.Name('self',ast.Load()), '_gil_global_'+node.id,ast.Load()),node)
            return node

    def global_type(name):
        return {int:'long long',float:'double',bool:'bint'}.get(spec.inputs.get(name),'object') if signatures else 'object'
    fields = '\n'.join('    cdef '+global_type(n)+' _gil_global_'+n for n in sorted(globals_))
    if typed_fields:
        fields += '\n' + '\n'.join(
            '    cdef ' + ('double[::1]' if kind == 'float64_1d' else 'double')
            + ' _gil_field_' + name + '_' + key
            for name, entries in sorted(typed_fields.items())
            for key, kind in sorted(entries.items()))
    source=source.replace('cdef class Kernel:\n', 'cdef class Kernel:\n'+fields+'\n',1)
    assignments='\n'.join(f'        self._gil_global_{n} = self._python_scope.namespace[{n!r}]' for n in sorted(globals_))
    if typed_fields:
        assignments += '\n' + '\n'.join(
            f"        self._gil_field_{name}_{key} = self._gil_global_{name}[{key!r}]"
            for name, entries in sorted(typed_fields.items())
            for key in sorted(entries))
    source=source.replace('        self._python_scope.select(self._records[policy], policy)',
                           '        self._python_scope.select(self._records[policy], policy)\n'+assignments,1)
    if manifest.get('native_scalar_outputs'):
        source += ('\n    def attach_python_scope_current(self):\n'
                   '        """Bind Python fallbacks after a native entry without clearing its cache."""\n'
                   '        self._require_idle()\n'
                   '        if self._python_scope is None or len(self._records) != 1:\n'
                   '            raise RuntimeError("Expected one retained ItemSpace and a Python receiver")\n'
                   '        self._python_scope.select(self._records[0], 0)\n'
                   + assignments + '\n'
                   '        self._frame._scope_valid = True\n')
    methods=[]
    for name,node in bodies.items():
        params=rows[name]['params']
        hints=signatures.get(name,{'parameters':{},'return':'object'})
        signature=', '.join(CTYPES[hints['parameters'].get(p['name'],'object')]+' '+p['name'] for p in params)
        result_type=CTYPES[hints['return']]
        methods.append(f'    cdef {result_type} _gil_formula_{name}(self'+(', '+signature if signature else '')+')'+(' except? -1' if name in numeric else '')+':')
        binder=Bind(spec.cells[name])
        for statement in node.body:
            if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Constant) and isinstance(statement.value.value,str):continue
            transformed = binder.visit(copy.deepcopy(statement))
            if transformed is not None:
                methods.extend(textwrap.indent(ast.unparse(ast.fix_missing_locations(transformed)), '        ').splitlines())
        if name in numeric:methods.append('        raise ValueError("Numeric Cell formula fell through without a result")')
        methods.append('')
        source=source.replace(f'entry.value = self._object_fn_{name}(',f'entry.value = self._gil_formula_{name}(')
        source=source.replace(f'value = self._object_fn_{name}(',f'value = self._gil_formula_{name}(')
        source=source.replace(f"            if self._python_scope.diagnostics:\n                self._python_scope.calls[{name!r}] += 1\n", '')
        rows[name]['calculator']='cython_object' if name in objects else 'cython_scalar'
        if name not in objects:
            # The numeric cache/miss entry already calls PythonScope.calculate.
            # Expose the one Cython body to that bridge without replacing the
            # entry or creating a second cache.
            methods += [f'    def _gil_entry_{name}(self'+(', '+signature if signature else '')+'): ',
                        f'        return self._gil_formula_{name}({", ".join(p["name"] for p in params)})', '']
    for name in sorted(native_called):
        hint = signatures[name]
        params = rows[name]['params']
        typed = ', '.join(CTYPES[hint['parameters'][p['name']]] + ' ' + p['name']
                          for p in params)
        args = ', '.join(p['name'] for p in params)
        methods.append(f"    cdef inline {CTYPES[hint['return']]} _native_cell_{name}(self"
                       + (', ' + typed if typed else '') + ') except? -1:')
        if manifest.get('native_error_abi', '').endswith('guard') or manifest.get('native_error_abi') == 'reentrant_guard_with_checked_python_miss':
            # The native Cell ABI can throw MxAbort. A direct call from a GIL
            # formula has no surrounding C++ guard and would terminate the
            # process on bounds/lookup errors. Keep the same scoped guard used
            # by public numeric entries, including legitimate -1 results.
            # _execute was emitted before mixed/object Cells were appended to
            # the manifest. Its identity domain contains only numeric rows.
            identity = [key for key in rows if key not in objects].index(name)
            methods += ['        cdef _Command command',
                        '        command.mode = 0',
                        f'        command.identity = {identity}']
            for position, param in enumerate(params):
                field = 'f' if param['type']['kind'] == 'float' else 'i'
                methods.append(f'        command.args[{position}].{field} = {param["name"]}')
            result_field = 'f' if hint['return'] == 'float' else 'i'
            methods += ['        self._invoke(&command)',
                        f'        return command.value.{result_field}']
        else:
            frame_pointer = ('self._frame' if manifest.get('physical_layout')
                             == 'retained_heap_numeric_memo_frame_with_python_owners'
                             else '&self._frame')
            methods.append(f"        return _cell_{name}({frame_pointer}"
                           + (', ' + args if args else '') + ')')
        methods.append('')
    manifest['curve_result_conversion'] = 'once on memo miss; one owned native Curve value'
    manifest['numeric_array_builders'] = 0
    manifest['numeric_buffer_backend'] = 'standard Cython declarations and memoryviews'
    manifest['python_semantic_power_sites'] = len(python_power_sites)
    if python_power_sites:
        source = 'cimport cython as _mxcf_cy\n' + source
    if cython_aliases:
        source = '\n'.join('cimport cython as ' + alias for alias in sorted(cython_aliases)) + '\n' + source
    for name, hint in signatures.items():
        if hint['return'] == 'str':
            source = source.replace('    cdef object _object_cell_' + name + '(',
                                    '    cdef str _object_cell_' + name + '(', 1)
    # Normalize typed Curve results exactly once, while publishing the owned
    # Cell value. Native and detached-Python callers then share that same
    # native object, rather than converting/checking on every numeric lookup.
    for name in sorted(curve_cells):
        marker = f'    cdef object _object_cell_{name}('
        begin = source.index(marker)
        import re
        match = re.search(r'^    (?:cdef |def )', source[begin+len(marker):], re.M)
        end = begin + len(marker) + match.start() if match else len(source)
        chunk = source[begin:end]
        chunk = chunk.replace(marker, f'    cdef Curve _object_cell_{name}(', 1)
        calc = '_gil_formula_' if name in bodies else '_object_fn_'
        lines = chunk.splitlines()
        expanded = []
        for line in lines:
            if line.strip() == 'return value':
                line = line.replace('return value', 'return <Curve>value')
            expanded.append(line)
            if f'value = self.{calc}{name}(' in line:
                indent = line[:len(line)-len(line.lstrip())]
                expanded += [indent+'if value is None: raise ValueError("A bound curve Cell cannot return None")',
                             indent+'if not isinstance(value, Curve):',
                             indent+'    value = Curve(*value.arrays)']
        source = source[:begin] + '\n'.join(expanded) + '\n\n' + source[end:]
    # Direct checked entrypoints: no string dispatch or per-call Python codec.
    for name in sorted(objects):
        row=rows[name]
        if any(p['type']['kind'] != 'object' for p in row['params']):continue
        if row['result']['kind'] != 'object':continue
        parts=[]
        for i,p in enumerate(row['params']):
            parts.append(p['name']+('='+p['default'] if p['default'] is not None else ''))
            if p.get('positional_only') and (i+1==len(row['params']) or not row['params'][i+1].get('positional_only')):parts.append('/')
        sig=', '.join(parts)
        args=', '.join(p['name'] for p in row['params'])
        methods += [f'    def _entry_{name}(self'+(', '+sig if sig else '')+'):',
            '        if not self._frame._scope_valid: raise RuntimeError("Select an evaluation scope first")',
            '        if self._object_key_guard or self._clearing_objects: raise RuntimeError("Cannot re-enter during object cache maintenance")',
            '        self._frame._entry_depth += 1', '        try:',
            f'            return self._object_cell_{name}({args})',
            '        finally:', '            self._frame._entry_depth -= 1', '']
    table_fields='\n'.join('    cdef _IndexedTable _gil_table_'+n for n in sorted(tables_))
    table_fields += '\n' + '\n'.join('    cdef object _gil_raw_table_' + n for n in sorted(raw_tables))
    source=source.replace('cdef class Kernel:\n','cdef class Kernel:\n'+table_fields+'\n',1)
    init='\n'.join(f"        self._gil_table_{n} = self._python_scope.namespace['data']._prepare_lookup({n!r}, _IndexedTable)" for n in sorted(tables_))
    init += '\n' + '\n'.join(f"        self._gil_raw_table_{n} = self._python_scope.namespace['data'].{n}()" for n in sorted(raw_tables))
    source=source.replace('        self._python_scope.select(self._records[policy], policy)',
        '        self._python_scope.select(self._records[policy], policy)\n'+init,1)
    if manifest.get('native_scalar_outputs'):
        source=source.replace('        self._python_scope.select(self._records[0], 0)',
                              '        self._python_scope.select(self._records[0], 0)\n'+init,1)
    manifest['bound_data_readers'] = sorted(raw_tables)
    if tables_:
        source='from ._table_lookup cimport IndexedTable as _IndexedTable\n'+source
    manifest['prepared_lookup_tables']=sorted(tables_)
    manifest['cython_object_cells']=sorted(set(bodies) & objects)
    if typed_fields:
        manifest['typed_mapping_inputs'] = typed_fields
    manifest['direct_object_entries']=[n for n in sorted(objects) if rows[n]['result']['kind']=='object' and all(p['type']['kind']=='object' for p in rows[n]['params'])]
    # A persistent ItemSpace owns an unchanged input frame between explicit
    # table/reference updates. Clear every Cell once without rebinding that frame.
    # The ordinary root reset/select protocol is intentionally unchanged.
    if manifest.get('python_cells'):
        methods += ['    def _reset_item_cache(self):',
                    '        cdef bint selected',
                    '        self._require_idle()',
                    '        selected = self._frame._scope_valid',
                    '        self._frame._scope_valid = False',
                    '        self._advance()',
                    '        self._frame._scope_valid = selected', '']
    else:
        methods += ['    def _reset_item_cache(self):',
                    '        self._advance()', '']
    source = source+'\n'+'\n'.join(methods)+'\n'
    # Keep object ownership/exception cleanup on the miss path, just as the
    # numeric memo does. There is still one authoritative owned singleton.
    hot = []
    for name in sorted(objects - numeric):
        if rows[name]['params'] or not rows[name]['cached']:
            continue
        kind = signatures.get(name, {}).get('return', 'object')
        ct = 'Curve' if kind == 'curve' else 'str' if kind == 'str' else 'object'
        source = source.replace(f'    cdef {ct} _object_cell_{name}(',
                                f'    cdef {ct} _object_compute_{name}(', 1)
        hot += [f'    cdef inline {ct} _object_cell_{name}(self):',
                f'        if self._object_singleton_{name} is not _DICT_EMPTY and self._object_singleton_{name} is not _DICT_BUSY:']
        if manifest['diagnostics']:
            hot += [f'            self._ocalls_{name} += 1', f'            self._ohits_{name} += 1',
                    '            self._frame._calls += 1']
        hot += [f'            return <{ct}>self._object_singleton_{name}',
                f'        return self._object_compute_{name}()', '']
    source += '\n' + '\n'.join(hot) + '\n'
    memo_signatures = {name: hint for name, hint in signatures.items()
                       if name in objects and rows[name]['cached']}
    if memo_signatures:
        from .typed_memo_codegen import augment as typed_augment
        source=typed_augment(source,manifest,spec,memo_signatures,bodies)
    return source
