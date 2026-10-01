"""Attach the native frame to the original compiler's public Space shell.

Discovery, specification precedence, public declarations, construction and
virtual dispatch belong to the original compiler. This adapter consumes its
ModuleInfo, emits a private kernel, and redirects exact public entries only.
Unsupported graphs retain generated Cython, not an interpreted replacement.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import textwrap
from pathlib import Path

from .contracts import KernelSpec
from .exported_fallback import _UnbindFormula

TYPES = {'double': float, 'long long': int, 'bint': bool}
BUILTINS = {'float', 'int', 'bool', 'abs', 'min', 'max', 'range',
            'ValueError', 'IndexError', 'ArithmeticError', 'RuntimeError'}


def _reason(fn, info):
    if info.is_locked:
        return 'locked Space'
    a = fn.args
    if a.defaults or a.kw_defaults or a.kwonlyargs or a.posonlyargs or a.vararg or a.kwarg:
        return 'non-simple signature'
    if any(arg.arg == 'state' or arg.arg.startswith('_mx_') for arg in a.args[1:]):
        return 'name overlaps the current native receiver namespace'
    forbidden = (ast.Try, ast.TryStar, ast.With, ast.Lambda, ast.Yield,
                 ast.YieldFrom, ast.GeneratorExp, ast.ListComp, ast.DictComp,
                 ast.SetComp, ast.Subscript, ast.Dict, ast.List, ast.Set,
                 ast.NamedExpr, ast.Delete, ast.Global, ast.Nonlocal,
                 ast.Tuple, ast.Assert, ast.Match)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and (node.id == 'state' or node.id.startswith('_mx_')):
            return 'name overlaps the current native receiver namespace'
        if isinstance(node, forbidden) or isinstance(node, ast.FunctionDef) and node is not fn:
            return 'effectful or object formula'
        if isinstance(node, ast.Attribute) and not (
                isinstance(node.value, ast.Name) and node.value.id == 'self'
                and not node.attr.startswith('_') and isinstance(node.ctx, ast.Load)):
            return 'non-scalar reference or attribute write'
        if isinstance(node, ast.Call) and node.keywords:
            return 'keyword dependency call'
        if isinstance(node, ast.Raise) and (node.cause is not None or
                not isinstance(node.exc, ast.Call) or
                not isinstance(node.exc.func, ast.Name) or
                any(not isinstance(a, ast.Constant) for a in node.exc.args)):
            return 'effectful exception construction'
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in BUILTINS:
            return 'unresolved or effectful call'
        # Keep the original power transformation until its contract is shared.
        if isinstance(node, ast.Pow):
            return 'power lowering not yet shared'
    return None


def plan(unit):
    """Use final original types/specifications; do not trace or rediscover Spaces."""
    classes = {c.name: c for c in ast.parse(unit.source).body if isinstance(c, ast.ClassDef)}
    plans, inventory = {}, []
    for cls_name, info in unit.module_info.classes.items():
        methods = {f.name: f for f in classes[cls_name].body if isinstance(f, ast.FunctionDef)}
        candidates, reasons, dependencies, reads = {}, {}, {}, {}
        for name, cell in info.cells.items():
            fn = methods.get('_f_' + name)
            reason = None
            if fn is None or not cell.has_typeinfo() or not cell.has_formula_def:
                reason = 'untraced or uncached Cell'
            elif cell.called_with_kwargs or cell.get_rettype_expr(c_style=True) not in TYPES:
                reason = 'object result or keyword public entry'
            elif any(cell.get_argtype_expr(p, c_style=True) != 'long long' for p in cell.params):
                reason = 'non-dense argument domain; retain original growing cache'
            elif len(cell.params) > 1:
                reason = 'multidimensional legacy storage migration not yet shared'
            else:
                reason = _reason(fn, info)
            if not reason and set(info.cells) & BUILTINS:
                reason = 'Cell/builtin name collision in free-function namespace'
            if not reason and cell.params:
                sizes = info.cells_arg_sizes[tuple(cell.params)]
                slots = 1
                for size in sizes:
                    slots *= size
                if any(size <= 0 or size > 1000000 for size in sizes) or slots > 20000000:
                    reason = 'dense layout outside current frame allocation contract'
            if not reason:
                dep = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
                       and isinstance(n.value, ast.Name) and n.value.id == 'self'}
                dependencies[name] = dep & info.cells.keys()
                reads[name] = dep - info.cells.keys()
                if any(k not in info.refs or info.refs[k].get_type_expr(c_style=True) not in TYPES
                       for k in reads[name]):
                    reason = 'non-scalar or unresolved input'
                else:
                    candidates[name] = fn
            if reason:
                reasons[name] = reason
        changed = True
        while changed:
            changed = False
            for name in list(candidates):
                missing = dependencies[name] - candidates.keys()
                if missing:
                    reasons[name] = 'dependency retains original Cython: ' + ', '.join(sorted(missing))
                    del candidates[name]
                    changed = True
        for name in info.cells:
            cell = info.cells[name]
            inventory.append({'space': info.fqname, 'cell': name,
                              'execution': 'frame_exact' if name in candidates else 'original_cython',
                              'reason': reasons.get(name),
                              'original_module_sha256': hashlib.sha256(unit.source.encode()).hexdigest(),
                              'formula_sha256': (hashlib.sha256(ast.unparse(methods['_f_' + name]).encode()).hexdigest()
                                                 if '_f_' + name in methods else None),
                              'public_return_type': cell.get_rettype_expr(c_style=True),
                              'public_arguments': {p: cell.get_argtype_expr(p, c_style=True) for p in cell.params}})
        if candidates:
            refs = set().union(*(reads[n] for n in candidates))
            plans[cls_name] = {'methods': candidates, 'info': info,
                               'inputs': {n: TYPES[info.refs[n].get_type_expr(c_style=True)]
                                          for n in sorted(refs)}}
    return plans, inventory


def emit_private_kernels(plans, package_path, package_name):
    """One build graph, with private kernel modules beneath the exported package."""
    from .compiler import build
    modules = []
    for cls_name, p in plans.items():
        source = ['from modelx_cython._frame_runtime.contracts import cell as _mx_shell_cell']
        domains = {}
        info = p['info']
        for name, original in sorted(p['methods'].items()):
            cell = info.cells[name]
            fn = copy.deepcopy(original)
            fn.name = name
            fn.args.args.pop(0)
            fn.decorator_list = [ast.Name('_mx_shell_cell', ast.Load())]
            for arg in fn.args.args:
                arg.annotation = ast.Name(TYPES[cell.get_argtype_expr(arg.arg, c_style=True)].__name__, ast.Load())
            fn.returns = ast.Name(TYPES[cell.get_rettype_expr(c_style=True)].__name__, ast.Load())
            fn = ast.fix_missing_locations(_UnbindFormula().visit(fn))
            source.append(ast.unparse(fn))
            if cell.params:
                domains[name] = {arg: (0, size) for arg, size in
                                 zip(cell.params, info.cells_arg_sizes[tuple(cell.params)])}
        name = '_mx_frame_' + cls_name.removeprefix('_c_')
        folder = package_path / name
        folder.mkdir()
        (folder / '__init__.py').write_text('', encoding='utf-8')
        formula_file = folder / '_formulas.py'
        formula_file.write_text('\n\n'.join(source) + '\n', encoding='utf-8')
        module_name = package_name + '.' + name + '._formulas'
        loader = importlib.util.spec_from_file_location(module_name, formula_file)
        module = importlib.util.module_from_spec(loader)
        sys.modules[module_name] = module
        try:
            loader.loader.exec_module(module)
            capacity = max([1] + [s for params in domains.values() for _, s in params.values()])
            spec = KernelSpec(module, p['inputs'], {}, max(1, capacity), '0',
                              cell_outputs=tuple(sorted(p['methods'])),
                              cache_domains=domains, original_space_shell=True)
            artifact = folder / 'artifact'
            build(spec, artifact, compile_native=False)
        finally:
            sys.modules.pop(module_name, None)
        manifest_file = artifact / 'manifest.json'
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        if manifest.get('python_cells'):
            raise ValueError('Original-shell admission disagrees with native emitter: ' + str(manifest['python_cells']))
        private = manifest['package']
        target = folder / private
        shutil.move(str(artifact / private), str(target))
        # Let original create_setup consume this extension in the same build.
        kernel = target / '_kernel.pyx'
        kernel.write_text('# distutils: language=c++\n' + kernel.read_text(encoding='utf-8'), encoding='utf-8')
        manifest['source_sha256'] = hashlib.sha256(kernel.read_bytes()).hexdigest()
        manifest['build_driver'] = 'original mx2cy package setup'
        manifest['original_space_shell'] = True
        manifest['package'] = package_name + '.' + name + '.' + private
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        p['kernel_module'] = manifest['package'] + '._kernel'
        modules.extend([Path(name) / private / '_lookups.py', Path(name) / private / '_kernel.pyx'])
    return modules


def transform(source, declarations, plans, original_methods):
    """Reuse original wrapper bodies on subclasses, original private ABI always."""
    if not plans:
        return source, declarations
    tree = ast.parse(source)
    owner_module = next(iter(plans.values()))['kernel_module'].split('.')[0] + '._mx_frame_owner'
    tree.body.insert(0, ast.parse(f'from {owner_module} import SpaceFrameOwner as _mx_SpaceFrameOwner').body[0])
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in plans:
            continue
        p = plans[cls.name]
        cls.body.insert(0, ast.AnnAssign(ast.Name('_mx_frame_owner', ast.Store()),
                                       ast.Name('object', ast.Load()), None, 1))
        marker = next(l for l in declarations.splitlines() if l.startswith('cdef class ' + cls.name + '('))
        declarations = declarations.replace(marker, marker + '\n    cdef object _mx_frame_owner', 1)
        inputs = '(' + ', '.join('self.' + k for k in p['inputs']) + (',' if p['inputs'] else '') + ')'
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            if fn.name in p['methods']:
                args = '(' + ', '.join(a.arg for a in fn.args.args[1:]) + (',' if len(fn.args.args) > 1 else '') + ')'
                head = ast.parse(f'''if type(self) is {cls.name}:
    if self._mx_frame_owner is None:
        self._mx_frame_owner = _mx_SpaceFrameOwner({p['kernel_module']!r}, {inputs})
    return self._mx_frame_owner.call({fn.name!r}, {args}, {inputs})''').body
                fn.body = head + fn.body
            elif fn.name.startswith('_f_') and fn.name[3:] in p['methods']:
                # Keep decorators and the original virtual declaration, but
                # restore its uncached body. Dependencies enter public wrappers.
                fn.body = copy.deepcopy(original_methods[cls.name][fn.name].body)
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n', declarations


def add_collect_hooks(source, plans):
    """Delegate lifetime to the existing exported collector and detachment rules."""
    tree = ast.parse(source)
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in plans:
            continue
        methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
        if 'collect' not in methods:
            continue
        snippets = {
            '_mx_collect_attach': '''_mx_frame_scratch = context.get((template, "frame"))
if _mx_frame_scratch is not None:
    _mx_frame_scratch.reset()
    self._mx_frame_owner = _mx_frame_scratch''',
            '_mx_collect_release': '''if self._mx_frame_owner is not None:
    context[(template, "frame")] = self._mx_frame_owner''',
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
    lines = ['    def bind_inputs(self, tuple record):']
    lines += [f'        self._frame.{name} = record[{i}]' for i, name in enumerate(manifest['inputs'])]
    if not manifest['inputs']:
        lines.append('        pass')
    lines += ['    def snapshot(self):', '        result = []', '        cdef Py_ssize_t i']
    for name, row in manifest['cells'].items():
        value = '_floats' if row['result']['kind'] == 'float' else '_ints'
        lines += [f"        for i in range({row['offset']}, {row['offset'] + row['slots']}):",
                  '            if self._frame._seen[i] == self._frame._epoch:',
                  f'                result.append((i, self._frame.{value}[i]))']
    lines += ['        return result', '    def restore(self, entries):', '        cdef Py_ssize_t i',
              '        for i, value in entries:']
    for j, row in enumerate(manifest['cells'].values()):
        value = '_floats' if row['result']['kind'] == 'float' else '_ints'
        lines += [f"            {'if' if j == 0 else 'elif'} {row['offset']} <= i < {row['offset'] + row['slots']}:",
                  f'                self._frame.{value}[i] = value']
    lines += ['            else: raise ValueError("Invalid completed cache slot")',
              '            self._frame._seen[i] = self._frame._epoch']
    return source.rstrip() + '\n\n' + '\n'.join(lines) + '\n'
