"""Adapted dev155 build harness for pristine latest modelx/modelx-cython.
Imports must be configured by the caller (see paths.json).
prepare returns (module, metadata, runtime); module.run_all() is timed.
Export/sample must already exist; sample_keys records the actual traced subset.
"""
from pathlib import Path
import ast, hashlib, json, time
from modelx_graph.modelx_cython_fast_backend import build_modelx_cython_fast_plan
from modelx_graph.mixed_recursive_native import MixedRecursiveCythonGenerator
from modelx_graph.build import build_extension


def _union_runner_source(program, output_targets, cover_output):
    """Expose additional requested roots from one covering native graph.

    The previous compiler historically selected one public output root even
    when its generated recursive program already contained the dependency
    Cells needed by other benchmark roots.  This helper adds thin batch
    drivers around those existing native functions so the harness can satisfy
    the one-build output-union contract without compiling a second module.

    This is reference-enabling donor code, not candidate architecture.
    """
    if not output_targets:
        return "", {cover_output: "run_all"}

    cells = {
        cell.source_name: cell
        for cell in program.formula_program.cells.values()
        if cell.execution_role == "formula"
    }
    runners = {cover_output: "run_all"}
    blocks = []

    dtype_info = {
        "float64": ("double", "np.float64"),
        "int64": ("long long", "np.int64"),
        "bool": ("unsigned char", "np.bool_"),
    }

    for target in output_targets:
        name = str(target["cell"])
        if name == cover_output:
            continue
        if name not in cells:
            raise RuntimeError(
                f"output union root {name!r} is not contained in covering graph {cover_output!r}"
            )
        cell = cells[name]
        if cell.dtype not in dtype_info:
            raise RuntimeError(f"output union root {name!r} has unsupported dtype {cell.dtype!r}")
        ctype, npdtype = dtype_info[cell.dtype]
        func = "run_output_" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
        call = target.get("call", {"kind": "no_args"})
        kind = call.get("kind", "no_args")

        if kind == "no_args":
            if cell.parameters:
                raise RuntimeError(
                    f"output union root {name!r} has parameters {cell.parameters!r} but target is no_args"
                )
            block = f'''\ncpdef object {func}():
    cdef cnp.ndarray arr = np.empty(_n, dtype={npdtype})
    cdef {ctype}[::1] outv = arr
    cdef Py_ssize_t i
    global _memo_error, _guard_error
    _memo_error = 0
    _guard_error = 0
    for i in range(_n):
        _next_epoch()
        outv[i] = r_{cell.uid}(i)
        if _memo_error != 0 or _guard_error != 0:
            break
    _fallback_raise_if_needed()
    if _memo_error != 0:
        raise RuntimeError(f'recursive memo failure code {{_memo_error}}')
    if _guard_error != 0:
        raise ValueError(f'native formula guard failed with code {{_guard_error}}')
    return arr
'''
        elif kind == "range_by_cell":
            if len(cell.parameters) != 1:
                raise RuntimeError(
                    f"range output union root {name!r} must have exactly one logical parameter"
                )
            stop_name = str(call["stop_cell"])
            stop = cells.get(stop_name)
            if stop is None or stop.parameters or stop.dtype != "int64":
                raise RuntimeError(
                    f"range output union root {name!r} requires scalar int64 stop cell {stop_name!r}"
                )
            start = int(call.get("start", 0))
            block = f'''\ncpdef object {func}():
    cdef list result = []
    cdef Py_ssize_t i
    cdef long long t
    cdef long long stop
    cdef cnp.ndarray arr
    cdef {ctype}[::1] outv
    global _memo_error, _guard_error
    _memo_error = 0
    _guard_error = 0
    for i in range(_n):
        _next_epoch()
        stop = <long long>r_{stop.uid}(i)
        if stop < {start}:
            raise ValueError('invalid output-union range')
        arr = np.empty(stop - {start}, dtype={npdtype})
        outv = arr
        for t in range({start}, stop):
            outv[t - {start}] = r_{cell.uid}(i, t)
            if _memo_error != 0 or _guard_error != 0:
                break
        if _memo_error != 0 or _guard_error != 0:
            break
        result.append(arr)
    _fallback_raise_if_needed()
    if _memo_error != 0:
        raise RuntimeError(f'recursive memo failure code {{_memo_error}}')
    if _guard_error != 0:
        raise ValueError(f'native formula guard failed with code {{_guard_error}}')
    return result
'''
        else:
            raise RuntimeError(f"unsupported output-union call kind {kind!r} for {name!r}")
        blocks.append(block)
        runners[name] = func
    return "".join(blocks), runners

def compile_exported(model, space, output_name, keys, export_path, sample_path, outdir, *, sample_keys=None, compatibility_mode='dev155_observed_trace_metadata_v1', capture_unmodified=False, output_targets=None):
    if compatibility_mode not in ('none', 'observed_trace_geometry', 'dev155_observed_trace_metadata_v1'):
        raise ValueError('Unknown compatibility_mode')
    export_path, sample_path, outdir = map(Path, (export_path, sample_path, outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    import modelx_graph
    compiler_root = Path(modelx_graph.__file__).parent
    compiler_digest = hashlib.sha256()
    for compiler_file in sorted(compiler_root.rglob('*.py')):
        compiler_digest.update(compiler_file.relative_to(compiler_root).as_posix().encode('utf-8'))
        compiler_digest.update(b'\0')
        compiler_digest.update(compiler_file.read_bytes())
        compiler_digest.update(b'\0')
    compiler_source_sha256 = compiler_digest.hexdigest()
    if isinstance(output_name, str):
        output_names = (output_name,)
    else:
        output_names = tuple(str(x) for x in output_name)
        if not output_names:
            raise ValueError('output_name union must not be empty')
    # The public report is the final H0.4/H0.5 root and covers the numeric
    # calculation graph below it.  Compile that graph once and expose other
    # requested roots from the same module.
    cover_output = output_names[-1]

    # Class name is exporter-defined: inspect all modules rather than assume Projection.
    class_name = '_c_' + space.name
    candidates = []
    for path in export_path.rglob('*.py'):
        if path.name in ('__init__.py', '_mx_sys.py', '_mx_io.py'): continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        if any(isinstance(n, ast.ClassDef) and n.name == class_name for n in tree.body):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f'Ambiguous exported class {class_name}: {candidates}')
    classes_path = candidates[0]
    module_name = '.'.join((export_path.name, *classes_path.relative_to(export_path).with_suffix('').parts))
    started = time.perf_counter()
    def make_plan():
        import sys
        for name in list(sys.modules):
            if name == export_path.name or name.startswith(export_path.name + '.'):
                del sys.modules[name]
        return build_modelx_cython_fast_plan(classes_path, model_path=export_path,
        sample_path=sample_path, spec_data={}, model=model, space=space,
        output_name=cover_output, sample_keys=tuple(keys if sample_keys is None else sample_keys), run_keys=tuple(keys),
        class_name=class_name, module_name=module_name)
    unmodified = {'status': 'not_checked', 'reason': 'avoid_duplicate_full_trace'}
    import modelx_graph.modelx_cython_fast_backend as backend
    from modelx_cython.cli import run_sample as upstream_run_sample
    adapted_run_sample = backend.run_sample
    if capture_unmodified or compatibility_mode == 'none':
        try:
            backend.run_sample = upstream_run_sample
            plan = make_plan()
            (outdir/'dev_unmodified_plan_manifest.json').write_text(json.dumps(plan.manifest(), indent=2, default=str))
            program = plan.build_mixed_recursive_program()
            unmodified = {'status': 'supported' if program.supported else 'unsupported', 'blockers': program.blockers}
        except Exception as exc:
            unmodified = {'status': 'error', 'type': type(exc).__name__, 'message': str(exc)}
            (outdir/'dev_unmodified_status.json').write_text(json.dumps(unmodified, indent=2))
            if compatibility_mode == 'none':
                raise
        finally:
            backend.run_sample = adapted_run_sample
    (outdir/'dev_unmodified_status.json').write_text(json.dumps(unmodified, indent=2))
    if compatibility_mode != 'none':
        plan = make_plan()
        compatibility_mode = 'dev155_observed_trace_metadata_v1'
    logger = plan.module_info.logger
    (outdir/'dev_trace_geometry.json').write_text(json.dumps(
        getattr(logger, 'compatibility_evidence', {}), indent=2, default=str))
    (outdir/'dev_plan_manifest.json').write_text(json.dumps(plan.manifest(), indent=2, default=str))
    program = plan.build_mixed_recursive_program()
    metadata = {'plan_seconds':time.perf_counter()-started,
                'compiler_source_sha256':compiler_source_sha256,
                'compiler_hash_scope':'modelx_graph relative .py paths and bytes at prepare entry',
                'compatibility_mode':compatibility_mode,
                'unmodified_frontend_status':unmodified,
                'execution_classes':plan.execution_class_plan.manifest(),
                'program':program.manifest(),
                'compile_optimization':'O2','openmp':False,'native_arch':False,
                'requested_output_roots': list(output_names),
                'cover_output_root': cover_output}
    (outdir/'dev_manifest.json').write_text(json.dumps(metadata, indent=2, default=str))
    if not program.supported:
        raise RuntimeError(f'modelx-cython extension unsupported: {program.blockers}')
    import uuid
    tag = 'mxe_' + uuid.uuid4().hex[:16]
    pyx = outdir/(tag+'.pyx')
    generator = MixedRecursiveCythonGenerator(program)
    source = generator.source(tag)
    extra_source, output_runners = _union_runner_source(program, output_targets, cover_output)
    if len(output_names) > 1:
        missing = [name for name in output_names if name not in output_runners]
        if missing:
            raise RuntimeError(
                f"single covering graph {cover_output!r} cannot expose requested output roots {missing!r}"
            )
    source += extra_source
    pyx.write_text(source, encoding='utf-8')
    declarations = generator.declaration_source()
    for _name, runner in output_runners.items():
        if runner != 'run_all':
            declarations += f'cpdef object {runner}()\n'
    pyx.with_suffix('.pxd').write_text(declarations, encoding='utf-8')
    metadata['output_runners'] = output_runners
    metadata['generated_pyx_sha256'] = hashlib.sha256(pyx.read_bytes()).hexdigest()
    (outdir/'source_formulas.json').write_text(json.dumps(
        {uid: ast.unparse(fn) for uid,fn in plan.formula_source_functions.items()}, indent=2))
    started = time.perf_counter()
    module, artifact, proc = build_extension(pyx, tag, outdir/'build',
        openmp=False, optimization='O2', native_arch=False)
    metadata['compile_seconds'] = time.perf_counter()-started
    metadata['artifact'] = str(artifact)
    (outdir/'build.log').write_text(proc.stdout+'\n'+proc.stderr)
    binding_started = time.perf_counter()
    bound = plan.bind_source_preserving_inputs()
    prepared_inputs = {spec.key:bound[spec.key] for spec in program.formula_program.inputs}
    module.init_inputs(prepared_inputs)
    runtime = None
    if hasattr(module, 'bind_fallback_runtime'):
        runtime = plan.build_fallback_runtime(model=model, space=space)
        module.bind_fallback_runtime(runtime)
    def reset():
        module.init_inputs(prepared_inputs)
        if runtime is not None:
            module.bind_fallback_runtime(runtime)
    module.reset = reset
    module._census_reset = reset
    metadata['binding_seconds'] = time.perf_counter() - binding_started
    metadata['initial_fallback_stats'] = runtime.stats() if runtime else None
    (outdir/'dev_manifest.json').write_text(json.dumps(metadata, indent=2, default=str))
    return module, metadata, runtime




