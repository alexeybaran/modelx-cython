# Copyright (c) 2023-2025 Fumito Hamamura <fumito.ham@gmail.com>

# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation version 3.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library.  If not, see <http://www.gnu.org/licenses/>.

"""Command-line interface for mx2cy.

This module implements the ``mx2cy`` command, which translates an
exported modelx model (a pure-Python package created by modelx's
``Model.export``) into a Cython package and compiles it.

Unless ``--compile-only`` is given, :func:`main_handler` copies the
model directory to ``<name>_cy``, runs the user's sample script under
tracing to collect runtime type information, rewrites the model
modules with Cython pure-Python-mode annotations (every space module,
traced or not; ``_mx_model`` modules only when traced), generates
matching ``.pxd`` files, and writes a ``setup.py``.  Unless ``--translate-only`` is
given, it then runs ``setup.py build_ext --inplace`` to build the
extension modules in place.
"""

import sys
import os
import textwrap
import pathlib
import shutil
import runpy
import ast
import argparse
import json
import dataclasses
import logging
import subprocess
from typing import IO, TYPE_CHECKING, Sequence, Optional, Tuple

from modelx_cython.consts import MX_MODEL_MOD, MX_SPACE_MOD, MX_SYS_MOD
from modelx_cython.config import TransSpec
from modelx_cython.tracer import trace_calls, MxCallTraceLogger, MxCodeFilter
from modelx_cython.builder import ModuleInfo
from modelx_cython.parser import ModuleVisitor
from modelx_cython.transformer import ModuleTransformer, PXDGenerator
from modelx_cython.usage import (
    analyze_usage, apply_verdicts, build_resolver)
from modelx_cython.powers import build_kind_maps

_logger = logging.getLogger(__name__)


def increment_backups(
        base_path: pathlib.Path,
        max_backups=1, nth=0):
    """Rotate backup copies of ``base_path``.

    If ``base_path`` exists, rename it to ``<base_path>_BAK1``, after
    first shifting any existing ``<base_path>_BAK<n>`` copy to
    ``_BAK<n+1>`` by calling itself recursively with ``nth + 1``.  The
    oldest copy, ``<base_path>_BAK<max_backups>``, is deleted
    (recursively if it is a directory) instead of being renamed
    further.

    Parameters
    ----------
    base_path : pathlib.Path
        Path whose backup copies are rotated.
    max_backups : int, optional
        Number of ``_BAK`` copies to keep (default 1).
    nth : int, optional
        Backup generation currently being shifted; used internally by
        the recursion.  Callers should leave this at 0.

    Raises
    ------
    ValueError
        If a path to be removed is neither a file nor a directory.
    """

    postfix = "_BAK" + str(nth) if nth else ""
    backup_path = pathlib.Path(str(base_path) + postfix)
    if backup_path.exists():
        if nth == max_backups:
            if backup_path.is_dir():
                shutil.rmtree(backup_path)
            elif backup_path.is_file():
                backup_path.unlink()
            else:
                raise ValueError("cannot remove '%s'" % str(backup_path))
        else:
            increment_backups(base_path, max_backups, nth + 1)
            next_backup = pathlib.Path(str(base_path) + "_BAK" + str(nth + 1))
            backup_path.rename(next_backup)


def run_sample(model_path: pathlib.Path, sample_path: str, new_model_name: str = None) -> MxCallTraceLogger:
    """Run the user's sample script while tracing calls into the model.

    Temporarily prepends the model's parent directory to ``sys.path``,
    then executes ``sample_path`` with :func:`runpy.run_path` (as
    ``__main__``) under :func:`~modelx_cython.tracer.trace_calls`, so
    that calls into modules of the ``model_path`` package are recorded
    by an :class:`~modelx_cython.tracer.MxCallTraceLogger`.

    Parameters
    ----------
    model_path : pathlib.Path
        Path to the exported model package to trace.
    sample_path : str
        Path to the sample script to execute.
    new_model_name : str, optional
        If given, the model (package) name recorded in the collected
        trace information is replaced with this name (e.g.
        ``<name>_cy``).

    Returns
    -------
    MxCallTraceLogger
        Logger holding the per-cells type information collected during
        the run.
    """

    module: str = model_path.name
    try:
        module_path = str(model_path.parent)
        sys.path.insert(0, module_path)

        logger = MxCallTraceLogger(module=module, new_model_name=new_model_name)
        with trace_calls(
            module=module,
            logger=logger,
            max_typed_dict_size=0,
            code_filter= MxCodeFilter()):
            runpy.run_path(sample_path, run_name="__main__")

    finally:
        assert sys.path.pop(0) == module_path

    return logger


class HandlerError(Exception):
    """Error type reserved for failures in the command-line handlers."""
    pass


@dataclasses.dataclass
class _TransUnit:
    """One module's parsed state between the build and transform phases.

    Attributes
    ----------
    fqname : str
        Fully qualified module name rooted at the model package name.
    source : str
        Original source code of the module.
    visitor : ModuleVisitor
        Lexical information parsed from ``source``.
    module_info : ModuleInfo
        Combined lexical, runtime and spec information for the module.
    abs_src_path : pathlib.Path
        Absolute path of the module's ``.py`` file.
    rel_src_path : pathlib.Path
        Path of the ``.py`` file relative to the working directory,
        as listed in the generated ``setup.py``.
    abs_pxd_path : pathlib.Path
        Absolute path of the ``.pxd`` file to generate.
    abs_init_path : pathlib.Path
        Absolute path of the ``__init__.pxd`` file to generate in the
        module's package directory.
    """
    fqname: str
    source: str
    visitor: ModuleVisitor
    module_info: ModuleInfo
    abs_src_path: pathlib.Path
    rel_src_path: pathlib.Path
    abs_pxd_path: pathlib.Path
    abs_init_path: pathlib.Path


def iter_module_files(model_path: pathlib.Path):
    """Yield (module fqname, path) for every model/space module under
    model_path. fqnames are rooted at model_path.name.

    Parameters
    ----------
    model_path : pathlib.Path
        Root directory of the model package to scan.

    Yields
    ------
    tuple of (str, pathlib.Path)
        Dotted module name and file path of each ``_mx_model.py`` and
        ``_mx_classes.py`` file, in sorted path order for each of the
        two module kinds.
    """
    for name in (MX_MODEL_MOD, MX_SPACE_MOD):
        for path in sorted(model_path.rglob(name + ".py")):
            rel = path.relative_to(model_path.parent).with_suffix("")
            yield ".".join(rel.parts), path


def main_handler(args: argparse.Namespace, stdout: IO[str], stderr: IO[str]) -> int:
    """Run the translate and compile pipeline for a parsed command line.

    Resolves ``args.model_path`` and derives the output package
    directory ``<name>_cy`` next to it, along with the ``setup.py``
    location (``args.setup`` if given, otherwise ``setup.py`` in the
    same parent directory).

    Unless ``args.compile_only`` is set, performs the translation:
    rotates backups of the output directory, copies the model there,
    copies the bundled ``_mx_sys.pxd`` into it, runs the sample script
    under tracing via :func:`run_sample`, and reads the spec file with
    :func:`ast.literal_eval` (skipped when ``args.no_spec`` is set; a
    missing spec file re-raises :class:`FileNotFoundError` with a hint
    to use ``--no-spec``).  Translation proper then runs in three
    phases: (1) parse every model/space module and build its
    :class:`~modelx_cython.builder.ModuleInfo` -- all space modules
    are included even when the sample never exercised their cells
    (other modules may reference their classes in ``.pxd``
    declarations), while an ``_mx_model`` module is included only
    when traced, since the transformer does not handle the model
    class; (2) statically analyze how array-returning cells are
    consumed and apply the verdicts so unsafe memoryview return types
    fall back to ``object``, and mark cells that are called with
    keyword arguments anywhere in the model so their public methods
    stay plain Python methods (C-level calls are positional-only);
    (3) rewrite each included module with
    :class:`~modelx_cython.transformer.ModuleTransformer`, write its
    ``.pxd`` and ``__init__.pxd`` files, and finally write the
    generated ``setup.py`` via :func:`create_setup`.  Model sources
    are read and written as UTF-8 regardless of locale.

    Unless ``args.translate_only`` is set, compiles the result with
    :func:`compile_main`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (see :func:`main`).
    stdout : IO[str]
        Output stream; accepted for interface compatibility but not
        used.
    stderr : IO[str]
        Error stream; accepted for interface compatibility but not
        used.

    Returns
    -------
    int
        0 when only translating, otherwise the exit code of the
        compilation subprocess.

    Raises
    ------
    FileNotFoundError
        If translation is performed (``--compile-only`` not set), the
        spec file does not exist, and ``--no-spec`` was not given.
    """

    orig_path = pathlib.Path(args.model_path).resolve()
    if getattr(args, "frame_collect", False):
        return _frame_collect_handler(args, orig_path)
    work_dir = orig_path.parent
    model_name = orig_path.name + "_cy"
    model_path = work_dir / model_name
    setup_file = pathlib.Path(args.setup) if args.setup else work_dir / "setup.py"

    if not args.compile_only:
        increment_backups(model_path)
        shutil.copytree(orig_path, model_path)
        shutil.copy(pathlib.Path(__file__).parent / (MX_SYS_MOD + ".pxd"), model_path)

        logger = run_sample(orig_path, args.sample, new_model_name=model_name)
        if args.no_spec:
            d = {}
        else:
            try:
                d = ast.literal_eval(pathlib.Path(args.spec).read_text(encoding="utf-8"))
            except FileNotFoundError as e:
                raise FileNotFoundError(f"{e}. Add '--no-spec' to omit the spec file.") from e

        spec = TransSpec(d)
        rel_model_path = model_path.relative_to(model_path.parent)

        modules = [rel_model_path / (MX_SYS_MOD + ".py")]

        # Phase 1: parse all sources and build all module infos.
        # Every model/space module is translated, including those whose
        # cells the sample never exercised (e.g. enum-only spaces):
        # other modules may reference their classes in pxd declarations,
        # so their pxd files must exist.
        units = []
        for m, src_path in iter_module_files(model_path):
            subs = m.split(".")
            assert subs.pop(0) == model_path.name
            assert subs[-1] in [MX_MODEL_MOD, MX_SPACE_MOD]
            if subs[-1] == MX_MODEL_MOD and m not in logger.modules:
                # The model module defines the model class, which the
                # transformer does not handle; it is translated only when
                # the sample traced cells in it.
                continue
            pxd_path = subs.copy()
            subs[-1] = subs[-1] + ".py"
            pxd_path[-1] = pxd_path[-1] + ".pxd"
            abs_src_path = model_path / "/".join(subs)
            rel_src_path = rel_model_path / "/".join(subs)
            abs_pxd_path = model_path / "/".join(pxd_path)
            abs_init_path = model_path / "/".join(subs[:-1] + ["__init__.pxd"])
            source = abs_src_path.read_text(encoding="utf-8")
            visitor = ModuleVisitor(module=m, source=source)
            module_info = ModuleInfo(m, visitor, logger, spec)
            for cls in sorted(visitor.locked_classes):
                if cls in module_info.classes:
                    _logger.info(
                        f"{m}.{cls} is locked: its cells run their formulas "
                        "under the model lock")
            units.append(_TransUnit(
                m, source, visitor, module_info,
                abs_src_path, rel_src_path, abs_pxd_path, abs_init_path))

        # Phase 2: classify cross-module usage of array-returning cells
        # so that get_rettype_expr can fall back to object where needed.
        pairs = [(u.visitor, u.module_info) for u in units]
        apply_verdicts(pairs, analyze_usage(pairs))

        # Cells called with keyword arguments anywhere in the model must
        # keep plain Python public methods: Cython C-level calls are
        # positional-only.
        kwarg_names = set().union(
            *(u.visitor.kwarg_called_names for u in units))
        for u in units:
            for cls_info in u.module_info.classes.values():
                for cells in cls_info.cells.values():
                    if cells.name in kwarg_names:
                        cells.called_with_kwargs = True
                        _logger.info(
                            f"{cells.fqname} stays a Python method because "
                            "a call with keyword arguments matches its name")

        # Every cells return type is final by now, so the operands of a
        # '**' can be typed: see modelx_cython.powers.
        module_infos = {u.module_info.fqname: u.module_info for u in units}
        resolver = build_resolver(module_infos)
        cells_kinds, ref_kinds = build_kind_maps(module_infos)

        if d.get("bind_enum_members", True):
            from modelx_cython.enum_bindings import configure as configure_enum_bindings
            configure_enum_bindings(units)

        if d.get("experimental_helpers", False):
            from modelx_cython.experimental_helpers import configure, emit_runtime
            configure(units)
            modules.extend(emit_runtime(model_path, units))
            helper_inventory = [row for u in units for cls in u.module_info.classes.values()
                                for row in cls.helper_info.inventory]
            (model_path / "_mx_helper_inventory.json").write_text(
                json.dumps({"schema": 1, "calls": helper_inventory}, indent=2), encoding="utf-8")

        # Final type/reference/cache authority shared by every emission choice.
        from modelx_cython.compilation_plan import build_package_plan
        package_plan = build_package_plan(units)
        (model_path / '_mx_compilation_plan.json').write_text(
            json.dumps(package_plan.inventory(), indent=2), encoding='utf-8')

        # Phase 3: transform and write out
        for u in units:
            trans = ModuleTransformer(
                u.source, u.module_info, resolver=resolver,
                cells_kinds=cells_kinds, ref_kinds=ref_kinds)
            shell_plans = {}
            if os.environ.get('MX_CY_ORIGINAL_FRAME_SHELL', '0') == '1':
                from modelx_cython._frame_runtime.space_shell import plan, emit_private_kernels
                shell_plans, shell_inventory = plan(u, package_plan=package_plan)
                u.abs_src_path.with_suffix('.frame-shell.json').write_text(
                    json.dumps({'schema': 1, 'cells': shell_inventory}, indent=2), encoding='utf-8')
                private_modules = emit_private_kernels(
                    shell_plans, u.abs_src_path.parent, u.fqname.rsplit('.', 1)[0],
                    formula_transform=trans._rewrite_powers)
                modules.extend(u.rel_src_path.parent / m for m in private_modules)
                if shell_plans:
                    shutil.copy(pathlib.Path(__file__).parent / '_frame_runtime' / 'space_owner.py',
                                model_path / '_mx_frame_owner.py')
                    # Existing checked receiver/codec utilities, without a
                    # CompiledModel/Space facade or an installed compiler.
                    support = model_path / '_mx_frame_support'
                    support.mkdir(exist_ok=True)
                    (support / '__init__.py').write_text('', encoding='utf-8')
                    support_names = ['python_calculator.py', 'contracts.py', 'enum_binding.py']
                    if any(p.get('prepared_spec') is not None for p in shell_plans.values()):
                        support_names.extend(['prepared_space_inputs.py', 'input_codec.py',
                                              'data_binding.py', 'lookups.py'])
                    for name in support_names:
                        shutil.copy(pathlib.Path(__file__).parent / '_frame_runtime' / name,
                                    support / name)
            pxd = PXDGenerator(u.module_info)

            u.abs_src_path.write_text(trans.transformed.code, encoding="utf-8")
            u.abs_pxd_path.write_text(pxd.code, encoding="utf-8")
            from modelx_cython.numeric_cache import transform as cache_transform
            cache_py, cache_pxd = cache_transform(
                u.abs_src_path.read_text(encoding="utf-8"), pxd.code, u.module_info,
                u.abs_src_path.with_name("_mx_numeric_cache.pxd"))
            u.abs_src_path.write_text(cache_py, encoding="utf-8")
            u.abs_pxd_path.write_text(cache_pxd, encoding="utf-8")
            if u.abs_src_path.with_name("_mx_numeric_cache.py").exists():
                modules.append(u.rel_src_path.with_name("_mx_numeric_cache.py"))


            from modelx_cython.checked_exceptions import transform as e4_transform
            import ast as e4_ast
            e4_src = u.abs_src_path.read_text(encoding="utf-8")
            e4_tree = e4_ast.parse(e4_src)
            if any(isinstance(c, e4_ast.ClassDef) and any(isinstance(f, e4_ast.FunctionDef) and not f.name.startswith('__') and f.returns is not None and e4_ast.unparse(f.returns) in ('_mx_cy.double', '_mx_cy.longlong') and any(e4_ast.unparse(d) in ('_mx_cy.ccall', '_mx_cy.cfunc') for d in f.decorator_list) for f in c.body) for c in e4_tree.body):
                e4_py, e4_pxd, _ = e4_transform(e4_src, u.abs_pxd_path.read_text(encoding="utf-8"))
                u.abs_src_path.write_text(e4_py, encoding="utf-8")
                u.abs_pxd_path.write_text(e4_pxd, encoding="utf-8")

            # Qualified native execution; explicit zero retains diagnostic comparison paths.
            import os as _connected_os
            if _connected_os.environ.get("MX_CY_CONNECTED_NATIVE", "1") == "1":
                from modelx_cython.object_key_cache import transform as index_transform
                index_py, index_pxd = index_transform(
                    u.abs_src_path.read_text(encoding="utf-8"),
                    u.abs_pxd_path.read_text(encoding="utf-8"), u.module_info,
                    u.abs_src_path.with_suffix(".object-index.json"))
                u.abs_src_path.write_text(index_py, encoding="utf-8")
                u.abs_pxd_path.write_text(index_pxd, encoding="utf-8")
                if u.abs_src_path.with_name("_mx_object_index.py").exists():
                    modules.append(u.rel_src_path.with_name("_mx_object_index.py"))

            # Native execution beneath ordinary Space/Cell entry points.
            import os as _native_os
            shell_original_methods = {}
            if shell_plans:
                shell_original_methods = {
                    cls.name: {fn.name: fn for fn in cls.body if isinstance(fn, ast.FunctionDef)}
                    for cls in ast.parse(u.abs_src_path.read_text(encoding='utf-8')).body
                    if isinstance(cls, ast.ClassDef)}
            if _native_os.environ.get("MX_CY_INTEGRATED_NATIVE", "1") == "1":
                from modelx_cython.integrated_native import transform as native_transform
                native_py, native_pxd = native_transform(
                    u.abs_src_path.read_text(encoding="utf-8"),
                    u.abs_pxd_path.read_text(encoding="utf-8"), u.module_info,
                    u.abs_src_path.with_suffix(".native.json"),
                    frame_cells={name: p['methods'] for name, p in shell_plans.items()} or None)
                u.abs_src_path.write_text(native_py, encoding="utf-8")
                u.abs_pxd_path.write_text(native_pxd, encoding="utf-8")
            if shell_plans:
                from modelx_cython._frame_runtime.space_shell import transform as shell_transform
                shell_py, shell_pxd = shell_transform(
                    u.abs_src_path.read_text(encoding='utf-8'),
                    u.abs_pxd_path.read_text(encoding='utf-8'), shell_plans, shell_original_methods)
                u.abs_src_path.write_text(shell_py, encoding='utf-8')
                u.abs_pxd_path.write_text(shell_pxd, encoding='utf-8')

            # Reuse scratch cache owners only for exported collect traversals.
            if _native_os.environ.get("MX_CY_COLLECT_REUSE", "1") == "1":
                from modelx_cython.collect_reuse import transform as collect_transform
                collect_py, collect_pxd = collect_transform(
                    u.abs_src_path.read_text(encoding="utf-8"),
                    u.abs_pxd_path.read_text(encoding="utf-8"),
                    u.abs_src_path.with_suffix(".collect.json"))
                u.abs_src_path.write_text(collect_py, encoding="utf-8")
                u.abs_pxd_path.write_text(collect_pxd, encoding="utf-8")

            if shell_plans and _native_os.environ.get('MX_CY_COLLECT_REUSE', '1') == '1':
                from modelx_cython._frame_runtime.space_shell import add_collect_hooks
                u.abs_src_path.write_text(add_collect_hooks(
                    u.abs_src_path.read_text(encoding='utf-8'), shell_plans), encoding='utf-8')

            # Experiment-only native traversal for one typed numeric sum leaf.
            if _native_os.environ.get("MX_CY_COLLECT_NATIVE", "0") == "1":
                from modelx_cython.collect_native import transform as native_collect_transform
                collect_py, collect_pxd = native_collect_transform(
                    u.abs_src_path.read_text(encoding="utf-8"),
                    u.abs_pxd_path.read_text(encoding="utf-8"),
                    u.abs_src_path.with_suffix(".collect-native.json"))
                u.abs_src_path.write_text(collect_py, encoding="utf-8")
                u.abs_pxd_path.write_text(collect_pxd, encoding="utf-8")

            u.abs_init_path.write_text("from . cimport _mx_classes", encoding="utf-8")
            modules.append(u.rel_src_path)

        if _native_os.environ.get("MX_CY_ORIGINAL_FRAME_SHELL", "0") == "1":
            from modelx_cython.prepared_io_cache import emit as emit_prepared_io
            emit_prepared_io(model_path, source_package=orig_path)
        create_setup(model_name, modules=modules, setup_file=setup_file)

    if args.translate_only:
        return 0
    else:
        return compile_main(work_dir, setup_file)


def _frame_collect_handler(args: argparse.Namespace, orig_path: pathlib.Path) -> int:
    """Build the experimental native frame from an ordinary exported model.

    Keep this entry separate from the established translator.  Unsupported
    model/collect shapes fail during source import; they never fall through to
    a different execution or cache path after the user requested the frame.
    The source handoff is retained beside the generated package for audit.
    """
    if not args.no_spec:
        raise HandlerError("--frame-collect requires --no-spec")
    if args.setup:
        raise HandlerError("--frame-collect generates its own setup.py")

    from modelx_cython._frame_runtime.modelx_frontend_exported import open_exported_source
    from modelx_cython._frame_runtime.trace_contracts import from_mx2cy_trace
    from modelx_cython._frame_runtime.compiler import build, compile_generated

    destination = orig_path.parent / (orig_path.name + "_cy")
    source_dir = orig_path.parent / (orig_path.name + "_cy_sources")
    if args.compile_only:
        compile_generated(destination)
        return 0
    if destination.exists() or source_dir.exists():
        raise FileExistsError("Frame output or source directory already exists")
    logger = run_sample(orig_path, args.sample)
    with open_exported_source(orig_path) as derive:
        contracts = from_mx2cy_trace(
            logger, derive.source_manifest, space=args.frame_space)
        spec = derive(source_dir, projection=args.frame_space,
                      point_table=args.frame_point_table,
                      steps=args.frame_steps, contracts=contracts)
    if args.translate_only:
        build(spec, destination, compile_native=False)
    else:
        build(spec, destination)
    return 0


# Mapping from string names to logging levels
LOG_LEVELS = {
    'notset': logging.NOTSET,
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


# Custom type function for argparse
def parse_log_level(value):
    """Convert a command-line value to a :mod:`logging` level.

    Used as the argparse ``type`` for ``--log-level``.  Accepts either
    a standard numeric level (e.g. ``"10"``) or a case-insensitive
    level name (e.g. ``"debug"``).

    Parameters
    ----------
    value : str
        Command-line argument to interpret.

    Returns
    -------
    int
        The corresponding logging level.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``value`` is neither a recognized numeric level nor a
        recognized level name.
    """
    try:
        # Try interpreting as an integer
        level = int(value)
        if level in LOG_LEVELS.values():
            return level
        raise ValueError
    except ValueError:
        # Try interpreting as a string key
        level_name = value.lower()
        if level_name in LOG_LEVELS:
            return LOG_LEVELS[level_name]
        raise argparse.ArgumentTypeError(f"Invalid log level: {value}")


def compile_main(work_dir: pathlib.Path, setup_file: pathlib.Path) -> int:
    """Compile the translated model with the generated ``setup.py``.

    Runs ``sys.executable <setup_file> build_ext --inplace`` in a
    subprocess with ``work_dir`` as the working directory and with
    ``work_dir`` prepended to the subprocess's ``PYTHONPATH``.

    Parameters
    ----------
    work_dir : pathlib.Path
        Directory containing the translated model package.
    setup_file : pathlib.Path
        Path to the ``setup.py`` script to run.

    Returns
    -------
    int
        Exit code of the subprocess.
    """

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = subprocess.run([sys.executable, str(setup_file), "build_ext", "--inplace"],
                         env=env, cwd=str(work_dir))
    return cmd.returncode


def main(argv: Sequence[str], stdout: IO[str], stderr: IO[str]) -> int:
    """Parse command-line arguments and run :func:`main_handler`.

    Defines the ``mx2cy`` argument parser: positional ``model_path``,
    ``--sample``, mutually exclusive ``--spec``/``--no-spec``,
    ``--setup``, mutually exclusive
    ``--translate-only``/``--compile-only``, and ``--log-level``.
    Configures the root logger with :func:`logging.basicConfig` at the
    requested level before delegating to :func:`main_handler`.

    Parameters
    ----------
    argv : Sequence[str]
        Command-line arguments, excluding the program name.
    stdout : IO[str]
        Output stream passed through to :func:`main_handler`.
    stderr : IO[str]
        Error stream passed through to :func:`main_handler`.

    Returns
    -------
    int
        Exit code returned by :func:`main_handler`.
    """

    parser = argparse.ArgumentParser(
        description="Translate an exported modelx model into Cython and compile it.",
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to an exported modelx model to translate into Cython",
    )

    parser.add_argument(
        "--sample",
        type=str,
        default="sample.py",
        help=(
            "Path to a sample file to run for collecting type information (default: sample.py)"
        )
    )

    spec_group = parser.add_mutually_exclusive_group()

    spec_group.add_argument(
        "--spec",
        type=str,
        default="spec.py",
        help=(
            "Path to a spec file for setting parameters (default: spec.py)"
        )
    )

    spec_group.add_argument(
        "--no-spec",
        action="store_true",
        default=False,
        help="Skip the spec file (default: False)"
    )

    parser.add_argument(
        "--setup",
        type=str,
        default="",
        help=(
            "Path to a setup file for Cython (default: setup.py)"
        )
    )

    parser.add_argument(
        "--frame-collect", action="store_true", default=False,
        help="Build the experimental source-preserved native collect frame",
    )
    parser.add_argument(
        "--frame-space", default="Projection",
        help="collect-enabled Space name for --frame-collect (default: Projection)",
    )
    parser.add_argument(
        "--frame-point-table", default=None,
        help="policy table name for --frame-collect when inference is ambiguous",
    )
    parser.add_argument(
        "--frame-steps", type=int, default=2048,
        help="compiled period capacity for --frame-collect (default: 2048)",
    )

    task_group = parser.add_mutually_exclusive_group()

    task_group.add_argument(
        "--translate-only",
        action="store_true",
        default=False,
        help="Perform translation only (default: False)",
    )

    task_group.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Perform compilation only (default: False)",
    )

    parser.add_argument(
        '--log-level',
        default=logging.WARNING,
        type=parse_log_level,
        help='Logging level: NOTSET(0), DEBUG(10), INFO(20), WARNING(30), ERROR(40), CRITICAL(50) (default: WARNING)'
    )
    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(level=args.log_level)

    return main_handler(args, stdout, stderr)


def create_setup(model_name: str, modules: Sequence[str], setup_file: pathlib.Path):
    """Write a ``setup.py`` that cythonizes the translated modules.

    The generated script calls :func:`setuptools.setup` with
    ``name=model_name`` and ``ext_modules`` produced by
    ``cythonize(..., annotate=True)`` over the given module paths, with
    the ``freethreading_compatible`` directive set so that importing the
    compiled model on a free-threaded build of Python does not re-enable
    the GIL (the directive needs Cython 3.1 or later and does nothing on
    a build with the GIL).  Any existing file at ``setup_file`` is
    overwritten.

    Parameters
    ----------
    model_name : str
        Package name of the translated model (e.g. ``<name>_cy``).
    modules : Sequence[pathlib.PurePath]
        Paths of the ``.py`` files to compile, relative to the
        directory where the script is run.  Despite the ``str``
        annotation, path objects are required, as each is written
        into the script via its ``as_posix`` method.
    setup_file : pathlib.Path
        Path of the ``setup.py`` file to write.
    """

    modules_str = textwrap.indent(",\n".join(
        ['"' + s.as_posix() + '"' for s in modules]
    ), " " * 8)

    setup_script = textwrap.dedent("""\
    import sys
    from setuptools import setup
    from Cython.Build import cythonize

    setup(
        name="{model_name}",
        ext_modules=cythonize([
    {modules_str}
            ],
            annotate=True,
            compiler_directives={{"freethreading_compatible": True}}
        )
    )
    """)

    setup_file.write_text(
        setup_script.format(
            model_name=model_name,
            modules_str=modules_str),
        encoding="utf-8")


def entry_point_main():
    """Wrapper for main() for setuptools console_script entry point."""
    # Since monkeytype needs to import the user's code (and possibly config
    # code), the user's code must be on the Python path. But when running the
    # CLI script, it won't be. So we add the current working directory to the
    # Python path ourselves.
    # sys.path.insert(0, os.getcwd())
    sys.exit(main(sys.argv[1:], sys.stdout, sys.stderr))
