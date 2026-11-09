"""Build-time Cython compatibility and standalone setup generation."""
import pathlib
import sys
import textwrap
from typing import Sequence
import libcst as cst

from .syntax import statements, walk


def create_setup(model_name: str, modules: Sequence[str], setup_file: pathlib.Path, source_root=None,
                 source_modules=None):
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

    # Cython's generated pickle state builds a recursive boolean chain over
    # native members. Provision its visitor stack from generated class shape,
    # rather than requiring a model-specific recursion setting from the user.
    source_root = pathlib.Path(source_root) if source_root is not None else setup_file.parent
    recursion_budget = sys.getrecursionlimit()
    if source_modules is None:
        source_modules = (cst.parse_module(source.read_text(encoding='utf-8'))
                          for module in modules if (source := source_root / module).suffix == '.py'
                          and source.is_file())
    for module in source_modules:
        for node in walk(module):
            if isinstance(node, cst.ClassDef):
                members = sum(isinstance(child, cst.AnnAssign) for child in statements(node.body))
                recursion_budget = max(recursion_budget, 64 * (members + 32))

    modules_str = textwrap.indent(",\n".join(
        ['"' + s.as_posix() + '"' for s in modules]
    ), " " * 8)

    setup_script = textwrap.dedent("""\
    from setuptools import setup
    from Cython.Build import cythonize
    from Cython.Compiler.PyrexTypes import CArrayType, c_int_type
    from pathlib import Path
    import modelx_cython
    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), {recursion_budget}))

    # Some Cython releases compare nested arrays without their inner lengths.
    # Their temporary allocator can then corrupt multidimensional pickle state.
    # Correct structural equality only when the installed release has the bug;
    # Cython's assignment compatibility and generated runtime stay unchanged.
    if CArrayType(CArrayType(c_int_type, 2), 2) == CArrayType(CArrayType(c_int_type, 5), 2):
        def _mx_array_type_equal(left, right):
            if not isinstance(right, CArrayType) or left.size != right.size:
                return False
            if left.base_type.is_array:
                return left.base_type == right.base_type
            return left.base_type.same_as(right.base_type)
        CArrayType.__eq__ = _mx_array_type_equal

    setup(
        name="{model_name}",
        ext_modules=cythonize([
    {modules_str}
            ],
            annotate=True,
            # PEP660 meta-path installs do not add the package root to sys.path.
            include_path=[str(Path(modelx_cython.__file__).resolve().parent.parent)],
            compiler_directives={{"freethreading_compatible": True}}
        )
    )
    """)

    setup_file.write_text(
        setup_script.format(
            model_name=model_name,
            modules_str=modules_str, recursion_budget=recursion_budget),
        encoding="utf-8")

