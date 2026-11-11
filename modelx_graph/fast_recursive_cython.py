"""Cython object-mode emitter for the static fast-recursive program.

This is a graph/ABI proof layer, not the final native numeric backend.  It compiles every
Cell in the static recursive closure to a C-level function while preserving unsupported
provider/Python operations as ordinary Python object operations.  Native provider/type
lowering can therefore be developed independently of recurrence scheduling.
"""
from __future__ import annotations

from pathlib import Path

from .fast_recursive_graph import FastRecursiveProgram
from .fast_recursive_python import emit_fast_recursive_python


def emit_fast_recursive_cython_object(program: FastRecursiveProgram) -> str:
    src = emit_fast_recursive_python(program)
    out: list[str] = [
        "# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False",
        "",
    ]
    for line in src.splitlines():
        # _calc_* and fr_* are internal C-level functions.  Their arguments intentionally
        # remain Python objects at this stage; this proves graph topology independently
        # of numeric/provider lowering.
        if line.startswith("def _calc_") or line.startswith("def fr_"):
            line = "cdef object " + line[len("def "):]
        out.append(line)
    return "\n".join(out) + "\n"


def write_fast_recursive_cython_object(program: FastRecursiveProgram, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(emit_fast_recursive_cython_object(program), encoding="utf-8")
    return out


__all__ = ["emit_fast_recursive_cython_object", "write_fast_recursive_cython_object"]
