"""Private prototype backend copied from the user-supplied semantic compiler.

This package is *not* the correctness frontend of modelx-graph-compiler.  It is
vendored only for the Stage-Part-1 experiment that measures the value of a
register-local/typed-snapshot batch kernel.  Unsupported cases must fall back to
CanonicalExecutionPlan/WholeArtifactProgram.  Part 2 is expected to port proven
optimizations into the canonical backend and delete this adapter.
"""
from .frontend import ModelxModelCompiler, ModelxCompileError
from .builder import NativeBackendBuilder, CompiledModelxBackend

__all__ = [
    "ModelxModelCompiler", "ModelxCompileError", "NativeBackendBuilder",
    "CompiledModelxBackend",
]
