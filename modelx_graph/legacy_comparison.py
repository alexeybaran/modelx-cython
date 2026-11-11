from __future__ import annotations

"""Explicit mature-IR comparison utilities.

Production canonical/Stage compilation does not import or call this module.  It
exists solely for regression/audit tooling that intentionally asks for the mature
ExecutableGraph/OptimizedProgram oracle.
"""

from dataclasses import dataclass
from typing import Any


class LegacyComparisonError(RuntimeError):
    pass


@dataclass(frozen=True)
class LegacyComparisonBundle:
    executable: Any
    optimized_program: Any
    optimization_level: str

    @property
    def formula_hash(self) -> str:
        return str(self.executable.formula_hash)


def build_legacy_comparison_bundle(
    canonical_model: Any,
    *,
    optimization_level: str = "O2",
) -> LegacyComparisonBundle:
    """Build mature comparison state from an explicitly comparison-enabled frontend.

    Recompile ``GraphModelCompiler(..., legacy_comparison=True)`` when the supplied
    canonical model came from the production Stage path.  This prevents audit code
    from accidentally turning mature construction back into an implicit production
    dependency.
    """
    executable = getattr(canonical_model, "executable", None)
    if executable is None:
        raise LegacyComparisonError(
            "legacy comparison is unavailable on a production canonical model; "
            "compile explicitly with legacy_comparison=True"
        )
    from .optimized_program import build_optimized_program

    level = str(optimization_level).upper()
    optimized = build_optimized_program(canonical_model, optimization_level=level)
    return LegacyComparisonBundle(
        executable=executable,
        optimized_program=optimized,
        optimization_level=level,
    )
