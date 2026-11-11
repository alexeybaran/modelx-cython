from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

import numpy as np


class FinalBackendError(RuntimeError):
    """Fail-closed error for final benchmark-profile selection."""


class ExecutionProfile(str, Enum):
    SCALAR = "scalar"
    VECTOR_PRESERVING = "vector_preserving"


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, np.ndarray):
        return tuple(int(x) for x in value.shape)
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(x) for x in shape)
    except Exception:
        return None


def _is_shaped_value(value: Any) -> bool:
    shape = _shape_tuple(value)
    if shape is None or shape == ():
        return False
    # A length-one vector is still a vector semantically. Do not silently
    # scalarize merely because the current fixture happens to contain one item.
    return True


def _value_kind(value: Any) -> str:
    shape = _shape_tuple(value)
    module = type(value).__module__.split(".", 1)[0]
    name = type(value).__name__
    if isinstance(value, np.ndarray):
        return f"ndarray:{value.dtype}:shape={shape}"
    if module == "pandas" and shape is not None:
        return f"pandas.{name}:shape={shape}"
    if isinstance(value, np.generic):
        return f"numpy_scalar:{value.dtype}"
    if shape not in (None, ()):
        return f"shaped:{type(value).__module__}.{name}:shape={shape}"
    return f"scalar:{type(value).__module__}.{name}"


@dataclass(frozen=True)
class FinalBackendProfile:
    """Structural benchmark profile for the frozen pre-benchmark compiler.

    The profile does not infer actuarial/domain meaning.  It only distinguishes
    scalar targets from shaped vector targets.  Vector targets deliberately keep
    NumPy/pandas FormulaOps as opaque Python vector kernels while Cython owns the
    canonical scheduling.  Scalar targets may use only already-proven optional
    optimizations.
    """

    execution_profile: ExecutionProfile
    target_value_kinds: tuple[str, ...]
    shaped_target_count: int
    use_frozen_numeric_references: bool
    use_register_regions: bool
    allow_pow: bool = False
    reason: str = ""

    @property
    def is_vector_preserving(self) -> bool:
        return self.execution_profile is ExecutionProfile.VECTOR_PRESERVING

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_profile": self.execution_profile.value,
            "target_value_kinds": list(self.target_value_kinds),
            "shaped_target_count": self.shaped_target_count,
            "use_frozen_numeric_references": self.use_frozen_numeric_references,
            "use_register_regions": self.use_register_regions,
            "allow_pow": self.allow_pow,
            "reason": self.reason,
        }


def choose_final_backend_profile(target_values: Iterable[Any]) -> FinalBackendProfile:
    values = tuple(target_values)
    if not values:
        raise FinalBackendError("at least one realized target value is required")
    kinds = tuple(_value_kind(v) for v in values)
    shaped = sum(_is_shaped_value(v) for v in values)
    if shaped:
        return FinalBackendProfile(
            execution_profile=ExecutionProfile.VECTOR_PRESERVING,
            target_value_kinds=kinds,
            shaped_target_count=shaped,
            use_frozen_numeric_references=False,
            use_register_regions=False,
            allow_pow=False,
            reason=(
                "one or more realized targets are shaped; preserve NumPy/pandas "
                "vector kernels and optimize canonical scheduling/cache ownership only"
            ),
        )
    return FinalBackendProfile(
        execution_profile=ExecutionProfile.SCALAR,
        target_value_kinds=kinds,
        shaped_target_count=0,
        use_frozen_numeric_references=True,
        use_register_regions=True,
        allow_pow=False,
        reason=(
            "all realized targets are scalar; enable only proven frozen numeric "
            "references and conservative auto-gated RegisterRegions"
        ),
    )
