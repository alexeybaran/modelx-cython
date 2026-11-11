from __future__ import annotations

"""Backend-neutral normalized operator descriptors.

These records describe execution permission for static/state-free operations after
full-domain proof.  They are deliberately separate from scheduling semantics: a
value may be state-free yet still have no approved backend operator.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedOperator:
    uid: str
    kind: str
    result_dtype: str
    result_shape: str
    static_inputs: tuple[tuple[str, Any], ...] = ()
    dynamic_inputs: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()
    state_semantic: str = "state_free"
    python_supported: bool = False
    cython_supported: bool = False
    capability_blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "kind": self.kind,
            "result_dtype": self.result_dtype,
            "result_shape": self.result_shape,
            "static_inputs": [[name, value] for name, value in self.static_inputs],
            "dynamic_inputs": list(self.dynamic_inputs),
            "provenance": list(self.provenance),
            "state_semantic": self.state_semantic,
            "python_supported": bool(self.python_supported),
            "cython_supported": bool(self.cython_supported),
            "capability_blockers": list(self.capability_blockers),
        }
