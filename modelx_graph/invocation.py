from __future__ import annotations

"""Backend-neutral invocation geometry for canonical Stage compilation.

RunDomain answers *which structural Space instances are executed*.
OutputInvocation answers *which output Cell and fixed argument tuple is requested*.
ResultDomain answers *how scalar results correspond to the declared RunDomain*.
Internal projection/recurrence axes remain owned by StageExecutionPlan.ExecutionDomain.
"""

from dataclasses import dataclass
import hashlib
import inspect
from typing import Any, Iterable


def complete_space_invocation_keys(space: Any, keys: Iterable[Any]) -> tuple[Any, ...]:
    """Bind omitted Space arguments using its declared Python defaults.

    The canonical keys are shared by tracing, input binding and fallback binding.
    Missing required arguments remain errors; no default values are invented.
    """
    params = tuple(space.parameters or ())
    keys = tuple(keys)
    if not params:
        return keys
    signature = inspect.signature(space.formula.func)
    completed = []
    for key in keys:
        args = key if isinstance(key, tuple) else (key,)
        bound = signature.bind(*args)
        bound.apply_defaults()
        values = tuple(bound.arguments[name] for name in params)
        completed.append(values[0] if len(params) == 1 else values)
    return tuple(completed)


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    raw = "|".join(repr(x) for x in parts).encode("utf-8", "backslashreplace")
    return f"{prefix}_{hashlib.sha1(raw).hexdigest()[:16]}"


def _normalize_literal(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "fixed output invocation arguments must be scalar literals; "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True)
class OutputInvocation:
    uid: str
    output_name: str
    argument_names: tuple[str, ...]
    argument_values: tuple[Any, ...]
    binding_kind: str = "fixed_tuple_v1"

    @classmethod
    def fixed(
        cls,
        output_name: str,
        argument_names: Iterable[str] = (),
        argument_values: Iterable[Any] = (),
    ) -> "OutputInvocation":
        names = tuple(str(x) for x in argument_names)
        values = tuple(_normalize_literal(x) for x in argument_values)
        if len(names) != len(values):
            raise ValueError(
                "OutputInvocation argument name/value cardinality mismatch: "
                f"{len(names)} != {len(values)}"
            )
        uid = _stable_id("output_invocation", (str(output_name), names, values, "fixed_tuple_v1"))
        return cls(uid, str(output_name), names, values)

    @property
    def arity(self) -> int:
        return len(self.argument_values)

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "output_name": self.output_name,
            "argument_names": list(self.argument_names),
            "argument_values": list(self.argument_values),
            "binding_kind": self.binding_kind,
        }


@dataclass(frozen=True)
class RunDomain:
    uid: str
    space_fullname: str
    itemspace_parameter_names: tuple[str, ...]
    run_keys: tuple[Any, ...]
    proof_kind: str = "finite_structural_run_domain_v1"

    @classmethod
    def finite(
        cls,
        *,
        space_fullname: str,
        itemspace_parameter_names: Iterable[str],
        run_keys: Iterable[Any],
        proof_kind: str = "finite_structural_run_domain_v1",
    ) -> "RunDomain":
        params = tuple(str(x) for x in itemspace_parameter_names)
        keys = tuple(run_keys)
        if not keys:
            raise ValueError("RunDomain requires at least one structural instance")
        if not params:
            if keys != ((),):
                raise ValueError(
                    "non-parameterized Space RunDomain must be the singleton key ()"
                )
        elif len(params) == 1:
            # Single-parameter keys are intentionally represented directly.
            pass
        else:
            for key in keys:
                if not isinstance(key, tuple) or len(key) != len(params):
                    raise ValueError(
                        f"RunDomain key {key!r} does not match parameters {params!r}"
                    )
        tokens = tuple(repr(x) for x in keys)
        uid = _stable_id(
            "run_domain",
            (str(space_fullname), params, len(keys), *tokens, str(proof_kind)),
        )
        return cls(uid, str(space_fullname), params, keys, str(proof_kind))

    @property
    def point_count(self) -> int:
        return len(self.run_keys)

    @property
    def run_key_tokens(self) -> tuple[str, ...]:
        return tuple(repr(x) for x in self.run_keys)

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "space_fullname": self.space_fullname,
            "itemspace_parameter_names": list(self.itemspace_parameter_names),
            "point_count": self.point_count,
            "run_key_tokens": list(self.run_key_tokens),
            "proof_kind": self.proof_kind,
        }


@dataclass(frozen=True)
class ResultDomain:
    uid: str
    result_count: int
    run_domain_uid: str
    result_key_tokens: tuple[str, ...]
    ordering: str = "run_domain_order_v1"

    @classmethod
    def scalar_per_run_key(cls, run_domain: RunDomain) -> "ResultDomain":
        tokens = run_domain.run_key_tokens
        uid = _stable_id(
            "result_domain",
            (run_domain.uid, len(tokens), *tokens, "run_domain_order_v1"),
        )
        return cls(uid, len(tokens), run_domain.uid, tokens)

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "result_count": int(self.result_count),
            "run_domain_uid": self.run_domain_uid,
            "result_key_tokens": list(self.result_key_tokens),
            "ordering": self.ordering,
        }
