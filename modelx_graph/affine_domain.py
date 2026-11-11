from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AffineIntForm:
    """Small integer affine form used only for semantic domain proofs.

    The form is ``constant + sum(coefficient[symbol] * symbol)``.  It deliberately
    knows nothing about modelx, source names, or scheduling.  Symbols and their
    domains are supplied by the semantic frontend.
    """

    constant: int = 0
    terms: tuple[tuple[str, int], ...] = ()

    @staticmethod
    def scalar(value: int) -> "AffineIntForm":
        return AffineIntForm(int(value), ())

    @staticmethod
    def symbol(name: str) -> "AffineIntForm":
        return AffineIntForm(0, ((str(name), 1),))

    @staticmethod
    def _freeze(constant: int, coeffs: Mapping[str, int]) -> "AffineIntForm":
        return AffineIntForm(
            int(constant),
            tuple(sorted((str(name), int(coeff)) for name, coeff in coeffs.items() if int(coeff) != 0)),
        )

    def coefficients(self) -> dict[str, int]:
        return dict(self.terms)

    def coefficient(self, symbol: str) -> int:
        return int(dict(self.terms).get(symbol, 0))

    def add(self, other: "AffineIntForm") -> "AffineIntForm":
        coeffs = self.coefficients()
        for name, coeff in other.terms:
            coeffs[name] = coeffs.get(name, 0) + int(coeff)
        return self._freeze(self.constant + other.constant, coeffs)

    def sub(self, other: "AffineIntForm") -> "AffineIntForm":
        return self.add(other.scale(-1))

    def scale(self, factor: int) -> "AffineIntForm":
        factor = int(factor)
        return self._freeze(
            self.constant * factor,
            {name: coeff * factor for name, coeff in self.terms},
        )

    def without(self, symbol: str) -> "AffineIntForm":
        return self._freeze(self.constant, {n: c for n, c in self.terms if n != symbol})

    def substitute(self, symbol: str, replacement: "AffineIntForm") -> "AffineIntForm":
        coeff = self.coefficient(symbol)
        if coeff == 0:
            return self
        return self.without(symbol).add(replacement.scale(coeff))


@dataclass(frozen=True)
class AffineRangeConstraint:
    """Inclusive affine bounds for one semantic coordinate.

    This is proof data only.  It must never be used to select physical iteration
    geometry; that remains StageExecutionPlan/ExecutionDomain authority.
    """

    coordinate_symbol: str
    lower: AffineIntForm | None = None
    upper: AffineIntForm | None = None


def affine_interval_bounds(
    form: AffineIntForm,
    symbol_bounds: Mapping[str, tuple[int, int]],
) -> tuple[int | None, int | None]:
    lo = int(form.constant)
    hi = int(form.constant)
    for symbol, coeff in form.terms:
        bounds = symbol_bounds.get(symbol)
        if bounds is None:
            return None, None
        sym_lo, sym_hi = int(bounds[0]), int(bounds[1])
        if coeff >= 0:
            lo += coeff * sym_lo
            hi += coeff * sym_hi
        else:
            lo += coeff * sym_hi
            hi += coeff * sym_lo
    return lo, hi


def affine_bounds_under_range(
    form: AffineIntForm,
    constraint: AffineRangeConstraint,
    symbol_bounds: Mapping[str, tuple[int, int]],
) -> tuple[int | None, int | None]:
    """Bound an affine query while preserving correlation with the coordinate range.

    The only relational reasoning admitted here is monotone substitution of the
    coordinate's affine lower/upper bounds.  Unsupported or one-sided cases remain
    unknown on the corresponding side.
    """

    coord = constraint.coordinate_symbol
    coeff = form.coefficient(coord)
    if coeff == 0:
        return affine_interval_bounds(form, symbol_bounds)

    remainder = form.without(coord)
    if coeff > 0:
        lower_form = None if constraint.lower is None else remainder.add(constraint.lower.scale(coeff))
        upper_form = None if constraint.upper is None else remainder.add(constraint.upper.scale(coeff))
    else:
        lower_form = None if constraint.upper is None else remainder.add(constraint.upper.scale(coeff))
        upper_form = None if constraint.lower is None else remainder.add(constraint.lower.scale(coeff))

    lo = None
    hi = None
    if lower_form is not None:
        lo, _ = affine_interval_bounds(lower_form, symbol_bounds)
    if upper_form is not None:
        _, hi = affine_interval_bounds(upper_form, symbol_bounds)
    return lo, hi
