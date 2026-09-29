"""Explicit kernel contracts, not an expression/type/category interpreter."""
from dataclasses import dataclass, field
from enum import IntEnum
import inspect


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class TableKey:
    """A scalar label encoded with a named CurveTable, not a formula inference."""
    table: str
    default: int | None = None

    def __post_init__(self):
        if not self.table.isidentifier(): raise ContractError('TableKey needs an input table name')
        if self.default is not None and (type(self.default) is not int or self.default >= 0):
            raise ContractError('A missing/default key must be a negative integer sentinel')


def cell(function=None):
    """Mark an ordinary modelx-style Cell formula.

    Caching is semantic and unconditional, as in normal modelx Cells. The compiler
    chooses the physical memo representation from the declared signature; authors do
    not specify cache policies.
    """
    def decorate(f):
        f.__kernel_cell__ = True
        return f
    return decorate if function is None else decorate(function)


@dataclass(frozen=True)
class KernelSpec:
    module: object
    inputs: dict
    outputs: dict
    steps: int
    length: str
    scalar_outputs: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    frontend: dict | None = None
    cell_outputs: tuple[str, ...] = ()
    allow_none: dict[str, bool] | None = None
    model: dict | None = None
    model_name: str | None = None
    # Build-time snapshot used only to create an importable compiled-model package.
    # It is deliberately excluded from equality/repr and never enters generated code.
    initial_tables: dict | None = field(default=None, repr=False, compare=False)
    cython_objects: bool = False
    cython_signatures: dict | None = None

    def __post_init__(self):
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or not 0 < self.steps <= 1000000:
            raise ContractError('steps must be an explicit positive bound <= 1,000,000')
        if self.allow_none is not None and any(type(v) is not bool for v in self.allow_none.values()):
            raise ContractError('allow_none values must be booleans')
        if not (self.outputs or self.cell_outputs):
            raise ContractError('Output columns or Cell entries must be explicit')
        if len(set(self.parameters)) != len(self.parameters) or any(k not in self.inputs for k in self.parameters):
            raise ContractError('Parameters must be unique declared input names')
        for name in self.inputs:
            if not name.isidentifier() or name.startswith('_'):
                raise ContractError('Input names must be public identifiers')

    @property
    def cells(self):
        return {k: v for k, v in vars(self.module).items()
                if inspect.isfunction(v) and hasattr(v, '__kernel_cell__')}

    @property
    def enums(self):
        return {k: v for k, v in vars(self.module).items()
                if inspect.isclass(v) and issubclass(v, IntEnum) and v is not IntEnum and v.__members__}
