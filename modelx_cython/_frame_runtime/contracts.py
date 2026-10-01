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


def cell(function=None, *, cached=True):
    """Mark an ordinary modelx-style Cell formula.

    Preserve the exported Cell's cache contract. The compiler chooses physical
    memo representation for cached Cells from the declared signature.
    """
    def decorate(f):
        f.__kernel_cell__ = True
        f.__kernel_cached__ = cached
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
    # Optional build contract for numeric fields of object-valued row references.
    # The model's formula source and public references remain unchanged.
    typed_mapping_inputs: dict[str, dict[str, str]] | None = None
    # Original compiler cache domains, rather than a projection-sized guess.
    cache_domains: dict | None = None
    # Internal integration contract; never changes the existing frame default.
    original_space_shell: bool = False
    # Prepared original-shell resources: shared snapshots, not live public refs.
    # Values are existing compiled helper class paths or 'object'.
    original_prepared_inputs: dict[str, str] | None = None
    original_helper_modules: dict[str, str] | None = None

    def __post_init__(self):
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or not 0 < self.steps <= 1000000:
            raise ContractError('steps must be an explicit positive bound <= 1,000,000')
        if self.allow_none is not None and any(type(v) is not bool for v in self.allow_none.values()):
            raise ContractError('allow_none values must be booleans')
        if not (self.outputs or self.cell_outputs):
            raise ContractError('Output columns or Cell entries must be explicit')
        if len(set(self.parameters)) != len(self.parameters) or any(k not in self.inputs for k in self.parameters):
            raise ContractError('Parameters must be unique declared input names')
        if self.original_prepared_inputs and (not self.original_space_shell or
                any(self.inputs.get(n) is not object for n in self.original_prepared_inputs)):
            raise ContractError('Prepared shell resources require original-shell object inputs')
        if self.original_helper_modules and not self.original_space_shell:
            raise ContractError('Compiled shell helpers require an original Space receiver')
        for name in self.inputs:
            if not name.isidentifier() or name.startswith('_'):
                raise ContractError('Input names must be public identifiers')
        for name, fields in (self.typed_mapping_inputs or {}).items():
            if self.inputs.get(name) is not object or not isinstance(fields, dict):
                raise ContractError('Typed mapping fields require a declared object input')
            for key, kind in fields.items():
                if not isinstance(key, str) or not key.isidentifier() or kind not in ('float', 'float64_1d'):
                    raise ContractError('Typed mapping fields require identifier keys and numeric kinds')
        aliases = ['_gil_field_' + name + '_' + key
                   for name, fields in (self.typed_mapping_inputs or {}).items()
                   for key in fields]
        if len(aliases) != len(set(aliases)):
            raise ContractError('Typed mapping field names produce a generated-name collision')

    @property
    def cells(self):
        from modelx_cython.formula_emitter import SourceFormula
        return {k: v for k, v in vars(self.module).items()
                if (inspect.isfunction(v) or isinstance(v, SourceFormula))
                and hasattr(v, '__kernel_cell__')}

    @property
    def enums(self):
        return {k: v for k, v in vars(self.module).items()
                if inspect.isclass(v) and issubclass(v, IntEnum) and v is not IntEnum and v.__members__}
