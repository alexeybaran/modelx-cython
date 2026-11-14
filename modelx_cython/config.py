# Copyright (c) 2023-2025 Fumito Hamamura <fumito.ham@gmail.com>

# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation version 3.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library.  If not, see <http://www.gnu.org/licenses/>.

"""Translation-spec handling for mx2cy.

This module defines :class:`TransSpec`, a thin wrapper around the
user-supplied spec dictionary read from the spec file passed to the
``mx2cy`` command (``--spec``).  The spec lets the user customize how
spaces and cells are translated to Cython, for example by fixing a
cells return type or declaring maximum cells parameter sizes.
"""

from typing import Union, Sequence, Mapping
import os
from dataclasses import dataclass
from modelx_cython.consts import (
    FILE_PREF,
    GLOBAL_PREF,
    MODULE_PREF,
    SPACE_PREF,
)


@dataclass(frozen=True)
class CompilerPolicy:
    """Resolve dependent diagnostic options once, before code generation."""
    automatic_optimizations: bool
    parallel_workers: int
    dense_bounds_proofs: bool
    dense_miss_binding: bool
    dense_one_axis: bool
    propagate_exact_receiver: bool
    native_cache_hits: str
    native_math: bool  # Guarded builtin rounding.
    # M1 (W1.1, default on since W1.5): '# cython: infer_types=True, overflowcheck=True'
    # on generated model modules.
    infer_types: bool = True
    # M1 (W1.3): lower pandas-spelled table reads to typed handle reads (table_reads.py).
    # On by default since the runtime handles (W1.2) landed (D45).
    typed_table_reads: bool = True
    # diag-4w (D71): the handle of a zero-argument table Cell is held per instance and read
    # borrowed (no Cell call, no reference count per read); false resolves it per call.
    held_table_handles: bool = True
    # auto-tables (D104): a plain pandas table that is not admitted, or pandas run on an admitted
    # table in a per-step cell (class C), stops the build instead of a warning.
    strict_tables: bool = False

    @classmethod
    def resolve(cls, data):
        automatic = data.get('automatic_optimizations', True)
        miss = data.get('dense_miss_binding', automatic)
        values = dict(automatic_optimizations=automatic,
            parallel_workers=data.get('parallel_workers', 1),
            dense_bounds_proofs=data.get('dense_bounds_proofs', automatic),
            dense_miss_binding=miss, dense_one_axis=data.get('dense_one_axis', automatic),
            propagate_exact_receiver=data.get('propagate_exact_receiver', automatic and miss),
            native_cache_hits=data.get('native_cache_hits', 'auto' if automatic and miss else 'none'),
            native_math=data.get('native_math', True),
            infer_types=data.get('infer_types', True),
            typed_table_reads=data.get('typed_table_reads', True),
            held_table_handles=data.get('held_table_handles', True),
            strict_tables=data.get('strict_tables', os.environ.get('MX_CY_STRICT_TABLES') == '1'))
        for name, value in values.items():
            if name not in ('parallel_workers', 'native_cache_hits') and type(value) is not bool:
                raise ValueError(f'{name} must be a boolean')
        if type(values['parallel_workers']) is not int or values['parallel_workers'] < 1:
            raise ValueError('parallel_workers must be a positive integer')
        if type(values['native_cache_hits']) is not str or values['native_cache_hits'] not in (
                'none', 'scalar', 'dense', 'both', 'auto'):
            raise ValueError('native_cache_hits must be none, scalar, dense, both or auto')
        if not miss:
            if values['propagate_exact_receiver']:
                raise ValueError('propagate_exact_receiver requires dense_miss_binding')
            if values['native_cache_hits'] != 'none':
                raise ValueError('native_cache_hits requires dense_miss_binding')
        return cls(**values)


class TransSpec:
    """Wrapper around the user-supplied translation-spec dictionary.

    The spec is a nested dict (read from the spec file with
    a safe literal decoder) in which each nesting level describes one
    space.  A space-level dict may have these keys:

    * ``"spaces"``: dict mapping child space names (without prefixes)
      to their own space-level dicts.
    * ``"cells_param_size"``: dict mapping a cells parameter name (or
      a tuple of parameter names) to its maximum size (or a tuple of
      sizes).  Used to size the fixed-length caches generated for
      cells with parameters; sampled maximum arguments can enlarge
      but not shrink these sizes. Also read at the model level (top of
      the spec) and in enclosing Spaces: the nearest declaration of a key
      wins. A tuple key applies to Cells with exactly that parameter
      tuple; a single-name key bounds that parameter's axis of every
      cached Cell in scope whose parameter tuple has no tuple entry.
    * ``"cells"``: dict mapping cells names to per-cells dicts.
    * ``"serial_collect"``: optional experimental scalar collection for an
      isolated slotted leaf Space. Use ``"reduce"`` for fresh receivers or
      ``"reuse"`` for one receiver. The collect Space owns policy execution:
      policies are independent and their live receiver and cache stay inside
      collect. In reuse mode, Reference bindings are captured at entry.
    * ``"serial_private_lookups"``: enable supported helpers' private warm
      path on a worker-owned serial collect receiver.
    * ``"parallel_collect_reuse"``: with ``"serial_collect": "reuse"``, give
      each ordered collect chunk its own reusable receiver. The same policy
      isolation and stable Reference contract applies across workers. Scalar
      chunk totals merge in domain order unless ``"parallel_collect_order"``
      says ``"completion"``.
    * ``"parallel_collect_order"``: ``"domain"`` (default with parallel reuse)
      or ``"completion"``. Domain order gives reproducible chunk merging at a
      fixed worker count and requires parallel reuse. It does not imply serial
      sum equivalence: the four-worker total can differ from the one-worker
      total in the last bits. Completion order merges each chunk as it finishes.
    * ``"param_bound_checks"``: dict mapping a sized parameter name to a
      zero-argument cached Cell of the Space whose value is the policy's
      need for that parameter (for example ``{"t": "proj_len"}``). The
      opt-in collect entry checks every policy against the bound before
      any projection and raises IndexError naming the policy.
    * ``"cells_params"`` (deprecated): older form of
      ``"cells_param_size"``; maps a parameter name (or tuple) to a
      dict with a ``"size"`` key.  Consulted only when
      ``"cells_param_size"`` is absent.

    A per-cells dict may have:

    * ``"native_numeric_cache"``: override automatic selection of native numeric
      key storage for an eligible cell. Other signatures use dictionaries.
      Both support compiled model pickling. Setting ``"numeric_cache_capacity"``
      also opts in unless ``"native_numeric_cache"`` is false.
    * ``"dense_window"``: a sequence of half-open ``(lower, stop)`` pairs
      in parameter order for a dense cache. The physical window must
      fit inside the existing logical bounds. A declared window overrides
      the automatic dense storage size ceiling for an eligible cached Cell.
      Exact spill keeps the original logical domain available.
    * ``"native_divisibility"``: enable the checked Cython zero-remainder
      comparison for proven signed-integer operands.
    * ``"memoize_power"``: enable a one-entry exact-operand memo for the
      existing typed power helper on a privately owned policy Space.
    * ``"return_type"``: either a type name accepted by
      :data:`modelx_cython.typedefs.str_to_type` (``"bool"``,
      ``"int"``, ``"float"``, ``"str"``, ``"object"``) that overrides
      the traced return type, or ``"memoryview"``, which forces a
      typed-memoryview return type for a real-valued numpy array
      return regardless of the usage analysis.  A spec return type
      takes effect only when the sample run produced type
      information for the cells; otherwise it is ignored (with a
      warning for ``"memoryview"``) and the return type falls back
      to ``object``.

    The ``"space_params"`` and ``"param_type"`` keys are defined
    below but are not currently consumed by the pipeline.

    The top-level ``"dense_one_axis"`` experiment separates checked and proven
    calls to an existing one-axis dense cache body.
    ``"propagate_exact_receiver"`` carries the receiver fact through those
    entries and private dense formula calls. ``"native_math"`` enables guarded
    builtin rounding by default.
    ``"infer_types"`` (default true): C type inference with overflow checks
    (``# cython: infer_types=True, overflowcheck=True``) in generated model modules;
    false keeps Cython's default (no inference of C types for untyped locals, no
    overflow checks).
    ``"typed_table_reads"`` (default true; false keeps reads as Python calls): lower
    ``.loc``/``.at``/``.index`` reads on traced tables to typed handle reads, and (D78) a
    literal field read of a zero-argument record Cell (``model_point()["f"]``, also through a
    local bound once to it) to a read of a record handle held per instance and a held field
    slot; ``int()``/``float()`` of the read convert natively. Other record reads stay Python
    calls and are reported.
    ``"strict_tables"`` (default false, or true when the environment sets
    ``MX_CY_STRICT_TABLES=1``; auto-tables, D104): a zero-argument cell returning a pandas
    DataFrame that is not admitted as a table, or pandas run on an admitted table in a cell with
    arguments (class C), stops the build with an error instead of a warning.
    ``"held_table_handles"`` (default true): the handle of a zero-argument table Cell
    read by a lowered read is resolved once per instance into a field (reset with the
    per-policy caches) and used borrowed; false resolves it on every call.
    ``"native_cache_hits"`` defaults to ``"auto"`` when automatic optimization
    and dense miss binding are enabled. It selects scalar hits and already-proven
    dense hits; explicit alternatives are ``"none"``, ``"scalar"``, ``"dense"``
    and ``"both"``. Hits require ``"dense_miss_binding"``. Misses retain their
    original entry and Cython exception handling.
    ``"automatic_optimizations": False`` disables automatic selection and the
    default dense entry optimizations on every platform. Individual diagnostic
    options can override these defaults, subject to their dependencies.

    Attributes
    ----------
    SPACES : str
        Key ``"spaces"`` holding child space specs.
    SPACE_PARAMS : str
        Key ``"space_params"`` (currently unused).
    CELLS : str
        Key ``"cells"`` holding per-cells specs.
    CELLS_PARAM_SIZE : str
        Key ``"cells_param_size"`` holding max parameter sizes.
    CELLS_PARAMS : str
        Deprecated key ``"cells_params"``.
    SIZE : str
        Deprecated key ``"size"`` used inside ``"cells_params"``.
    RET_T : str
        Key ``"return_type"`` in a per-cells spec.
    RET_MEMORYVIEW : str
        Value ``"memoryview"`` for ``"return_type"`` forcing
        memoryview emission.
    PARAM_T : str
        Key ``"param_type"`` (currently unused).
    """

    SPACES = "spaces"
    SPACE_PARAMS = "space_params"
    CELLS = "cells"
    CELLS_PARAM_SIZE = "cells_param_size"
    CELLS_PARAMS = "cells_params"   # deprecated
    PARAM_BOUND_CHECKS = "param_bound_checks"
    SIZE = "size"   # deprecated
    RET_T = "return_type"
    DENSE_WINDOW = "dense_window"
    RET_MEMORYVIEW = "memoryview"   # RET_T value forcing memoryview emission
    PARAM_T = "param_type"

    def __init__(self, data: dict) -> None:
        """Store the spec dictionary.

        Parameters
        ----------
        data : dict
            The nested spec dictionary as described in the class
            docstring.
        """
        
        self._data = data
        self.policy = CompilerPolicy.resolve(data)
        self.automatic_optimizations = self.policy.automatic_optimizations

    def get_declared_sizes(self, object_path: str):
        """Return ``(tuples, names)`` of the ``"cells_param_size"`` entries in
        force for a Space: ``{parameter tuple: (sizes, where)}`` and
        ``{parameter name: (size, where)}``. The model level comes first, then
        each enclosing Space down to the addressed one; a nearer declaration
        of the same key wins. A level without ``"cells_param_size"`` uses its
        deprecated ``"cells_params"`` sizes, as for one Space."""
        tuples, names = {}, {}

        def merge(data, where):
            declared = data.get(self.CELLS_PARAM_SIZE)
            if declared is None:
                declared = {keys: config[self.SIZE]
                            for keys, config in data.get(self.CELLS_PARAMS, {}).items()
                            if self.SIZE in config}
            for keys, stops in declared.items():
                if isinstance(keys, tuple):
                    tuples[keys] = tuple(stops), where
                else:
                    names[keys] = stops, where

        merge(self._data, 'model')
        data, path = self._data, []
        for name in object_path.split(".")[1:]:
            for prefix in (MODULE_PREF, SPACE_PREF):
                if name[:len(prefix)] == prefix:
                    name = name[len(prefix):]
                    break
            else:
                continue
            data = (data.get(self.SPACES) or {}).get(name) or {}
            if not data:
                break
            path.append(name)
            merge(data, '.'.join(path))
        return tuples, names

    def get_spec(self, object_path: str):
        """Return the space-level spec dict for a dotted object path.

        The first component of *object_path* (the model package name)
        is discarded.  Each remaining component is interpreted by its
        prefix: components starting with ``_mx_`` (module file names
        such as ``_mx_classes``) are skipped; components starting
        with ``_m_`` (subpackage) or ``_c_`` (space class) have their
        prefix stripped and are looked up under the ``"spaces"`` key
        of the current level.  Components with no recognized prefix
        are ignored.

        Parameters
        ----------
        object_path : str
            Dotted path of a module, class, or space in the exported
            model package.

        Returns
        -------
        dict
            The spec dict for the addressed space, or an empty dict
            if any looked-up name is missing from the spec.

        Examples
        --------
        Paths such as the following address the ``Projection`` space
        and the ``Parent.Child`` space respectively::

            PkgName._mx_classes._c_Projection
            PkgName._m_Parent._m_Child
        """
        def get_subspace(data, name):
            """Return the child-space spec *name* of *data*, or {}."""
            spaces = data.get(self.SPACES)
            if spaces and spaces.get(name):
                return spaces.get(name)
            else:
                return {}
            
        names = object_path.split(".")[1:]  # Remove first
        data = self._data
        while names:
            name = names.pop(0)
            if name[:len(FILE_PREF)] == FILE_PREF:
                continue
            elif name[:len(MODULE_PREF)] == MODULE_PREF:
                name = name[len(MODULE_PREF):]
                data_ = get_subspace(data, name)
                if data_:
                    data = data_
                else:
                    return {}
            elif name[:len(SPACE_PREF)] == SPACE_PREF:
                name = name[len(SPACE_PREF):]
                data_ = get_subspace(data, name)
                if data_:
                    data = data_
                else:
                    return {}

        return data
