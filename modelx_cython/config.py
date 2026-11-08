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
from dataclasses import dataclass
import sys
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
    native_math: bool

    @classmethod
    def resolve(cls, data):
        automatic = data.get('automatic_optimizations', True)
        miss = data.get('dense_miss_binding', automatic or sys.platform == 'win32')
        values = dict(automatic_optimizations=automatic,
            parallel_workers=data.get('parallel_workers', 1),
            dense_bounds_proofs=data.get('dense_bounds_proofs', automatic),
            dense_miss_binding=miss, dense_one_axis=data.get('dense_one_axis', automatic),
            propagate_exact_receiver=data.get('propagate_exact_receiver', automatic and miss),
            native_cache_hits=data.get('native_cache_hits', 'auto' if automatic and miss else 'none'),
            native_math=data.get('native_math', automatic))
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
    ``ast.literal_eval``) in which each nesting level describes one
    space.  A space-level dict may have these keys:

    * ``"spaces"``: dict mapping child space names (without prefixes)
      to their own space-level dicts.
    * ``"cells_param_size"``: dict mapping a cells parameter name (or
      a tuple of parameter names) to its maximum size (or a tuple of
      sizes).  Used to size the fixed-length caches generated for
      cells with parameters; sampled maximum arguments can enlarge
      but not shrink these sizes.
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
      chunk totals merge in completion order.
    * ``"cells_params"`` (deprecated): older form of
      ``"cells_param_size"``; maps a parameter name (or tuple) to a
      dict with a ``"size"`` key.  Consulted only when
      ``"cells_param_size"`` is absent.

    A per-cells dict may have:

    * ``"native_numeric_cache"``: explicitly select experimental numeric key
      storage for an eligible cell. The default dictionary supports compiled
      model pickling. Setting ``"numeric_cache_capacity"`` also opts in unless
      ``"native_numeric_cache"`` is false.
    * ``"dense_window"``: a sequence of half-open ``(lower, stop)`` pairs
      in parameter order for a dense cache. The physical window must
      fit inside the existing logical bounds. Exact spill keeps the
      original logical domain available.
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
    scalar lowering for ``round``, ``math.floor``, and ``math.ceil``.
    ``"native_cache_hits"`` opts into direct reads of existing native cache
    fields at proven private calls: ``"scalar"``, ``"dense"`` or ``"both"``.
    It defaults to ``"none"`` and requires ``"dense_miss_binding"``. Misses
    retain their original entry and Cython exception handling.

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
