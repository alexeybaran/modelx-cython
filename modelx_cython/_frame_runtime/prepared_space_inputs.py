"""Native resources behind original Space classes, without a model facade.

The generated package supplies already resolved tables and reference values.
This module neither discovers Spaces nor evaluates or caches Cell formulas.
Replacing a resource binding does not invalidate completed Cell results.
"""
import numbers

from .contracts import ContractError
from .data_binding import pack_tables, prepare_inputs
from .input_codec import NativeInputCodec


class PreparedSpaceInputs(NativeInputCodec):
    """One atomically prepared binding shared by independently owned caches.

    The caller owns synchronization with original public references. Tables use
    the established pack_tables snapshot contract; scalar references remain a
    separate entry-boundary responsibility. No implicit preparation happens at
    collect entry, and no selected-policy state is shared between owners.
    """

    def __init__(self, manifest, lookups, tables, *, references=None):
        self.manifest = manifest
        self._lookups = lookups
        self._record_layout()
        self.replace(tables, references=references)

    def _record_layout(self):
        infos = self.manifest['inputs']
        locations = {name: i for i, name in enumerate(infos)}
        self._parameter_locations = {binding['name']: (name, locations[name], infos[name])
            for name, binding in self.manifest['model']['bindings'].items()
            if binding['kind'] == 'parameter'}

    def replace(self, tables, *, references=None):
        """Prepare a complete replacement before publishing any resource."""
        references = dict(references or {})
        packed = pack_tables(self.manifest['model'], tables)
        prepared = prepare_inputs(self.manifest, tables, references=references,
                                  packed=packed, prepared=True)
        records = self._validate(prepared)
        # Owners keep the old native objects alive through their own record.
        # A failed conversion leaves this binding and all their caches intact.
        self._records = records
        self._packed = packed

    def record(self, position, *, parameters=None):
        """Bind the selected row and actual original dynamic parameters.

        Selection and required parameter conversion belong inside calculation.
        Existing owners retain their record; this does not reset their caches.
        """
        if (isinstance(position, bool) or not isinstance(position, numbers.Integral)
                or not 0 <= position < len(self._records)):
            raise IndexError('Policy position outside the prepared binding')
        record = self._records[position]
        if not parameters:
            return record
        declared = self._parameter_locations
        if set(parameters) - declared.keys():
            raise ContractError('Unknown dynamic Space parameter')
        raw = None
        updated = list(record)
        for parameter, value in parameters.items():
            name, position, info = declared[parameter]
            if info['kind'] == 'table_key' and raw is None:
                raw = dict(zip(self.manifest['inputs'], record))
            updated[position] = self._scalar_value(name, info, value, raw)
        return tuple(updated)

    def __getstate__(self):
        # Module objects are not pickle state. Completed native resources own
        # their own ordinary reductions; preserve sharing through pickle's memo.
        return self.manifest, self._lookups.__name__, self._records, self._packed

    def __setstate__(self, state):
        from importlib import import_module
        self.manifest, module, self._records, self._packed = state
        self._lookups = import_module(module)
        self._record_layout()
