"""Shared native input conversion; no model facade or Cell execution."""
import math
import numbers
import numpy as np
from .contracts import ContractError
from .data_binding import PreparedModelRows
from .lookups import (Curve, CurveTable, IndexedFloat1, IndexedFloat2,
                      IndexedInt1, IndexedInt2)

_INDEXED_ARRAYS = {cls.__name__: cls for cls in
                   (IndexedInt1, IndexedFloat1, IndexedInt2, IndexedFloat2)}


class NativeInputCodec:
    """Validate complete input records for a generated native layout.

    Consumers supply ``manifest`` and the generated ``_lookups`` module.
    Conversion is atomic: no receiver or cache is changed here.
    """
    def _validate(self, records):
        if isinstance(records, PreparedModelRows):
            return self._validate_prepared(records)
        converted = []
        shared = {}
        immutable_last = {}
        infos = self.manifest['inputs']
        for record in records:
            if set(record) != set(infos):
                raise ContractError(f'Input names differ: missing={set(infos)-set(record)}, extra={set(record)-set(infos)}')
            row = []
            for key, ti in infos.items():
                v = record[key]
                kind = ti['kind']
                if kind in ('curve', 'curve_table', 'indexed_array'):
                    expected_type = (Curve if kind == 'curve' else CurveTable
                                     if kind == 'curve_table' else _INDEXED_ARRAYS[ti['ctype']])
                    if type(v) is not expected_type:
                        raise ContractError(f'{key}: expected an immutable prepared native input')
                    if id(v) not in shared:
                        constructor = getattr(self._lookups, expected_type.__name__) if self.manifest['lookups'] == 'native' else expected_type
                        shared[id(v)] = constructor(*v.arrays, v.labels) if kind == 'curve_table' else constructor(*v.arrays)
                    v = shared[id(v)]
                else:
                    # Model-level references often repeat the exact immutable
                    # Python object across thousands of policy rows. Reuse its
                    # checked conversion, but never memoize mutable/custom
                    # numbers or table keys whose meaning depends on a record.
                    if kind != 'table_key' and type(v) in (int, float, bool, str):
                        last = immutable_last.get(key)
                        if last is not None and last[0] is v:
                            v = last[1]
                        else:
                            converted_value = self._scalar_value(key, ti, v, record)
                            immutable_last[key] = (v, converted_value)
                            v = converted_value
                    else:
                        v = self._scalar_value(key, ti, v, record)
                row.append(v)
            converted.append(tuple(row))
        if not converted:
            raise ContractError('At least one policy is required')
        return tuple(converted)

    def _validate_prepared(self, prepared):
        """Validate model-produced fixed inputs once and varying inputs per row.

        The marker is constructed only after ``prepare_inputs`` has copied the
        same fixed values into every row. External records never use this path.
        Build all converted rows before publishing a new native binding.
        """
        rows = prepared.rows
        if not rows:
            raise ContractError('At least one policy is required')
        infos = self.manifest['inputs']
        varying = [(position, key, ti) for position, (key, ti) in
                   enumerate(infos.items()) if key not in prepared.fixed_names]
        first_row = dict(rows[0])
        native_varying = [(position, key, ti) for position, key, ti in varying
                          if ti['kind'] in ('curve', 'curve_table', 'indexed_array')]
        for _, key, ti in native_varying:
            if ti['kind'] != 'indexed_array':
                raise ContractError('Unexpected varying prepared curve input')
            first_row[key] = _INDEXED_ARRAYS[ti['ctype']](first_row[key])
        first = self._validate([first_row])[0]
        converted = [first]
        shared_arrays = {}
        for record in rows[1:]:
            values = list(first)
            for position, key, ti in varying:
                if ti['kind'] == 'indexed_array':
                    source = record[key]
                    prior = shared_arrays.get(id(source))
                    if prior is not None and prior[0] is source:
                        values[position] = prior[1]
                    else:
                        wrapper = getattr(self._lookups, ti['ctype'])(source)
                        shared_arrays[id(source)] = (source, wrapper)
                        values[position] = wrapper
                else:
                    values[position] = self._scalar_value(key, ti, record[key], record)
            converted.append(tuple(values))
        return tuple(converted)

    @staticmethod
    def _scalar_value(key, ti, v, record=None):
        kind = ti['kind']
        if kind == 'object':
            import builtins
            typ = type(None) if ti['python_type'] == 'NoneType' else getattr(builtins, ti['python_type'])
            if not isinstance(v, typ):
                raise ContractError(f'{key}: expected {ti["python_type"]}')
            return v
        if kind == 'table_key':
            if ti['default'] is not None and type(v) is int and v == ti['default']: return v
            if record is None or ti['table'] not in record:
                raise ContractError(f'{key}: missing declared key-domain table')
            try: return record[ti['table']].code(v)
            except (KeyError,TypeError,ValueError) as e: raise ContractError(f'{key}: unknown table label {v!r}') from e
        if kind == 'float':
            # Original typed fields produce exact builtins. Their declared
            # representation needs the same value checks, not repeated ABC
            # discovery. Subclasses and custom numbers retain normal conversion.
            if (type(v) not in (float, int) and
                    (isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real))) or not math.isfinite(v):
                raise ContractError(f'{key}: finite real scalar required')
            v = float(v)
        elif kind == 'bool':
            if not isinstance(v, (bool, np.bool_)):
                raise ContractError(f'{key}: boolean required')
            v = bool(v)
        else:
            if type(v) is not int and (isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral)):
                raise ContractError(f'{key}: integral scalar required')
            v = int(v)
            if not -(1 << 63) <= v < (1 << 63):
                raise ContractError(f'{key}: outside signed int64')
            if kind == 'enum' and v not in ti['members'].values():
                raise ContractError(f'{key}: outside declared category domain')
        return v

