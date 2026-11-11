"""Trace-evidence compatibility for upstream modelx-cython.

Adds conservative geometry and minimum integer argument observations required by
this compiler, without editing upstream or altering its translation/runtime.
The additional fields are observations of sample returns, not unseen-domain proof.
"""
from __future__ import annotations
import numbers
import runpy
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from modelx_cython.tracer import MxCallTraceLogger, MxCodeFilter, trace_calls

COMPATIBILITY_MODE = 'dev155_observed_trace_metadata_v1'

def observed_metadata(traces):
    values = [getattr(t, 'ret_val', None) for t in traces]
    result = {'array_shape': None, 'numeric_series_evidence': None,
              'numeric_sequence_shape': None, 'min_args': {}}
    if values and all(type(v) in (list, tuple) for v in values):
        try:
            arrays = [np.asarray(v) for v in values]
            shapes = {tuple(v.shape) for v in arrays}
            if len(shapes) == 1 and all(v.dtype.kind in 'biuf' for v in arrays):
                result['numeric_sequence_shape'] = next(iter(shapes))
        except (TypeError, ValueError):
            pass  # ragged/non-numeric sequences remain opaque objects
    if values and all(isinstance(v, np.ndarray) for v in values):
        shapes = {tuple(v.shape) for v in values}
        if len(shapes) == 1:
            result['array_shape'] = next(iter(shapes))
    if values and all(isinstance(v, pd.Series) for v in values):
        first = values[0]
        if all(isinstance(v.dtype, np.dtype) and v.dtype.kind in 'biuf'
               and v.index.is_unique and v.index.equals(first.index)
               and tuple(v.shape) == tuple(first.shape) for v in values):
            result['numeric_series_evidence'] = (
                np.result_type(*(v.dtype for v in values)).type,
                tuple(first.shape), tuple(first.index))
    names = set().union(*(set(t.arg_vals) for t in traces)) if traces else set()
    for name in names - {'self'}:
        args = [t.arg_vals.get(name) for t in traces]
        if all(isinstance(a, numbers.Integral) for a in args):
            result['min_args'][name] = min(int(a) for a in args)
    return result

class EvidenceLogger(MxCallTraceLogger):
    def flush(self):
        extra = {name: observed_metadata(traces) for name, traces in self._traces.items()}
        super().flush()
        for name, metadata in extra.items():
            if self.new_name:
                name = self.new_name + '.' + name.split('.', 1)[1]
            info = self.cells_info.get(name)
            if info is not None:
                for key, value in metadata.items():
                    setattr(info, key, value)
        self.compatibility_mode = COMPATIBILITY_MODE
        self.compatibility_evidence = extra

def run_sample(model_path, sample_path, new_model_name=None):
    """Same sample protocol as upstream CLI, with an evidence-aware logger."""
    model_path = Path(model_path)
    module_path = str(model_path.parent)
    sys.path.insert(0, module_path)
    try:
        logger = EvidenceLogger(module=model_path.name, new_model_name=new_model_name)
        with trace_calls(module=model_path.name, logger=logger,
                         max_typed_dict_size=0, code_filter=MxCodeFilter()):
            runpy.run_path(str(sample_path), run_name='__main__')
    finally:
        assert sys.path.pop(0) == module_path
    return logger
