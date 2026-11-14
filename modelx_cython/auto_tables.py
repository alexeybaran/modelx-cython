"""Plain pandas input tables, admitted on their traced value (auto-tables, D104).

While the sample runs, every zero-argument formula of the exported model is wrapped at class
level (no file is edited): its value goes through ``runtime.table_admission.admit``.

- An admitted DataFrame is replaced, for the rest of the sample, by the helper Table ``admit``
  builds, and the tracer records that Table as the formula's result (``substitute``). Tracing,
  native binding, schema and every downstream trace are then those of ``pdn.table(frame)``. The
  compiled cell needs no generated code of its own: its binder converts the formula's DataFrame
  with the same ``admit`` at run time (``_native_inputs.table_from_value``).
- A frame failing a condition keeps its value (plain pandas, as live) and is reported with each
  reason; its reads are the pandas receivers of the report (class B or C).
- The same frame returned by several cells is admitted once (by identity); a cell admitted more
  than once in the sample (a table built per policy) is reported.
- A pandas Series is never admitted; it is counted.

``FALLBACK_HOOK`` of ``modelx_helpers.immutable_inputs`` is set while the sample runs: each pandas
fallback on an admitted table is counted against the traced formula running it and its line
(the tracer's own stack of active calls), so the report gives call counts per read site.
"""

import collections
import contextlib
import importlib
import logging
import sys

from modelx_cython.consts import FORMULA_PREF, MX_MODEL_MOD, MX_SPACE_MOD, SPACE_PREF

_logger = logging.getLogger(__name__)


class SampleAdmission:
    """What admission saw while one sample ran."""

    def __init__(self):
        self.tables = {}        # id(frame) -> (frame, table, first cell)
        self.paths = {}         # id(table) -> (table, first cell)
        self.cells = {}         # cell path -> {'status', 'count', 'reasons', 'funcname', ...}
        self.series = {}        # cell path -> calls
        self.fallbacks = collections.Counter()      # (formula funcname, line, operation) -> calls
        self.tracer = None      # the active MxCallTracer (its ``traces``: frame -> CallTrace)

    # ------------------------------------------------------------ values
    def observe(self, value, path, funcname=None):
        """The value the sample continues with for the zero-argument cell ``path``."""
        module = type(value).__module__
        if module.split('.')[0] != 'pandas':
            first = self.paths.get(id(value))
            if first is not None and first[0] is value and first[1] != path:   # an admitted table passed on
                record = self.cells.setdefault(path, {'status': 'admitted', 'count': 0, 'reasons': [],
                                                      'funcname': funcname, 'same_frame_as': first[1]})
                record['count'] += 1
            return value
        import pandas as pd
        if isinstance(value, pd.Series):
            self.series[path] = self.series.get(path, 0) + 1
            return value
        if not isinstance(value, pd.DataFrame):
            return value
        record = self.cells.setdefault(path, {'status': None, 'count': 0, 'reasons': [], 'funcname': funcname})
        record['count'] += 1
        known = self.tables.get(id(value))
        if known is not None and known[0] is value:     # the same frame returned by another cell
            record['status'] = 'admitted'
            if known[2] != path:
                record['same_frame_as'] = known[2]
            return known[1]
        from modelx_cython.runtime.table_admission import TableAdmissionError, admit
        try:
            table = admit(value, path)
        except TableAdmissionError as error:
            record['status'] = 'not admitted'
            record['reasons'] = list(error.reasons)
            return value
        record['status'] = 'admitted'
        if getattr(type(value.index), '__name__', '') == 'RangeIndex' and value.index.name is None:
            record['row_numbers_as_keys'] = True
        self.tables[id(value)] = (value, table, path)
        self.paths[id(table)] = (table, path)
        return table

    def substitute(self, traces):
        """Replace each traced admitted frame by its table (before the traces are merged); the count."""
        count = 0
        for trace in traces:
            known = self.tables.get(id(trace.ret_val))
            if known is not None and known[0] is trace.ret_val:
                trace.ret_val = known[1]
                count += 1
        return count

    # ------------------------------------------------------------ pandas fallbacks
    def fallback(self, operation):
        tracer = self.tracer
        frames = getattr(tracer, 'traces', None)
        if not frames:
            return
        frame = next(reversed(frames))      # the innermost active traced call
        trace = frames[frame]
        line = frame.f_lineno - frame.f_code.co_firstlineno + 1
        self.fallbacks[trace.funcname, line, operation] += 1

    def record_for(self, funcname):
        """The admission record of the zero-argument formula ``funcname`` (tracer name), or None."""
        for record in self.cells.values():
            if record.get('funcname') == funcname:
                return record
        return None

    def fallback_counts(self, funcname):
        """``{line: {operation: calls}}`` for one traced formula."""
        result = {}
        for (name, line, operation), calls in self.fallbacks.items():
            if name == funcname:
                result.setdefault(line, {})[operation] = calls
        return result

    # ------------------------------------------------------------ report
    def summary(self, rename=None):
        rename = rename or (lambda name: name)
        admitted = sorted(rename(p) for p, r in self.cells.items() if r['status'] == 'admitted')
        return {
            'admitted': admitted,
            'not_admitted': {rename(p): r['reasons'] for p, r in sorted(self.cells.items())
                             if r['status'] == 'not admitted'},
            'built_more_than_once': {rename(p): r['count'] for p, r in sorted(self.cells.items())
                                     if r['status'] == 'admitted' and r['count'] > 1},
            'same_frame_as': {rename(p): rename(r['same_frame_as']) for p, r in sorted(self.cells.items())
                              if r.get('same_frame_as')},
            'row_numbers_as_keys': sorted(rename(p) for p, r in self.cells.items() if r.get('row_numbers_as_keys')),
            'series_cells': {rename(p): n for p, n in sorted(self.series.items())},
        }

    def warnings(self):
        """Build warnings: one per failed condition, per table built more than once, per Series cell."""
        result = []
        for path, record in sorted(self.cells.items()):
            if record['status'] == 'not admitted':
                result += [f'{path}: not admitted as a table: {reason}' for reason in record['reasons']]
            elif record['count'] > 1:
                result.append(f'{path}: admitted as a table, but built {record["count"]} times in the sample '
                              '(a table built per policy?): build it once in a Space without per-policy reset '
                              'and select with .loc[key]')
        return result


def _formula_classes(package):
    for name, module in list(sys.modules.items()):
        if module is None or not (name == package or name.startswith(package + '.')):
            continue
        if name.rsplit('.', 1)[-1] not in (MX_MODEL_MOD, MX_SPACE_MOD):
            continue
        for cls in list(vars(module).values()):
            if isinstance(cls, type) and cls.__module__ == name and cls.__name__.startswith(SPACE_PREF):
                yield module, cls


def _path(module, cls, formula):
    """``Space.cell`` with the owning Spaces (``Outer.Inner.cell``), as the build report names cells."""
    owners = [part[len('_m_'):] for part in module.__name__.split('.')[1:-1] if part.startswith('_m_')]
    return '.'.join((*owners, cls.__name__[len(SPACE_PREF):], formula[len(FORMULA_PREF):]))


@contextlib.contextmanager
def sample_admission(package):
    """Wrap the zero-argument formulas of the imported model ``package`` for one sample run."""
    from modelx_helpers import immutable_inputs
    admission = SampleAdmission()
    importlib.import_module(package)
    wrapped = []
    for module, cls in _formula_classes(package):
        for name, function in list(vars(cls).items()):
            code = getattr(function, '__code__', None)
            if not name.startswith(FORMULA_PREF) or code is None or code.co_argcount != 1:
                continue
            wrapped.append((cls, name, function))
            setattr(cls, name, _wrapper(admission, function, _path(module, cls, name)))
    previous = immutable_inputs.FALLBACK_HOOK
    immutable_inputs.FALLBACK_HOOK = admission.fallback
    try:
        yield admission
    finally:
        immutable_inputs.FALLBACK_HOOK = previous
        for cls, name, function in wrapped:
            setattr(cls, name, function)


def _wrapper(admission, function, path):
    funcname = f'{function.__module__}.{function.__qualname__}'

    def formula(self):
        return admission.observe(function(self), path, funcname)
    formula.__wrapped__ = function      # the tracer resolves the traced code through it
    formula.__name__ = function.__name__
    formula.__qualname__ = function.__qualname__
    return formula


# ---------------------------------------------------------------- the per-cell report
PANDAS_C = ('{cell}: {read} (formula line {line}) runs pandas on an admitted table on every call '
            '({calls}); the result equals live. Native forms: T.loc[key, "col"], key in T.index, '
            'T.index.asof(x), or select once per policy in a cell without arguments')


def classify(cells, sites, lowered, counts):
    """Entries of ``table_admission.reads`` for one formula.

    ``sites`` are TableReadSite objects on admitted-table receivers (every use of the receiver),
    ``lowered`` the ids of the nodes lowered natively, ``counts`` the formula's pandas fallbacks
    in the sample by line. Class A: lowered; B: not lowered, in a cell without arguments; C: not
    lowered, in a cell with arguments.
    """
    per_step = cells.has_args()
    entries = []
    for site in sites:
        if id(site.node) in lowered:
            klass = 'A'
        else:
            klass = 'C' if per_step else 'B'
        entry = {'read': site.read, 'kind': site.kind, 'line': site.line, 'class': klass}
        if klass != 'A':
            calls = sum(counts.get(site.line, {}).values())
            entry['pandas_calls_in_sample'] = calls
            if not calls:
                entry['note'] = ('not reached by the sample, or answered by the table without pandas; '
                                 'gives the pandas result when reached')
        entries.append(entry)
    return entries


def cell_report(cell):
    """The ``table_admission`` entry of a cell in ``_mx_admission.json``, or None."""
    from modelx_cython.admission import PANDAS_RECEIVER
    entry = {}
    record = getattr(cell, 'admission_record', None)
    if record:
        entry['status'] = record['status']
        entry['built_in_sample'] = record['count']
        for key in ('reasons', 'same_frame_as', 'row_numbers_as_keys'):
            if record.get(key):
                entry[key] = record[key]
    reads = getattr(cell, 'table_admission_reads', None)
    if reads:
        entry['reads'] = reads
    per_step = cell.has_args()
    pandas = [{'read': e['read'], 'kind': e['kind'], 'line': e['line'], 'class': 'C' if per_step else 'B',
               'receiver': 'pandas value (not admitted)'}
              for e in getattr(cell, 'table_reads_untyped', ()) or () if e.get('reason') == PANDAS_RECEIVER]
    if pandas:
        entry['pandas_reads'] = pandas
    return entry or None


def display_name(fqname):
    """``Outer.Space.cell`` for a report key such as ``pkg._mx_classes._c_Space._f_cell``."""
    from modelx_cython.runtime.table_admission import display_name as name
    return name(fqname)


def per_step_warnings(name, entry):
    """Build warnings for the class C reads of one cell's ``table_admission`` entry."""
    name = display_name(name)
    result = []
    for read in entry.get('reads', ()):
        if read['class'] == 'C':
            calls = read.get('pandas_calls_in_sample', 0)
            result.append(PANDAS_C.format(cell=name, read=read['read'], line=read['line'],
                                          calls=f'{calls} calls in the sample' if calls
                                          else 'not reached by the sample'))
    return result
