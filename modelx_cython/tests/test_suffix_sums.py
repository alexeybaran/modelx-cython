import ast
import subprocess
import sys

import modelx as mx

from modelx_cython.suffix_sums import _pattern


def test_recognize_right_associated_only():
    template = """\
if k >= self.end():
    return 0.0
if _mx_exact_receiver:
    return TERM
return self.value(k) + self.total(k + 1)
"""
    for term, expected in (
        ("self.value(k) + entry(self, k + 1)", True),
        ("entry(self, k + 1) + self.value(k)", False),
        ("self.value(k) + entry(self, k + 2)", False),
    ):
        body = ast.parse(template.replace("TERM", term)).body
        assert (_pattern(body, "k", "entry") is not None) is expected


def test_compiled_suffix_preserves_cached_values_and_fallback(tmp_path):
    model = mx.new_model("SuffixSum")
    try:
        run = model.new_space("Run", formula="lambda size, fail: None")
        run.new_cells("end", "lambda: size")
        run.new_cells("term", """def term(k):
    if fail and k == 3:
        raise ValueError('term failed')
    return 1.0 if k % 2 else 1e16
""")
        run.new_cells("total", """def total(k):
    if k >= end():
        return 0.0
    return term(k) + total(k + 1)
""")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "spec.py").write_text(repr({
        "spaces": {"Run": {"cells_param_size": {"k": 9},
                           "cells": {"total": {"iterative_suffix_sum": True}}}},
    }))
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model as m\nm.Run(8, False).total(0)\n")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported"], cwd=tmp_path, check=True)
    source = (tmp_path / "exported_cy" / "_mx_classes.py").read_text()
    assert "_mx_suffix_malloc" in source
    probe = """\
from exported_cy import mx_model as m
from exported import mx_model as live
for size in (7, 8):
    compiled = m.Run(size, False)
    expected = live.Run(size, False)
    assert compiled.total(4) == expected.total(4)
    assert compiled.total(0) == expected.total(0)
    assert [compiled.total(k) for k in range(size+1)] == [expected.total(k) for k in range(size+1)]
try:
    m.Run(9, False).total(0)
except IndexError as error:
    assert 'Run.total' in str(error)
else:
    raise AssertionError('expected terminal index outside the declared cache')
failing = m.Run(7, True)
failing.end()
try:
    failing.total(0)
except ValueError as error:
    assert str(error) == 'term failed'
else:
    raise AssertionError('expected term failure')
assert failing.total(4) == live.Run(7, False).total(4)
"""
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, check=True)
