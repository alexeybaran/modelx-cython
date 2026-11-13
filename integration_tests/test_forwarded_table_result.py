"""Keep one owned native table result across ordinary Cell forwarding."""

import json
import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd
from modelx_helpers import pdn
from modelx_helpers.immutable_inputs import Table


def test_native_table_forwarding(tmp_path):
    model = mx.new_model("ForwardedTable")
    try:
        inputs = model.new_space("Inputs")
        inputs.owner = Table((0, 1), ((2.0,), (5.0,)), ("rate",))
        inputs.pdn = pdn
        inputs.source_frame = pd.DataFrame({"rate": [2.0, 5.0]}, index=[0, 1])
        inputs.new_cells("constructed", "lambda: pdn.table(source_frame)")
        inputs.new_cells("forwarded_constructor", "lambda: constructed()")
        inputs.new_cells("conditional_constructor", "lambda groups: pdn.table(source_frame, groups=groups)")
        inputs.new_cells("direct", '''\
def direct() -> "modelx_helpers.immutable_inputs.Table":
    return owner
''')
        inputs.new_cells("shared", "lambda: direct()")
        inputs.new_cells("unannotated", "lambda: owner")
        calculation = model.new_space("Calculation")
        calculation.inputs = inputs
        calculation.new_cells("direct_source", '''\
def direct_source() -> "modelx_helpers.immutable_inputs.Table":
    return inputs.direct()
''')
        calculation.new_cells("first_hop", "lambda: inputs.shared()")
        calculation.new_cells("second_hop", '''\
def second_hop():
    """Forward the result without changing its owner."""
    return first_hop()
''')
        calculation.new_cells("object_hop", '''\
def object_hop() -> object:
    return first_hop()
''')
        calculation.new_cells("read_direct", 'lambda month: direct_source().lookup_float(month, "rate")')
        calculation.new_cells("read_first", 'lambda month: first_hop().lookup_float(month, "rate")')
        calculation.new_cells("read_second", 'lambda month: second_hop().lookup_float(month, "rate")')
        assert calculation.first_hop() is calculation.second_hop() is inputs.owner
        assert calculation.read_first(0) == calculation.read_second(0) == 2.0

        for name in ("left", "right"):
            model.export(tmp_path / name, locked_spaces=["Inputs"])
    finally:
        model.close()

    for name in ("left", "right"):
        sample = tmp_path / "sample.py"
        sample.write_text(textwrap.dedent(f'''\
            from {name} import mx_model as m
            c = m.Calculation
            assert c.direct_source() is c.first_hop() is c.second_hop() is c.object_hop() is m.Inputs.direct()
            assert m.Inputs.unannotated() is m.Inputs.owner
            assert m.Inputs.constructed() is m.Inputs.forwarded_constructor()
            assert m.Inputs.conditional_constructor(()).lookup_float(0, "rate") == 2.0
            for t, value in enumerate((2.0, 5.0)):
                assert c.read_direct(t) == c.read_first(t) == c.read_second(t) == value
            '''), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "modelx_cython", name, "--no-spec"],
                       cwd=tmp_path, check=True)
        assert "owned-table-1" in (
            tmp_path / f"{name}_cy/_mx_provider_guard.py").read_text(encoding="utf-8")
        report = json.loads((tmp_path / f"{name}_cy/_mx_admission.json").read_text(encoding="utf-8"))
        cells = {key.rsplit("._f_", 1)[-1]: row for key, row in report.items()}
        annotation_reason = (
            "native provider result requires a matching return annotation or direct forwarding from a typed Cell"
        )
        for cell in ("direct", "shared", "direct_source", "first_hop", "second_hop"):
            assert cells[cell]["return_type"].endswith(".NativeTable")
            assert cells[cell]["cache"] == "scalar"
            assert annotation_reason not in cells[cell]["reasons"]
        assert cells["object_hop"]["return_type"] == "object"
        assert cells["unannotated"]["return_type"] == "object"
        assert cells["constructed"]["return_type"].endswith(".NativeTable")
        assert cells["forwarded_constructor"]["return_type"].endswith(".NativeTable")
        assert cells["conditional_constructor"]["return_type"] == "object"
        assert annotation_reason in cells["unannotated"]["reasons"]
        operations = cells["read_direct"]["native_numeric_helpers"]
        assert operations
        assert operations == cells["read_first"]["native_numeric_helpers"]
        assert operations == cells["read_second"]["native_numeric_helpers"]

    probe = textwrap.dedent('''\
        import gc
        from importlib.machinery import ExtensionFileLoader
        import left_cy._mx_classes as left_classes
        import right_cy._mx_classes as right_classes
        from left_cy import mx_model as left
        from right_cy import mx_model as right
        from modelx_helpers.immutable_inputs import Table

        assert isinstance(left_classes.__loader__, ExtensionFileLoader)
        assert isinstance(right_classes.__loader__, ExtensionFileLoader)

        old = type(left)()
        old.Inputs.owner = Table((0, 1), ((2.0,), (5.0,)), ("rate",))
        table = old.Inputs.direct()
        owner = table.__modelx_cython_owner__()
        assert table.__modelx_table_owner__() is owner
        prepared = tuple(owner._native_cache.items())
        assert prepared
        assert old.Inputs.shared() is table
        assert old.Calculation.direct_source() is old.Calculation.first_hop() is table
        assert old.Calculation.second_hop() is table
        assert old.Calculation.object_hop() is table
        assert tuple(owner._native_cache) == tuple(key for key, _ in prepared)
        assert all(owner._native_cache[key] is resource for key, resource in prepared)

        def unexpected_python_lookup(*args, **kwargs):
            raise AssertionError("typed table reads should not call the Python owner method")

        lookup = Table.lookup_float
        Table.lookup_float = unexpected_python_lookup
        try:
            assert old.Calculation.read_direct(0) == old.Calculation.read_first(0) == 2.0
            assert old.Calculation.read_second(0) == 2.0
        finally:
            Table.lookup_float = lookup
        owner_id = id(owner)
        del owner, prepared, old
        gc.collect()

        other = type(right)()
        other.Inputs.owner = table
        rebound = other.Inputs.direct()
        assert type(rebound) is not type(table)
        assert rebound.__modelx_table_owner__() is table.__modelx_table_owner__()
        assert id(rebound.__modelx_table_owner__()) == owner_id
        assert other.Calculation.first_hop() is other.Calculation.second_hop() is rebound
        assert other.Calculation.read_first(1) == other.Calculation.read_second(1) == 5.0

        fresh = type(left)()
        fresh.Inputs.owner = Table((0, 1), ((7.0,), (11.0,)), ("rate",))
        assert fresh.Calculation.read_first(0) == fresh.Calculation.read_second(0) == 7.0
        assert table.lookup_float(0, "rate") == 2.0
        assert fresh.Calculation.second_hop().__modelx_table_owner__() is fresh.Inputs.owner

        class OverriddenTable(Table):
            def lookup_float(self, key, column, default=None):
                return 23.0

        fallback = type(left)()
        fallback.Inputs.owner = OverriddenTable((0, 1), ((7.0,), (11.0,)), ("rate",))
        assert fallback.Calculation.read_first(1) == fallback.Calculation.read_second(1) == 23.0

        invalid = type(left)()
        invalid.Inputs.owner = object()
        try:
            invalid.Calculation.second_hop()
        except TypeError as exc:
            assert "Inputs.direct" in str(exc) and "return_type='object'" in str(exc)
        else:
            raise AssertionError("a changed typed result must fail at its binding boundary")
        ''')
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, check=True)
