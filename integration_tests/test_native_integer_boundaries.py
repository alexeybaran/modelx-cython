"""Native integer boundaries reject lossy conversions with rebuild guidance."""

import json
import operator
import subprocess
import sys
import textwrap

import modelx as mx
import pytest


def run_python(work, *args):
    result = subprocess.run([sys.executable, "-B", *args], cwd=work,
                            capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def integer_domain_build(tmp_path_factory):
    work = tmp_path_factory.mktemp("integer-domains")
    model = mx.new_model("IntegerDomains")
    try:
        projection = model.new_space("Outer").new_space(
            "Projection", refs={"payload": {"value": 2}, "amount": 30,
                                "get_item": operator.getitem})
        projection.new_cells("object_result", 'lambda: payload["value"]')
        projection.new_cells("conditional_result", 'lambda: payload["value"] if amount > 0 else 0')
        projection.new_cells("helper_result", 'lambda: get_item(payload, "value")')
        projection.new_cells("read_amount", "lambda: amount")
        policies = model.new_space("Policies", formula="lambda policy: {}")
        policies.new_cells("number", "lambda: policy")
        assert policies[4].number() == 4
        assert projection.conditional_result() == 2
        assert projection.object_result() == projection.helper_result() == 2
        assert projection.read_amount() == 30
        model.export(work / "exported")
    finally:
        model.close()

    (work / "sample.py").write_text(textwrap.dedent("""\
        import numpy as np
        from exported import mx_model

        projection = mx_model.Outer.Projection
        for count in (0, 2, 6, np.int64(4)):
            mx_model.Policies[count].number()
        projection.object_result()
        projection.conditional_result()
        projection.helper_result()
        projection.read_amount()
        """), encoding="utf-8")
    run_python(work, "-m", "modelx_cython", "exported", "--sample", "sample.py", "--no-spec")

    report = json.loads((work / "exported_cy/_mx_admission.json").read_text(encoding="utf-8"))

    def admission(cell):
        result, = (row for name, row in report.items() if name.endswith(f"._f_{cell}"))
        return result

    for name in ("object_result", "conditional_result", "helper_result", "read_amount"):
        assert admission(name)["return_type"] == "long long"

    (work / "check_boundaries.py").write_text(textwrap.dedent(r"""
        from importlib import import_module
        from importlib.machinery import ExtensionFileLoader
        import math
        import sys

        import numpy as np
        from exported import mx_model as exported
        from exported_cy import mx_model as compiled

        case = sys.argv[1]
        prefix = "Outer.Projection."
        if case == "reference":
            # An eager setter can name the Reference instead of the consuming Cell.
            cell_paths = (prefix + "amount", prefix + "read_amount")
        else:
            cell_paths = (prefix + case,)

        classes = import_module(type(compiled.Outer.Projection).__module__)
        assert isinstance(classes.__loader__, ExtensionFileLoader), classes.__file__

        def calculate(model, value):
            projection = type(model)().Outer.Projection
            if case == "reference":
                projection.amount = value
                return projection.read_amount()
            projection.payload = {"value": value}
            return getattr(projection, case)()

        valid = (2, np.int64(4), -1, -(1 << 63), np.int64((1 << 63) - 1))
        for value in valid:
            assert calculate(exported, value) == value, (case, value)
            assert calculate(compiled, value) == value, (case, value)

        invalid = (2.5, np.float64(2.5), float("nan"), 1 << 70,
                   -(1 << 70), np.uint64(1 << 63))
        for value in invalid:
            exported_result = calculate(exported, value)
            if math.isnan(value):
                assert math.isnan(exported_result), (case, value)
            else:
                assert exported_result == value, (case, value)
            detail = f"{case}, value={value!r}"
            try:
                actual = calculate(compiled, value)
            except (TypeError, ValueError, OverflowError) as error:
                message = str(error)
                assert any(path in message for path in cell_paths), (detail, message)
                assert "Rebuild with a wider sample or spec" in message, (detail, message)
            else:
                raise AssertionError(f"{detail}: compiled calculation returned {actual!r}")
        """), encoding="utf-8")
    return work


@pytest.mark.parametrize("case", [
    "object_result",
    "conditional_result",
    "helper_result",
    "reference",
])
def test_native_integer_boundary(integer_domain_build, case):
    run_python(integer_domain_build, "check_boundaries.py", case)


def test_integer_itemspace_parameter(integer_domain_build):
    run_python(integer_domain_build, "-c", textwrap.dedent("""
        import numpy as np
        from exported_cy import mx_model

        policies = mx_model.Policies
        assert policies[np.int64(4)].number() == 4
        for bad in (2.5, float("nan"), 1 << 70, np.uint64(1 << 63)):
            try:
                policies[bad]
            except (TypeError, OverflowError) as error:
                assert "Policies" in str(error) and "wider sample or spec" in str(error)
            else:
                raise AssertionError(f"ItemSpace accepted an out-of-domain key {bad!r}")
        """))
