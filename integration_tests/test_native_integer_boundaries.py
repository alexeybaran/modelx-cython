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
            "Projection", refs={"payload": {"value": 2}, "amount": 30, "divisor": -1, "shift": 1,
                                "get_item": operator.getitem})
        # An object result keeps the argument checks independent of native
        # result admission and dense cache range checks.
        projection.new_cells("receive", 'lambda count: {"count": count}')
        projection.new_cells("receive_default", 'lambda count=4: {"count": count}')
        projection.new_cells("internal_argument", 'lambda: receive(payload["value"])')
        projection.new_cells("internal_operand", 'lambda: receive(payload["value"] + 1)')
        projection.new_cells("internal_keyword", 'lambda: receive_default(count=payload["value"])')
        projection.new_cells("object_result", 'lambda: payload["value"]')
        projection.new_cells("conditional_result", 'lambda: payload["value"] if amount > 0 else 0')
        projection.new_cells("helper_result", 'lambda: get_item(payload, "value")')
        projection.new_cells("read_amount", "lambda: amount")
        projection.new_cells("increment_amount", "lambda: amount + 1")
        projection.new_cells("decrement_amount", "lambda: amount - 1")
        projection.new_cells("square_amount", "lambda: amount * amount")
        projection.new_cells("divide_amount", "lambda: amount // divisor")
        projection.new_cells("left_shift", "lambda: amount << shift")
        projection.new_cells("right_shift", "lambda: amount >> shift")
        projection.new_cells("remainder", "lambda: amount % divisor")
        projection.new_cells("negate_amount", "lambda: -amount")
        projection.new_cells("local_amount", """def local_amount():
    adjusted = amount - 1
    return adjusted + 1
""")
        projection.new_cells("native_operand", "lambda: receive(amount + 1)")
        projection.new_cells("inline_receive", "lambda count: count + 1")
        projection.new_cells("inline_operand", 'lambda: {"count": inline_receive(amount * 4)}')
        projection.new_cells("shift_argument", 'lambda: {"count": inline_receive(amount << shift)}')
        policies = model.new_space("Policies", formula="lambda policy: {}")
        policies.new_cells("number", "lambda: policy")
        assert policies[4].number() == 4
        assert projection.receive(2) == projection.internal_argument() == {"count": 2}
        assert projection.internal_operand() == {"count": 3}
        assert projection.receive_default() == {"count": 4}
        assert projection.internal_keyword() == {"count": 2}
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
            projection.receive(count)
            projection.receive_default(count)
            mx_model.Policies[count].number()
        projection.receive_default()
        projection.internal_argument()
        projection.internal_operand()
        projection.internal_keyword()
        projection.object_result()
        projection.conditional_result()
        projection.helper_result()
        projection.read_amount()
        projection.increment_amount()
        projection.decrement_amount()
        projection.square_amount()
        projection.divide_amount()
        projection.left_shift()
        projection.right_shift()
        projection.remainder()
        projection.negate_amount()
        projection.local_amount()
        projection.native_operand()
        projection.inline_receive(0)
        projection.inline_operand()
        projection.shift_argument()
        """), encoding="utf-8")
    run_python(work, "-m", "modelx_cython", "exported", "--sample", "sample.py", "--no-spec")

    report = json.loads((work / "exported_cy/_mx_admission.json").read_text(encoding="utf-8"))

    def admission(cell):
        result, = (row for name, row in report.items() if name.endswith(f"._f_{cell}"))
        return result

    assert admission("receive")["arguments"]["count"] == "long long"
    assert admission("receive")["return_type"] == "object"
    assert admission("receive_default")["arguments"]["count"] == "long long"
    assert admission("inline_receive")["inline_entry"]
    for name in ("object_result", "conditional_result", "helper_result", "read_amount"):
        assert admission(name)["return_type"] == "long long"

    (work / "check_boundaries.py").write_text(textwrap.dedent(r"""
        from importlib import import_module
        from importlib.machinery import ExtensionFileLoader
        import math
        import re
        import sys

        import numpy as np
        from exported import mx_model as exported
        from exported_cy import mx_model as compiled

        case = sys.argv[1]
        prefix = "Outer.Projection."
        positional_cases = {"public_argument", "internal_argument", "internal_operand"}
        keyword_cases = {"public_keyword", "internal_keyword"}
        receiving_cases = positional_cases | keyword_cases
        if case in positional_cases:
            cell_paths = (prefix + "receive",)
        elif case in keyword_cases:
            cell_paths = (prefix + "receive_default",)
        elif case == "reference":
            # An eager setter can name the Reference instead of the consuming Cell.
            cell_paths = (prefix + "amount", prefix + "read_amount")
        else:
            cell_paths = (prefix + case,)

        classes = import_module(type(compiled.Outer.Projection).__module__)
        assert isinstance(classes.__loader__, ExtensionFileLoader), classes.__file__

        def calculate(model, value, warm_cache=False):
            projection = type(model)().Outer.Projection
            if warm_cache:
                if case in keyword_cases:
                    assert projection.receive_default() == {"count": 4}
                    projection.receive_default(2)
                else:
                    projection.receive(3 if case == "internal_operand" else 2)
            if case == "public_argument":
                return projection.receive(value)["count"]
            if case == "public_keyword":
                assert projection.receive_default() == {"count": 4}
                return projection.receive_default(count=value)["count"]
            if case == "reference":
                projection.amount = value
                return projection.read_amount()
            projection.payload = {"value": value}
            result = getattr(projection, case)()
            return result["count"] if case in receiving_cases else result

        def expected(value):
            return value + 1 if case == "internal_operand" else value

        valid = (2, np.int64(4))
        if case not in receiving_cases:
            valid += (-1, -(1 << 63), np.int64((1 << 63) - 1))
        for value in valid:
            assert calculate(exported, value) == expected(value), (case, value)
            assert calculate(compiled, value) == expected(value), (case, value)

        invalid = (2.5, np.float64(2.5), float("nan"), 1 << 70,
                   -(1 << 70), np.uint64(1 << 63))
        for value in invalid:
            raw = expected(value)
            exported_result = calculate(exported, value)
            if math.isnan(raw):
                assert math.isnan(exported_result), (case, value)
            else:
                assert exported_result == raw, (case, value)
            for warm_cache in ((False, True) if case in receiving_cases else (False,)):
                detail = f"{case}, value={value!r}, warm_cache={warm_cache}"
                try:
                    actual = calculate(compiled, value, warm_cache)
                except (TypeError, ValueError, OverflowError) as error:
                    message = str(error)
                    assert any(path in message for path in cell_paths), (detail, message)
                    assert re.search(
                        r"\b(?:sample|spec(?:ification)?|return_type|param_type)\b", message, re.I
                    ), (detail, message)
                    assert re.search(
                        r"\b(?:rebuild|recompile|retrace|wider|widen|broader)\b", message, re.I
                    ), (detail, message)
                else:
                    raise AssertionError(f"{detail}: compiled calculation returned {actual!r}")
        """), encoding="utf-8")
    return work


@pytest.mark.parametrize("case", [
    "public_argument",
    "public_keyword",
    "internal_argument",
    "internal_operand",
    "internal_keyword",
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


def test_private_native_entry_checks_integer_argument(integer_domain_build):
    run_python(integer_domain_build, "-c", textwrap.dedent("""\
        from exported import mx_model as exported
        from exported_cy import mx_model as compiled

        reference = type(exported)().Outer.Projection
        projection = type(compiled)().Outer.Projection
        assert reference.inline_operand() == projection.inline_operand() == {"count": 121}

        reference = type(exported)().Outer.Projection
        projection = type(compiled)().Outer.Projection
        reference.amount = projection.amount = 1 << 62
        assert reference.inline_operand() == {"count": (1 << 64) + 1}
        assert projection.inline_receive(0) == 1
        try:
            projection.inline_operand()
        except OverflowError as error:
            assert "Outer.Projection.inline_receive" in str(error)
        else:
            raise AssertionError("private entry accepted a wrapped integer argument")
        """))


def test_native_signed_arithmetic_cannot_wrap_result(integer_domain_build):
    run_python(integer_domain_build, "-c", textwrap.dedent("""\
        from exported import mx_model as exported
        from exported_cy import mx_model as compiled

        maximum, minimum = (1 << 63) - 1, -(1 << 63)
        cases = (
            ("increment_amount", maximum, maximum - 1),
            ("decrement_amount", minimum, minimum + 1),
            ("square_amount", 3037000500, 1 << 31),
            ("divide_amount", minimum, minimum + 1),
            ("negate_amount", minimum, minimum + 1),
        )
        for cell, overflow, valid in cases:
            reference = type(exported)().Outer.Projection
            projection = type(compiled)().Outer.Projection
            reference.amount = projection.amount = valid
            assert getattr(projection, cell)() == getattr(reference, cell)()
            reference = type(exported)().Outer.Projection
            projection = type(compiled)().Outer.Projection
            reference.amount = projection.amount = overflow
            assert not minimum <= getattr(reference, cell)() <= maximum
            try:
                getattr(projection, cell)()
            except OverflowError as error:
                assert f"Outer.Projection.{cell}" in str(error)
            else:
                raise AssertionError(f"{cell} wrapped before conversion")

        reference = type(exported)().Outer.Projection
        projection = type(compiled)().Outer.Projection
        reference.amount = projection.amount = maximum
        assert reference.native_operand() == {"count": maximum + 1}
        try:
            projection.native_operand()
        except OverflowError as error:
            assert "Outer.Projection.receive" in str(error)
        else:
            raise AssertionError("A native operand wrapped before argument conversion")

        reference = type(exported)().Outer.Projection
        projection = type(compiled)().Outer.Projection
        reference.amount = projection.amount = minimum
        assert reference.local_amount() == minimum
        try:
            projection.local_amount()
        except OverflowError as error:
            assert "Outer.Projection.local_amount" in str(error)
        else:
            raise AssertionError("The native local accepted an overflowing intermediate")
        """))


@pytest.mark.parametrize("cell, amount, other, expected", [
    ("left_shift", 1 << 62, 1, "overflow"),
    ("left_shift", 30, -1, "negative shift"),
    ("left_shift", -3, 2, -12),
    ("left_shift", -1, 63, -(1 << 63)),
    ("left_shift", 0, 64, 0),
    ("left_shift", 1, 64, "overflow"),
    ("right_shift", -30, -1, "negative shift"),
    ("right_shift", -(1 << 63), 1, -(1 << 62)),
    ("right_shift", -30, 64, -1),
    ("right_shift", 30, 64, 0),
    ("shift_argument", 30, 1, {"count": 61}),
    ("shift_argument", 1 << 62, 2, "overflow"),
    ("remainder", -(1 << 63), -1, 0),
    ("remainder", -30, 7, 5),
    ("remainder", 30, 0, "division by zero"),
])
def test_native_signed_shift_and_remainder(integer_domain_build, cell, amount, other, expected):
    run_python(integer_domain_build, "-c", textwrap.dedent(f"""\
        from exported_cy import mx_model

        projection = mx_model.Outer.Projection
        projection.amount = {amount}
        projection.shift = projection.divisor = {other}
        cell = {cell!r}
        expected = {expected!r}
        try:
            actual = getattr(projection, cell)()
        except OverflowError as error:
            assert expected == "overflow", str(error)
            name = "inline_receive" if cell == "shift_argument" else cell
            assert f"Outer.Projection.{{name}}" in str(error)
        except ValueError as error:
            assert expected == "negative shift" and str(error) == "negative shift count"
        except ZeroDivisionError:
            assert expected == "division by zero"
        else:
            assert actual == expected, (cell, actual, expected)
        """))
