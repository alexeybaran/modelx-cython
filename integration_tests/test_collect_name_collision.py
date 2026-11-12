import subprocess
import sys
import textwrap

import modelx as mx


def test_compiled_collect_cell_and_aggregation_remain_callable(tmp_path):
    model = mx.new_model("CollectCollision")
    try:
        plain = model.new_space("Plain")
        plain.new_cells("collect", "lambda: 4.0")
        keys = (1, 2)
        policies = model.new_space(
            "Policy",
            formula="lambda policy_id: None",
            refs={"keys": keys},
            collect_over={"policy_id": keys},
            collect_outputs='lambda: {"sum": {"amount": amount()}}',
        )
        policies.new_cells("collect", "lambda: 5.0")
        policies.new_cells("amount", "lambda: collect() + policy_id")
        model.export(tmp_path / "exported")
    finally:
        model.close()

    (tmp_path / "sample.py").write_text(textwrap.dedent("""\
        from exported import mx_model
        mx_model.Plain.collect()
        mx_model.Policy.collect()
        mx_model.Policy[2].amount()
        mx_model.Policy._mx_collect()
        """), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-B", "-m", "modelx_cython", "exported", "--no-spec"],
        cwd=tmp_path,
        check=True,
    )
    source = (tmp_path / "exported_cy" / "_mx_classes.py").read_text(encoding="utf-8")
    assert "evaluate_serial=self._mx_evaluate_serial_collect" in source.replace(" ", "")

    (tmp_path / "probe.py").write_text(textwrap.dedent("""\
        from importlib.machinery import ExtensionFileLoader
        from exported_cy import _mx_classes, mx_model

        assert isinstance(_mx_classes.__loader__, ExtensionFileLoader)
        assert mx_model.Plain.collect() == 4.0
        assert mx_model.Policy.collect() == 5.0
        assert mx_model.Policy._cells["collect"]() == 5.0
        assert mx_model.Policy[2].collect() == 5.0
        assert mx_model.Policy[2].amount() == 7.0
        assert mx_model.Policy._mx_collect().sum.amount == 13.0
        """), encoding="utf-8")
    subprocess.run([sys.executable, "-B", "probe.py"], cwd=tmp_path, check=True)
