"""Verify private Collect reuse with rebound object inputs and native output domains."""

import subprocess
import sys
import textwrap

import modelx as mx
import pytest


@pytest.mark.parametrize('dependency', ['row', 'cached_collect', 'unhashable_default'])
def test_factory_private_refs(tmp_path, dependency):
    model = mx.new_model("PrivateFactory")
    use_collect = dependency != 'row'
    admitted = dependency == 'cached_collect'
    try:
        factory = ('lambda case, point, options=[]: None' if dependency == 'unhashable_default'
                   else 'lambda case, point: None' if use_collect else None)
        data = model.new_space("Data", refs={"weights": ((3.0,), (5.0,), (7.0,), (11.0,)),
                                            "points": (0, 1, 2, 3)},
                               formula=factory)
        if use_collect:
            data.new_cells('domain', 'lambda: points')
            data.new_cells('row', 'lambda: (weights[point][0] + 100.0 * case,)')
            data.set_collect(over={'point': data.domain}, outputs=lambda: {"individual": {"row": row()}})
            row = 'record.row'
            history = '\n    record = data.collect(case).individual[point]'
        else:
            data.new_cells("row", "lambda i: weights[i]")
            row, history = 'data.row(point)', ''
        run = model.new_space(
            "Runs", refs={"data": data, "row": (0.0,), "points": (0, 1, 2, 3)},
            formula=f"""def factory(case, point):
    if case < 0:
        raise ValueError('negative case'){history}
    return {{'refs': {{'row': {row}}}}}
""",
        )
        run.new_cells("value", "lambda t: row[0] + t")
        if admitted:
            run.new_cells('end', 'lambda: case + 1')
            run.new_cells('total', """def total():
    result = 0.0
    for k in range(end()):
        result += value(k)
    return result
""")
        else:
            run.new_cells("total", "lambda: value(case)")
        run.new_cells("steps", "lambda case: range(case)")
        run.set_collect(over={"point": run.points},
                        outputs=lambda: {"sum": {"total": total(), "series": {t: value(t) for t in steps(case)}}})
        model.export(tmp_path / "exported")
    finally:
        model.close()
    source = (tmp_path / "exported" / "_mx_sys.py").read_text()
    if "_mx_collect_packed_sums_v1 = True" not in source:
        pytest.skip("requires exporter with packed Collect sums")
    options = {"factory_collect_reuse": True, "factory_collect_private_refs": True,
               "factory_collect_subtotals": True, "serial_collect": "reduce",
               'factory_collect_hold_individuals': use_collect,
               'factory_collect_bind_fixed_refs': admitted}
    (tmp_path / "spec.py").write_text(repr({"spaces": {"Runs": options}, "cells_param_size": {"t": 8, "case": 8, 'k': 9}}))
    (tmp_path / "sample.py").write_text("from exported import mx_model as m\nm.Runs.collect(2)\n")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported"], cwd=tmp_path, check=True)
    generated = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert ('_mx_factory_individuals' in generated) == admitted
    assert ("refs['row'])" in generated) == admitted
    probe = textwrap.dedent(f"""\
        from exported_cy import mx_model as m
        from exported_cy._mx_collect_semantics import NativeCollectedSeries
        options = m.Runs._mx_collect_options_v2()
        assert options[1] and options[2]
        direct = m.Runs[2, 0]
        use_collect = {use_collect!r}
        admitted = {admitted!r}
        assert direct.total() == (612. if admitted else 205. if use_collect else 5.)
        if admitted:
            memo = [None]
            m.Runs._mx_space_formula(2, 0, _mx_factory_individuals=memo)
            if type(m.Data).__flags__ & 256 and type(m.Data).__dictoffset__ == 0:
                assert memo[0] is m.Data.collect(2).individual
                result = m.Runs._mx_space_formula(2, 1, _mx_factory_individuals=memo)
                assert result['refs']['row'] == (205.,)
        for workers, case in ((1, 2), (4, 3)):
            m.collect_workers = workers
            result = m.Runs.collect(case)
            total = 26. + (400. * case if use_collect else 0.)
            expected = (case + 1) * total + 2 * case * (case + 1) if admitted else total + 4 * case
            assert result.sum.total == expected
            assert type(result.sum.series) is NativeCollectedSeries
            assert list(result.sum.series.values()) == [total + 4 * t for t in range(case)]
        try:
            m.Runs.collect(-1)
        except ValueError as error:
            assert str(error) == 'negative case'
        else:
            raise AssertionError('factory validation bypassed')
        assert m.Runs[2, 0] is direct
        """)
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, check=True)
