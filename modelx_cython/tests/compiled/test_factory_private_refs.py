"""Verify private Collect evaluation with rebound object inputs and series outputs."""

import subprocess
import sys
import textwrap

import modelx as mx
import pytest


@pytest.mark.parametrize('dependency', ['row', 'cached_collect', 'unhashable_default'])
def test_factory_private_refs(tmp_path, dependency):
    model = mx.new_model("PrivateFactory")
    use_collect = dependency != 'row'
    accumulates = dependency == 'cached_collect'
    needs_options = dependency == 'unhashable_default'
    try:
        factory = ('lambda case, point, options=[]: None' if needs_options
                   else 'lambda case, point: None' if use_collect else None)
        data = model.new_space("Data", refs={"weights": ((3.0,), (5.0,), (7.0,), (11.0,)),
                                            "points": (0, 1, 2, 3)},
                               formula=factory)
        if use_collect:
            data.new_cells('domain', 'lambda: points')
            data.new_cells('row', 'lambda: (weights[point][0] + 100.0 * case,)')
            data.set_collect(over={'point': data.domain}, outputs=lambda: {"individual": {"row": row()}})
            row = 'record.row'
            args = 'case, ()' if needs_options else 'case'
            history = f'\n    record = data.collect({args}).individual[point]'
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
        if accumulates:
            run.new_cells('end', 'lambda: case + 1')
            run.new_cells('total', """def total():
    result = 0.0
    for k in range(end()):
        result += value(k)
    return result
""")
        else:
            run.new_cells("total", "lambda: value(case)")
        # The uniform series axis must also accept the case rejected by the factory.
        run.set_collect(over={"point": "points"},
                        outputs=lambda: {"sum": {"total": total(), "series": {t: value(t) for t in range(case)}}})
        model.export(tmp_path / "exported", locked_spaces=['Data'])
    finally:
        model.close()
    (tmp_path / "spec.py").write_text(repr({"cells_param_size": {"t": 8, "case": 8, 'k': 9}}), encoding="utf-8")
    (tmp_path / "sample.py").write_text("from exported import mx_model as m\nm.Runs.collect(2)\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported"], cwd=tmp_path, check=True)
    probe = textwrap.dedent(f"""\
        from exported import mx_model as export
        from exported_cy import mx_model as m
        assert type(m.Runs)._mx_collect_hook_version == 1
        direct = m.Runs[2, 0]
        use_collect = {use_collect!r}
        accumulates = {accumulates!r}
        needs_options = {needs_options!r}
        assert direct.total() == export.Runs[2, 0].total() == (612. if accumulates else 205. if use_collect else 5.)
        if use_collect:
            args = (2, ()) if needs_options else (2,)
            assert m.Data.collect(*args).individual[1].row == (205.,)
            assert m.Data.collect(*args) is m.Data.collect(*args)
        if needs_options:
            for model in (export, m):
                try:
                    model.Data.collect(2)
                except TypeError as error:
                    assert 'unhashable' in str(error)
                else:
                    raise AssertionError('an unhashable Collect argument was accepted')
        for workers, case in ((1, 2), (4, 3)):
            m._mx_collect_workers = workers
            result = m.Runs.collect(case)
            live = export.Runs.collect(case)
            total = 26. + (400. * case if use_collect else 0.)
            expected = (case + 1) * total + 2 * case * (case + 1) if accumulates else total + 4 * case
            assert result.sum.total == live.sum.total == expected
            assert result.sum.series[1] == result.sum.series(1)
            assert dict(result.sum.series) == dict(live.sum.series)
            assert list(result.sum.series.values()) == [total + 4 * t for t in range(case)]
        try:
            m.Runs.collect(-1)
        except ValueError as error:
            assert str(error) == 'negative case'
        else:
            raise AssertionError('factory validation bypassed')
        assert m.Runs[2, 0] is direct
        assert m.Runs.collect(2).sum.total == export.Runs.collect(2).sum.total
        """)
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, check=True)
