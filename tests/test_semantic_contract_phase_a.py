from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelx_graph.semantic_contract import SemanticContractError, build_semantic_facade


def _frontend_cell(
    uid: str,
    name: str,
    *,
    result_dtype: str = "float64",
    parameters=(),
    parameter_categories=(),
    result_category=None,
    ndim: int = 0,
    shape=None,
):
    return SimpleNamespace(
        uid=uid,
        source_name=name,
        source_fullname=f"m.Space._f_{name}",
        result_dtype=result_dtype,
        parameters=tuple(SimpleNamespace(name=n, dtype=d) for n, d in parameters),
        parameter_category_domains=tuple(parameter_categories),
        result_category_domain=result_category,
        result_ndim=ndim,
        result_shape=(() if ndim == 0 and shape is None else shape),
        result_shape_source="test_trace",
    )


def _catalog_cell(
    uid: str,
    name: str,
    *,
    dtype: str = "float64",
    parameters=(),
    parameter_categories=(),
    result_category=None,
    ndim: int = 0,
    shape=None,
):
    names = tuple(n for n, _ in parameters)
    dtypes = tuple(d for _, d in parameters)
    domains = dict(parameter_categories)
    return SimpleNamespace(
        uid=uid,
        source_name=name,
        source_fullname=f"m.Space._f_{name}",
        dtype=dtype,
        parameters=names,
        parameter_dtypes=dtypes,
        parameter_category_domains=tuple(domains.get(n) for n in names),
        result_category_domain=result_category,
        logical_result_ndim=ndim,
        logical_result_shape=shape,
    )


def _execution_row(
    uid: str,
    name: str,
    *,
    execution_class="N0",
    parameter_dtypes=(),
    result_dtype="float64",
    value_abi="native_scalar",
):
    memo = SimpleNamespace(
        kind="scalar" if not parameter_dtypes else "dense",
        eligible=True,
        physicalization_ready=True,
        bounds=tuple((0, 4) for _ in parameter_dtypes),
        size=1 if not parameter_dtypes else 5 ** len(parameter_dtypes),
        hash_capacity=0,
    )
    return SimpleNamespace(
        uid=uid,
        source_name=name,
        source_fullname=f"m.Space._f_{name}",
        execution_class=execution_class,
        parameter_dtypes=tuple(parameter_dtypes),
        result_dtype=result_dtype,
        value_abi=value_abi,
        native_scalar_abi=(value_abi == "native_scalar"),
        bridgeable_python_fallback=(value_abi != "unsupported"),
        memo=memo,
        direct_blockers=(),
        classification_reasons=("test",),
    )


def _plan(*, cells, catalog, execution_rows, output_uid, ranked_output=None, category_ids=()):
    return SimpleNamespace(
        cells=tuple(cells),
        cross_space_cells=(),
        source_transparent_cell_catalog=dict(catalog),
        execution_class_plan=SimpleNamespace(cells=dict(execution_rows)),
        output_uid=output_uid,
        ranked_output=ranked_output,
        category_plan=SimpleNamespace(
            encoding=SimpleNamespace(domains={domain_id: object() for domain_id in category_ids})
        ),
    )


def test_category_semantics_are_separate_from_int64_physical_abi():
    front = _frontend_cell(
        "u_grade", "grade_share",
        parameters=(("grade", "int64"),),
        parameter_categories=(("grade", "cat_grade"),),
    )
    catalog = _catalog_cell(
        "u_grade", "grade_share",
        parameters=(("grade", "int64"),),
        parameter_categories=(("grade", "cat_grade"),),
    )
    row = _execution_row(
        "u_grade", "grade_share", parameter_dtypes=("int64",)
    )
    facade = build_semantic_facade(_plan(
        cells=(front,),
        catalog={"u_grade": catalog},
        execution_rows={"u_grade": row},
        output_uid="u_grade",
        category_ids=("cat_grade",),
    ))

    signature = facade.signatures["u_grade"]
    assert signature.parameters[0].value.logical_dtype == "category"
    assert signature.parameters[0].value.category_domain_id == "cat_grade"
    assert facade.lowerings["u_grade"].parameter_abi_dtypes == ("int64",)
    assert facade.lowerings["u_grade"].key_abi == "native_tuple"


def test_ranked_output_gets_authoritative_point_axis_but_equal_length_peer_does_not():
    output = _frontend_cell(
        "u_out", "pv", result_dtype="float64", ndim=1, shape=(3,)
    )
    peer = _frontend_cell(
        "u_peer", "age", result_dtype="int64", ndim=1, shape=(3,)
    )
    output_catalog = _catalog_cell(
        "u_out", "pv", dtype="float64", ndim=1, shape=(3,)
    )
    peer_catalog = _catalog_cell(
        "u_peer", "age", dtype="int64", ndim=1, shape=(3,)
    )
    ranked = SimpleNamespace(
        dtype="float64",
        ndim=1,
        shape=(3,),
        axis_keys=(101, 102, 103),
        point_frame_ref="model_point_table",
        authority="test_ranked_trace",
    )
    facade = build_semantic_facade(_plan(
        cells=(output, peer),
        catalog={"u_out": output_catalog, "u_peer": peer_catalog},
        execution_rows={
            "u_out": _execution_row("u_out", "pv", value_abi="ranked_array"),
            "u_peer": _execution_row("u_peer", "age", result_dtype="int64", value_abi="ranked_array"),
        },
        output_uid="u_out",
        ranked_output=ranked,
    ))

    out_axis = facade.signatures["u_out"].result.axes[0]
    peer_axis = facade.signatures["u_peer"].result.axes[0]
    assert out_axis.kind == "point"
    assert out_axis.keys == (101, 102, 103)
    assert peer_axis.kind == "unresolved"
    assert peer_axis.axis_id != out_axis.axis_id
    assert peer_axis.size == out_axis.size == 3


def test_lowering_is_physical_and_does_not_change_semantic_rank():
    front = _frontend_cell("u", "f", result_dtype="float64", ndim=1, shape=(2,))
    catalog = _catalog_cell("u", "f", dtype="float64", ndim=1, shape=(2,))
    ranked = SimpleNamespace(
        dtype="float64", ndim=1, shape=(2,), axis_keys=(0, 1),
        point_frame_ref=None, authority="test",
    )
    row = _execution_row("u", "f", value_abi="ranked_array")
    facade = build_semantic_facade(_plan(
        cells=(front,), catalog={"u": catalog}, execution_rows={"u": row},
        output_uid="u", ranked_output=ranked,
    ))
    assert facade.signatures["u"].result.ndim == 1
    assert facade.lowerings["u"].result_value_abi == "ranked_array"
    assert facade.lowerings["u"].execution_backend == "native"


def test_frontend_catalog_disagreement_fails_closed():
    front = _frontend_cell("u", "f", result_dtype="float64")
    catalog = _catalog_cell("u", "f", dtype="int64")
    with pytest.raises(SemanticContractError, match="result dtype mismatch"):
        build_semantic_facade(_plan(
            cells=(front,), catalog={"u": catalog},
            execution_rows={"u": _execution_row("u", "f")}, output_uid="u",
        ))


def test_manifest_is_deterministic_and_does_not_embed_axis_keys():
    front = _frontend_cell("u", "f", ndim=1, shape=(3,))
    catalog = _catalog_cell("u", "f", ndim=1, shape=(3,))
    ranked = SimpleNamespace(
        dtype="float64", ndim=1, shape=(3,), axis_keys=("a", "b", "c"),
        point_frame_ref="points", authority="trace",
    )
    plan = _plan(
        cells=(front,), catalog={"u": catalog},
        execution_rows={"u": _execution_row("u", "f", value_abi="ranked_array")},
        output_uid="u", ranked_output=ranked,
    )
    first = build_semantic_facade(plan).manifest()
    second = build_semantic_facade(plan).manifest()
    assert first == second
    axis = next(iter(first["axes"].values()))
    assert axis["key_count"] == 3
    assert "key_sha256" not in axis
    assert "keys" not in axis
    assert first["role"] == "behavior_preserving_semantic_facade_not_yet_codegen_authority"
