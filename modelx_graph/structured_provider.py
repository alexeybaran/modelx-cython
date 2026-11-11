from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


class StructuredProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class StructuredOp:
    """One typed relational/row-domain operator in the structured-provider IR.

    ``StructuredFramePlan.kind`` is retained for compatibility with the dev149
    handoff, while this record makes the semantic operator explicit and stable
    enough for diagnostics, manifests and future rule registries.  The payload
    contains only compile-time primitive metadata; pandas/modelx objects never
    become part of this IR.
    """

    kind: str
    parameters: tuple[tuple[str, Any], ...] = ()

    @staticmethod
    def source(ref_name: str, source_kind: str) -> "StructuredOp":
        return StructuredOp(
            "source", (("ref", str(ref_name)), ("source_kind", str(source_kind)))
        )

    @staticmethod
    def left_join(right_ref: str, on: str) -> "StructuredOp":
        return StructuredOp(
            "left_join", (("right_ref", str(right_ref)), ("on", str(on)))
        )

    @staticmethod
    def row_map(row_map: "RowMap") -> "StructuredOp":
        return StructuredOp("row_map", (("row_map_kind", row_map.kind),))

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parameters": {name: value for name, value in self.parameters},
        }


@dataclass(frozen=True)
class StructuredColumnType:
    """Static schema/type evidence for one structured-provider column.

    This is intentionally closer to a compiler type than to a NumPy dtype.
    ``physical_dtype`` is the ABI representation consumed by generated code;
    ``logical_kind`` records the source semantics that produced it.  Finite
    categories therefore remain distinguishable from ordinary integer columns
    even though both are physically encoded as int64.
    """

    logical_kind: str  # bool | int64 | float64 | category
    physical_dtype: str  # int64 | float64
    nullable: bool
    source_dtype: str
    category_labels: tuple[Any, ...] = ()

    def with_nullable(self, nullable: bool = True) -> "StructuredColumnType":
        return replace(self, nullable=bool(nullable or self.nullable))

    def manifest(self) -> dict[str, Any]:
        return {
            "logical_kind": self.logical_kind,
            "physical_dtype": self.physical_dtype,
            "nullable": bool(self.nullable),
            "source_dtype": self.source_dtype,
            "category_labels": [repr(x) for x in self.category_labels],
        }


@dataclass(frozen=True)
class StructuredSourceDemand:
    """Backward-propagated source demand for one projected structured field."""

    source_ref: str
    columns: tuple[str, ...] = ()
    needs_index: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "columns": list(self.columns),
            "needs_index": bool(self.needs_index),
        }


@dataclass(frozen=True)
class RowMap:
    """Logical mapping from output rows to one parent structured provider.

    The map is deliberately small and composable.  It describes geometry only;
    no pandas/DataFrame object is required by generated runtime code.
    """

    kind: str
    output_size: int
    parent_size: int
    repeat: int = 1
    gather: tuple[int, ...] = ()

    @staticmethod
    def identity(size: int) -> "RowMap":
        return RowMap("identity", int(size), int(size))

    @staticmethod
    def repeat_each(parent_size: int, repeat: int) -> "RowMap":
        parent_size = int(parent_size)
        repeat = int(repeat)
        if parent_size < 0 or repeat <= 0:
            raise StructuredProviderError("invalid repeat-each row map")
        return RowMap("repeat_each", parent_size * repeat, parent_size, repeat=repeat)

    @staticmethod
    def gather_map(parent_size: int, gather: Sequence[int], *, kind: str = "gather") -> "RowMap":
        values = tuple(int(x) for x in gather)
        if any(x < 0 or x >= int(parent_size) for x in values):
            raise StructuredProviderError("gather row map exceeds parent domain")
        return RowMap(str(kind), len(values), int(parent_size), gather=values)

    @staticmethod
    def tile(parent_size: int, repeats: int) -> "RowMap":
        parent_size = int(parent_size)
        repeats = int(repeats)
        if parent_size < 0 or repeats <= 0:
            raise StructuredProviderError("invalid tile row map")
        return RowMap.gather_map(
            parent_size, tuple(range(parent_size)) * repeats, kind="tile"
        )

    @staticmethod
    def affine(
        parent_size: int, output_size: int, *, start: int = 0, step: int = 1
    ) -> "RowMap":
        parent_size = int(parent_size); output_size = int(output_size)
        start = int(start); step = int(step)
        if output_size < 0:
            raise StructuredProviderError("invalid affine row-map size")
        return RowMap.gather_map(
            parent_size, (start + step * i for i in range(output_size)), kind="affine"
        )

    @staticmethod
    def filter_mask(parent_size: int, mask: Sequence[bool]) -> "RowMap":
        flags = tuple(bool(x) for x in mask)
        if len(flags) != int(parent_size):
            raise StructuredProviderError("filter mask size disagrees with parent domain")
        return RowMap.gather_map(
            int(parent_size), (i for i, keep in enumerate(flags) if keep), kind="filter"
        )

    @staticmethod
    def concat(parent_size: int, parts: Sequence["RowMap"]) -> "RowMap":
        parent_size = int(parent_size)
        indices: list[int] = []
        for part in parts:
            if int(part.parent_size) != parent_size:
                raise StructuredProviderError("concat row maps must share one parent domain")
            indices.extend(part.indices())
        return RowMap.gather_map(parent_size, indices, kind="concat")

    @staticmethod
    def cartesian_left(left_size: int, right_size: int) -> "RowMap":
        # Projection of a row-major Cartesian(left, right) domain onto left.
        return RowMap.repeat_each(int(left_size), int(right_size))

    @staticmethod
    def cartesian_right(left_size: int, right_size: int) -> "RowMap":
        # Projection of a row-major Cartesian(left, right) domain onto right.
        return RowMap.tile(int(right_size), int(left_size))

    def indices(self) -> tuple[int, ...]:
        if self.kind == "identity":
            return tuple(range(self.parent_size))
        if self.kind == "repeat_each":
            return tuple(i for i in range(self.parent_size) for _ in range(self.repeat))
        if self.gather:
            return self.gather
        if self.output_size == 0:
            return ()
        raise StructuredProviderError(f"row map {self.kind!r} has no index representation")

    def compose(self, child: "RowMap", *, kind: str = "gather") -> "RowMap":
        if child.parent_size != self.output_size:
            raise StructuredProviderError("row-map composition domain mismatch")
        parent_indices = self.indices()
        return RowMap.gather_map(
            self.parent_size, (parent_indices[i] for i in child.indices()), kind=kind
        )

    def apply(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.shape[0] != self.parent_size:
            raise StructuredProviderError(
                f"row map {self.kind} expected parent shape {(self.parent_size,)}, got {arr.shape}"
            )
        if self.kind == "identity":
            return arr
        if self.kind == "repeat_each":
            return np.repeat(arr, self.repeat)
        if self.gather or self.output_size == 0:
            return arr[np.asarray(self.indices(), dtype=np.int64)]
        raise StructuredProviderError(f"unknown row map kind {self.kind!r}")

    def manifest(self) -> dict[str, Any]:
        data = {
            "kind": self.kind,
            "output_size": self.output_size,
            "parent_size": self.parent_size,
        }
        if self.kind == "repeat_each":
            data["repeat"] = self.repeat
        if self.gather or self.kind in {"gather", "tile", "affine", "filter", "concat"}:
            data["gather_size"] = len(self.gather)
            payload = ",".join(str(int(x)) for x in self.gather).encode("ascii")
            data["gather_sha256"] = hashlib.sha256(payload).hexdigest()
        return data


@dataclass(frozen=True)
class StructuredFramePlan:
    """Preparation-only columnar semantics for a structured provider.

    Supported nodes are intentionally generic relational/row-domain operators:
    source frame, left join, and row-map transforms.  Generated formulas consume
    individual typed columns; they never receive this object or a pandas frame.
    """

    kind: str
    output_size: int
    columns: tuple[str, ...]
    source_ref: str | None = None
    parent: "StructuredFramePlan | None" = None
    right_ref: str | None = None
    join_on: str | None = None
    row_map: RowMap | None = None
    provenance: tuple[str, ...] = ()
    op: StructuredOp | None = None

    @staticmethod
    def source(ref_name: str, frame: Any) -> "StructuredFramePlan":
        columns, size = _source_schema(frame)
        return StructuredFramePlan(
            kind="source",
            output_size=size,
            columns=columns,
            source_ref=str(ref_name),
            provenance=(f"source:{ref_name}:{_source_kind(frame)}",),
            op=StructuredOp.source(str(ref_name), _source_kind(frame)),
        )

    @staticmethod
    def left_join(
        parent: "StructuredFramePlan", right_ref: str, right: Any, *, on: str
    ) -> "StructuredFramePlan":
        if not _is_dataframe(right):
            raise StructuredProviderError(f"{right_ref!r} is not a pandas DataFrame")
        if on not in parent.columns:
            raise StructuredProviderError(f"join key {on!r} is absent from left provider")
        if not getattr(right.index, "is_unique", False):
            raise StructuredProviderError("structured left-join requires a unique right index")
        overlap = set(parent.columns) & set(map(str, right.columns))
        if overlap:
            raise StructuredProviderError(
                "structured left-join overlapping columns require explicit suffix semantics: "
                + ",".join(sorted(overlap))
            )
        columns = tuple(parent.columns) + tuple(str(x) for x in right.columns)
        return StructuredFramePlan(
            kind="left_join",
            output_size=parent.output_size,
            columns=columns,
            parent=parent,
            right_ref=str(right_ref),
            join_on=str(on),
            row_map=RowMap.identity(parent.output_size),
            provenance=(*parent.provenance, f"left_join:{right_ref}:on={on}"),
            op=StructuredOp.left_join(str(right_ref), str(on)),
        )

    @staticmethod
    def remap(parent: "StructuredFramePlan", row_map: RowMap, *, note: str) -> "StructuredFramePlan":
        if row_map.parent_size != parent.output_size:
            raise StructuredProviderError("row-map parent size disagrees with provider size")
        return StructuredFramePlan(
            kind="row_map",
            output_size=row_map.output_size,
            columns=parent.columns,
            parent=parent,
            row_map=row_map,
            provenance=(*parent.provenance, note),
            op=StructuredOp.row_map(row_map),
        )

    def source_demands(self, fields: Sequence[str]) -> tuple[StructuredSourceDemand, ...]:
        """Propagate selected output fields back to primitive structured sources.

        This is the structured-provider analogue of relational column pruning.
        It does not change dev149 execution semantics by itself; it makes the
        demanded source schema explicit so binding and future runtime lowering do
        not have to treat every column in a DataFrame as semantically live.
        """

        requested = tuple(dict.fromkeys(str(field) for field in fields))
        unknown = tuple(field for field in requested if field not in self.columns)
        if unknown:
            raise StructuredProviderError(
                f"structured provider has no demanded columns {unknown!r}"
            )

        merged: dict[str, tuple[set[str], bool]] = {}

        def add(ref_name: str, columns: Sequence[str], *, needs_index: bool = False) -> None:
            existing, old_index = merged.get(str(ref_name), (set(), False))
            existing.update(str(x) for x in columns)
            merged[str(ref_name)] = (existing, bool(old_index or needs_index))

        def visit(plan: "StructuredFramePlan", wanted: tuple[str, ...]) -> None:
            if not wanted:
                return
            if plan.kind == "source":
                if plan.source_ref is None:
                    raise StructuredProviderError("structured source lost its reference")
                add(plan.source_ref, wanted)
                return
            if plan.parent is None:
                raise StructuredProviderError("structured provider lost its parent")
            if plan.kind == "row_map":
                visit(plan.parent, wanted)
                return
            if plan.kind == "left_join":
                if plan.right_ref is None or plan.join_on is None:
                    raise StructuredProviderError("left-join provider metadata is incomplete")
                left = tuple(field for field in wanted if field in plan.parent.columns)
                right = tuple(field for field in wanted if field not in plan.parent.columns)
                if right:
                    # The left join key is a hidden dependency of every right-side
                    # projection, and the right index is part of the join schema.
                    visit(plan.parent, tuple(dict.fromkeys((*left, plan.join_on))))
                    add(plan.right_ref, right, needs_index=True)
                else:
                    visit(plan.parent, left)
                return
            raise StructuredProviderError(f"unknown structured frame kind {plan.kind!r}")

        visit(self, requested)
        return tuple(
            StructuredSourceDemand(
                source_ref=ref_name,
                columns=tuple(sorted(columns)),
                needs_index=needs_index,
            )
            for ref_name, (columns, needs_index) in sorted(merged.items())
        )

    def row_map_chain(self) -> tuple[RowMap, ...]:
        """Return output-to-source row-domain transforms in execution order."""

        maps: list[RowMap] = []
        plan: StructuredFramePlan | None = self
        while plan is not None:
            if plan.kind == "row_map" and plan.row_map is not None:
                maps.append(plan.row_map)
            plan = plan.parent
        maps.reverse()
        return tuple(maps)

    def column_type(
        self,
        field: str,
        refs: Mapping[str, Any],
        *,
        labels: Sequence[Any] | None = None,
        cache: dict[Any, Any] | None = None,
    ) -> StructuredColumnType:
        """Infer one column ABI type without materializing its expanded row domain."""

        field = str(field)
        if field not in self.columns:
            raise StructuredProviderError(f"structured provider has no column {field!r}")
        key = ("structured_column_type", id(self), field, None if labels is None else tuple(labels))
        if cache is not None and key in cache:
            return cache[key]

        if self.kind == "source":
            if self.source_ref is None or self.source_ref not in refs:
                raise StructuredProviderError("structured source reference is unavailable")
            frame = refs[self.source_ref]
            columns, size = _source_schema(frame)
            if columns != self.columns or size != self.output_size:
                raise StructuredProviderError(
                    f"structured source {self.source_ref!r} schema/row domain changed after compilation"
                )
            result = infer_column_type(_source_column(frame, field), labels=labels)
        else:
            if self.parent is None:
                raise StructuredProviderError("structured provider lost its parent")
            if self.kind == "row_map":
                result = self.parent.column_type(field, refs, labels=labels, cache=cache)
            elif self.kind == "left_join":
                if field in self.parent.columns:
                    result = self.parent.column_type(field, refs, labels=labels, cache=cache)
                else:
                    if self.right_ref is None or self.join_on is None or self.right_ref not in refs:
                        raise StructuredProviderError("left-join provider metadata is incomplete")
                    right = refs[self.right_ref]
                    if not _is_dataframe(right) or not getattr(right.index, "is_unique", False):
                        raise StructuredProviderError("right join source changed after compilation")
                    right_field = _stringified_key(tuple(right.columns), field)
                    left_key = self.parent.column_values(self.join_on, refs, cache=cache)
                    indexer = _left_join_indexer(self, right, left_key, cache=cache)
                    # Type the *reachable join result*, not the entire right
                    # table.  This preserves dev149's selected-domain category
                    # evidence without materializing any later repeat/gather row
                    # expansion.  It is the relational equivalent of predicate
                    # pushdown before schema inference.
                    if np.all(indexer >= 0):
                        reachable = np.asarray(right[right_field].to_numpy())[indexer]
                    else:
                        reachable = right.reindex(left_key)[right_field].to_numpy()
                    result = infer_column_type(np.asarray(reachable), labels=labels)
            else:
                raise StructuredProviderError(f"unknown structured frame kind {self.kind!r}")

        if cache is not None:
            cache[key] = result
        return result

    def column_values(
        self,
        field: str,
        refs: Mapping[str, Any],
        *,
        cache: dict[Any, Any] | None = None,
    ) -> np.ndarray:
        field = str(field)
        if field not in self.columns:
            raise StructuredProviderError(f"structured provider has no column {field!r}")
        cache_key = ("structured_column_values", id(self), field)
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        if self.kind == "source":
            if self.source_ref is None or self.source_ref not in refs:
                raise StructuredProviderError("structured source reference is unavailable")
            frame = refs[self.source_ref]
            columns, size = _source_schema(frame)
            if columns != self.columns or size != self.output_size:
                raise StructuredProviderError(
                    f"structured source {self.source_ref!r} schema/row domain changed after compilation"
                )
            result = _source_column(frame, field)
            if cache is not None:
                cache[cache_key] = result
            return result

        if self.parent is None:
            raise StructuredProviderError("structured provider lost its parent")

        if self.kind == "row_map":
            if self.row_map is None:
                raise StructuredProviderError("row-map provider has no row map")
            result = self.row_map.apply(
                self.parent.column_values(field, refs, cache=cache)
            )
            if cache is not None:
                cache[cache_key] = result
            return result

        if self.kind == "left_join":
            if field in self.parent.columns:
                result = self.parent.column_values(field, refs, cache=cache)
                if cache is not None:
                    cache[cache_key] = result
                return result
            if self.right_ref is None or self.join_on is None or self.right_ref not in refs:
                raise StructuredProviderError("left-join provider metadata is incomplete")
            right = refs[self.right_ref]
            if not _is_dataframe(right) or not getattr(right.index, "is_unique", False):
                raise StructuredProviderError("right join source changed after compilation")
            left_key = self.parent.column_values(self.join_on, refs, cache=cache)
            indexer = _left_join_indexer(self, right, left_key, cache=cache)
            # For the common complete-key case, direct NumPy gather preserves the
            # right column dtype and lets all joined fields share one indexer.
            # Missing-key joins retain pandas' exact reindex promotion/null rules.
            try:
                right_field = _stringified_key(tuple(right.columns), field)
                if np.all(indexer >= 0):
                    result = np.asarray(right[right_field].to_numpy())[indexer]
                else:
                    result = right.reindex(left_key)[right_field].to_numpy()
            except Exception as exc:
                raise StructuredProviderError(
                    f"cannot bind joined structured column {field!r}"
                ) from exc
            if len(result) != self.output_size:
                raise StructuredProviderError("joined structured column changed output row domain")
            result = np.asarray(result)
            if cache is not None:
                cache[cache_key] = result
            return result

        raise StructuredProviderError(f"unknown structured frame kind {self.kind!r}")

    def encoded_column_values(
        self,
        field: str,
        refs: Mapping[str, Any],
        *,
        labels: Sequence[Any] | None = None,
        cache: dict[Any, Any] | None = None,
    ) -> tuple[str, np.ndarray, bool]:
        """Bind one primitive column, encoding before row expansion when safe.

        In dev149, category/object values flowed through ``RowMap.apply`` and
        were encoded only after reaching the full ranked domain.  For repeated
        policy/scenario domains this creates large temporary object arrays.  The
        physical code domain is invariant under deterministic row maps, so source
        or joined values can be encoded first and the primitive codes remapped.
        """

        field = str(field)
        labels_token = None if labels is None else tuple(labels)
        cache_key = ("structured_encoded_column", id(self), field, labels_token)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        if field not in self.columns:
            raise StructuredProviderError(f"structured provider has no column {field!r}")

        if self.kind == "source":
            values = self.column_values(field, refs, cache=cache)
            result = encode_column(values, labels=labels_token)
        else:
            if self.parent is None:
                raise StructuredProviderError("structured provider lost its parent")
            if self.kind == "row_map":
                if self.row_map is None:
                    raise StructuredProviderError("row-map provider has no row map")
                dtype, parent_values, nullable = self.parent.encoded_column_values(
                    field, refs, labels=labels_token, cache=cache
                )
                result = (dtype, self.row_map.apply(parent_values), nullable)
            elif self.kind == "left_join":
                if field in self.parent.columns:
                    result = self.parent.encoded_column_values(
                        field, refs, labels=labels_token, cache=cache
                    )
                else:
                    if self.right_ref is None or self.join_on is None or self.right_ref not in refs:
                        raise StructuredProviderError("left-join provider metadata is incomplete")
                    right = refs[self.right_ref]
                    if not _is_dataframe(right) or not getattr(right.index, "is_unique", False):
                        raise StructuredProviderError("right join source changed after compilation")
                    left_key = self.parent.column_values(self.join_on, refs, cache=cache)
                    indexer = _left_join_indexer(self, right, left_key, cache=cache)
                    right_field = _stringified_key(tuple(right.columns), field)
                    if np.all(indexer >= 0):
                        # Encode only the reachable right rows.  A frozen category
                        # domain may intentionally exclude labels present elsewhere
                        # in the right table but unreachable from this left domain.
                        reachable = np.asarray(right[right_field].to_numpy())[indexer]
                        result = encode_column(reachable, labels=labels_token)
                    else:
                        # Missing join keys can change pandas dtype/null semantics,
                        # so retain exact reindex behavior before physical encoding.
                        joined = right.reindex(left_key)[right_field].to_numpy()
                        result = encode_column(np.asarray(joined), labels=labels_token)
            else:
                raise StructuredProviderError(f"unknown structured frame kind {self.kind!r}")

        dtype, encoded, nullable = result
        encoded = np.ascontiguousarray(encoded)
        if encoded.shape != (self.output_size,):
            raise StructuredProviderError(
                f"structured encoded column {field!r} produced shape {encoded.shape}, "
                f"expected {(self.output_size,)}"
            )
        result = (dtype, encoded, nullable)
        if cache is not None:
            cache[cache_key] = result
        return result

    def manifest(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "output_size": self.output_size,
            "columns": list(self.columns),
            "provenance": list(self.provenance),
        }
        if self.source_ref is not None:
            data["source_ref"] = self.source_ref
        if self.right_ref is not None:
            data["right_ref"] = self.right_ref
        if self.join_on is not None:
            data["join_on"] = self.join_on
        if self.row_map is not None:
            data["row_map"] = self.row_map.manifest()
        if self.op is not None:
            data["op"] = self.op.manifest()
        if self.parent is not None:
            data["parent"] = self.parent.manifest()
        return data


def _stringified_key(keys: Sequence[Any], field: str) -> Any:
    matches = [key for key in keys if str(key) == str(field)]
    if len(matches) != 1:
        raise StructuredProviderError(
            f"structured field {field!r} is absent or ambiguous after string normalization"
        )
    return matches[0]


def _source_kind(value: Any) -> str:
    if _is_dataframe(value):
        return "dataframe"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, np.ndarray) and value.dtype.names:
        return "structured_ndarray"
    return type(value).__name__


def _source_schema(value: Any) -> tuple[tuple[str, ...], int]:
    if _is_dataframe(value):
        columns = tuple(str(x) for x in value.columns)
        if len(set(columns)) != len(columns):
            raise StructuredProviderError("structured DataFrame columns are ambiguous after string normalization")
        return columns, int(len(value))
    if isinstance(value, Mapping):
        columns = tuple(str(x) for x in value.keys())
        if len(set(columns)) != len(columns):
            raise StructuredProviderError("structured mapping keys are ambiguous after string normalization")
        sizes: set[int] = set()
        for key in value:
            arr = np.asarray(value[key])
            if arr.ndim != 1:
                raise StructuredProviderError(
                    f"structured mapping column {key!r} must be rank-1"
                )
            sizes.add(int(arr.shape[0]))
        if len(sizes) > 1:
            raise StructuredProviderError("structured mapping columns have different row domains")
        return columns, (next(iter(sizes)) if sizes else 0)
    if isinstance(value, np.ndarray) and value.dtype.names:
        columns = tuple(str(x) for x in value.dtype.names)
        if len(set(columns)) != len(columns):
            raise StructuredProviderError("structured ndarray fields are ambiguous after string normalization")
        return columns, int(value.shape[0])
    raise StructuredProviderError(
        f"unsupported structured source type {type(value).__module__}.{type(value).__name__}"
    )


def _source_column(value: Any, field: str) -> np.ndarray:
    if _is_dataframe(value):
        key = _stringified_key(tuple(value.columns), field)
        return np.asarray(value[key].to_numpy())
    if isinstance(value, Mapping):
        key = _stringified_key(tuple(value.keys()), field)
        return np.asarray(value[key])
    if isinstance(value, np.ndarray) and value.dtype.names:
        key = _stringified_key(tuple(value.dtype.names), field)
        return np.asarray(value[key])
    raise StructuredProviderError("unsupported structured source")


def _is_dataframe(value: Any) -> bool:
    return (
        type(value).__name__ == "DataFrame"
        and type(value).__module__.startswith("pandas.")
    )


def _left_join_indexer(
    plan: StructuredFramePlan,
    right: Any,
    left_key: np.ndarray,
    *,
    cache: dict[Any, Any] | None = None,
) -> np.ndarray:
    key = ("structured_left_join_indexer", id(plan))
    if cache is not None and key in cache:
        return cache[key]
    try:
        indexer = np.asarray(right.index.get_indexer(left_key), dtype=np.int64)
    except Exception as exc:
        raise StructuredProviderError("cannot compute structured left-join indexer") from exc
    if indexer.shape != (plan.output_size,):
        raise StructuredProviderError("structured left-join indexer changed output row domain")
    if cache is not None:
        cache[key] = indexer
    return indexer


def infer_column_type(
    values: np.ndarray, *, labels: Sequence[Any] | None = None
) -> StructuredColumnType:
    """Infer static logical/physical column type without expanding row maps."""

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise StructuredProviderError(f"structured column must be rank-1, got {arr.shape}")
    source_dtype = str(arr.dtype)
    kind = arr.dtype.kind
    if kind == "b":
        return StructuredColumnType(
            logical_kind="bool",
            physical_dtype="int64",
            nullable=False,
            source_dtype=source_dtype,
        )
    if kind in "iu":
        return StructuredColumnType(
            logical_kind="int64",
            physical_dtype="int64",
            nullable=False,
            source_dtype=source_dtype,
        )
    if kind in "fc":
        nullable = False
        try:
            nullable = bool(np.any(np.isnan(arr)))
        except Exception:
            nullable = False
        return StructuredColumnType(
            logical_kind="float64",
            physical_dtype="float64",
            nullable=nullable,
            source_dtype=source_dtype,
        )

    semantic_labels, nullable = column_category(arr)
    chosen = tuple(semantic_labels if labels is None else labels)
    if not chosen:
        raise StructuredProviderError(
            f"structured object column needs a proven finite category domain, dtype={arr.dtype}"
        )
    # Validate the frozen category domain now, while still avoiding any row-map
    # expansion.  The same domain is rechecked by encode_column during binding.
    allowed = set(chosen)
    for value in arr.tolist():
        if _is_missing(value):
            nullable = True
            continue
        if value not in allowed:
            raise StructuredProviderError(
                f"structured category value {value!r} is outside frozen domain {chosen!r}"
            )
    return StructuredColumnType(
        logical_kind="category",
        physical_dtype="int64",
        nullable=nullable,
        source_dtype=source_dtype,
        category_labels=chosen,
    )


def column_category(values: np.ndarray) -> tuple[tuple[Any, ...], bool]:
    """Return finite semantic labels for one object/category column.

    Numeric/bool columns deliberately return no category labels.  Missing values
    are represented separately so physical category codes can reserve -1 without
    adding NaN to the semantic label domain.
    """
    arr = np.asarray(values)
    if arr.dtype.kind in "iufcb":
        return (), False
    labels: list[Any] = []
    nullable = False
    for value in arr.tolist():
        if _is_missing(value):
            nullable = True
            continue
        if not isinstance(value, (str, bool, int, float)):
            raise StructuredProviderError(
                f"structured category contains unsupported value {type(value).__name__}"
            )
        if value not in labels:
            labels.append(value)
    if not labels:
        return (), nullable
    return tuple(labels), nullable


def encode_column(
    values: np.ndarray, *, labels: Sequence[Any] | None = None
) -> tuple[str, np.ndarray, bool]:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise StructuredProviderError(f"structured column must be rank-1, got {arr.shape}")
    if arr.dtype.kind in "iub":
        return "int64", np.asarray(arr, dtype=np.int64), False
    if arr.dtype.kind in "fc":
        return "float64", np.asarray(arr, dtype=np.float64), False
    semantic_labels, nullable = column_category(arr)
    chosen = tuple(semantic_labels if labels is None else labels)
    if not chosen:
        raise StructuredProviderError(
            f"structured object column needs a proven finite category domain, dtype={arr.dtype}"
        )
    code = {label: i for i, label in enumerate(chosen)}
    encoded = np.empty(arr.shape[0], dtype=np.int64)
    for i, value in enumerate(arr.tolist()):
        if _is_missing(value):
            encoded[i] = -1
        elif value in code:
            encoded[i] = code[value]
        else:
            raise StructuredProviderError(
                f"structured category value {value!r} is outside frozen domain {chosen!r}"
            )
    return "int64", encoded, nullable


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except Exception:
        return False
