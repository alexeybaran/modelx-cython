"""Describe a collect calculation graph using existing exported Cell sources.

This is metadata for a future native context emitter. Object-valued Cells are
kept as graph nodes and boundaries; their presence never excludes a model.
No type is guessed from formula arithmetic or sampled return values.
"""

import ast
import hashlib
from pathlib import Path


_KINDS = {
    "double": "float", "float": "float",
    "longlong": "int", "long": "int", "int": "int",
    "bint": "bool", "bool": "bool", "str": "str", "object": "object",
}


def _source(path):
    data = Path(path).read_bytes()
    return ast.parse(data.decode("utf-8")), hashlib.sha256(data).hexdigest()


def _space(tree, name):
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "_c_" + name]
    if len(classes) != 1:
        raise ValueError("expected one exported Space class: " + name)
    return classes[0]


def _kind(annotation):
    if annotation is None:
        return "object"
    if isinstance(annotation, ast.Attribute):
        name = annotation.attr
    elif isinstance(annotation, ast.Name):
        name = annotation.id
    else:
        name = ""
    return _KINDS.get(name, "object")


def _calls(body, names):
    result = set()
    for statement in body:
        for node in ast.walk(statement):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self" and node.func.attr in names):
                result.add(node.func.attr)
    return result


def build_manifest(export_source, cython_source, *, space="Projection"):
    """Return a deterministic source- and signature-pinned collect graph."""
    export, export_hash = _source(export_source)
    compiled, compiled_hash = _source(cython_source)
    export_space, compiled_space = _space(export, space), _space(compiled, space)
    formulas = {node.name[3:]: node for node in export_space.body
                if isinstance(node, ast.FunctionDef) and node.name.startswith("_f_")}
    typed = {node.name[3:]: node for node in compiled_space.body
             if isinstance(node, ast.FunctionDef) and node.name.startswith("_f_")}
    if "collect_outputs" not in formulas:
        raise ValueError("collect_outputs formula is required")
    names = set(formulas)
    roots = _calls(formulas["collect_outputs"].body, names)
    roots.discard("collect_outputs")
    if not roots:
        raise ValueError("collect_outputs has no Cell root")
    seen = set()
    pending = list(roots)
    edges = {}
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        deps = _calls(formulas[name].body, names)
        edges[name] = sorted(deps)
        pending.extend(deps - seen)
    rows = {}
    for name in sorted(seen):
        fn = typed.get(name)
        if fn is None:
            raise ValueError("missing compiled Cell signature: " + name)
        source_arguments = [arg.arg for arg in formulas[name].args.posonlyargs
                            + formulas[name].args.args if arg.arg != "self"]
        typed_arguments = [arg.arg for arg in fn.args.posonlyargs + fn.args.args
                           if arg.arg != "self"]
        if source_arguments != typed_arguments:
            raise ValueError("compiled Cell arguments differ: " + name)
        rows[name] = {
            "result": _kind(fn.returns),
            "arguments": {arg.arg: _kind(arg.annotation)
                          for arg in fn.args.posonlyargs + fn.args.args
                          if arg.arg != "self"},
            "dependencies": edges[name],
        }
    return {
        "schema": 1, "space": space,
        "export_sha256": export_hash, "cython_sha256": compiled_hash,
        "roots": sorted(roots), "cells": rows,
        "numeric": sorted(name for name, row in rows.items()
                          if row["result"] in ("float", "int", "bool")
                          and all(kind in ("float", "int", "bool")
                                  for kind in row["arguments"].values())),
        "object_edges": sorted(name for name, row in rows.items()
                               if row["result"] not in ("float", "int", "bool")
                               or any(kind not in ("float", "int", "bool")
                                      for kind in row["arguments"].values())),
    }
