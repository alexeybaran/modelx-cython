"""Emit the importable model package shell around a compiled artifact.

The generated package mirrors modelx export/modelx-cython usage: importing the
package exposes an already-bound model object. ``CompiledModel`` remains an
internal runtime primitive rather than a constructor normal model users call.
"""
from __future__ import annotations

import hashlib
import keyword
import pickle
from pathlib import Path

from modelx_cython._frame_runtime.contracts import ContractError


_TABLES_FILE = "_mx_tables.pkl"


def add_public_package(destination, manifest, tables, model_name):
    """Embed initial Data and write the modelx-style package ``__init__``.

    ``destination`` is the package directory itself, just as ``Model_nomx_cy``
    is the package directory produced by modelx-cython. The compiled kernel stays
    in its private hash-named child package.
    """
    if not manifest.get("model"):
        return
    if tables is None:
        raise ContractError("A model artifact requires its initial Data snapshot")

    destination = Path(destination)
    package_name = destination.name
    if not package_name.isidentifier() or keyword.iskeyword(package_name):
        raise ContractError(
            "Compiled model destination must have an importable package name "
            "(for example Model_nomx_cy)"
        )

    if not isinstance(model_name, str) or not model_name.isidentifier() or keyword.iskeyword(model_name):
        raise ContractError("Compiled model_name must be an importable Python identifier")

    payload = pickle.dumps(tables, protocol=pickle.HIGHEST_PROTOCOL)
    (destination / _TABLES_FILE).write_bytes(payload)
    manifest["public_api"] = {
        "schema": "mxcf.package.v1",
        "model_name": model_name,
        "aliases": ["mx_model", model_name],
        "tables_file": _TABLES_FILE,
        "tables_sha256": hashlib.sha256(payload).hexdigest(),
    }

    init_source = f'''"""Generated compiled model package; do not edit by hand."""\nfrom modelx_cython.tests.reference_frame.model import _load_embedded_model as _mx_load_model\n\nmx_model = {model_name} = _mx_load_model(__file__)\n__all__ = ["mx_model", {model_name!r}]\n\ndel _mx_load_model\n'''
    (destination / "__init__.py").write_text(init_source)
