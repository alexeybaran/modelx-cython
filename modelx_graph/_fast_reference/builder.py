from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CompiledModelxBackend:
    compiler: object
    module: object
    build_dir: Path
    module_name: str

    def run(self, *, threads: int = 1):
        """Bind current live modelx References and execute native backend."""
        inputs = self.compiler.bind_inputs()
        return self.module.run(**inputs, threads=threads)

    def explain(self, cell_name: str):
        return self.compiler.explain(cell_name)

    @property
    def manifest(self):
        return self.compiler.manifest()


class NativeBackendBuilder:
    """Build the serial Part-1 Cython extension from a ModelxModelCompiler."""

    def __init__(self, compiler):
        self.compiler = compiler

    def build(self, build_dir, module_name="modelx_native_backend", *, native_arch=True):
        build_dir = Path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        pyx = build_dir / f"{module_name}.pyx"
        manifest = build_dir / f"{module_name}.manifest.json"
        setup_py = build_dir / "setup.py"
        self.compiler.generate_cython(pyx, module_name=module_name)
        self.compiler.write_manifest(manifest)

        march = ", '-march=native'" if native_arch else ""
        setup_py.write_text(f'''\
from setuptools import setup, Extension\nfrom Cython.Build import cythonize\nimport numpy as np\next = Extension(\n    {module_name!r},\n    [{str(pyx.name)!r}],\n    include_dirs=[np.get_include()],\n    extra_compile_args=['-O3'{march}],\n)\nsetup(name={module_name!r}, ext_modules=cythonize([ext], compiler_directives={{'language_level':3}}))\n''')
        env = dict(os.environ)
        # Ensure generated Cython can resolve package imports only at Python-call boundary.
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=build_dir, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        matches = list(build_dir.glob(module_name + "*.so")) + list(build_dir.glob(module_name + "*.pyd"))
        if not matches:
            raise RuntimeError(f"Compiled extension not found in {build_dir}")
        ext_path = matches[0]
        spec = importlib.util.spec_from_file_location(module_name, ext_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return CompiledModelxBackend(self.compiler, mod, build_dir, module_name)
