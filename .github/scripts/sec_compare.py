"""Build one released lifelib SEC model with mx2cy and compare it with live modelx.

    python sec_compare.py <work dir> [model] [product]

The model's library is copied by ``lifelib.create('uslib')``.  The model is exported by
upstream modelx and compiled with its own ``sample_sec.py`` and ``spec_sec.py``; then the
library's own check (``sec_support.compiled_mismatches``: every point swept, every
cached value and raise of modelx against the compiled package, in a child process) must
find nothing, to the library's relative tolerance ``COMPILED_RTOL`` (1e-12), not bit for
bit: the cut's own report compares at rel 1e-10, and MYGA_US_SEC differs from modelx in the
last bit on a few values on Linux x86-64.  Exits non-zero on a build failure or a mismatch.
"""
import os
import pathlib
import subprocess
import sys

import lifelib
import modelx as mx

work = pathlib.Path(sys.argv[1]).resolve()
model = sys.argv[2] if len(sys.argv) > 2 else 'Term_US_SEC'
product = sys.argv[3] if len(sys.argv) > 3 else 'term_life'

if sys.version_info < (3, 11):
    # lifelib's harness runs its child as ``python -P``, an option added in 3.11; before it, the
    # child runs without it (the only difference: the runner's directory is first on sys.path).
    _run = subprocess.run

    def _run_without_safe_path(args, *rest, **kwargs):
        if isinstance(args, list) and args[1:2] == ['-P']:
            args = args[:1] + args[2:]
        return _run(args, *rest, **kwargs)
    subprocess.run = _run_without_safe_path

lib = work / 'uslib'
if not lib.exists():
    lifelib.create('uslib', lib)
sys.path.insert(0, str(lib / 'tests'))
import sec_support                                   # noqa: E402

folder = lib / 'products' / product
nomx = f'{model}_nomx'
mx.read_model(folder / model).export(folder / nomx)
for m in list(mx.get_models()):
    mx.get_models()[m].close()
env = dict(os.environ, PYTHONIOENCODING='utf-8')
env['PYTHONPATH'] = str(folder) + os.pathsep + env.get('PYTHONPATH', '')
build = subprocess.run([sys.executable, '-m', 'modelx_cython', nomx, '--sample', 'sample_sec.py',
                        '--spec', 'spec_sec.py'], cwd=folder, env=env)
if build.returncode:
    sys.exit(f'mx2cy failed on {model}: exit {build.returncode}')
package = folder / f'{nomx}_cy'
built = sec_support.Build(model, folder, package, (package / '_mx_classes.pxd').read_text(encoding='utf-8'))
case = sec_support.compiled_case(built, work / 'case')
lines = sec_support.compiled_mismatches(built, case, exact=False)
print('\n'.join(lines) or f'{model}: compiled package matches live modelx on every point')
sys.exit(1 if lines else 0)
