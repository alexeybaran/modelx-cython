"""Fault injection is confined to disposable generated test artifacts."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys


def test_native_allocation_failure_and_destruction(tmp_path):
    import modelx as mx
    m=mx.new_model('FaultStorage');s=m.new_space('S');s.counts={}
    s.new_cells(name='value',formula="def value(x):\n counts[x]=counts.get(x,0)+1\n return x-1.0\n")
    allocation=(Path(__file__).parents[1]/'cache_storage.py').exists()
    sample='for x in range(4): p.value(x)' if allocation else 'for x in (0.25,1.25): p.value(x)'
    m.export(tmp_path/'model');m.close()
    (tmp_path/'sample.py').write_text('from model import mx_model\np=mx_model.S\n'+sample+'\n')
    env=os.environ.copy();env['PYTHONPATH']=str(tmp_path)+os.pathsep+env.get('PYTHONPATH','')
    def run(cmd,label):
        r=subprocess.run(cmd,cwd=tmp_path,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (tmp_path/(label+'.log')).write_text(r.stdout)
        assert r.returncode==0,r.stdout[-12000:]
    run([sys.executable,'-m','modelx_cython','model','--sample','sample.py','--no-spec','--translate-only'],'translate')
    package=tmp_path/'model_cy'
    (package/'_mx_fault.pxd').write_text('cdef void* mx_alloc(size_t) noexcept nogil\ncdef void* mx_calloc(size_t,size_t) noexcept nogil\ncdef void mx_free(void*) noexcept nogil\ncpdef int mx_state(int) noexcept\n')
    (package/'_mx_fault.pyx').write_text('''from libc.stdlib cimport malloc, calloc, free
cdef int fail_next = 0
cdef int owned = 0
cdef void* mx_alloc(size_t n) noexcept nogil:
    global fail_next, owned
    cdef void* p
    if fail_next:
        fail_next = 0
        return NULL
    p = malloc(n)
    if p != NULL: owned += 1
    return p
cdef void* mx_calloc(size_t n, size_t size) noexcept nogil:
    global fail_next, owned
    cdef void* p
    if fail_next:
        fail_next = 0
        return NULL
    p = calloc(n,size)
    if p != NULL: owned += 1
    return p
cdef void mx_free(void* p) noexcept nogil:
    global owned
    if p != NULL: owned -= 1
    free(p)
cpdef int mx_state(int fail) noexcept:
    global fail_next
    fail_next = fail
    return owned
''')
    p=package/'_mx_classes.py';src=p.read_text()
    if allocation:
        support=package/'_mx_value_storage.py'
        owner=support.read_text()
        owner=owner.replace('import cython as _mx_cy', 'import cython as _mx_cy\nfrom cython.cimports.model_cy._mx_fault import mx_alloc, mx_free')
        owner=owner.replace('    pass', '    def __init__(self):\n        self._test_token = mx_alloc(1)\n        if self._test_token == _mx_cy.NULL:\n            raise MemoryError()\n    def __dealloc__(self):\n        mx_free(self._test_token)')
        support.write_text(owner)
        decl=support.with_suffix('.pxd');text=decl.read_text()
        text=text.replace('cdef class _mx_values_c_S:', 'cdef class _mx_values_c_S:\n    cdef void* _test_token')
        decl.write_text(text)
    else:
        support=package/'_mx_numeric_cache.pxd'
        support.write_text(support.read_text().replace('from libc.stdlib cimport calloc, free','from model_cy._mx_fault cimport mx_calloc as calloc, mx_free as free'))
        owner=package/'_mx_numeric_cache.py'
        owner.write_text(owner.read_text().replace('from cython.cimports.libc.stdlib import free as _mx_free','from cython.cimports.model_cy._mx_fault import mx_free as _mx_free'))
    for kernel in package.rglob('_kernel.pyx'):
        text = kernel.read_text()
        text = text.replace('from libc.stdlib cimport calloc, free',
                            'from model_cy._mx_fault cimport mx_calloc as calloc, mx_free as free')
        kernel.write_text(text)
    src+='\nfrom cython.cimports.model_cy._mx_fault import mx_state\n@_mx_cy.ccall\ndef fault_state(fail: _mx_cy.int) -> _mx_cy.int:\n    return mx_state(fail)\n'
    p.write_text(src)
    p=package/'_mx_classes.pxd';p.write_text(p.read_text()+'\ncpdef int fault_state(int fail)\n')
    # Consume the actual translation union, including private kernels. A fixed
    # legacy list would skip the storage implementation under test.
    setup_tree = ast.parse((tmp_path/'setup.py').read_text())
    cython_call = next(n for n in ast.walk(setup_tree) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name) and n.func.id == 'cythonize')
    modules = ast.literal_eval(cython_call.args[0]) + ['model_cy/_mx_fault.pyx']
    (tmp_path/'setup_fault.py').write_text(
        "from setuptools import setup\nfrom Cython.Build import cythonize\n"
        + "setup(ext_modules=cythonize(" + repr(modules) + "))\n")
    run([sys.executable,'setup_fault.py','build_ext','--inplace'],'build')
    verify='''import gc
from model_cy import mx_model
from model_cy._mx_classes import fault_state
m=type(mx_model)();p=m.S
assert fault_state(0)==0
'''
    if allocation:
        verify+='''fault_state(1)
try:p.value(0)
except MemoryError:pass
else:raise AssertionError('allocation failure not propagated')
assert fault_state(0)==0
assert p.value(0)==-1.0 and p.counts[0]==2
assert p.value(0)==-1.0 and p.counts[0]==2
assert fault_state(0)==1
'''
    else:
        # Inline keys allocate only at promotion/growth; inject at that boundary.
        manifest = next(package.glob('_mx_frame_S/manifest.json'))
        row = json.loads(manifest.read_text())['cells']['value']
        growth_count = 2 * row['slots'] if row.get('native_prefix') and row['hashed'] else 8
        verify+='''for i in range(GROWTH_COUNT):assert p.value(i+0.25)==i-0.75
assert fault_state(0)==1
fault_state(1)
try:p.value(GROWTH_COUNT+0.25)
except MemoryError:pass
else:raise AssertionError('growth failure not propagated')
assert fault_state(0)==1
for i in range(GROWTH_COUNT):assert p.value(i+0.25)==i-0.75 and p.counts[i+0.25]==1
assert p.value(GROWTH_COUNT+0.25)==GROWTH_COUNT-0.75 and p.counts[GROWTH_COUNT+0.25]==2
assert fault_state(0)==1
'''
        verify = verify.replace('GROWTH_COUNT', str(growth_count))
    verify+='''del p,m
gc.collect()
assert fault_state(0)==0
'''
    (tmp_path/'verify.py').write_text(verify);run([sys.executable,'verify.py'],'verify')
