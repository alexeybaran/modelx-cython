"""Native category metadata must not change Python comparison semantics."""
import subprocess
import sys


def test_category_binding_guards(tmp_path):
    from modelx_cython.connected_categories import Categories, SUPPORT
    from modelx_cython.integrated_native import ENUM_COMPARE_SUPPORT
    c=Categories('_S',{}, {},None);c.entries['fixture']='_mx_cat_S_0'
    fields,functions=c.emit()
    (tmp_path/'support.pxd').write_text(ENUM_COMPARE_SUPPORT+SUPPORT)
    (tmp_path/'cats.pxd').write_text('cdef class _S:\n'+fields+'\n')
    (tmp_path/'cats.py').write_text('''import cython as _mx_cy
from cython.cimports.support import _mx_enum_equal, _mx_category_standard, _mx_category_code
@_mx_cy.cclass
class _S: pass
_owner: _S = _S()
def compare(a,b,invert=False):
    return bool(_mx_cat_S_0(_owner,a,b,invert))
'''+ '\n'.join(functions))
    (tmp_path/'setup.py').write_text("from setuptools import setup\nfrom Cython.Build import cythonize\nsetup(ext_modules=cythonize('cats.py'))\n")
    def run(args):
        p=subprocess.run([sys.executable,*args],cwd=tmp_path,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert p.returncode==0,p.stdout
    run(['setup.py','build_ext','--inplace'])
    (tmp_path/'check.py').write_text('''from cats import compare
from enum import IntEnum
class A(IntEnum): X=1; Y=-1; Z=0; HUGE=2**100
class B(IntEnum): X=1
for repeat in range(2):
 for a in (A.X,A.Y,A.Z,A.HUGE,B.X,1,0,-1,2**100,None,'1'):
  for b in (A.X,A.Y,A.Z,A.HUGE,B.X,1,0,-1,2**100,None,'1'):
   assert compare(a,b)==bool(a==b)
   assert compare(a,b,True)==bool(a!=b)
assert compare(A.X,B.X)
B.__eq__=lambda a,b:False
assert compare(A.X,B.X)==bool(A.X==B.X)
assert not compare(B.X,A.X)
class Bad:
 def __eq__(self,other):raise ValueError('same failure')
try:compare(Bad(),A.X)
except ValueError:pass
else:raise AssertionError('lost exception')
# Evicting derived metadata may run an int subclass finalizer. Its effects
# must be visible before the comparison of the new arguments.
class V(int): pass
class Old(int):
 def __del__(self):V.__eq__=lambda a,b:False
old=Old(9);assert compare(old,9);del old
v=V(1)
assert not compare(v,1)
print('category guards PASS')
''')
    run(['check.py'])
