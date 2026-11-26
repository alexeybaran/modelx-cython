import runpy, sys
sys.setrecursionlimit(50000)
runpy.run_path(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.py') else 'sample_sec.py', run_name='__main__')
from exported import mx_model as m
print('EXPORTED', repr(m.Book.total()))
