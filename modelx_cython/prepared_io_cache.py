"""Verified acceleration of exported IO loading, retaining changed-file reads.

The existing exported reader produces the snapshot during compilation. Runtime
loading verifies the complete source bytes and IO declarations before using it;
a changed resource uses the original reader. Every model gets new owned values.
"""
import ast
import hashlib
import importlib
import json
import pickle
from pathlib import Path


RUNTIME = '''"""Verified IO snapshot; original readers handle any changed source."""
import hashlib
import json
import pickle

def load(path, ios, iospecs):
    receipt = json.loads((path / '_mx_io_snapshot.json').read_text(encoding='utf-8'))
    current = hashlib.sha256(pickle.dumps((ios, iospecs), protocol=4)).hexdigest()
    if current != receipt['declarations_sha256']:
        return None
    try:
        for name, digest in receipt['sources'].items():
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
                return None
        payload = (path / '_mx_io_snapshot.pkl').read_bytes()
    except OSError:
        return None
    if hashlib.sha256(payload).hexdigest() != receipt['payload_sha256']:
        return None
    return pickle.loads(payload)
'''


def emit(package, *, source_package=None):
    package = Path(package)
    sys_path = package / '_mx_sys.py'
    tree = ast.parse(sys_path.read_text(encoding='utf-8'))
    base = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseModel'), None)
    method = next((n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == '_mx_load_io'), None) if base else None
    io_path = package / '_mx_io.py'
    if method is None or not io_path.exists():
        return None
    source_name = Path(source_package).name if source_package is not None else package.name
    exported = importlib.import_module(source_name + '._mx_sys')
    declarations = importlib.import_module(source_name + '._mx_io')
    if not declarations.ios:
        return None
    # No custom IO backend, hidden client ownership or arbitrary class state is
    # assumed serializable. Unsupported resources keep their original loader.
    if any(row['type'] != 'PandasIO' for row in declarations.ios.values()) or any(
            row['type'] != 'PandasData' for row in declarations.iospecs.values()):
        return None
    sources = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
               for name in declarations.ios}
    original_specs = pickle.dumps((declarations.ios, declarations.iospecs), protocol=4)
    ios = {name: exported.io_types[row['type']](name, package / name, row)
           for name, row in declarations.ios.items()}
    data = {key: exported.iospec_types[row['type']](ios[row['io']],
             pickle.loads(pickle.dumps(row['kwargs']))).load_value()
            for key, row in declarations.iospecs.items()}
    if any(hasattr(value, '_mx_dataclient') for value in data.values()):
        return None
    if any(hashlib.sha256((package / name).read_bytes()).hexdigest() != digest
           for name, digest in sources.items()):
        raise RuntimeError('IO sources changed while preparing their verified snapshot')
    assignment = next(n for n in method.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == 'io_data' for t in n.targets))
    assignment.value = ast.parse('_mx_cached_io(self.path, _mx_io.ios, _mx_io.iospecs)', mode='eval').body
    guard = next(n for n in method.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'has_io')
    guard.test = ast.parse('has_io and io_data is None', mode='eval').body
    guard.body.insert(0, ast.parse('io_data = {}').body[0])
    method.body.insert(method.body.index(guard) + 1, ast.parse('if io_data is None:\n    io_data = {}').body[0])
    tree.body.insert(0, ast.parse('from ._mx_prepared_io import load as _mx_cached_io').body[0])
    sys_path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + '\n', encoding='utf-8')
    payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    (package / '_mx_io_snapshot.pkl').write_bytes(payload)
    receipt = dict(sources=sources, declarations_sha256=hashlib.sha256(original_specs).hexdigest(),
                   payload_sha256=hashlib.sha256(payload).hexdigest(),
                   contract='Verify source bytes and declarations; changed sources use original loader')
    (package / '_mx_io_snapshot.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (package / '_mx_prepared_io.py').write_text(RUNTIME, encoding='utf-8')
    return receipt
