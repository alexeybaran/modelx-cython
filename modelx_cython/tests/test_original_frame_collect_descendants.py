"""Nested original Space shells participate in the collection owner lifecycle."""
import json

from test_original_frame_shell import _run


def test_nested_collection_reuses_descendant_owner_and_expires_transient_cache(tmp_path):
    import modelx as mx

    model = mx.new_model('NestedCollectShell')
    policies = model.new_space(
        'Policies',
        formula='def Policies(policy_id, shock=0.0):\n return {"refs": {"offset": float(policy_id) + shock}}\n',
        refs={'offset': 0.0, 'policy_ids': (1, 2, 3), 'hook': None},
        collect_over={'policy_id': (1, 2, 3)},
        collect_outputs='lambda: {"sum": {"total": observed()}, "individual": {"values": {0: Child.fib(0), 7: Child.fib(7)}}}',
    )
    child = policies.new_space('Child')
    child.Policy = policies
    child.new_cells('seed', 'def seed():\n return Policy.offset\n')
    child.new_cells('fib', 'def fib(n):\n return seed() if n < 2 else 0.5 * (fib(n-1) + fib(n-2))\n')
    policies.new_cells('observed', 'def observed():\n if hook is not None: hook(Child)\n return Child.fib(7)\n')
    model.export(tmp_path / 'model')
    model.close()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\ns=mx_model.Policies\n'
        'for i in (1,2,3):\n s[i].observed()\n'
        's.collect()\n', encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--no-spec',
                    '--sample', str(tmp_path / 'sample.py')], 'build')
    receipts = list((tmp_path / 'model_cy').rglob('*.frame-shell.json'))
    rows = [row for path in receipts for row in json.loads(path.read_text())['cells']]
    assert any(row['space'].endswith('._c_Child') and row['cell'] == 'fib'
               and row['execution'] == 'frame_exact' for row in rows)
    (tmp_path / 'verify.py').write_text('''
import pickle
from model_cy import mx_model
from model_cy._mx_frame_owner import SpaceFrameOwner

created = []
original_init = SpaceFrameOwner.__init__
def record(self, *args):
    original_init(self, *args)
    if '_mx_frame_Child.' in self.module: created.append(self)
SpaceFrameOwner.__init__ = record
m = type(mx_model)(); s = m.Policies
# Ordinary descendant Spaces retain their independent caches.
for i in (1,2,3): assert s[i].Child.fib(7) == float(i)
assert len(created) == 3
created.clear()
held = []
s.hook = held.append
result = s.collect()
assert result.sum.total == 6.0
assert len(created) == 1, ('descendant owner was not reused', len(created))
assert len(held) == 3 and len({id(space) for space in held}) == 3
for i in (1,2,3):
    assert result.individual[i].values == {0: float(i), 7: float(i)}
    # Detached outputs and ordinary policy results survive scratch reuse.
    assert s[i].Child.fib(7) == float(i)
for space in held:
    for n in (0,7):
        try: space.fib(n)
        except RuntimeError as error:
            assert str(error) == 'Collection policy cache has expired'
        else: raise AssertionError('an expired Space read another policy cache')
assert s.collect() is result
assert len(held) == 3 and len(created) == 1
changed = s.collect(5.0)
assert changed.sum.total == 21.0
assert changed.individual[1].values == {0: 6.0, 7: 6.0}
assert result.individual[1].values == {0: 1.0, 7: 1.0}
restored = pickle.loads(pickle.dumps(result))
assert restored.sum.total == 6.0
assert restored.individual[3].values == {0: 3.0, 7: 3.0}
# Failure releases the transient descendants; the next invocation starts clean.
failed = []
def fail(space):
    failed.append(space)
    space.fib(0)
    raise ValueError('nested collection retry')
u = type(mx_model)(); u.Policies.hook = fail
for _ in range(2):
    try: u.Policies.collect()
    except ValueError as error: assert str(error) == 'nested collection retry'
    else: raise AssertionError('collection swallowed its failure')
assert len(failed) == 2
for space in failed:
    try: space.fib(0)
    except RuntimeError: pass
    else: raise AssertionError('failed invocation left shared cache accessible')
u.Policies.hook = None
assert u.Policies.collect().sum.total == 6.0
print('nested owner reuse, cache expiry, detachment, direct ownership and retry passed')
''', encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')
