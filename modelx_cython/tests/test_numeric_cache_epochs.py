"""Growing exact-key caches reset without revisiting completed entry storage."""
from modelx_cython.numeric_cache import support
from test_original_frame_shell import _run


def test_numeric_epoch_growth_collision_reset_and_wrap(tmp_path):
    code = '''cimport cython
from libc.stdint cimport uint64_t
from libc.stdlib cimport calloc, free
from libc.string cimport memset
cdef union _mx_bits:
    double d
    uint64_t u
'''
    code += support('_keys', ['double'], 'double', native=True, epochs=True)
    code += '''
def verify():
    cdef _keys_cache c
    cdef Py_ssize_t found
    cdef int i
    memset(&c, 0, sizeof(_keys_cache))
    try:
        assert _keys_find(&c, 0.0) == -1
        for i in range(3000):
            _keys_put(&c, i * 0.25, i - 1.0)
        assert len(_keys_dictionary(&c)) == 3000
        for i in range(3000):
            found = _keys_find(&c, i * 0.25)
            assert found >= 0 and c.items[found].value == i - 1.0
        found = _keys_find(&c, -0.0)
        assert c.items[found].value == -1.0
        _keys_reset(&c)
        assert c.count == 0 and not _keys_dictionary(&c)
        assert _keys_find(&c, 0.0) == -1
        # Grow after reset: old tagged entries must never be copied as results.
        for i in range(10000):
            _keys_put(&c, -i * 0.25, i + 0.5)
        assert len(_keys_dictionary(&c)) == 10000
        assert _keys_find(&c, 0.25) == -1
        for i in range(10000):
            found = _keys_find(&c, -i * 0.25)
            assert found >= 0 and c.items[found].value == i + 0.5
        # Include a live entry tagged with the value that would be reused on wrap.
        _keys_reset(&c)
        c.epoch = 1
        _keys_put(&c, 123.0, -1.0)
        c.epoch = <uint64_t>-1
        c.count = 0
        _keys_put(&c, 234.0, 0.0)
        _keys_reset(&c)
        assert c.epoch == 1 and not _keys_dictionary(&c)
        assert _keys_find(&c, 123.0) == -1
        assert _keys_find(&c, 234.0) == -1
        _keys_put(&c, -0.0, -1.0)
        found = _keys_find(&c, 0.0)
        assert c.items[found].value == -1.0
    finally:
        free(c.items)
'''
    (tmp_path/'epoch.pyx').write_text(code, encoding='utf-8')
    (tmp_path/'setup.py').write_text(
        "from setuptools import setup\nfrom Cython.Build import cythonize\n"
        "setup(ext_modules=cythonize('epoch.pyx'))\n", encoding='utf-8')
    _run(tmp_path, ['setup.py', 'build_ext', '--inplace'], 'build')
    (tmp_path/'verify.py').write_text('import epoch\nepoch.verify()\n', encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')
