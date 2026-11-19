"""Evidence for ordinary Cython formulas after retirement of the reduction emitter."""
import re


def assert_compiled_generator_formulas(folder, names):
    source=(folder/'exported_cy/_mx_classes.c').read_text(encoding='utf-8')
    assert '__pyx_gb_' in source, 'Cython generator bodies were not emitted'
    for name in names:
        pattern=r'^static[^\n]*__pyx_(?:f|pf)_[^\n]*_f_'+re.escape(name)+r'\([^;]*?\) \{'
        assert re.search(pattern,source,re.MULTILINE), name
    # This proves compiled bodies, not zero object iteration or runtime-domain
    # admission. The unchanged differential/error/override probes prove behavior.
