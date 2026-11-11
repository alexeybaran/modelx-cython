"""Convenience API for isolated, snapshot-based compiled valuations."""
from dataclasses import dataclass
from pathlib import Path
import shutil
from .api import compile_exported
from .modelx_cython_bridge import _stage_external_model_files

@dataclass
class CompiledValuation:
    """One compiled module and its bound input snapshot.

    Call reset before each complete valuation. Input changes require a new build.
    A valuation is not thread-safe. Separate builds use distinct native modules.
    """
    module: object
    metadata: dict
    runtime: object

    def reset(self):
        self.module.reset()

    def calculate(self):
        return self.module.run_all()

def build_valuation(model, space, output, keys, directory, *, sample_keys=None):
    """Export, trace, compile and bind a zero-argument output Cells.

    `space` is a live modelx Space; `keys` identifies its requested ItemSpaces.
    For a non-parameterized Space use `[()]`. Sample keys must cover relevant
    branches and categories; validate the entire run against modelx before use.
    Compilation is serial and requires a supported C/C++ compiler.
    """
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    export=directory/'nomx'
    if export.exists():
        raise FileExistsError('Use an empty build directory for each valuation')
    _stage_external_model_files(model,directory)
    if model.path:
        for folder in Path(model.path).parent.iterdir():
            if folder.is_dir() and not (folder/'__init__.py').exists() and folder.name!='__pycache__':
                for file in folder.rglob('*'):
                    if file.is_file() and file.suffix.lower() in {'.csv','.xlsx','.xls','.json','.pickle','.pkl','.npy','.npz'}:
                        target=directory/file.relative_to(Path(model.path).parent)
                        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    model.export(export)
    # modelx fullname includes the model name, whereas the export exposes mx_model.
    space_path=space.fullname.split('.',1)[1]
    keys=tuple(keys);sample_keys=keys if sample_keys is None else tuple(sample_keys)
    calls=[]
    for key in sample_keys:
        access='mx_model.'+space_path
        if space.parameters:access+='['+repr(key)+']'
        calls.append(access+'.'+output+'()')
    sample=directory/'sample.py'
    sample.write_text('from nomx import mx_model\n'+'\n'.join(calls)+'\n',encoding='utf-8')
    return CompiledValuation(*compile_exported(model,space,output,keys,export,sample,
                            directory/'native',sample_keys=sample_keys))
