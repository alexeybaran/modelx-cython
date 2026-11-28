# Installation

## Installing modelx-cython

To install modelx-cython, run the following command:

```console
$ pip install modelx-cython
```

To build models on a free-threaded interpreter, install with
`pip install 'modelx-cython[free-threaded]'`. This selects Cython 3.3.2 or later.

If you are on Anaconda, install it from conda-forge instead:

```console
$ conda install -c conda-forge modelx-cython
```

Installing the package makes the {doc}`mx2cy <cli>` command available as an
executable script.

For a source build before a Collect-capable modelx release is available, build
and install both checkouts together. Supply a modelx checkout that implements
`Space.set_collect()` and the version-1 exported Collect hook.

```console
$ python -m pip install /path/to/modelx /path/to/modelx-cython
```

On free-threaded Python, select `/path/to/modelx-cython[free-threaded]` as
the compiler argument.

## Requirements

modelx-cython requires Python 3.10 or newer, and depends on the following
packages, which are installed automatically when you install modelx-cython
with `pip` or `conda`:

* [modelx](https://github.com/fumitoh/modelx) `>=0.33.1.dev0` — for exporting
  models to pure-Python packages, with or without `__slots__` on the Space
  classes
* [Cython](https://cython.org/) v3.2.4+ — for compiling the translated
  models. The generated build script sets the `freethreading_compatible`
  directive.
* [setuptools](https://setuptools.pypa.io/) — for building the extension
  modules
* [libcst](https://libcst.readthedocs.io/) — for parsing and transforming
  the model source code
* [MonkeyType](https://monkeytype.readthedocs.io/) — for tracing runtime
  types
* [NumPy](https://numpy.org/) — for handling array-valued cells

For models exported with `locked_spaces`, see {doc}`freethreading`.

## C compiler

Compiling the translated model requires a C compiler that Python's build
tool chain can find:

* **Windows**: Microsoft Visual C++ (install
  [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)).
  Run `mx2cy` from a command prompt where the MSVC environment is set up
  (for example, a *Developer Command Prompt*), or let setuptools locate the
  compiler automatically.
* **Linux**: gcc (for example, `sudo apt install build-essential` on
  Debian/Ubuntu)
* **macOS**: clang (install the Xcode command-line tools with
  `xcode-select --install`)

If you only want to translate a model without compiling it, pass
`--translate-only` to `mx2cy`; no C compiler is needed in that case.
