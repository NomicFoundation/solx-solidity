.. _mlir-getting-started

===============
Getting started
===============

Build
-----
Build `era-compiler-llvm <https://github.com/matter-labs/era-compiler-llvm>`_
checked out at branch ``app-mlir`` with mlir and lld projects enabled. Make sure
rtti is on:

  .. code-block:: bash

    cmake <llvm> \
      ...
      '-DLLVM_ENABLE_PROJECTS=mlir;lld' \
      '-DLLVM_ENABLE_RTTI=ON' \

    ninja or make or ...


Then build this project using:


  .. code-block:: bash

    cmake <solidity> \
      ...
      '-DMLIR_DIR=<llvm-build>/lib/cmake/mlir' \
      '-DLLD_DIR=<llvm-build>/lib/cmake/lld'
      # Where <llvm-build> is the *absolute* path to the llvm build.

    ninja or make or ...

Style
-----
- More or less `mlir style guide
  <https://mlir.llvm.org/getting_started/DeveloperGuide/#style-guide>`_, which
  is the llvm style with some modification like using camelBack, no const
  correctness (`see <https://mlir.llvm.org/docs/Rationale/UsageOfConst>`_) etc.

- Some quirks I personally found helpful: Builder instances are named ``b`` and
  Rewriter instances are named ``r``. This makes the ir building/rewriting code
  look a little more concise and cleaner where the intent like "create an op",
  "replace an op" is more clearer. This is a little like llvm's ``IRBuilder``
  instances named as "irb".

- Tests are written to be as minimal as possible. No licenses, unnecessary
  specifiers (like pure) that doesn't affect the lowering etc., and the code
  might not make sense. The focus is purely on the behaviour of the codegen,
  with maximum coverage. A bit of a duh, but when you fix a bug, always add a
  minimal reproducer test!

- Location tracking is (``--mmlir --mlir-print-debuginfo``) mandatory in every
  source (non-mlir) FileCheck tests.

- Like in llvm, assert liberally. Unimplemented sections should assert-fail with
  an "NYI" ("Not yet implemented") message.

Before committing
-----------------
Once a patch is ready, you should do the following:

1. Testing
~~~~~~~~~~
- Run ``test/updFileCheckTest.py`` on each FileCheck test files under
  ``test/lit/mlirCodegen`` and check if the updates are correct.  I do something
  like this to update all tests in parallel:

  .. code-block:: bash

    find test/lit/mlirCodegen -type f \( -name '*.sol' -o -name '*.yul' \) \
      | xargs -n1 -P$(nproc) \
        test/updFileCheckTest.py --path "<solc-build>/solc"

    find test/lit/mlirCodegen -type f -name '*.mlir' \
      | xargs -n1 -P$(nproc) \
        test/updFileCheckTest.py --path "<solc-build>/tools/solOpt"

- Run the build rule ``check-solidity-mlir`` to run the lit test suite (and make
  sure it passes, duh).

- Run semantic tests using soltest. For this you need to build `evmone
  <https://github.com/ethereum/evmone>`_ using simple cmake, nothing fancy,
  which will build the ``libevmone.so`` used for the semantic testing. Then run:

  .. code-block:: bash

    soltest -t semanticTests/mlir -- --testpath <test-path> --vm <libevmone.so-path>.

  Where ``<test-path>`` is ``<solc-root>/test``. Note that "semanticTests/mlir"
  is a filter. You can restrict to, say, one test with ``-t
  semanticTests/mlir/<file>``

2. Add tests
~~~~~~~~~~~~
- Add a new test under ``test/lit/mlirCodegen`` that tests your change. Follow
  the existing formats of adding both a high level dialect test and standard
  dialect test. It's better to generate the check-lines using
  ``updFileCheckTest.py`` as our mlir dialect representations are not fully
  stable yet.

  (We don’t test dialects/ir/asm below the standard dialects - the lowering
  chain beneath them is already covered by llvm and mlir test suites. But we do
  keep one or two full-lowering smoke tests around for good measure.)

- If possible, add a semantic test under ``test/libsolidity/semanticTests/mlir``
  following the format of other tests there.

3. Linting
~~~~~~~~~~
- Once all the tests pass, if possible, fix all compiler warnings.

- clang-format your changes using something like:

  .. code-block:: bash

    git clang-format

  Don’t run clang-format on upstream solc files! Follow the surrounding style
  instead - afaik, upstream solc doesn't use auto-formatters.

- Then, optionally, run clang-tidy on your changes. I just run this:

  .. code-block:: bash

    find libsolidity/codegen/mlir -name '*.cpp' \
      | xargs -n1 -P"$(nproc)" clang-tidy -p <build-path>

Intro to the codegen
--------------------
``libsolidity/codegen/mlir`` has all the mlir lowering code, dialect
definitions, transformations etc. The code in there is built into the
``libsolidity`` library by ``libsolidity/CMakeLists.txt``. From now on, all file
paths are relative to ``libsolidity/codegen/mlir`` (unless stated otherwise).

Dialects
~~~~~~~~
``Sol/`` directory defines the sol dialect, which is a high level dialect to
represent solidity. Its type-system is defined in ``Sol/SolBase.td`` and its
operations in ``Sol/SolOps.td``. Similarly ``Yul/`` for the yul dialect, which
is more or less the yul ir from solc. Both dialects are under the ``mlir``
namespaces (``mlir::sol`` and ``mlir::yul``)

``mlir-tblgen`` generates the C++ into the build directory's
``libsolidity/codegen/mlir/<dialect>/*.inc`` which can be consulted when the
build errors are harder to follow.

The lowering of high level dialect for evm is under ``Target/EVM``. evm specific
code and utilities are under the ``evm`` namespace.

Pipeline
~~~~~~~~
``SolidityToMLIR.cpp`` lowers the solidity ast to the sol dialect and
``YulToMLIR.cpp`` lowers the yul ast to the yul dialect. ``Passes.cpp`` track
the pass-manager that further lowers the high level dialects. The functions
under the ``CompilerStack`` are integrated with solc's core pipeline. We try to
minimize adding functions in ``CompilerStack``, and instead use the
``solidity::mlirgen`` namespace for things like parsing mlir options, pass
management etc. This namespace is almost like a dumping ground for everything in
the mlir codegen that's not in an mlir dialect and neither in the ``evm``
namespace.

Builder extensions
~~~~~~~~~~~~~~~~~~
There are generic and target specific utilities. The generic builder extension
in the ``solidity::mlirgen`` namespace (called ``BuilderExt``) a builder that
generates frequently used snippets that are not evm specific. While the ``evm``
namespace has its own builder for building frequently generated snippets that
are evm specific. Frequently occurring ir generations should ideally go in one
of these builder extensions.
