.. _mlir-motivation:

Why mlir?
=========

Limitations of via-ir
---------------------
Yul (in the default dialect) is an excellent syntax sugar for evm assembly. But
as a compiler-ir, it lacks a robust type-system and high level operations.
Because of this, everything is in i256 and it forces the codegen to expand high
level constructs like contract definitions, memory allocation etc. to a low
level representation. For instance, the memory allocation is lowered to a free
pointer based manipulation which makes it harder to analyze; Contract
definitions are lowered to creation and runtime code sections, which might not
make sense for creation/runtime context insensitive analysis and
transformations.

via-ir (and the legacy pipeline) extensively uses a library of textual yul
builtin functions in the codegen. This makes the solc lowering easy to read and
maintain, but the preprocessing and compilation of the textual ir can contribute
to compile-time. Ideally, this shouldn't happen since the compiler, not the
user, is generating those.

ssa, symbols, attributes
------------------------
mlir is in ssa and it provides an infrastructure to do things like walking
through the users of an ssa definition, replacing all uses, block args (mlir
version of "phi").

Operations or ops can define and reference symbols which are tracked by symbol
table of an outer op. This, unlike ssa variables that are local to blocks,
allows cross block scope. This can be used for representing references to
contract, state variables, functions etc. in solidity.

Ops can track a static dictionary of attributes that can handle numbers,
strings, mlir types and so on.

Here's a sample snippet of mlir in the sol dialect that demonstrates ssa
variables, symbol references and attributes:

.. code-block:: mlir

  #Cancun = #sol<EvmVersion Cancun>
  #Contract = #sol<ContractKind Contract>
  #NonPayable = #sol<StateMutability NonPayable>
  module attributes {sol.evm_version = #Cancun} {
    sol.contract @C_13 {
      sol.func @f_12(%arg0: ui256) -> ui256 attributes {state_mutability = #NonPayable} {
        ...
        %2 = sol.call @f_12(%1) : (ui256) -> ui256
        sol.return %2 : ui256
      }
    } {interface_fns = [{selector = -1277270901 : i32, sym = @f_12, type = (ui256) -> ui256}], kind = #Contract}
  }


Semantics
---------
Custom operations and types-system can be defined in tablegen with minimal
effort. By default, it'll provide the build APIs, getters and setters for
operands and attributes, printing and parsing support for the default textual
assembly format.

You can optionally add verifier hooks in C++ that can verify the semantics,
traits and interfaces that can let analysis reason "opaquely", a custom asm
format (simple ones in tablegen; complex ones in C++).

Here's a simple example of defining the require op (for solidity's require
statements) in tablegen:

.. code-block:: text

  def Sol_RequireOp : Sol_Op<"require"> {
    let arguments = (ins I1:$cond, DefaultValuedStrAttr<StrAttr, "">:$msg);
    let assemblyFormat = "$cond `,` $msg attr-dict";
  }

With this snippet, the tablegen backend will generate C++ code that allows us to
build this operation in memory, parse and print this in text. The operation can
be serialized by the `inbuilt bytecode format
<https://mlir.llvm.org/docs/BytecodeFormat/>`_.

Interfaces
----------
Interfaces, like the ones in OOP, can a abstract a set of similar operations so
that transformations and analysis won't need to learn the semantics of all the
ops. See, for instance, `sol dialect interfaces
<https://github.com/matter-labs/era-solidity/blob/mlir/libsolidity/codegen/mlir/Sol/SolInterfaces.td>`_.

High level ir
-------------
mlir allows us to create a domain specific ir that's close to the ast. This
allows high level optimizations.

For instance, the memory allocation can be represented as an op that yields a
pointer to allocation instead of the low level free-ptr manipulation in yul.
This allows to do things like :ref:`mlir-lifetime`, optimal lowering of
multi-dim static array (linear layout instead of array of pointers) etc.

Another instance is checked arithmetic, which can be represented as it is
without expansion (sol.cadd, sol.csub etc.). This allows the optimizer to try
and convert them to unchecked version (sol.add, sol.sub etc.) (if legal) without
pattern-matching the expansion of checked arithmetic (see
:ref:`mlir-checked-arith`). The preservation of the original types (instead of
extending everything to i256) is important to this.

Another instance is optimizing for size in the presence of modifiers. The
default solc pipeline behaviour of inlining all modifiers is beneficial for
speed optimization. mlir can represent modifiers as it is and this allows us to
transform them to plain functions (legality depends on the placeholders). This
can be help size optimizations by skipping inlining, but without breaking the
speed optimization as we can still inline them in the pipeline.  See the
`modifier lowering
<https://github.com/matter-labs/era-solidity/blob/mlir/libsolidity/codegen/mlir/ModifierOpLowering.cpp>`_

Pass infrastructure
-------------------
The pass infrastructure supports debug logging, statistics, printing ir after
every pass or every change, compile-time analysis. See `mlir-opt cli
<https://mlir.llvm.org/docs/Tutorials/MlirOpt/#useful-cli-flags>`_

Progressive conversion
----------------------
mlir has a notion of dialects that can group related ops, attributes and
type-system. This allows progressive lowering from one level of abstraction to
another.  mlir provides a dialect conversion infrastructure which allows you to
write pattern rewrites in C++ or in `pdll <https://mlir.llvm.org/docs/PDLL/>`_,
and optionally a type-converter that's responsible for legalizing types.

Reuse across frontends
----------------------
For example, we could lower vyper into a dedicated vyper mlir dialect and then
translate it into the sol dialect, thereby reusing all existing optimizations
and lowering passes. This might require introducing additional intermediate
dialects or further generalizing the sol dialect. With the right abstraction
boundaries, however, we avoid starting from scratch for each new solidity-like
language.

Out of the box support for llvm
-------------------------------
Upstream mlir provides the llvm dialect which is more or less the llvm-ir in the
mlir representation. Standard dialects in upstream like arith (arithmetic ops),
scf (structured control flow like if/else, while etc.), cf (branch ops), func
(function ops) can be converted to the llvm dialect (some indirectly). We just
have to translate the downstream dialects to them (or directly to llvm dialect
(or ir via translation)). This is a major advantage if you have a working llvm
backend.
