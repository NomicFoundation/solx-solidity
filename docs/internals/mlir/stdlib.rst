.. _mlir-stdlib:

Bytecode stdlib
===============
Simple lowering should continue using OpBuilder. But lengthy lowering might be
easier to maintain by linking to a stdlib in mlir bytecode.

A stdlib func op can represent a snippet of ir.  Each target could have
pre-compiled bytecode for each stdlib func op.

Pre-processing
--------------
Conditional usage of OpBuilder can be mimiced with templatized stdlib like
solc's Whiskers.

Problems with reference
-----------------------
We can't reference the ir entities from a stdlib source in C++ like the
OpBuilder apporach.

EraVM problem
-------------
Some yul ops' lowering for EraVM is creation/runtime context sensitive. So we
need to inline the stdlib calls so that the later codegen gets the context from
the callee.
