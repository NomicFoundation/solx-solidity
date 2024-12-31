Bytecode stdlib
===============
Simple lowering should continue using OpBuilder. But lengthy lowering might be
easier to maintain by linking to a stdlib in mlir bytecode.

A stdlib func op can represent a snippet of ir.  Each target could have
pre-compiled bytecode for each stdlib func op.

EraVM problem
-------------
Some yul ops' lowering for EraVM is creation/runtime context sensitive. So we
need to inline the stdlib calls so that the later codegen gets the context from
the callee.

Size optimization?
------------------
Using stdlib calls without inlining can let llvm have finer control on inlining.
This might benefit optimizations for size. The "EraVM problem" has to be fixed
some other way in this case.
