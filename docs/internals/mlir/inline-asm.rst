.. _mlir-inline-asm:

Inline assembly
===============
sol.inline_asm blocks are non-speculatable by default and are assumed to have
all memory and storage effects.  We could have an analysis that refines the
attributes by analyzing its body. We could add additional attributes like if it
transfers the control-flow outside (via external calls like call, delegate) etc.
