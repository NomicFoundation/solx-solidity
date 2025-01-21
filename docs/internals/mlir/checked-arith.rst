.. _mlir-checked-arith:

Optimizing checked arithmetic
=============================

Optimal lowering
----------------
We can generate lesser checks for signed arithmetic if we know the value of 1
operand (see YulUtilFunctions for unary expression lowering like
incrementCheckedFunction).  We should integrate this in the sol dialect's binary
arithmetic op lowering (after canonicalizing the constant arg position).


Analysis
--------
We could write an analysis that tracks the "known information" about a value at
a program point (like insignificant bits, known zeros/ones) like llvm's
`value-tracking
<https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Analysis/ValueTracking.h>`_.
Representing the source integral types in the ir (instead of extending
everything to i256) can help the analysis. For instance, such analysis can say
that the ``a + b`` in the following code will never overflow.

.. code-block:: solidity

  function g(uint a, uint b) returns (uint) { return a + b; }
  function f(uint32 a, uint32 b) returns(uint) { return g(a, b); }

It must, of course, using the call-graph check all the callsites (and also be
conservative in the presence of indirect calls). We could also perform function
specialization for a favourable callsite. But it would be constrained by the
size optimization.
