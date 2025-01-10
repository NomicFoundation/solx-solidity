.. _opt-checked-arith:

Optimizing checked arithmetic
=============================

We could write an analysis that tracks the "known information" about a value at
a program point (like insignificant bits, known zeros/ones) like llvm's
`value-tracking <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Analysis/ValueTracking.h>`_.
Representing the source integral types in the ir (instead of extending
everything to i256) can help the analysis. For instance, such analysis can say
that the `a + b` in the following code will never overflow.

.. code-block:: solidity
  function g(uint a, uint b) returns (uint) { return a + b; }
  function f(uint32 a, uint32 b) returns(uint) { return g(a, b); }

It must, of course, using the call-graph check all the callsites (and also be
conservative in the presence of indirect calls). We could also perform function
specialization for a favourable callsite. But it would be constrained by the
size optimization.
