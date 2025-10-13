.. _mlir-storage-promotion:

Storage promotion
=================
Storage access is expensive and it's beneficial to promote it to stack and/or
memory when the access is hot.

How?
----
In every kind of promotion, we replace all load/stores with stack/memory
load/stores.

The promotion can be done like:
(a) If the state-variable is read, then copy the variable to stack/memory in the
function entry.
(b) If the state-variable is written to, then copy the stack/memory variable
back to storage in the function exit.
(c) If both, then do (a) and (b)

Legal?
------
We can't promote if there's an inline-asm (anywhere in the function including
the call-graph from it) with:
- storage access instructions referencing the state-variable of interest
  (in)directly.
- does a delegatecall (since storage is shared)

Profitable?
-----------
The promotion might regress if the access is cold as the promotion code is not
cheap. We could start with only promoting state-variables accesses that happen
in a loop. Unfortunately, this misses promoting accesses in no-loop functions
that are called from loops.
