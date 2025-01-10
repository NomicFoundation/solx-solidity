.. _mlir-lifetime:

Lifetime
========

The original pipeline doesn't track lifetimes of allocations and hence it
doesn't free them when they go out of scope. For most allocations, the generated
ir keeps updating the free-ptr. For some cases, the pipeline generates unbounded
allocations by either delaying the free-ptr update or skipping it (if it can
statically know that it's the last allocation before termination).  This can be
seen in the following forwarding revert lowering.

.. code-block:: yul

  function <functionName>() {
    let pos := <allocateUnbounded>()
    returndatacopy(pos, 0, returndatasize())
    revert(pos, returndatasize())
  }

The current mlir lowering lowers memory allocations to the `sol.malloc` op which
yields a high level address type (see :ref:`mlir-addr-calc`).  This allows us to
detect local allocations like:

.. code-block:: solidity

   {
     uint[10] memory a;
     // ...
   }

and convert them to unbounded allocation, which does a faster allocation + takes
care of the deallocation requirement (since we're not updating the free-ptr at
all). There are challenges like:

- Needs escape analysis for legality. Unlike RAII, the allocations can "escape"
  to global variables etc. and we shouldn't free them even if they go out of
  scope.
- Can only free contiguous allocations starting from the free ptr. We can't use
  unbounded allocations in the following:

.. code-block:: solidity

   {
     <alloc-a>; // local
     <alloc-b>; // global
   }

but we can reorder the allocations so that the local ones are contiguous from
the free-ptr.
