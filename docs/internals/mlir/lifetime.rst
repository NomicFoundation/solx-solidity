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

Optimizing out the free-ptr
---------------------------
We could have a pass that analyzes ops with a malloc trait

- Tags offsets, if possible, relative to the free address.
- Generates the free op when they die.

.. code-block:: mlir

  sol.contract {
    sol.func @f() {
      %1 = sol.malloc {offset = 0} : !sol.array<2 x ui256, Memory>
      %2 = sol.malloc {offset = 2} : !sol.array<? x ui256, Memory>
      %3 = sol.malloc {offset = ?} : !sol.array<? x ui256, Memory>
      %4 = sol.encode ... {offset = ?} : !sol.string<Memory>
      %5 = sol.data_loc_cast ... {offset = ?} : !sol.string<Memory>
      ...
      sol.free %1
      sol.free %2
      ...
    }
  }

We don't need to generate the free-ptr iff:

1. All malloc trait ops are local and their offsets are known. **OR** If there
   are no ops with a malloc trait
2. No dynamic state variable arrays. This should be done by 1 since we generate
   the getters in the ast lowering
3. No free-ptr access in inline-asm

There are some potential problems with this:

- Since we're reusing heap, we might need to explicitly generate a memset zero
  loop for the allocations and/or make sure the array is written before any
  reads.  I couldn't find this rule in :ref:`arrays`, so it's not clear if we
  can get away without the memset zero.
- If a callee and caller has malloc ops with constant offsets, the callee needs
  to allocate after the caller's memory if the call doesn't happen after freeing
  all caller's allocation. The offset could be passed as an arg as
  it can be different for each callsite. But it might make things worse as we
  might need to propagate it further if a function reachable (in the callgraph)
  from the callee generates allocation. Better to bail out in such cases?
