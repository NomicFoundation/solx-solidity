.. _mlir-fuse-free-ptr:

Optimizing free-ptr modification
================================

Given:

.. code-block:: mlir

  %a = malloc : !array<2 x *, Memory>
  %b = malloc : !array<3 x *, Memory>

we can convert it to:

.. code-block:: mlir

  %a = upd_free_ptr 2 : i256
  %b = upd_free_ptr 3 : i256

where ``upd_free_ptr <x>`` adds the old free-ptr with ``<x>`` and stores it
back, but returns the original free-ptr (like a post-increment). While we could
instead have separate load and store free-ptr ops, we would then need to track
the add ops that advances the free-ptr, which I think is unnecessary. This
representation is relatively easier to analyze and it still allows representing
unbounded allocation with ``upd_free_ptr 0``.

Then we can eliminate the redundant free-ptr modification by rewriting it to:

.. code-block:: mlir

  %a = upd_free_ptr 5 : i256
  %b = add %a, 2
