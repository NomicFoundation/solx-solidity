.. _mlir-calldata:

Reuse calldata copies
=====================

When a function needs to materialize `msg.data` (or a slice of it) in memory,
the compiler must ensure we do not emit redundant `calldatacopy` sequences.
Multiple consumers of the same calldata view should share a single memory
allocation and copy whenever possible.
