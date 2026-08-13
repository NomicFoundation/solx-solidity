// Contract-less source units must survive linking actions (print-obj used to
// hit printJob's llvm_unreachable). No output to check; the run not crashing
// is the test.
// RUN: solc --mlir-action=print-obj --mlir-target=evm %s

function ret_42() pure returns (uint256) {
  return 42;
}
