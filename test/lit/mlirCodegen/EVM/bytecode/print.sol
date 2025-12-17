// RUN: solc --mlir-action=print-obj --mlir-target=evm %s | FileCheck %s

contract C { function ret_42() public returns (uint256) { return 42; } }

// CHECK: Binary:
// CHECK: {{[0-9a-f]+}}
// CHECK: Binary of the runtime part:
// CHECK: {{[0-9a-f]+}}
