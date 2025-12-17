// RUN: solc --strict-assembly --mlir-action=print-obj --mlir-target=evm %s | FileCheck %s

object "Test" {
  code {
    return(10, 11)
  }
  object "Test_deployed" {
    code {
      return(20, 21)
    }
  }
}

// CHECK: Binary representation:
// CHECK: {{[0-9a-f]+}}
