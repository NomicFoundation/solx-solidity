// XFAIL: *
// RUN: solc --strict-assembly --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

object "Test" {
  code {
    setimmutable(10, "foo", 0x20)
  }
  object "Test_deployed" {
    code {
      let a := loadimmutable("foo")
    }
  }
}
