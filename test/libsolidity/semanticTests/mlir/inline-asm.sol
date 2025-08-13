contract C {
  function f() public returns (uint) {
    uint a = 0;
    assembly {
      a := 1
    }
    return a;
  }
}

// ====
// compileViaMlir: true
// ----
// f() -> 1
