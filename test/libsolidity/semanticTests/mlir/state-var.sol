contract C {
  uint public m = 7;
  uint public m2 = f();

  function f() public returns (uint) { return m + m2; }
}

// ====
// compileViaMlir: true
// ----
// f() -> 14
