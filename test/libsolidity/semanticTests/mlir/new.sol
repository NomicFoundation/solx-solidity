contract C {
  function f() public returns (uint) { return 42; }
}

contract D {
  function g() public returns (uint) { return new C().f(); }
}

// ====
// compileViaMlir: true
// ----
// g() -> 42
