contract C {
  function f() public returns (uint) { return 42; }
}

contract D {
  function g() public returns (uint) {
    C c1 = new C();
    C c2 = c1;
    require(address(c1).code.length > 50);
    return 1;
  }
}

// ====
// compileViaMlir: true
// ----
// g() -> 1
