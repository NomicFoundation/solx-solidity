contract C {
  uint public x;

  modifier m() {
    _;
    x += 100;
  }

  function f(bool c) public m returns (uint r) {
    x = 1;
    if (c) return 42;
    x = 2;
    return 7;
  }
}

// ====
// compileViaMlir: true
// ----
// f(bool): true -> 42
// x() -> 101
// f(bool): false -> 7
// x() -> 102
