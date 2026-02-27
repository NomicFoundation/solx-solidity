contract C {
  uint x;

  function f() public {}

  function get() public returns (C) {
    x = 42;
    return this;
  }

  function selector() public returns (bytes4) {
    return this.f.selector;
  }

  function sideEffect() public returns (uint) {
    get().f.selector;
    return x;
  }
}

// ====
// compileViaMlir: true
// ----
// selector() -> left(0x26121ff0)
// sideEffect() -> 42
