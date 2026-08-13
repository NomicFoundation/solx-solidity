contract C {
  uint public x;

  function set(uint v) internal {
    x = v;
  }

  function f(uint v) public {
    if (v > 10) {
      return set(100);
    }
    return set(v);
  }
}

// ====
// compileViaMlir: true
// ----
// f(uint256): 5 ->
// x() -> 5
// f(uint256): 20 ->
// x() -> 100
