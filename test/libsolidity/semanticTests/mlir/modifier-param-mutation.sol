contract C {
  modifier clamp(uint v) {
    if (v > 10) v = 10;
    _;
  }

  function f(uint v) public clamp(v) returns (uint) {
    return v;
  }
}

// ====
// compileViaMlir: true
// ----
// f(uint256): 5 -> 5
// f(uint256): 50 -> 50
