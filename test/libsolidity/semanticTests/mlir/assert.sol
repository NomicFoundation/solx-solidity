contract C {
  function f(uint a) public returns (uint) {
    assert(a > 42);
    return a;
  }
}

// ====
// compileViaMlir: true
// ----
// f(uint256): 100 -> 100
// f(uint256): 10 -> FAILURE, hex"4e487b71", 0x01
