contract C {
  function f(uint a, uint b) public returns (uint) {
    uint r = 1;
    for (uint i = 0; i < a; i += 1) {
      if (i == b)
        break;
      r += r;
    }
    return r;
  }
}

// ====
// compileViaMlir: true
// ----
// f(uint256,uint256): 20, 10 -> 1024
