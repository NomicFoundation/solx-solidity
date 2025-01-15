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

  function g(uint a) public returns (uint) {
    uint r = 1;
    do {
      r = 2;
    } while (false);

    uint i = 0;
    while (i < a) {
      if (i < 99) {
        r += 2;
        i += 1;
        continue;
      }
      r = 3;
    }
    return r;
  }
}

// ====
// compileViaMlir: true
// ----
// f(uint256,uint256): 20, 10 -> 1024
// g(uint256): 10 -> 22
