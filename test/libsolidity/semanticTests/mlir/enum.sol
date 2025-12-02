contract C {
  enum E { A, B, C }

  function id(E a) public returns (E) {
    return a;
  }

  function cmp(E a, E b) public returns (bool, bool) {
    return (a == b, a < b);
  }

  function fromUint(uint x) public returns (E) {
    return E(x);
  }

  function toUint(E x) public returns (uint) {
    return uint(x);
  }
}

// ====
// compileViaMlir: true
// ----
// id(uint8): 0 -> 0
// id(uint8): 2 -> 2
// id(uint8): 3 -> FAILURE, hex"4e487b71", 0x21
// cmp(uint8,uint8): 0, 1 -> false, true
// cmp(uint8,uint8): 2, 2 -> true, false
// toUint(uint8): 2 -> 2
// fromUint(uint256): 1 -> 1
// fromUint(uint256): 3 -> FAILURE, hex"4e487b71", 0x21
