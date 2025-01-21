contract C {
  function uc(uint a, uint b) public returns (uint) {
    uint r = 0;
    unchecked {
    r += a + b;
    r *= a * b;
    r -= a - b;
    r = (a + r) * 2;
    }
    return r;
  }

  function c(uint a, uint b) public returns (uint) {
    return a + b;
  }

  function uci8(uint8 a) public returns (uint8) {
    unchecked { return a + 1; }
  }

  function ci8(uint8 a) public returns (uint8) {
    return a + 1;
  }

  function neg(int a) public returns (int) {
    return -a;
  }

  function inc(uint a) public returns (uint) {
    uint r = a++;
    r += a;
    return ++r;
  }
}

// ====
// compileViaMlir: true
// ----
// uc(uint256,uint256): 4, 2 -> 100
// c(uint256,uint256): -2, 1 -> -1
// c(uint256,uint256): -2, 2 -> FAILURE, hex"4e487b71", 0x11
// uci8(uint8): 255 -> 0
// ci8(uint8): 255 -> FAILURE, hex"4e487b71", 0x11
// neg(int256): 1 -> -1
// inc(uint256): 0 -> 2
