contract C {
  function uc(uint a, uint b) public returns (uint) {
    uint r = 0;
    unchecked {
    r += a + b;
    r *= a * b;
    r -= a - b;
    r = (a + r) * 2;
    r = r / 2;
    r /= 2;
    r = r % 13;
    r %= 13;
    }
    return r;
  }

  function c(uint a, uint b) public returns (uint) {
    return a + b;
  }

  function ofa(uint8 a) public returns (uint8) {
    return a + 1;
  }

  function ofs(int8 a) public returns (int8) {
    return a - 1;
  }

  function ofm(int8 a) public returns (int8) {
    return a * 2;
  }

  function ofd(int8 a) public returns (int8) {
    return a / -1;
  }

  function dbz(int8 a) public returns (int8) {
    return a / 0;
  }

  function neg(int a) public returns (int) {
    return -a;
  }

  function inc(int a) public returns (int) {
    int r = a++;
    r += a;
    return ++r;
  }

  function dec(int8 a) public returns (int8) {
    return a--;
  }

  function shl(uint a, uint b) public returns (uint) {
    return a << b;
  }

  function shr(uint a, uint b) public returns (uint) {
    return a >> b;
  }

  function shl8(uint8 a, uint b) public returns (uint8) {
    return a << b;
  }

  function shr8(uint8 a, uint b) public returns (uint8) {
    return a >> b;
  }

  function bit(int a, int b) public returns (int, int, int) {
    return (a & b, a | b, a ^ b);
  }

  function bit8(int8 a, int8 b) public returns (int8, int8, int8) {
    return (a & b, a | b, a ^ b);
  }
}

// ====
// compileViaMlir: true
// ----
// uc(uint256,uint256): 4, 2 -> 12
// c(uint256,uint256): -2, 1 -> -1
// c(uint256,uint256): -2, 2 -> FAILURE, hex"4e487b71", 0x11
// ofa(uint8): 255 -> FAILURE, hex"4e487b71", 0x11
// ofs(int8): -128 -> FAILURE, hex"4e487b71", 0x11
// ofm(int8): 127 -> FAILURE, hex"4e487b71", 0x11
// ofd(int8): -128 -> FAILURE, hex"4e487b71", 0x11
// dbz(int8): 1 -> FAILURE, hex"4e487b71", 0x12
// neg(int256): 1 -> -1
// inc(int256): 0 -> 2
// dec(int8): -128 -> FAILURE, hex"4e487b71", 0x11
// shl(uint256,uint256): 4, 1 -> 8
// shr(uint256,uint256): 4, 1 -> 2
// shl8(uint8,uint256): 1, 8 -> 0
// shr8(uint8,uint256): 1, 8 -> 0
// bit(int256,int256): 6, 3 -> 2, 7, 5
// bit8(int8,int8): 6, 3 -> 2, 7, 5
