contract C {
  function b(bytes1 a) public returns (bytes1) {
    return a;
  }
  function b2i(bytes1 a) public returns (uint8) {
    return uint8(a);
  }
  function i2b(uint8 a) public returns (bytes1) {
    return bytes1(a);
  }
  function bm(bytes memory a) public returns (bytes memory) {
    return a;
  }
  function ld(bytes memory a) public returns (bytes1) {
    return a[0];
  }
  function st(bytes memory a) public returns (bytes memory) {
    uint8 b = 0x61;
    a[0] = bytes1(b);
    return a;
  }
  function l(bytes memory a) public returns (uint) {
    return a.length;
  }
}

// ====
// compileViaMlir: true
// ----
// b(bytes1): "a" -> 0x6100000000000000000000000000000000000000000000000000000000000000
// b2i(bytes1): "a" -> 0x61
// i2b(uint8): 1 -> 0x0100000000000000000000000000000000000000000000000000000000000000
// bm(bytes): 32, 3, hex"AB33BB" -> 32, 3, left(0xAB33BB)
// ld(bytes): 32, 5, "hello" -> 0x6800000000000000000000000000000000000000000000000000000000000000
// st(bytes): 32, 5, "hello" -> 32, 5, 0x61656c6c6f000000000000000000000000000000000000000000000000000000
// l(bytes): 32, 5, "hello" -> 5
