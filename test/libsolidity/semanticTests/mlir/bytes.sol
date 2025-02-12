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
}

// ====
// compileViaMlir: true
// ----
// b(bytes1): "a" -> 0x6100000000000000000000000000000000000000000000000000000000000000
// b2i(bytes1): "a" -> 0x61
// i2b(uint8): 1 -> 0x0100000000000000000000000000000000000000000000000000000000000000
