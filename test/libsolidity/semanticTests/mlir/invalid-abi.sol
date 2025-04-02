contract C {
  function f(uint[] memory a) public returns (uint[] memory) {
    return a;
  }
  function g(string memory a) public returns (string memory) {
    return a;
  }
  function h(uint[][] memory a) public returns (uint[][] memory) {
    return a;
  }
}

// ====
// compileViaMlir: true
// ----
// f(uint256[]): 32, 3, 1, 2 -> FAILURE, hex"08c379a0", 0x20, 0x20, "ABI decoding: invalid array size"
// g(string): 32, 33, "Hello" -> FAILURE, hex"08c379a0", 0x20, 0x27, "ABI decoding: invalid byte array", " length"
// h(uint256[][]): 32, 2, 0x200, 0xa0, 2, 1, 2, 2, 9, 8 -> FAILURE, hex"08c379a0", 0x20, 0x22, "ABI decoding: invalid array offs", "et"
