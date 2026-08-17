contract C {
  function f(uint256[] calldata a) external pure returns (uint256) {
    return a.length;
  }

  function g(bytes calldata b) external pure returns (uint256) {
    return b.length;
  }

  function h(uint256[2] calldata a) external pure returns (uint256) {
    return a[1];
  }
}

// ====
// compileViaMlir: true
// ----
// f(uint256[]): 0x20, 2, 7, 8 -> 2
// f(uint256[]): 0x20, 3, 7, 8 -> FAILURE
// f(uint256[]): 0x60, 0 -> FAILURE
// f(uint256[]): 0x10000000000000000, 0 -> FAILURE
// f(uint256[]): 0x20, 0x10000000000000000 -> FAILURE
// g(bytes): 0x20, 3, hex"aabbcc0000000000000000000000000000000000000000000000000000000000" -> 3
// g(bytes): 0x20, 0x21, hex"aabbcc0000000000000000000000000000000000000000000000000000000000" -> FAILURE
// g(bytes): 0x20, 0x10000000000000000 -> FAILURE
// g(bytes): 0x20, 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe0 -> FAILURE
// h(uint256[2]): 1 -> FAILURE
