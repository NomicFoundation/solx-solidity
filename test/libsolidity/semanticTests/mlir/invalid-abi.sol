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
  function e() public returns (uint256) {
    bytes memory data = hex"00000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000000";
    abi.decode(data, (string[2]));
    return 1;
  }
}

// ====
// compileViaMlir: true
// ----
// f(uint256[]): 32, 3, 1, 2 -> FAILURE, hex"08c379a0", 0x20, 0x2b, "ABI decoding: invalid calldata a", "rray stride"
// g(string): 32, 33, "Hello" -> FAILURE, hex"08c379a0", 0x20, 0x27, "ABI decoding: invalid byte array", " length"
// g(string): 0x10000000000000000 -> FAILURE, hex"08c379a0", 0x20, 0x22, "ABI decoding: invalid tuple offs", "et"
// h(uint256[][]): 32, 2, 0x200, 0xa0, 2, 1, 2, 2, 9, 8 -> FAILURE, hex"08c379a0", 0x20, 0x2b, "ABI decoding: invalid calldata a", "rray offset"
// h(uint256[][]): 32, 1, 0x10000000000000000 -> FAILURE, hex"08c379a0", 0x20, 0x2b, "ABI decoding: invalid calldata a", "rray offset"
// h(uint256[][]): 32, 1, 32, 3618502788666131106986593281521497120414687020801267626233049500247285301248 -> FAILURE, hex"4e487b71", 0x41
// e() -> FAILURE, hex"08c379a0", 0x20, 0x2b, "ABI decoding: invalid calldata a", "rray stride"
