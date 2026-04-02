contract C {
  struct S {
    uint256 a;
    string b;
  }

  function f(uint[] memory a) public returns (uint[] memory) {
    return a;
  }

  function g(string memory a) public returns (string memory) {
    return a;
  }

  function h(uint[][] memory a) public returns (uint[][] memory) {
    return a;
  }

  function c(uint[] calldata a) external returns (uint256) {
    return a[0];
  }

  function d(uint256[][2] calldata a) external returns (uint256) {
    return 1;
  }

  function e() public returns (uint256) {
    bytes memory data = hex"00000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000000";
    abi.decode(data, (string[2]));
    return 1;
  }

  function sc(S calldata s) external returns (uint256) {
    return s.a;
  }

  function ec(S calldata s) external returns (bytes memory) {
    return abi.encode(s);
  }

  function ec_arr(uint256[][2] calldata x) external returns (bytes memory) {
    return abi.encode(x);
  }

  function ec_arr_dyn(uint256[][] calldata x) external returns (bytes memory) {
    return abi.encode(x);
  }

  function ec_arr_nested(uint256[][2][] calldata x) external returns (bytes memory) {
    return abi.encode(x);
  }

  function sm(S memory s) external returns (uint256) {
    return s.a;
  }
}

// ====
// compileViaMlir: true
// revertStrings: debug
// ----
// f(uint256[]): 32, 3, 1, 2 -> FAILURE, hex"08c379a0", 0x20, 0x2b, "ABI decoding: invalid calldata a", "rray stride"
// g(string): 32, 33, "Hello" -> FAILURE, hex"08c379a0", 0x20, 0x27, "ABI decoding: invalid byte array", " length"
// g(string): 0x10000000000000000 -> FAILURE, hex"08c379a0", 0x20, 0x22, "ABI decoding: invalid tuple offs", "et"
// h(uint256[][]): 32, 2, 0x200, 0xa0, 2, 1, 2, 2, 9, 8 -> FAILURE, hex"08c379a0", 0x20, 0x2b, "ABI decoding: invalid calldata a", "rray offset"
// h(uint256[][]): 32, 1, 0x10000000000000000 -> FAILURE, hex"08c379a0", 0x20, 0x2b, "ABI decoding: invalid calldata a", "rray offset"
// h(uint256[][]): 32, 1, 32, 3618502788666131106986593281521497120414687020801267626233049500247285301248 -> FAILURE, hex"4e487b71", 0x41
// c(uint256[]): 0x20, 0x8000000000000000000000000000000000000000000000000000000000000000 -> FAILURE, hex"08c379a0", 0x20, 0x2b, "ABI decoding: invalid calldata a", "rray length"
// d(uint256[][2]): 0x20, 0x40 -> FAILURE, hex"08c379a0", 0x20, 0x2b, "ABI decoding: invalid calldata a", "rray stride"
// e() -> FAILURE, hex"08c379a0", 0x20, 0x2b, "ABI decoding: invalid calldata a", "rray stride"
// sc((uint256,string)): 0x20, 1 -> FAILURE, hex"08c379a0", 0x20, 0x27, "ABI decoding: struct calldata to", "o short"
// ec((uint256,string)): 0x20, 1, 0x100, 0x80 -> FAILURE, hex"08c379a0", 0x20, 0x1e, "Invalid calldata access offset"
// ec((uint256,string)): 0x20, 1, 0x40, 0x10000000000000000 -> FAILURE, hex"08c379a0", 0x20, 0x1e, "Invalid calldata access length"
// ec((uint256,string)): 0x20, 1, 0x40, 2 -> FAILURE, hex"08c379a0", 0x20, 0x1e, "Invalid calldata access stride"
// ec_arr(uint256[][2]): 0x20, 0x100, 0x40 -> FAILURE, hex"08c379a0", 0x20, 0x1e, "Invalid calldata access offset"
// ec_arr(uint256[][2]): 0x20, 0x40, 0x80, 0x10000000000000000 -> FAILURE, hex"08c379a0", 0x20, 0x1e, "Invalid calldata access length"
// ec_arr(uint256[][2]): 0x20, 0x40, 0x80, 2, 1 -> FAILURE, hex"08c379a0", 0x20, 0x1e, "Invalid calldata access stride"
// ec_arr_dyn(uint256[][]): 0x20, 1, 0x100 -> FAILURE, hex"08c379a0", 0x20, 0x1e, "Invalid calldata access offset"
// ec_arr_dyn(uint256[][]): 0x20, 1, 0x20, 0x10000000000000000 -> FAILURE, hex"08c379a0", 0x20, 0x1e, "Invalid calldata access length"
// ec_arr_dyn(uint256[][]): 0x20, 1, 0x20, 2, 1 -> FAILURE, hex"08c379a0", 0x20, 0x1e, "Invalid calldata access stride"
// ec_arr_nested(uint256[][2][]): 0x20, 1, 0x100 -> FAILURE, hex"08c379a0", 0x20, 0x1e, "Invalid calldata access offset"
// ec_arr_nested(uint256[][2][]): 0x20, 1, 0x20, 0x40, 0x80, 0x10000000000000000 -> FAILURE, hex"08c379a0", 0x20, 0x1e, "Invalid calldata access length"
// ec_arr_nested(uint256[][2][]): 0x20, 1, 0x20, 0x40, 0x80, 2, 1 -> FAILURE, hex"08c379a0", 0x20, 0x1e, "Invalid calldata access stride"
// sm((uint256,string)): 0x20, 1 -> FAILURE, hex"08c379a0", 0x20, 0x23, "ABI decoding: struct data too sh", "ort"
// sm((uint256,string)): 0x20, 1, 0x10000000000000000 -> FAILURE, hex"08c379a0", 0x20, 0x23, "ABI decoding: invalid struct off", "set"
