contract C {
  enum E { A, B, C }

  function int8_min() public pure returns (int8) {
    return type(int8).min;
  }

  function int8_max() public pure returns (int8) {
    return type(int8).max;
  }

  function uint8_min() public pure returns (uint8) {
    return type(uint8).min;
  }

  function uint8_max() public pure returns (uint8) {
    return type(uint8).max;
  }

  function int256_min() public pure returns (int256) {
    return type(int256).min;
  }

  function int256_max() public pure returns (int256) {
    return type(int256).max;
  }

  function uint256_min() public pure returns (uint256) {
    return type(uint256).min;
  }

  function uint256_max() public pure returns (uint256) {
    return type(uint256).max;
  }

  function enum_min() public pure returns (E) {
    return type(E).min;
  }

  function enum_max() public pure returns (E) {
    return type(E).max;
  }
}

// ====
// compileViaMlir: true
// ----
// int8_min() -> -128
// int8_max() -> 127
// uint8_min() -> 0
// uint8_max() -> 255
// int256_min() -> -57896044618658097711785492504343953926634992332820282019728792003956564819968
// int256_max() -> 57896044618658097711785492504343953926634992332820282019728792003956564819967
// uint256_min() -> 0
// uint256_max() -> 115792089237316195423570985008687907853269984665640564039457584007913129639935
// enum_min() -> 0
// enum_max() -> 2

