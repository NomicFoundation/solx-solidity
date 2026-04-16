contract C {
  enum E { A, B, C }

  function add(uint a, uint b) external pure returns (uint) {
    return a + b;
  }

  function loadCalldataByte(bytes calldata data) external pure returns (bytes1) {
    return data[0];
  }

  function loadCalldataBool(bool[] calldata values) external pure returns (bool) {
    return values[0];
  }

  function loadCalldataUint8(uint8[] calldata values) external pure returns (uint8) {
    return values[0];
  }

  function loadCalldataInt8(int8[] calldata values) external pure returns (int8) {
    return values[0];
  }

  function loadCalldataEnum(E[] calldata values) external pure returns (uint8) {
    return uint8(values[0]);
  }

  function loadCalldataAddress(address[] calldata values) external pure returns (address) {
    return values[0];
  }

  function loadCalldataBytes4(bytes4[] calldata values) external pure returns (bytes4) {
    return values[0];
  }

  function loadCalldataBytes31(bytes31[] calldata values) external pure returns (bytes31) {
    return values[0];
  }

  function loadCalldataFnPtrHelper(function(uint, uint) external pure returns (uint)[] calldata values) external view returns (uint) {
    return values[0](20, 22);
  }

  function loadCalldataFnPtrSuccess() external returns (uint) {
    function(uint, uint) external pure returns (uint)[] memory values =
      new function(uint, uint) external pure returns (uint)[](1);
    values[0] = this.add;
    return this.loadCalldataFnPtrHelper(values);
  }
}

// ====
// compileViaMlir: true
// ----
// loadCalldataByte(bytes): 32, 1, "X" -> "X"
// loadCalldataBool(bool[]): 0x20, 1, 1 -> true
// loadCalldataBool(bool[]): 0x20, 1, 2 -> FAILURE
// loadCalldataUint8(uint8[]): 0x20, 1, 0x34 -> 0x34
// loadCalldataUint8(uint8[]): 0x20, 1, 0x1234 -> FAILURE
// loadCalldataInt8(int8[]): 0x20, 1, 0x7f -> 127
// loadCalldataInt8(int8[]): 0x20, 1, 0x1234 -> FAILURE
// loadCalldataEnum(uint8[]): 0x20, 1, 2 -> 2
// loadCalldataEnum(uint8[]): 0x20, 1, 3 -> FAILURE
// loadCalldataAddress(address[]): 0x20, 1, 0x1234123412341234123412341234123412341234 -> 0x1234123412341234123412341234123412341234
// loadCalldataAddress(address[]): 0x20, 1, 0x10000000000000000000000000000000000000000 -> FAILURE
// loadCalldataBytes4(bytes4[]): 0x20, 1, left(0x12345678) -> left(0x12345678)
// loadCalldataBytes4(bytes4[]): 0x20, 1, 0x1234567800000000000000000000000000000000000000000000000000000001 -> FAILURE
// loadCalldataBytes31(bytes31[]): 0x20, 1, left(0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f) -> left(0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f)
// loadCalldataBytes31(bytes31[]): 0x20, 1, 0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1fff -> FAILURE
// loadCalldataFnPtrSuccess() -> 42
// loadCalldataFnPtrHelper(function[]): 0x20, 1, 0x1234123412341234123412341234123412341234112233440000000000000001 -> FAILURE
