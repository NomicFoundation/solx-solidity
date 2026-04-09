contract C {
  bytes s;

  constructor() {
    s = hex"123456";
  }

  function hCalldata(uint8[] calldata x) external pure returns (bytes memory) {
    return abi.encode(x);
  }

  function iCalldata(uint8[] calldata x) external view returns (bytes memory) {
    return this.hCalldata(x);
  }

  function hStorage(bytes calldata x) external pure returns (bytes memory) {
    return abi.encode(x);
  }

  function iStorage() external view returns (bytes memory) {
    return this.hStorage(s);
  }
}

// ====
// compileViaMlir: true
// ----
// hCalldata(uint8[]): 32, 3, 23, 42, 87 -> 32, 160, 32, 3, 23, 42, 87
// iCalldata(uint8[]): 32, 3, 23, 42, 87 -> 32, 160, 32, 3, 23, 42, 87
// hCalldata(uint8[]): 32, 3, 0xFF23, 0x1242, 0xAB87 -> FAILURE
// iCalldata(uint8[]): 32, 3, 0xAB23, 0x1242, 0xFF87 -> FAILURE
// hStorage(bytes): 32, 3, hex"123456" -> 32, 96, 32, 3, left(0x123456)
// iStorage() -> 32, 96, 32, 3, left(0x123456)
