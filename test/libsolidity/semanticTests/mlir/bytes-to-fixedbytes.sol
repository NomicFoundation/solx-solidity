contract C {
  bytes emptyStorage = "";
  bytes shortStorage = "abcdefghabcdefghabcdefghabcdefg";
  bytes longStorage = "abcdefghabcdefghabcdefghabcdefghX";

  function fromMemory16(bytes memory m) public pure returns (bytes16) {
    return bytes16(m);
  }

  function fromMemoryCleanup(bytes memory m) public pure returns (bytes16) {
    assembly {
      mstore(m, 14)
    }
    return bytes16(m);
  }

  function fromMemory32(bytes memory m) public pure returns (bytes32) {
    return bytes32(m);
  }

  function fromCalldata16(bytes calldata c) external pure returns (bytes16) {
    return bytes16(c);
  }

  function fromCalldata32(bytes calldata c) external pure returns (bytes32) {
    return bytes32(c);
  }

  function fromStorageEmpty() external view returns (bytes32) {
    return bytes32(emptyStorage);
  }

  function fromStorageShort() external view returns (bytes32) {
    return bytes32(shortStorage);
  }

  function fromStorageLong() external view returns (bytes32) {
    return bytes32(longStorage);
  }
}

// ====
// compileViaMlir: true
// ----
// fromMemory16(bytes): 0x20, 16, "abcdefghabcdefgh" -> "abcdefghabcdefgh"
// fromMemoryCleanup(bytes): 0x20, 16, "abcdefghabcdefgh" -> "abcdefghabcdef\0\0"
// fromMemory32(bytes): 0x20, 0 -> 0
// fromCalldata16(bytes): 0x20, 15, "abcdefghabcdefgh" -> "abcdefghabcdefg\0"
// fromCalldata32(bytes): 0x20, 0 -> 0
// fromStorageEmpty() -> 0
// fromStorageShort() -> "abcdefghabcdefghabcdefghabcdefg\0"
// fromStorageLong() -> "abcdefghabcdefghabcdefghabcdefgh"
