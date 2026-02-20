contract C {
  function at_u8(uint8[] memory x, uint i) public returns (uint8) {
    return x[i];
  }

  function at_u32(uint32[] memory x, uint i) public returns (uint32) {
    return x[i];
  }

  function at_u64(uint64[] memory x, uint i) public returns (uint64) {
    return x[i];
  }

  function at_u128(uint128[] memory x, uint i) public returns (uint128) {
    return x[i];
  }
}

// ====
// compileViaMlir: true
// ----
// at_u8(uint8[],uint256): 0x40, 2, 3, 11, 22, 33 -> 33
// at_u32(uint32[],uint256): 0x40, 1, 3, 11, 22, 33 -> 22
// at_u64(uint64[],uint256): 0x40, 0, 3, 0x1111, 0x2222, 0x3333 -> 0x1111
// at_u128(uint128[],uint256): 0x40, 2, 3, 0x1111, 0x2222, 0x3333 -> 0x3333
