contract C {
  uint256[] dyn256;
  uint128[] dyn128;
  uint8[] dyn8;

  uint256[10] static256x10;
  uint128[3] static128x3;
  uint8[31] static8x31;
  uint8[32] static8x32;
  uint8[33] static8x33;

  constructor() {
    dyn256.push(0x11);
    dyn256.push(0x22);

    dyn128.push(0x1234);
    dyn128.push(0x5678);
    dyn128.push(0x9abc);

    dyn8.push(1);
    dyn8.push(2);
    dyn8.push(3);
    dyn8.push(4);

    for (uint256 i = 0; i < 10; ++i)
      static256x10[i] = i + 1;

    for (uint256 i = 0; i < 3; ++i)
      static128x3[i] = uint128((i + 1) * 16);

    for (uint256 i = 0; i < 31; ++i)
      static8x31[i] = uint8(i + 1);

    for (uint256 i = 0; i < 32; ++i)
      static8x32[i] = uint8(i + 1);

    for (uint256 i = 0; i < 33; ++i)
      static8x33[i] = uint8(i + 1);
  }

  function ep_u256_array_dynamic_storage() public view returns (bytes memory) {
    return abi.encodePacked(dyn256);
  }

  function ep_u128_array_dynamic_storage() public view returns (bytes memory) {
    return abi.encodePacked(dyn128);
  }

  function ep_u8_array_dynamic_storage() public view returns (bytes memory) {
    return abi.encodePacked(dyn8);
  }

  function ep_u256_array_static_storage() public view returns (bytes memory) {
    return abi.encodePacked(static256x10);
  }

  function ep_u128_array_static_storage() public view returns (bytes memory) {
    return abi.encodePacked(static128x3);
  }

  function ep_u8_array_static_storage_31() public view returns (bytes memory) {
    return abi.encodePacked(static8x31);
  }

  function ep_u8_array_static_storage_32() public view returns (bytes memory) {
    return abi.encodePacked(static8x32);
  }

  function ep_u8_array_static_storage_33() public view returns (bytes memory) {
    return abi.encodePacked(static8x33);
  }
}

// ====
// compileViaMlir: true
// ----
// constructor() ->
// ep_u256_array_dynamic_storage() -> 32, 64, 17, 34
// ep_u128_array_dynamic_storage() -> 32, 96, 4660, 22136, 39612
// ep_u8_array_dynamic_storage() -> 32, 128, 1, 2, 3, 4
// ep_u256_array_static_storage() -> 32, 320, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
// ep_u128_array_static_storage() -> 32, 96, 16, 32, 48
// ep_u8_array_static_storage_31() -> 32, 992, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
// ep_u8_array_static_storage_32() -> 32, 1024, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
// ep_u8_array_static_storage_33() -> 32, 1056, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
