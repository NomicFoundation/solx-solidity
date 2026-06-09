contract C {
  function int24_min() public pure returns (int24) { return type(int24).min; }

  function int24_max() public pure returns (int24) { return type(int24).max; }

  function int72_min() public pure returns (int72) { return type(int72).min; }

  function int72_max() public pure returns (int72) { return type(int72).max; }

  function int136_min() public pure returns (int136) { return type(int136).min; }

  function int136_max() public pure returns (int136) { return type(int136).max; }

  function rational_shr() public pure returns (uint8) { return 0x4200 >> 8; }

  function rational_sub() public pure returns (int72) { return 2**71 - 1; }

  function rational_lt() public pure returns (bool) { return 1 < 2; }

  function rational_eq() public pure returns (bool) { return 3 == 3; }
}

// ====
// compileViaMlir: true
// ----
// int24_min() -> -8388608
// int24_max() -> 8388607
// int72_min() -> -2361183241434822606848
// int72_max() -> 2361183241434822606847
// int136_min() -> -43556142965880123323311949751266331066368
// int136_max() -> 43556142965880123323311949751266331066367
// rational_shr() -> 66
// rational_sub() -> 2361183241434822606847
// rational_lt() -> true
// rational_eq() -> true
