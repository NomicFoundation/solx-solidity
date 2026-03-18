contract C {
  struct Big {
    uint256 id;
    uint256 code;
    string note;
    uint256[2] fixedArr;
    uint256[] dynArr;
  }

  struct Nested {
    uint256 salt;
    Big payload;
    uint256 extra;
  }

  function mem_big(Big memory s) public pure returns (uint256, uint256, string memory, uint256[2] memory, uint256[] memory) {
    return (s.id, s.code, s.note, s.fixedArr, s.dynArr);
  }

  function mem_nested(Nested memory s) public pure returns (uint256, uint256) {
    return (s.salt, s.extra);
  }

  function cd_big(Big calldata s) public pure returns (uint256, uint256) {
    return (s.id, s.code);
  }

  function cd_nested(Nested calldata s) public pure returns (uint256, uint256) {
    return (s.salt, s.extra);
  }
}

// ====
// compileViaMlir: true
// ----
// mem_big((uint256,uint256,string,uint256[2],uint256[])): 0x20, 7, 8, 0xc0, 41, 42, 0x100, 3, "abc", 3, 11, 12, 13 -> 7, 8, 0xc0, 41, 42, 0x100, 3, "abc", 3, 11, 12, 13
// cd_big((uint256,uint256,string,uint256[2],uint256[])): 0x20, 7, 8, 0xc0, 41, 42, 0x100, 3, "abc", 3, 11, 12, 13 -> 7, 8
// mem_nested((uint256,(uint256,uint256,string,uint256[2],uint256[]),uint256)): 0x20, 100, 0x60, 200, 5, 6, 0xc0, 21, 22, 0x100, 2, "xy", 2, 31, 32 -> 100, 200
// cd_nested((uint256,(uint256,uint256,string,uint256[2],uint256[]),uint256)): 0x20, 100, 0x60, 200, 5, 6, 0xc0, 21, 22, 0x100, 2, "xy", 2, 31, 32 -> 100, 200
