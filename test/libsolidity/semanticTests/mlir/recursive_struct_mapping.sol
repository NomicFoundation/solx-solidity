// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

struct S {
  uint32 z;
  mapping(uint8 => S) rec;
}

contract C {
  S data;

  function set() public {
    data.z = 2;
    S storage inner = data.rec[0];
    inner.z = 3;
    inner.rec[0].z = inner.rec[1].z + 1;
  }

  function check() public view returns (uint32, uint32, uint32, uint32) {
    return (
      data.z,
      data.rec[0].z,
      data.rec[0].rec[0].z,
      data.rec[0].rec[1].z
    );
  }
}
// ====
// compileViaMlir: true
// ----
// check() -> 0, 0, 0, 0
// set() ->
// check() -> 2, 3, 1, 0
