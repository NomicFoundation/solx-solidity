contract C {
  uint8 s0;
  bool s1;
  bool[] s2;
  bool[40] s3;

  function dirtyS1Byte() public {
    assembly {
      let slot := s1.slot
      let off := s1.offset
      let cur := sload(slot)
      let dirty := shl(mul(off, 8), 0x80)
      sstore(slot, or(cur, dirty))
    }
  }

  function setS1(bool v) public {
    s1 = v;
  }

  function getS1() public view returns (bool) {
    return s1;
  }

  function initS2() public {
    s2.push();
    s2.push();
    s2.push();
    s2.push();
    s2.push();
    s2.push();
  }

  function dirtyS2Byte5() public {
    assembly {
      let arrSlot := s2.slot
      mstore(0x00, arrSlot)
      let base := keccak256(0x00, 0x20)
      let slot := add(base, div(5, 32))
      let off := mod(5, 32)
      let cur := sload(slot)
      let dirty := shl(mul(off, 8), 0x80)
      sstore(slot, or(cur, dirty))
    }
  }

  function setS2_5(bool v) public {
    s2[5] = v;
  }

  function getS2_5() public view returns (bool) {
    return s2[5];
  }

  function dirtyS3Byte5() public {
    assembly {
      let base := s3.slot
      let slot := add(base, div(5, 32))
      let off := mod(5, 32)
      let cur := sload(slot)
      let dirty := shl(mul(off, 8), 0x80)
      sstore(slot, or(cur, dirty))
    }
  }

  function setS3_5(bool v) public {
    s3[5] = v;
  }

  function getS3_5() public view returns (bool) {
    return s3[5];
  }
}

// ====
// compileViaMlir: true
// ----
// dirtyS1Byte() ->
// getS1() -> 1
// setS1(bool): false ->
// getS1() -> 0
// dirtyS1Byte() ->
// setS1(bool): true ->
// getS1() -> 1
// initS2() ->
// dirtyS2Byte5() ->
// getS2_5() -> 1
// setS2_5(bool): false ->
// getS2_5() -> 0
// dirtyS2Byte5() ->
// setS2_5(bool): true ->
// getS2_5() -> 1
// dirtyS3Byte5() ->
// getS3_5() -> 1
// setS3_5(bool): false ->
// getS3_5() -> 0
// dirtyS3Byte5() ->
// setS3_5(bool): true ->
// getS3_5() -> 1
