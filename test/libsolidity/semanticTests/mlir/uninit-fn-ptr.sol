contract C {
  // Never assigned: their types have no candidate functions, so both calls
  // dispatch over an empty set.
  function (uint) internal returns (uint) f1;
  function () internal f2;

  function a(uint x) public returns (uint) { return f1(x); }
  function b() public { f2(); }
}

// ====
// compileViaMlir: true
// ----
// a(uint256): 1 -> FAILURE, hex"4e487b71", 0x51
// b() -> FAILURE, hex"4e487b71", 0x51
