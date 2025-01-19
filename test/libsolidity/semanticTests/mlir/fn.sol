contract C {
  uint s;
  function f() public { s = 42; }
  function g() public returns (uint) { return s; }
  function i0(uint a) public returns (uint) { return a; }
  function i1(uint a, uint b) public returns (uint) { return a + b; }
  function b(bool a) public returns (bool) { return a; }
  function i8(uint8 a) public returns (uint8) { return a; }
}

// ====
// compileViaMlir: true
// ----
// f()
// g() -> 42
// i0(uint256): 1 -> 1
// i1(uint256,uint256): 2, 1 -> 3
// b(bool): 0 -> false
// i8(uint8): 255 -> 255
// i8(uint8): 256 -> FAILURE
