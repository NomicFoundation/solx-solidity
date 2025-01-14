contract C {
  uint s;
  function f() public { s = 42; }
  function g() public returns (uint) { return s; }
  function h(uint a) public returns (uint) { return a; }
  function i(uint a, uint b) public returns (uint) { return a + b; }
}

// ====
// compileViaMlir: true
// ----
// f()
// g() -> 42
// h(uint256): 1 -> 1
// i(uint256,uint256): 2, 1 -> 3
