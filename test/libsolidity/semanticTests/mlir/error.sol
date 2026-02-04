contract C {
  error E(uint a);
  error E2(uint a, uint b);
  error Empty();

  function f(uint a) public { revert E(a); }
  function g(uint a, uint b) public { revert E2(a, b); }
  function h() public { revert Empty(); }
}

// ====
// compileViaMlir: true
// ----
// f(uint256): 1 -> FAILURE, hex"002ff067", 1
// f(uint256): 42 -> FAILURE, hex"002ff067", 42
// g(uint256,uint256): 1, 2 -> FAILURE, hex"058c8f48", 1, 2
// h() -> FAILURE, hex"3db2a12a"
