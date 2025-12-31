contract C {
  error E(uint a);
  function f(uint a) public { revert E(a); }
}

// TODO:
// function g(uint a) public { require(a != 0, E(a)); }
// g(uint256): 0 -> FAILURE, hex"002ff067", 0
// ====
// compileViaYul: true
// ----
// f(uint256): 1 -> FAILURE, hex"002ff067", 1
