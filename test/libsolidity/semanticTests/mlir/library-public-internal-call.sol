library L {
  function pub(uint x) public pure returns (uint) {
    return x * 2;
  }

  function f(uint x) internal pure returns (uint) {
    return pub(x) + 1;
  }
}

contract C {
  function run(uint x) public pure returns (uint) {
    return L.f(x);
  }
}

// ====
// compileViaMlir: true
// ----
// run(uint256): 5 -> 11
