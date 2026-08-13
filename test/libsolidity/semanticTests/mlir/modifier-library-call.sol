library L {
  function check(uint x) internal pure returns (bool) {
    return x < 10;
  }
}

contract C {
  modifier m(uint x) {
    require(L.check(x));
    _;
  }

  function f(uint x) public m(x) returns (uint) {
    return x + 1;
  }
}

// ====
// compileViaMlir: true
// ----
// f(uint256): 5 -> 6
// f(uint256): 10 -> FAILURE
