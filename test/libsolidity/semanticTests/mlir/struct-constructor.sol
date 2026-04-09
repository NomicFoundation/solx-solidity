contract C {
  struct S {
    uint256 a;
    uint256 b;
  }

  function positional() public pure returns (uint256, uint256) {
    S memory s = S(7, 11);
    return (s.a, s.b);
  }

  function named() public pure returns (uint256, uint256) {
    S memory s = S({b: 13, a: 17});
    return (s.a, s.b);
  }
}

// ====
// compileViaMlir: true
// ----
// positional() -> 7, 11
// named() -> 17, 13
