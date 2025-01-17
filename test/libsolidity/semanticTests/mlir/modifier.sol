contract C {
  uint m;
  modifier m0(uint a) {
    require(a == 1);
    _;
    a += 1;
    _;
  }

  modifier m1(uint a) {
    require(a == 1);
    m += 10;
    _;
  }

  function f(uint a) m0(a) m1(a) public returns (uint) {
    require(a == 1);
    return m;
  }
}

// ====
// compileViaMlir: true
// ----
// f(uint256): 1 -> 20
