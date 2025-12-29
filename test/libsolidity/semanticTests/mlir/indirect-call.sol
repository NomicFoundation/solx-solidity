contract C {
  function f(uint a) internal returns (uint) { return a + 10; }
  function g(uint a) internal returns (uint) { return a + 20; }
  function m(uint a, uint b) public returns (uint) {
    function(uint) internal returns (uint) p;
    if (a == 0)
      p = f;
    else if (a == 1)
      p = g;
    return p(b);
  }

  function (uint) internal returns (uint) s0;
  function (uint) internal returns (uint) s1 = f;
  function n(bool a) public returns (uint) {
    if (a)
      return s1(0);
    return s0(0);
  }
}

// ====
// compileViaMlir: true
// ----
// m(uint256,uint256): 0, 1 -> 11
// m(uint256,uint256): 1, 2 -> 22
// m(uint256,uint256): 2, 3 -> FAILURE, hex"4e487b71", 0x51
// n(bool): true -> 10
// n(bool): false -> FAILURE, hex"4e487b71", 0x51
