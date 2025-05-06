contract C {
  uint[5] s1;
  function f_s1() public returns (uint) {
    s1[0] = 1;
    return s1[0];
  }

  uint[] d1;
  function f_d1(uint a) public returns (uint) {
    if (a != 0) {
      d1.push();
      d1.push() = a;
      d1.push(1 + d1[0] + d1[1]);
    }
    return d1[2];
  }

  function g_d1() public returns (uint) {
    d1.pop();
    return 0;
  }

  uint[3][2] s2;
  function f_s2() public returns (uint) {
  unchecked {
    for (uint i = 0; i < 2; ++i)
      for (uint j = 0; j < 3; ++j)
        s2[i][j] = i*10 + j;
    return s2[1][2];
  }
  }

  uint[3][] d2;
  function f_d2() public returns (uint) {
    d2.push()[0] = 1;
    d2[0][1] = 2;
    return d2[0][0] + d2[0][1];
  }
}

// ====
// compileViaMlir: true
// ----
// f_s1() -> 1
// g_d1() -> FAILURE, hex"4e487b71", 0x31
// f_d1(uint256): 0 -> FAILURE, hex"4e487b71", 0x32
// f_d1(uint256): 1 -> 2
// g_d1() -> 0
// f_s2() -> 12
// f_d2() -> 3
