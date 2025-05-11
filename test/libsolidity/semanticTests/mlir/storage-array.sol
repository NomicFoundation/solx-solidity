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

  uint[2][] d2;
  function f_d2() public returns (uint[2][] memory) {
    d2.push()[0] = 0x10;
    d2[0][1] = 0x20;
    d2.push();
    d2[1][0] = 0x30;
    return d2;
  }

  // uint[] m;
  // function f() public returns (uint[] memory) {
  //   m.push() = 1;
  //   // FIXME:
  //   // m.push(2);
  //   return m;
  // }
}

// f() -> 0x20, 2, 1, 2
// ====
// compileViaMlir: true
// ----
// f_s1() -> 1
// g_d1() -> FAILURE, hex"4e487b71", 0x31
// f_d1(uint256): 0 -> FAILURE, hex"4e487b71", 0x32
// f_d1(uint256): 1 -> 2
// g_d1() -> 0
// f_s2() -> 12
// f_d2() -> 0x20, 2, 0x10, 0x20, 0x30, 0
