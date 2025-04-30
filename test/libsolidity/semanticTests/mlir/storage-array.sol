contract C {
  uint[5] s1;
  uint[] d1;

  function f_s1() public returns (uint) {
    s1[0] = 1;
    return s1[0];
  }
  function f_d1(uint a) public returns (uint) {
    if (a != 0) {
      d1.push();
      d1.push() = a;
      d1.push(1 + d1[0] + d1[1]);
    }
    return d1[2];
  }
}

// ====
// compileViaMlir: true
// ----
// f_s1() -> 1
// f_d1(uint256): 0 -> FAILURE, hex"4e487b71", 0x32
// f_d1(uint256): 1 -> 2
