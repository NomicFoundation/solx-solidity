contract C {
  uint[5] s1;
  uint[] d1;

  function f_s1() public returns (uint) {
    s1[0] = 1;
    return s1[0];
  }
  function f_d1() public returns (uint) {
    // TODO:
    // d1.push(1);
    return d1[0];
  }
}

// ====
// compileViaMlir: true
// ----
// f_s1() -> 1
// f_d1() -> FAILURE, hex"4e487b71", 0x32
