contract C {
  function b1() public returns (bytes1) {
    bytes1 x = "a";
    return x;
  }

  function b4Short() public returns (bytes4) {
    bytes4 x = "ab";
    return x;
  }

  function b4Full() public returns (bytes4) {
    bytes4 x = "wxyz";
    return x;
  }
}

// ====
// compileViaMlir: true
// ----
// b1() -> 0x6100000000000000000000000000000000000000000000000000000000000000
// b4Short() -> left(0x61620000)
// b4Full() -> left(0x7778797a)
