contract C {
  modifier m(bool c) {
    if (c) _;
  }

  function f(bool c) public m(c) returns (uint[] memory a, string memory s) {
    a = new uint[](2);
    a[0] = 1;
    a[1] = 2;
    s = "abc";
  }
}

// ====
// compileViaMlir: true
// ----
// f(bool): true -> 0x40, 0xa0, 2, 1, 2, 3, "abc"
// f(bool): false -> 0x40, 0x60, 0, 0
