contract C {
  bytes str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
              "cccccccccccccccccccccccccccccccc"
              "ddddddddddddddddddddddddddddddddz";

  string hexStr = hex"48656C6C6F";

  string[3] strArr = ["A", "B", "C"];

  function i1() public returns (bytes memory) {
    return str;
  }

  function i2() public returns (bytes memory) {
    bytes memory s = "hello";
    return s;
  }

  function i3() public returns (bytes memory) {
    bytes memory s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    return s;
  }

  function i4() public returns (bytes memory) {
    bytes memory s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab";
    return s;
  }

  function i5() public returns (bytes memory) {
    bytes memory s = "";
    return s;
  }

  function i6() public returns (string memory) {
    return hexStr;
  }

  function i7() public returns (string memory) {
    string[3] memory temp = ["X", "Y", "Z"];
    return temp[1];
  }

  function i8() public returns (bytes memory) {
    bytes memory s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                     "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                     "cccccccccccccccccccccccccccccccc"
                     "ddddddddddddddddddddddddddddddddx";
    return s;
  }

  function i9() public returns (bytes memory) {
    str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
          "cccccccccccccccccccccccccccccccc"
          "ddddddddddddddddddddddddddddddddy";
    return str;
  }

  function i10() public returns (string memory, string memory, string memory) {
    return (strArr[0], strArr[1], strArr[2]);
  }

  // 33 bytes: first two-chunk inline case (genStringStore runs 2 iterations,
  // second chunk has 1 data byte + 31 zero-padding bytes).
  function i11() public returns (bytes memory) {
    bytes memory s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" "b";
    return s;
  }

  // 128 bytes: exactly at the inline/CODECOPY threshold (> 128 uses CODECOPY,
  // so 128 bytes is the largest literal that goes through genStringStore).
  function i12() public returns (bytes memory) {
    bytes memory s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                     "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                     "cccccccccccccccccccccccccccccccc"
                     "dddddddddddddddddddddddddddddddd";
    return s;
  }

  // Return a literal directly without assigning to a local variable.
  function i13() public returns (bytes memory) {
    return "hello";
  }

  // string memory literal (not bytes memory).
  function i14() public returns (string memory) {
    string memory s = "hello";
    return s;
  }

  // Two different literals materialised in the same function — each must get
  // its own independent memory allocation with correct content.
  function i15() public returns (bytes memory, bytes memory) {
    bytes memory a = "foo";
    bytes memory b = "bar";
    return (a, b);
  }

  // Hex literal in memory — same bytes as i6's storage value, verifies the
  // inline path produces correct output for hex notation.
  function i16() public returns (bytes memory) {
    bytes memory s = hex"48656c6c6f";
    return s;
  }
}

// ====
// compileViaMlir: true
// ----
// i1() -> 32, 129, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "cccccccccccccccccccccccccccccccc", "dddddddddddddddddddddddddddddddd", "z"
// i2() -> 32, 5, "hello"
// i3() -> 32, 31, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
// i4() -> 32, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
// i5() -> 32, 0
// i6() -> 32, 5, "Hello"
// i7() -> 32, 1, "Y"
// i8() -> 32, 129, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "cccccccccccccccccccccccccccccccc", "dddddddddddddddddddddddddddddddd", "x"
// i9() -> 32, 129, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "cccccccccccccccccccccccccccccccc", "dddddddddddddddddddddddddddddddd", "y"
// i10() -> 96, 160, 224, 1, "A", 1, "B", 1, "C"
// i11() -> 32, 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "b"
// i12() -> 32, 128, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "cccccccccccccccccccccccccccccccc", "dddddddddddddddddddddddddddddddd"
// i13() -> 32, 5, "hello"
// i14() -> 32, 5, "hello"
// i15() -> 64, 128, 3, "foo", 3, "bar"
// i16() -> 32, 5, "Hello"
