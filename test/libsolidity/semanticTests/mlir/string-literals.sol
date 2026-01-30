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
