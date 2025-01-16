contract C {
  function e(bool a, uint b) public returns (uint) {
    require(a, "foobar");
    return 2 - b;
  }

  function f(bool a, uint b) public returns (uint) {
    uint r = 0;
    try this.e(a, b) returns (uint ret) {
      r += ret;
    } catch Panic (uint code) {
      r += code;
    }

    return r;
  }

  function g() public returns (string memory) {
    string memory r;

    try this.e(false, 0) {
    } catch Error (string memory message) {
      r = message;
    }

    return r;
  }
}

// FIXME:
// f(bool,uint256): false, 1
// ====
// compileViaMlir: true
// ----
// f(bool,uint256): true, 1 -> 1
// f(bool,uint256): true, 3 -> 0x11
// g() -> 0x20, 6, "foobar"
