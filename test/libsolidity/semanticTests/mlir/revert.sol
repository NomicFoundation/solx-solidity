contract C {
  error E(uint b);

  function f() public {
    revert();
  }
  function g() public {
    revert("error message");
  }
  function h() public{
    revert E(2);
  }
}

// ====
// compileViaMlir: true
// ----
// f() -> FAILURE
// g() -> FAILURE, hex"08c379a0", 0x20, 13, "error message"
// h() -> FAILURE, hex"002ff067", 2
