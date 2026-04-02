contract C {
  function f() public pure {
    revert();
  }

  function g() public pure {
    revert("error message");
  }

  function h(bool ok) public pure {
    require(ok, "foobar");
  }
}

// ====
// compileViaMlir: true
// revertStrings: strip
// ----
// f() -> FAILURE
// g() -> FAILURE
// h(bool): false -> FAILURE
