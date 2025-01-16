contract C {
  function f(bool a) public returns (bool) {
    require(a);
    return a;
  }
  function g(bool a) public returns (bool) {
    require(a, "foobar");
    return a;
  }
}

// ====
// compileViaMlir: true
// ----
// f(bool): true -> true
// f(bool): false -> FAILURE
// g(bool): false -> FAILURE, hex"08c379a0", 0x20, 6, "foobar"
