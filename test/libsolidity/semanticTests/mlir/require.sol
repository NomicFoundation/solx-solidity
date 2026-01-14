contract C {
  error E(uint b);

  function f(bool a) public returns (bool) {
    require(a);
    return a;
  }
  function g(bool a) public returns (bool) {
    require(a, "foobar");
    return a;
  }
  function h(bool a) public returns (bool) {
    require(a, E(2));
    return a;
  }
}

// ====
// compileViaMlir: true
// ----
// f(bool): true -> true
// f(bool): false -> FAILURE
// g(bool): false -> FAILURE, hex"08c379a0", 0x20, 6, "foobar"
// h(bool): false -> FAILURE, hex"002ff067", 2
