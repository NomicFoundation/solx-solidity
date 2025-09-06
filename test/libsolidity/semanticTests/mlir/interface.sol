abstract contract A {
  function g() internal returns (uint) { return 42; }
}

interface I {
  function f() external returns (uint);
}

contract C is I, A {
  function f() external virtual override returns (uint) { return g(); }
}

// ====
// compileViaMlir: true
// ----
// f() -> 42
