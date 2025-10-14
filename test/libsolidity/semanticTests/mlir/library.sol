library L {
  function f(uint a) internal returns (uint) { return a + 1; }
  function g(uint a) external returns (uint) { return f(a); }
}

contract C {
  function m() external returns (uint) { return L.g(1); }
}

// ====
// compileViaMlir: true
// ----
// library: L
// m() -> 2
