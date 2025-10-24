library L {
  function f(uint a) external view returns (uint) { return a; }
}

contract C {
  function l() external returns (uint) { return L.f(1); }
}

// ====
// compileViaMlir: true
// ----
// library: L
// l() -> 1
