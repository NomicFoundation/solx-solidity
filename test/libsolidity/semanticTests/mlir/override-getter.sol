contract A {
  function f() external view virtual returns (uint) { return 1; }
}

contract B is A {
  uint public override f = 2;
}

// ====
// compileViaMlir: true
// ----
// f() -> 2
