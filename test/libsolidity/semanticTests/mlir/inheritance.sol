contract C1 {
  uint public m;
  function f() public returns (uint) { return g(); }
  function g() internal virtual returns (uint) { return 0x11; }
}

contract C0 is C1 {
  function g() internal override returns (uint) { return 0x01; }
  function h() public returns (uint) { return 0x02; }
}

// ====
// compileViaMlir: true
// ----
// m() -> 0
// f() -> 1
// h() -> 2
