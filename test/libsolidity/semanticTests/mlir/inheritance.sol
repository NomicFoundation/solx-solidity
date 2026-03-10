contract C1 {
  uint public m;
  function f() public returns (uint) { return g(); }
  function g() internal virtual returns (uint) { return 0x11; }
  function e() public virtual returns (uint) { return 1; }
  function j() public virtual returns (uint) { return 10; }
}

contract C0 is C1 {
  function g() internal override returns (uint) { return 0x01; }
  function h() public returns (uint) { return 0x02; }
  function e() public override returns (uint) { return 2; }
  function j() public override returns (uint) { return super.j() + 5; }
}

// ====
// compileViaMlir: true
// ----
// m() -> 0
// f() -> 1
// h() -> 2
// e() -> 2
// j() -> 15
