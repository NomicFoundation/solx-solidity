contract C {
  uint public Cm;
  uint public Cn;
  constructor () {
    Cm = 10;
    Cn = 11;
  }

  function f() public returns (uint) { return Cm; }
}

contract D is C {
  uint public Dm = f();
  uint public Dn = Cn;
}

// ====
// compileViaMlir: true
// ----
// Dm() -> 0
// Cm() -> 10
// Cn() -> 11
// Dn() -> 0
