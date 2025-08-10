contract E {
  uint public m;
  constructor(uint a) { m += a * 100; }
}

contract D is E {
  constructor(uint a, uint b) E(a + 2) { m += a * 10; }
}

contract C is D {
  constructor(uint a) D(a + 1, a + 9) { m += a; }
}

// ====
// compileViaMlir: true
// ----
// constructor(): 1 ->
// m() -> 421
