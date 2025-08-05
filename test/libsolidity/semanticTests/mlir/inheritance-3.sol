contract D {
  uint public m;
  constructor() { m = 11; }
}

contract C is D {
  constructor() { m = 10; }
}

// ====
// compileViaMlir: true
// ----
// m() -> 10
