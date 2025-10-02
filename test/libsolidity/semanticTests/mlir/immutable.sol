contract C {
  uint immutable i;
  uint immutable j;
  constructor() { i = 1; j = 2; }
  function f() public returns (uint) {
    uint ret;
    if (i == 2) // > 1 use of an immutable
      ret = i - j;
    else
      ret = i + j;
    return ret;
  }
}

// ====
// compileViaYul: false
// ----
// f() -> 3
