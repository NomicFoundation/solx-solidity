contract C {
  uint public n;
  uint public sum;

  modifier twice() {
    _;
    _;
  }

  modifier add(uint v) {
    sum += v;
    _;
  }

  function g() internal returns (uint) {
    n += 1;
    return n;
  }

  function f() public twice add(g()) returns (uint) {
    return sum;
  }
}

// ====
// compileViaMlir: true
// ----
// f() -> 3
// n() -> 2
// sum() -> 3
