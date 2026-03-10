contract A {}

contract B is A {
  function b_to_a() external returns (A) {
    return A(this);
  }
}

// ====
// compileViaMlir: true
// ----
// b_to_a() -> 0xc06afe3a8444fc0004668591e8306bfb9968e79e
