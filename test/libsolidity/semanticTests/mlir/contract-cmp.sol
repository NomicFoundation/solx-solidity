contract C {
  function eq(C a, C b) external pure returns (bool) {
    return a == b;
  }

  function ne(C a, C b) external pure returns (bool) {
    return a != b;
  }
}

// ====
// compileViaMlir: true
// ----
// eq(address,address): 1, 1 -> true
// eq(address,address): 1, 2 -> false
// ne(address,address): 1, 2 -> true
// ne(address,address): 1, 1 -> false
