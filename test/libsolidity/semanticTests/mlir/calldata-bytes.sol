contract C {
  function len(bytes calldata a) external returns (uint) {
    return a.length;
  }
}

// ====
// compileViaMlir: true
// ----
// len(bytes): 0x20, 3, "abc" -> 3
