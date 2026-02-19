contract C {
  function len() external returns (uint) {
    return msg.data.length;
  }
}

// ====
// compileViaMlir: true
// ----
// len() -> 4
