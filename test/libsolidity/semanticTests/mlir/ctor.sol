contract C {
  string m;
  constructor(string memory a) {
    m = a;
  }

  function getM() public returns (string memory) {
    return m;
  }
}

// ====
// compileViaMlir: true
// ----
// constructor(): 0x20, 5, "Hello" ->
// getM() -> 0x20, 5, "Hello"
