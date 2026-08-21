contract C {
  function counter() public pure returns (uint8) {
    uint8 last = 0;
    for (uint8 i = 0; i < 5; ++i) {
      last = i + 1;
    }
    return last;
  }

  function inclusive(uint8 limit) public pure returns (uint8) {
    uint8 last = 0;
    for (uint8 i = 0; i <= limit; ++i) {
      last = i;
    }
    return last;
  }

  function compound() public pure returns (uint8) {
    uint8 last = 0;
    for (uint8 i = 250; i < 255; i += 10) {
      last = i;
    }
    return last;
  }

  function decrement() public pure returns (uint8) {
    uint8 last = 0;
    for (uint8 i = 0; i < 5; --i) {
      last = i;
    }
    return last;
  }

  function widened() public pure returns (uint8) {
    uint256 limit = 300;
    uint8 last = 0;
    for (uint8 i = 0; i < limit; ++i) {
      last = i;
    }
    return last;
  }

  function bodyWrite() public pure returns (uint8) {
    uint8 last = 0;
    for (uint8 i = 0; i < 5; ++i) {
      last = i;
      i = 255;
    }
    return last;
  }

  // An unchecked block still wraps the step.
  function uncheckedStep() public pure returns (uint8) {
    unchecked {
      uint8 last = 0;
      for (uint8 i = 254; i != 3; ++i) {
        last = i;
      }
      return last;
    }
  }
}

// ====
// compileViaMlir: true
// ----
// counter() -> 5
// inclusive(uint8): 255 -> FAILURE, hex"4e487b71", 0x11
// compound() -> FAILURE, hex"4e487b71", 0x11
// decrement() -> FAILURE, hex"4e487b71", 0x11
// widened() -> FAILURE, hex"4e487b71", 0x11
// bodyWrite() -> FAILURE, hex"4e487b71", 0x11
// uncheckedStep() -> 2
