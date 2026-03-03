contract C {
  function shl_asm() public pure returns (uint) {
    uint r;
    assembly {
      r := shl(1, 4)
    }
    return r;
  }

  function shr_asm() public pure returns (uint) {
    uint r;
    assembly {
      r := shr(1, 4)
    }
    return r;
  }

  function sar_asm() public pure returns (int) {
    int r;
    assembly {
      r := sar(1, sub(0, 4))
    }
    return r;
  }
}

// ====
// compileViaMlir: true
// ----
// shl_asm() -> 8
// shr_asm() -> 2
// sar_asm() -> -2
