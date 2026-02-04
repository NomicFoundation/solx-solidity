contract C {
  function slice(uint[] calldata arr, uint start, uint end)
      public pure returns (uint[] calldata) {
    return arr[start:end];
  }

  function sliceFromStart(uint[] calldata arr, uint end)
      public pure returns (uint[] calldata) {
    return arr[:end];
  }

  function sliceToEnd(uint[] calldata arr, uint start)
      public pure returns (uint[] calldata) {
    return arr[start:];
  }

  function sliceFull(uint[] calldata arr)
      public pure returns (uint[] calldata) {
    return arr[:];
  }

  function sliceLen(uint[] calldata arr, uint start, uint end)
      public pure returns (uint) {
    uint[] calldata s = arr[start:end];
    return s.length;
  }
}

// ====
// compileViaMlir: true
// ----
// slice(uint256[],uint256,uint256): 0x60, 1, 3, 4, 10, 20, 30, 40 -> 0x20, 2, 20, 30
// sliceFromStart(uint256[],uint256): 0x40, 2, 4, 10, 20, 30, 40 -> 0x20, 2, 10, 20
// sliceToEnd(uint256[],uint256): 0x40, 2, 4, 10, 20, 30, 40 -> 0x20, 2, 30, 40
// sliceFull(uint256[]): 0x20, 4, 10, 20, 30, 40 -> 0x20, 4, 10, 20, 30, 40
// sliceLen(uint256[],uint256,uint256): 0x60, 1, 3, 4, 10, 20, 30, 40 -> 2
