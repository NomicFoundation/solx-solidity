contract C {
  function ei(uint ui, uint8 ui8, int32 si32) public returns (bytes memory) {
    bytes memory a = abi.encode(si32); // Tests the free-ptr update.
    return abi.encode(ui, ui8, si32);
  }

  function di(bytes memory a) public returns (uint, uint8, int32) {
    return abi.decode(a, (uint, uint8, int32));
  }
}

// ====
// compileViaMlir: true
// ----
// ei(uint256,uint8,int32): 1, 2, -1 -> 32, 96, 1, 2, -1
// di(bytes): 32, 96, 1, 2, -1 -> 1, 2, -1
