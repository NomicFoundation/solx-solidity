contract C {
  enum E { A, B, C }

  function add(uint a, uint b) external pure returns (uint) {
    return a + b;
  }

  function loadMemoryByte() public pure returns (bytes1) {
    bytes memory data = new bytes(1);
    assembly {
      mstore(add(data, 0x20), shl(248, 0x58))
    }
    return data[0];
  }

  function storeMemoryByte() public pure returns (bytes1) {
    bytes memory data = new bytes(1);
    data[0] = "X";
    return data[0];
  }

  function loadMemoryBool() public pure returns (bool) {
    bool[] memory values = new bool[](1);
    assembly {
      mstore(add(values, 0x20), 2)
    }
    return values[0];
  }

  function storeMemoryBool(bool value) public pure returns (bool) {
    bool[] memory values = new bool[](1);
    values[0] = value;
    return values[0];
  }

  function loadMemoryUint8() public pure returns (uint8) {
    uint8[] memory values = new uint8[](1);
    assembly {
      mstore(add(values, 0x20), 0x1234)
    }
    return values[0];
  }

  function storeMemoryUint8(uint8 value) public pure returns (uint8) {
    uint8[] memory values = new uint8[](1);
    values[0] = value;
    return values[0];
  }

  function loadMemoryInt8() public pure returns (int8) {
    int8[] memory values = new int8[](1);
    assembly {
      mstore(add(values, 0x20), 0x1ff)
    }
    return values[0];
  }

  function storeMemoryInt8(int8 value) public pure returns (int8) {
    int8[] memory values = new int8[](1);
    values[0] = value;
    return values[0];
  }

  function loadMemoryEnum() public pure returns (uint8) {
    E[] memory values = new E[](1);
    assembly {
      mstore(add(values, 0x20), 2)
    }
    return uint8(values[0]);
  }

  function storeMemoryEnum() public pure returns (uint8) {
    E value;
    E[] memory values = new E[](1);
    assembly {
      value := 2
    }
    values[0] = value;
    return uint8(values[0]);
  }

  function loadMemoryEnumInvalid() public pure returns (uint8) {
    E[] memory values = new E[](1);
    assembly {
      mstore(add(values, 0x20), 3)
    }
    return uint8(values[0]);
  }

  function storeMemoryEnumInvalid() public pure returns (uint8) {
    E value;
    E[] memory values = new E[](1);
    assembly {
      value := 3
    }
    values[0] = value;
    return uint8(values[0]);
  }

  function loadMemoryAddress() public pure returns (address) {
    address[] memory values = new address[](1);
    assembly {
      mstore(add(values, 0x20), or(shl(200, 1), 0x1234123412341234123412341234123412341234))
    }
    return values[0];
  }

  function storeMemoryAddress() public pure returns (address) {
    address value;
    address[] memory values = new address[](1);
    assembly {
      value := or(shl(200, 1), 0x1234123412341234123412341234123412341234)
    }
    values[0] = value;
    return values[0];
  }

  function loadMemoryBytes4() public pure returns (bytes4) {
    bytes4[] memory values = new bytes4[](1);
    assembly {
      mstore(add(values, 0x20), 0x12345678ffffffffffffffffffffffffffffffffffffffffffffffffffffffff)
    }
    return values[0];
  }

  function storeMemoryBytes4() public pure returns (bytes4) {
    bytes4 value;
    bytes4[] memory values = new bytes4[](1);
    assembly {
      value := 0x12345678ffffffffffffffffffffffffffffffffffffffffffffffffffffffff
    }
    values[0] = value;
    return values[0];
  }

  function loadMemoryBytes31() public pure returns (bytes31) {
    bytes31[] memory values = new bytes31[](1);
    assembly {
      mstore(add(values, 0x20), 0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1fff)
    }
    return values[0];
  }

  function storeMemoryBytes31() public pure returns (bytes31) {
    bytes31 value;
    bytes31[] memory values = new bytes31[](1);
    assembly {
      value := 0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1fff
    }
    values[0] = value;
    return values[0];
  }

  function loadMemoryFnPtr() public view returns (uint) {
    function(uint, uint) external pure returns (uint)[] memory values =
      new function(uint, uint) external pure returns (uint)[](1);
    values[0] = this.add;
    assembly {
      mstore(add(values, 0x20), or(mload(add(values, 0x20)), 1))
    }
    return values[0](20, 22);
  }

  function storeMemoryFnPtr() public view returns (bool) {
    function(uint, uint) external pure returns (uint) value = this.add;
    function(uint, uint) external pure returns (uint)[] memory values =
      new function(uint, uint) external pure returns (uint)[](1);
    assembly {
      value.address := or(value.address, shl(160, sub(0, 1)))
      value.selector := or(value.selector, shl(32, sub(0, 1)))
    }
    values[0] = value;
    uint raw;
    assembly {
      raw := mload(add(values, 0x20))
    }
    return uint64(raw) == 0;
  }
}

// ====
// compileViaMlir: true
// ----
// loadMemoryByte() -> "X"
// storeMemoryByte() -> "X"
// loadMemoryBool() -> true
// storeMemoryBool(bool): true -> true
// loadMemoryUint8() -> 0x34
// storeMemoryUint8(uint8): 0x34 -> 0x34
// loadMemoryInt8() -> -1
// storeMemoryInt8(int8): -1 -> -1
// loadMemoryEnum() -> 2
// storeMemoryEnum() -> 2
// loadMemoryEnumInvalid() -> FAILURE, hex"4e487b71", 0x21
// storeMemoryEnumInvalid() -> FAILURE, hex"4e487b71", 0x21
// loadMemoryAddress() -> 0x1234123412341234123412341234123412341234
// storeMemoryAddress() -> 0x1234123412341234123412341234123412341234
// loadMemoryBytes4() -> left(0x12345678)
// storeMemoryBytes4() -> left(0x12345678)
// loadMemoryBytes31() -> left(0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f)
// storeMemoryBytes31() -> left(0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f)
// loadMemoryFnPtr() -> 42
// storeMemoryFnPtr() -> true
