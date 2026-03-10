contract C {
  uint public counter;

  function for_brk(uint a, uint b) public returns (uint) {
    uint r = 1;
    for (uint i = 0; i < a; ++i) {
      if (i == b)
        break;
      r += r;
    }
    return r;
  }

  function while_cont(uint a) public returns (uint) {
    uint r = 1;
    do {
      r = 2;
    } while (false);

    uint i = 0;
    while (i < a) {
      if (i < 99) {
        r += 2;
        i += 1;
        continue;
      }
      r = 3;
    }
    return r;
  }

  function tern(bool c, uint a, uint b) public pure returns (uint) {
    return c ? a : b;
  }

  function tern_bytes(bool c, bytes4 a, bytes4 b) public pure returns (bytes4) {
    return c ? a : b;
  }

  function tern_cast(bool c, uint8 a, uint256 b) public pure returns (uint256) {
    return c ? a : b;
  }

  function tern_short_circuit(bool c) public returns (uint, uint) {
    counter = 0;
    uint result = c ? ++counter : ++counter;
    return (result, counter);
  }

  function tern_tuple(bool c, uint a, uint b, uint x, uint y) public pure returns (uint, uint) {
    return c ? (a, b) : (x, y);
  }

  function tern_const(bool c) public pure returns (uint) {
    return c ? 1 : 2;
  }
}

// ====
// compileViaMlir: true
// ----
// for_brk(uint256,uint256): 20, 10 -> 1024
// while_cont(uint256): 10 -> 22
// tern(bool,uint256,uint256): true, 10, 20 -> 10
// tern(bool,uint256,uint256): false, 10, 20 -> 20
// tern_bytes(bool,bytes4,bytes4): true, left(0x12345678), left(0xaabbccdd) -> left(0x12345678)
// tern_bytes(bool,bytes4,bytes4): false, left(0x12345678), left(0xaabbccdd) -> left(0xaabbccdd)
// tern_cast(bool,uint8,uint256): true, 42, 1000 -> 42
// tern_cast(bool,uint8,uint256): false, 42, 1000 -> 1000
// tern_short_circuit(bool): true -> 1, 1
// tern_short_circuit(bool): false -> 1, 1
// tern_tuple(bool,uint256,uint256,uint256,uint256): true, 10, 20, 30, 40 -> 10, 20
// tern_tuple(bool,uint256,uint256,uint256,uint256): false, 10, 20, 30, 40 -> 30, 40
// tern_const(bool): true -> 1
// tern_const(bool): false -> 2
