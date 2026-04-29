contract C {
  event E(address indexed a, uint b);
  event DirtyAddress(address indexed a);
  event DirtyUint(uint8 indexed a);

  function f(address a, uint b) public {
    emit E(a, b);
  }

  function dirtyAddress() public {
    address a;
    assembly {
      a := not(0)
    }
    emit DirtyAddress(a);
  }

  function dirtyUint() public {
    uint8 a;
    assembly {
      a := not(0)
    }
    emit DirtyUint(a);
  }
}

// ====
// compileViaMlir: true
// ----
// f(address,uint256): 1, 2 ->
// ~ emit E(address,uint256): #0x01, 0x02
// dirtyAddress() ->
// ~ emit DirtyAddress(address): #0xffffffffffffffffffffffffffffffffffffffff
// dirtyUint() ->
// ~ emit DirtyUint(uint8): #0xff
