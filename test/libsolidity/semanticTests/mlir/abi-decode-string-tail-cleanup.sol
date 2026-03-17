contract C {
  function tail_after_decode_memory() public returns (uint256) {
    bytes memory enc = abi.encode("a");
    uint256 tail;

    assembly {
      // Force decode allocation to reuse this dirty chunk.
      let p := mload(0x40)
      mstore(add(p, 0x20), not(0))
      mstore(add(p, 0x40), not(0))
      mstore(0x40, p)
    }

    string memory s = abi.decode(enc, (string));

    assembly {
      tail := mload(add(add(s, 0x20), mload(s)))
    }
    return tail;
  }
}

// ====
// compileViaMlir: true
// ----
// tail_after_decode_memory() -> 0
