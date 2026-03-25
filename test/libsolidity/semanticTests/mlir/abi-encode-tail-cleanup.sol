contract C {
  function tail_after_encode_bytes_memory(bytes memory x) public pure returns (uint256) {
    bytes memory out;

    assembly {
      // Force abi.encode allocation to reuse dirty memory.
      let p := mload(0x40)
      mstore(add(p, 0x60), not(0))
      mstore(add(p, 0x80), not(0))
      mstore(0x40, p)
    }

    out = abi.encode(x);
    uint256 tail;
    assembly {
      let payload := add(out, 0x20)
      let len := mload(add(payload, 0x20))
      tail := mload(add(add(payload, 0x40), len))
    }
    return tail;
  }

  function tail_after_encode_bytes_calldata(bytes calldata x) external pure returns (uint256) {
    bytes memory out;

    assembly {
      // Force abi.encode allocation to reuse dirty memory.
      let p := mload(0x40)
      mstore(add(p, 0x60), not(0))
      mstore(add(p, 0x80), not(0))
      mstore(0x40, p)
    }

    out = abi.encode(x);
    uint256 tail;
    assembly {
      let payload := add(out, 0x20)
      let len := mload(add(payload, 0x20))
      tail := mload(add(add(payload, 0x40), len))
    }
    return tail;
  }

  function tail_after_encode_string_memory(string memory x) public pure returns (uint256) {
    bytes memory out;

    assembly {
      // Force abi.encode allocation to reuse dirty memory.
      let p := mload(0x40)
      mstore(add(p, 0x60), not(0))
      mstore(add(p, 0x80), not(0))
      mstore(0x40, p)
    }

    out = abi.encode(x);
    uint256 tail;
    assembly {
      let payload := add(out, 0x20)
      let len := mload(add(payload, 0x20))
      tail := mload(add(add(payload, 0x40), len))
    }
    return tail;
  }

  function tail_after_encode_string_calldata(string calldata x) external pure returns (uint256) {
    bytes memory out;

    assembly {
      // Force abi.encode allocation to reuse dirty memory.
      let p := mload(0x40)
      mstore(add(p, 0x60), not(0))
      mstore(add(p, 0x80), not(0))
      mstore(0x40, p)
    }

    out = abi.encode(x);
    uint256 tail;
    assembly {
      let payload := add(out, 0x20)
      let len := mload(add(payload, 0x20))
      tail := mload(add(add(payload, 0x40), len))
    }
    return tail;
  }

  function tail_after_packed_bytes_memory(bytes memory x) public pure returns (uint256) {
    bytes memory out;

    assembly {
      // Force abi.encodePacked allocation to reuse dirty memory.
      let p := mload(0x40)
      mstore(add(p, 0x20), not(0))
      mstore(add(p, 0x40), not(0))
      mstore(0x40, p)
    }

    out = abi.encodePacked(x);
    uint256 tail;
    assembly {
      tail := mload(add(add(out, 0x20), mload(out)))
    }
    return tail;
  }

  function tail_after_packed_bytes_calldata(bytes calldata x) external pure returns (uint256) {
    bytes memory out;

    assembly {
      // Force abi.encodePacked allocation to reuse dirty memory.
      let p := mload(0x40)
      mstore(add(p, 0x20), not(0))
      mstore(add(p, 0x40), not(0))
      mstore(0x40, p)
    }

    out = abi.encodePacked(x);
    uint256 tail;
    assembly {
      tail := mload(add(add(out, 0x20), mload(out)))
    }
    return tail;
  }

  function tail_after_packed_string_memory(string memory x) public pure returns (uint256) {
    bytes memory out;

    assembly {
      // Force abi.encodePacked allocation to reuse dirty memory.
      let p := mload(0x40)
      mstore(add(p, 0x20), not(0))
      mstore(add(p, 0x40), not(0))
      mstore(0x40, p)
    }

    out = abi.encodePacked(x);
    uint256 tail;
    assembly {
      tail := mload(add(add(out, 0x20), mload(out)))
    }
    return tail;
  }

  function tail_after_packed_string_calldata(string calldata x) external pure returns (uint256) {
    bytes memory out;

    assembly {
      // Force abi.encodePacked allocation to reuse dirty memory.
      let p := mload(0x40)
      mstore(add(p, 0x20), not(0))
      mstore(add(p, 0x40), not(0))
      mstore(0x40, p)
    }

    out = abi.encodePacked(x);
    uint256 tail;
    assembly {
      tail := mload(add(add(out, 0x20), mload(out)))
    }
    return tail;
  }
}

// ====
// compileViaMlir: true
// ----
// tail_after_encode_bytes_memory(bytes): 0x20, 3, "abc" -> 0
// tail_after_encode_bytes_calldata(bytes): 0x20, 3, "abc" -> 0
// tail_after_encode_string_memory(string): 0x20, 3, "abc" -> 0
// tail_after_encode_string_calldata(string): 0x20, 3, "abc" -> 0
// tail_after_packed_bytes_memory(bytes): 0x20, 3, "abc" -> 0
// tail_after_packed_bytes_calldata(bytes): 0x20, 3, "abc" -> 0
// tail_after_packed_string_memory(string): 0x20, 3, "abc" -> 0
// tail_after_packed_string_calldata(string): 0x20, 3, "abc" -> 0
