interface I {
  function f() external;
}

contract C {
  function interfaceId() public pure returns (bytes4) {
    return type(I).interfaceId;
  }
}

// ====
// compileViaMlir: true
// ----
// interfaceId() -> left(0x26121ff0)
