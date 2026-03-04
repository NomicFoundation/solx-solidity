contract A {
  function f() public {}
}

contract C {
  function creation_code_length() public pure returns (bool) {
    return type(A).creationCode.length > 0;
  }

  function creation_code_compare() public pure returns (bool) {
    bytes memory code = type(A).creationCode;
    return keccak256(code) == keccak256(type(A).creationCode);
  }
}

// ====
// compileViaMlir: true
// ----
// creation_code_length() -> true
// creation_code_compare() -> true
