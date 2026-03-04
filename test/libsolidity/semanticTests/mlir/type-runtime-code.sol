contract A {
  function f() public {}
}

contract C {
  function runtime_code_length() public pure returns (bool) {
    return type(A).runtimeCode.length > 0;
  }

  function runtime_code_compare() public pure returns (bool) {
    bytes memory code = type(A).runtimeCode;
    return keccak256(code) == keccak256(type(A).runtimeCode);
  }
}

// ====
// compileViaMlir: true
// ----
// runtime_code_length() -> true
// runtime_code_compare() -> true
