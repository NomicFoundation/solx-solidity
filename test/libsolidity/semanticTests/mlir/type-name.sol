abstract contract A {
  function fa() public virtual;
}

interface I {
  function fi() external;
}

contract C {}

contract Test {
  function c() public pure returns (string memory) {
    return type(C).name;
  }

  function a() public pure returns (string memory) {
    return type(A).name;
  }

  function i() public pure returns (string memory) {
    return type(I).name;
  }

  function names_match() public pure returns (bool) {
    return
        keccak256(bytes(type(C).name)) == keccak256(bytes("C")) &&
        keccak256(bytes(type(A).name)) == keccak256(bytes("A")) &&
        keccak256(bytes(type(I).name)) == keccak256(bytes("I"));
  }
}

// ====
// compileViaMlir: true
// ----
// c() -> 0x20, 1, "C"
// a() -> 0x20, 1, "A"
// i() -> 0x20, 1, "I"
// names_match() -> true
