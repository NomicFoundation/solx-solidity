contract C {
  bytes data;

  constructor() {
    data = "hello";
  }

  function readBytes1(bytes memory a) public returns (bytes1) {
    return a[1];
  }

  function readBytes2(bytes memory a) public returns (bytes2) {
    return bytes2(a[1]);
  }

  function storeBytesMemory(bytes memory a, bytes1 x) public returns (bytes1) {
    a[4] = x;
    return a[4];
  }

  function readStorage() public returns (bytes1) {
    return data[1];
  }

  function writeStorage(bytes1 x) public returns (bytes1) {
    data[4] = x;
    return data[4];
  }

  function storeLiteral(bytes memory a) public returns (bytes1) {
    a[0] = "X";
    return a[0];
  }
}
// ====
// compileViaMlir: true
// ----
// readBytes1(bytes): 32, 5, "hello" -> "e"
// readBytes2(bytes): 32, 5, "hello" -> left(0x6500)
// storeBytesMemory(bytes,bytes1): 64, "X", 5, "hello" -> "X"
// readStorage() -> "e"
// writeStorage(bytes1): "X" -> "X"
// storeLiteral(bytes): 32, 5, "hello" -> "X"
