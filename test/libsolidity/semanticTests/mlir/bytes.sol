contract C {
  bytes str;
  bytes str2;

  function b(bytes1 a) public returns (bytes1) {
    return a;
  }
  function b2i(bytes1 a) public returns (uint8) {
    return uint8(a);
  }
  function i2b(uint8 a) public returns (bytes1) {
    return bytes1(a);
  }
  function bm(bytes memory a) public returns (bytes memory) {
    return a;
  }
  function ld(bytes memory a) public returns (bytes1) {
    return a[0];
  }
  function st(bytes memory a) public returns (bytes memory) {
    uint8 b = 0x61;
    a[0] = bytes1(b);
    return a;
  }
  function l(bytes memory a) public returns (uint) {
    return a.length;
  }
  function ls(bytes memory a) public returns (uint) {
    str2 = a;
    return str2.length;
  }
  function push_to_empty() public returns (bytes memory) {
    str.push(0x78);
    return str;
  }
  function push(bytes memory a) public returns (bytes memory) {
    str = a;
    str.push(0x78);
    return str;
  }
  function push2(bytes memory a, bytes1 b) public returns (bytes memory) {
    str = a;
    str.push(b);
    return str;
  }
  function pop(bytes memory a) public returns (bytes memory) {
    str = a;
    str.pop();
    return str;
  }
  function pop2(bytes memory a) public returns (bytes memory) {
    str = a;
    str.pop();
    return str;
  }
}

// ====
// compileViaMlir: true
// ----
// b(bytes1): "a" -> 0x6100000000000000000000000000000000000000000000000000000000000000
// b2i(bytes1): "a" -> 0x61
// i2b(uint8): 1 -> 0x0100000000000000000000000000000000000000000000000000000000000000
// bm(bytes): 32, 3, hex"AB33BB" -> 32, 3, left(0xAB33BB)
// ld(bytes): 32, 5, "hello" -> 0x6800000000000000000000000000000000000000000000000000000000000000
// st(bytes): 32, 5, "hello" -> 32, 5, 0x61656c6c6f000000000000000000000000000000000000000000000000000000
// l(bytes): 32, 5, "hello" -> 5
// ls(bytes): 32, 0, "" -> 0
// ls(bytes): 32, 5, "hello" -> 5
// ls(bytes): 32, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" -> 32
// push_to_empty() -> 0x20, 1, "x"
// push(bytes): 32, 5, "hello" -> 0x20, 6, "hellox"
// push(bytes): 32, 31, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" -> 0x20, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaax"
// push(bytes): 32, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" -> 0x20, 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "x"
// push2(bytes, bytes1): 0x40, "!", 5, "hello" -> 0x20, 6, "hello!"
// pop2(bytes): 32, 5, "hello" -> 0x20, 4, "hell"
// pop2(bytes): 32, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" -> 0x20, 31, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
// pop2(bytes): 32, 31, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaabc" -> 0x20, 30, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
// pop2(bytes): 32, 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaabcd", "e" -> 0x20, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaabcd"
