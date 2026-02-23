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
  function bc(bytes calldata a) public returns (bytes memory) {
    return a;
  }
  function bsm(bytes calldata a) public returns (bytes memory) {
    str = a;
    return str;
  }
  function ss() public returns (bytes memory) {
    str2 = str;
    return str2;
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
  function pop_empty() public returns (uint) {
    str.pop();
    return str.length;
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
  function push2(bytes memory a) public returns (bytes memory) {
    str = a;
    str.push() = 0x79;
    return str;
  }
  function push3(bytes memory a, bytes1 b) public returns (bytes memory) {
    str = a;
    str.push(b);
    return str;
  }
  function idx_read(bytes memory a, uint idx) public returns (bytes1) {
    str = a;
    return str[idx];
  }
  function idx_write(bytes memory a, uint idx, bytes1 c) public returns (bytes memory) {
    str = a;
    str[idx] = c;
    return str;
  }
  function pop(bytes memory a) public returns (bytes memory) {
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
// bc(bytes): 32, 3, hex"AB33BB" -> 32, 3, left(0xAB33BB)
// ld(bytes): 32, 5, "hello" -> 0x6800000000000000000000000000000000000000000000000000000000000000
// st(bytes): 32, 5, "hello" -> 32, 5, 0x61656c6c6f000000000000000000000000000000000000000000000000000000
// l(bytes): 32, 5, "hello" -> 5
// pop_empty() -> FAILURE, hex"4e487b71", 0x31
// ls(bytes): 32, 0, "" -> 0
// ls(bytes): 32, 5, "hello" -> 5
// ls(bytes): 32, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" -> 32
// ls(bytes): 32, 31, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" -> 31
// push_to_empty() -> 0x20, 1, "x"
// push(bytes): 32, 5, "hello" -> 0x20, 6, "hellox"
// ss() -> 0x20, 6, "hellox"
// push(bytes): 32, 31, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" -> 0x20, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaax"
// ss() -> 0x20, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaax"
// push(bytes): 32, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" -> 0x20, 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "x"
// ss() -> 0x20, 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "x"
// push(bytes): 32, 63, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" -> 0x20, 64, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaax"
// bsm(bytes): 32, 0, "" -> 0x20, 0
// bsm(bytes): 32, 5, "hello" -> 0x20, 5, "hello"
// bsm(bytes): 32, 31, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" -> 0x20, 31, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
// bsm(bytes): 32, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" -> 0x20, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
// push2(bytes): 32, 5, "hello" -> 0x20, 6, "helloy"
// push2(bytes): 32, 31, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" -> 0x20, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaay"
// push2(bytes): 32, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" -> 0x20, 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "y"
// push3(bytes,bytes1): 64, "!", 5, "hello" -> 0x20, 6, "hello!"
// push3(bytes,bytes1): 64, "!", 31, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" -> 0x20, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!"
// push3(bytes,bytes1): 64, "!", 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" -> 0x20, 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "!"
// pop(bytes): 32, 5, "hello" -> 0x20, 4, "hell"
// pop(bytes): 32, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" -> 0x20, 31, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
// pop(bytes): 32, 31, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaabc" -> 0x20, 30, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
// pop(bytes): 32, 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaabcd", "e" -> 0x20, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaabcd"
// pop(bytes): 32, 1, "a" -> 0x20, 0
// idx_read(bytes,uint256): 64, 2, 1, "A" -> FAILURE, hex"4e487b71", 0x32
// idx_read(bytes,uint256): 64, 0, 5, "hello" -> "h"
// idx_read(bytes,uint256): 64, 1, 5, "hello" -> "e"
// idx_read(bytes,uint256): 64, 4, 5, "hello" -> "o"
// idx_read(bytes,uint256): 64, 10, 32, "aaaaaaaaaaxaaaaaaaaaaaaaaaaaaaab" -> "x"
// idx_read(bytes,uint256): 64, 32, 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" -> FAILURE, hex"4e487b71", 0x32
// idx_read(bytes,uint256): 64, 31, 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaX", "Y" -> "X"
// idx_read(bytes,uint256): 64, 32, 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaX", "Y" -> "Y"
// idx_write(bytes,uint256, bytes1): 96, 0, "H", 5, "hello" -> 0x20, 5, "Hello"
// idx_write(bytes,uint256, bytes1): 96, 10, "x", 32, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" -> 0x20, 32, "aaaaaaaaaaxaaaaaaaaaaaaaaaaaaaab"
// idx_write(bytes,uint256, bytes1): 96, 4, "X", 5, "hello" -> 0x20, 5, "hellX"
// idx_write(bytes,uint256, bytes1): 96, 31, "Z", 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaX", "Y" -> 0x20, 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaZ", "Y"
// idx_write(bytes,uint256, bytes1): 96, 32, "Z", 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaX", "Y" -> 0x20, 33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaX", "Z"
// idx_write(bytes,uint256, bytes1): 96, 5, "X", 5, "hello" -> FAILURE, hex"4e487b71", 0x32
