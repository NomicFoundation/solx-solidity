contract C {
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

  function eq3232(bytes32 a, bytes32 b) public returns (bool) {
    return a == b;
  }

  function eq45(bytes4 a, bytes5 b) public returns (bool) {
    return a == b;
  }

  function eq(bytes4 a, bytes4 b) public returns (bool) {
    return a == b;
  }

  function ne(bytes4 a, bytes4 b) public returns (bool) {
    return a != b;
  }

  function lt(bytes2 a, bytes2 b) public returns (bool) {
    return a < b;
  }

  function lte(bytes2 a, bytes2 b) public returns (bool) {
    return a <= b;
  }

  function gt(bytes2 a, bytes2 b) public returns (bool) {
    return a > b;
  }

  function gte(bytes2 a, bytes2 b) public returns (bool) {
    return a >= b;
  }

  function b45(bytes4 a) public returns (bytes5) {
    return bytes5(a);
  }

  function b54(bytes5 a) public returns (bytes4) {
    return bytes4(a);
  }

  function and45(bytes4 a, bytes5 b) public returns (bytes5) {
    return a & b;
  }

  function or45(bytes4 a, bytes5 b) public returns (bytes5) {
    return a | b;
  }

  function xor45(bytes4 a, bytes5 b) public returns (bytes5) {
    return a ^ b;
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
// b45(bytes4): left(0x01020304) -> left(0x0102030400)
// b54(bytes5): left(0x0102030405) -> left(0x01020304)
// and45(bytes4,bytes5): left(0x12345678), left(0xAAAABBBBCC) -> left(0x0220123800)
// or45(bytes4,bytes5): left(0x12345678), left(0xAAAABBBBCC) -> left(0xBABEFFFBCC)
// xor45(bytes4,bytes5): left(0x12345678), left(0xAAAABBBBCC) -> left(0xB89EEDC3CC)
// eq3232(bytes32,bytes32): "abcdefghijklmnopqrstuvwxyzabcdef", "abcdefghijklmnopqrstuvwxyzabcdef" -> true
// eq45(bytes4,bytes5): left(0x01020304), left(0x0102030400) -> true
// eq45(bytes4,bytes5): left(0x01020304), left(0x0102030405) -> false
// eq(bytes4,bytes4): left(0x01020304), left(0x01020304) -> true
// eq(bytes4,bytes4): left(0x01020304), left(0x01020305) -> false
// ne(bytes4,bytes4): left(0x01020304), left(0x01020305) -> true
// lt(bytes2,bytes2): "aa", "ab" -> true
// lte(bytes2,bytes2): "ab", "ab" -> true
// gt(bytes2,bytes2): "ab", "aa" -> true
// gte(bytes2,bytes2): "ab", "ab" -> true
