enum E { A, B, C }

contract C {
  uint256 transient a;
  uint8 transient b;
  uint16 transient c;
  bool transient d;
  bytes4 transient e;
  bytes32 transient f;
  address transient g;
  int256 transient h;
  int8 transient i;
  E transient j;

  function test_uint() public returns (uint256, uint8, uint16) {
    a = 1;
    b = 2;
    c = 3;
    return (a, b, c);
  }

  function test_bool() public returns (bool) {
    d = true;
    return d;
  }

  function test_bytesN(bytes4 b4, bytes32 b32) public returns (bytes4, bytes32) {
    e = b4;
    f = b32;
    return (e, f);
  }

  function test_address(address addr) public returns (address) {
    g = addr;
    return g;
  }

  function test_int() public returns (int256, int8) {
    h = -100;
    i = -10;
    return (h, i);
  }

  function test_enum(E val) public returns (E) {
    j = val;
    return j;
  }
}

// ====
// compileViaMlir: true
// ----
// test_uint() -> 1, 2, 3
// test_bool() -> true
// test_bytesN(bytes4, bytes32): 0xdeadbeef00000000000000000000000000000000000000000000000000000000, 0x1234 -> 0xdeadbeef00000000000000000000000000000000000000000000000000000000, 0x1234
// test_address(address): 0x1234567890123456789012345678901234567890 -> 0x1234567890123456789012345678901234567890
// test_int() -> -100, -10
// test_enum(uint8): 1 -> 1
