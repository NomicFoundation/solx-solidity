interface I {
  function foo(uint256 x) external;
}

contract C {
  enum E { A, B, C }

  function ei(uint ui, uint8 ui8, int32 si32) public returns (bytes memory) {
    bytes memory a = abi.encode(si32); // Tests the free-ptr update.
    return abi.encode(ui, ui8, si32);
  }

  function di(bytes memory a) public returns (uint, uint8, int32) {
    return abi.decode(a, (uint, uint8, int32));
  }

  function ep(uint ui, uint8 ui8, int32 si32) public returns (bytes memory) {
    return abi.encodePacked(ui, ui8, si32);
  }

  function ep_empty() public returns (bytes memory) {
    return abi.encodePacked();
  }

  function ep_u256(uint256 x) public returns (bytes memory) {
    return abi.encodePacked(x);
  }

  function ep_u256x2(uint256 x, uint256 y) public returns (bytes memory) {
    return abi.encodePacked(x, y);
  }

  function ep_u8_len(uint8 x) public returns (uint) {
    return abi.encodePacked(x).length;
  }

  function ep_u8(uint8 x) public returns (bytes memory) {
    return abi.encodePacked(x);
  }

  function ep_u24_u96_u136(uint24 x, uint96 y, uint136 z) public returns (bytes memory) {
    return abi.encodePacked(x, y, z);
  }

  function ep_i8(int8 x) public returns (bytes memory) {
    return abi.encodePacked(x);
  }

  function ep_i16(int16 x) public returns (bytes memory) {
    return abi.encodePacked(x);
  }

  function ep_enum(E e) public returns (bytes memory) {
    return abi.encodePacked(e);
  }

  function ep_bool_u8(bool x, uint8 y) public returns (bytes memory) {
    return abi.encodePacked(x, y);
  }

  function ep_bytesN(bytes2 a, bytes1 b) public returns (bytes memory) {
    return abi.encodePacked(a, b);
  }

  function ep_bytes_only(bytes memory a) public returns (bytes memory) {
    return abi.encodePacked(a);
  }

  function ep_bytes_calldata(bytes calldata a) public returns (bytes memory) {
    return abi.encodePacked(a);
  }

  function ep_bytes_u8(bytes memory a, uint8 x) public returns (bytes memory) {
    return abi.encodePacked(a, x);
  }

  function ep_bytes_concat(bytes memory a, bytes memory b) public returns (bytes memory) {
    return abi.encodePacked(a, b);
  }

  function ep_u8_array_dynamic_local() public returns (bytes memory) {
    uint8[] memory a = new uint8[](3);
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    return abi.encodePacked(a);
  }

  function ep_u8_array_dynamic(uint8[] memory x) public returns (bytes memory) {
    return abi.encodePacked(x);
  }

  function ep_u8_array_dynamic_calldata(uint8[] calldata x) public returns (bytes memory) {
    return abi.encodePacked(x);
  }

  function ep_u16_static() public returns (bytes memory) {
    uint16[2] memory a;
    a[0] = 0x0102;
    a[1] = 0x0304;
    return abi.encodePacked(a);
  }

  function ep_string(string memory s) public returns (bytes memory) {
    return abi.encodePacked(s);
  }

  function ep_string_calldata(string calldata s) public returns (bytes memory) {
    return abi.encodePacked(s);
  }

  function ep_bytes32(bytes32 b) public returns (bytes memory) {
    return abi.encodePacked(b);
  }

  function ep_u8_cast_from_u16(uint16 x) public returns (bytes memory) {
    return abi.encodePacked(uint8(x));
  }

  function ews_len(bytes4 sel, uint256 x) public returns (uint256) {
    return abi.encodeWithSelector(sel, x).length;
  }

  function ews_bytes_u256(bytes4 sel, uint256 x) public returns (bytes memory) {
    return abi.encodeWithSelector(sel, x);
  }

  function ews_bytes_only(bytes4 sel) public returns (bytes memory) {
    return abi.encodeWithSelector(sel);
  }

  function ews_bytes_bytes(bytes4 sel, bytes memory data) public returns (bytes memory) {
    return abi.encodeWithSelector(sel, data);
  }

  function ews_constant() public returns (bytes memory) {
    return abi.encodeWithSelector(0x12345678);
  }

  function ews_sel_u256(uint256 x) public returns (bytes memory) {
    return abi.encodeWithSelector(I.foo.selector, x);
  }

  function ewsig_literal_u256(uint256 x) public returns (bytes memory) {
    return abi.encodeWithSignature("bar(uint256)", x);
  }

  function ewsig_literal() public returns (bytes memory) {
    return abi.encodeWithSignature("bar()");
  }

  function ewsig_runtime_memory_uint256(string memory sig, uint256 x) public returns (bytes memory) {
    return abi.encodeWithSignature(sig, x);
  }

  function ewsig_runtime_calldata_uint256(string calldata sig, uint256 x) public returns (bytes memory) {
    return abi.encodeWithSignature(sig, x);
  }

  function ewsig_runtime_calldata(string calldata sig) public returns (bytes memory) {
    return abi.encodeWithSignature(sig);
  }

  function foo(uint256) public {}

  function bar(uint256, uint8) public {}

  function ec_non_tuple(uint256 x) public returns (bytes memory) {
    return abi.encodeCall(this.foo, x);
  }

  function ec_tuple(uint256 x, uint8 y) public returns (bytes memory) {
    return abi.encodeCall(this.bar, (x, y));
  }

  function ec_decl(uint256 x) public returns (bytes memory) {
    return abi.encodeCall(I.foo, x);
  }

  function ec_ews(uint256 x) public returns (bool) {
    bytes memory a = abi.encodeCall(I.foo, x);
    bytes memory b = abi.encodeWithSelector(I.foo.selector, x);
    return keccak256(a) == keccak256(b);
  }
}

// ====
// compileViaMlir: true
// ----
// ei(uint256,uint8,int32): 1, 2, -1 -> 32, 96, 1, 2, -1
// di(bytes): 32, 96, 1, 2, -1 -> 1, 2, -1
// ep(uint256,uint8,int32): 1, 2, -1 -> 32, 37, 1, left(0x02ffffffff)
// ep_empty() -> 32, 0
// ep_u256(uint256): 1 -> 32, 32, 1
// ep_u256x2(uint256,uint256): 1, 2 -> 32, 64, 1, 2
// ep_u8_len(uint8): 1 -> 1
// ep_u8(uint8): 1 -> 32, 1, left(0x01)
// ep_u24_u96_u136(uint24,uint96,uint136): 0x010203, 0x0405060708090a0b0c0d0e0f, 0x101112131415161718191a1b1c1d1e1f20 -> 32, 32, 0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20
// ep_i8(int8): -1 -> 32, 1, left(0xff)
// ep_i16(int16): -1 -> 32, 2, left(0xffff)
// ep_enum(uint8): 1 -> 32, 1, left(0x01)
// ep_bool_u8(bool,uint8): true, 2 -> 32, 2, left(0x0102)
// ep_bool_u8(bool,uint8): false, 2 -> 32, 2, left(0x0002)
// ep_bytesN(bytes2,bytes1): "ab", "c" -> 32, 3, left(0x616263)
// ep_bytes_only(bytes): 0x20, 2, "ab" -> 32, 2, left(0x6162)
// ep_bytes_only(bytes): 0x20, 0 -> 32, 0
// ep_bytes_calldata(bytes): 0x20, 2, "ab" -> 32, 2, left(0x6162)
// ep_bytes_u8(bytes,uint8): 0x40, 1, 2, "ab" -> 32, 3, left(0x616201)
// ep_bytes_concat(bytes,bytes): 0x40, 0x80, 2, "ab", 2, "cd" -> 32, 4, left(0x61626364)
// ep_u8_array_dynamic_local() -> 32, 96, 1, 2, 3
// ep_u8_array_dynamic(uint8[]): 0x20, 3, 97, 98, 99 -> 32, 96, 97, 98, 99
// ep_u8_array_dynamic_calldata(uint8[]): 0x20, 3, 97, 98, 99 -> 32, 96, 97, 98, 99
// ep_u16_static() -> 32, 64, 0x0102, 0x0304
// ep_string(string): 0x20, 3, "abc" -> 32, 3, left(0x616263)
// ep_string_calldata(string): 0x20, 3, "abc" -> 32, 3, left(0x616263)
// ep_bytes32(bytes32): left(0x61) -> 32, 32, left(0x61)
// ep_u8_cast_from_u16(uint16): 0xff01 -> 32, 1, left(0x01)
// ews_len(bytes4,uint256): left(0x01020304), 1 -> 36
// ews_bytes_u256(bytes4,uint256): left(0x01020304), 1 -> 0x20, 0x24, left(0x01020304), left(0x00000001)
// ews_bytes_only(bytes4): left(0x01020304) -> 32, 4, left(0x01020304)
// ews_bytes_bytes(bytes4,bytes): left(0x01020304), 0x20, 3, "abc" -> 0x20, 0x64, left(0x01020304), left(0x00000020), left(0x00000020), left(0x00000003)
// ews_constant() -> 0x20, 4, left(0x12345678)
// ews_sel_u256(uint256): 1 -> 0x20, 0x24, left(0x2fbebd38), left(0x00000001)
// ewsig_literal_u256(uint256): 1 -> 0x20, 0x24, left(0x0423a132), left(0x00000001)
// ewsig_literal() -> 0x20, 4, left(0xfebb0f7e)
// ewsig_runtime_memory_uint256(string,uint256): 0x40, 1, 3, "abc" -> 0x20, 0x24, left(0x4e03657a), left(0x00000001)
// ewsig_runtime_calldata_uint256(string,uint256): 0x40, 1, 3, "abc" -> 0x20, 0x24, left(0x4e03657a), left(0x00000001)
// ewsig_runtime_calldata(string): 0x20, 3, "abc" -> 0x20, 4, left(0x4e03657a)
// ec_non_tuple(uint256): 1 -> 0x20, 0x24, left(0x2fbebd38), left(0x00000001)
// ec_tuple(uint256,uint8): 1, 2 -> 0x20, 0x44, left(0x06450a21), left(0x00000001), left(0x00000002)
// ec_decl(uint256): 1 -> 0x20, 0x24, left(0x2fbebd38), left(0x00000001)
// ec_ews(uint256): 1 -> 1
