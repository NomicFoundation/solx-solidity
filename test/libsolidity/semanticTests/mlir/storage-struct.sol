contract C {
  // Packed struct (all fields in one slot)
  struct Packed {
    uint8 a;
    uint8 b;
    uint8 c;
    uint8 d;
  }
  Packed packed;
  function set_packed(uint8 a, uint8 b, uint8 c, uint8 d) public {
    packed.a = a;
    packed.b = b;
    packed.c = c;
    packed.d = d;
  }
  function get_packed() public view returns (uint8, uint8, uint8, uint8) {
    return (packed.a, packed.b, packed.c, packed.d);
  }

  // Non-packed struct (fields span multiple slots)
  struct NonPacked {
    uint256 x;
    uint256 y;
  }
  NonPacked non_packed;
  function set_non_packed(uint256 x, uint256 y) public {
    non_packed.x = x;
    non_packed.y = y;
  }
  function get_non_packed() public view returns (uint256, uint256) {
    return (non_packed.x, non_packed.y);
  }

  // Mixed struct (packed and non-packed fields)
  struct Mixed {
    uint8 a;
    uint256 b;
    uint8 c;
  }
  Mixed mixed;
  function set_mixed(uint8 a, uint256 b, uint8 c) public {
    mixed.a = a;
    mixed.b = b;
    mixed.c = c;
  }
  function get_mixed() public view returns (uint8, uint256, uint8) {
    return (mixed.a, mixed.b, mixed.c);
  }

  // Nested struct
  struct Inner {
    uint8 x;
    uint8 y;
  }
  struct Outer {
    Inner inner;
    uint8 z;
  }
  Outer nested;
  function set_nested(uint8 x, uint8 y, uint8 z) public {
    nested.inner.x = x;
    nested.inner.y = y;
    nested.z = z;
  }
  function get_nested() public view returns (uint8, uint8, uint8) {
    return (nested.inner.x, nested.inner.y, nested.z);
  }

  // Struct with value + reference types
  struct MixedRef {
    uint8 x;
    uint256[] arr;
    uint8 y;
  }
  MixedRef mixed_ref;
  function set_mixed_ref(uint8 x, uint8 y) public {
    mixed_ref.x = x;
    mixed_ref.y = y;
  }
  function push_mixed_ref(uint256 val) public {
    mixed_ref.arr.push(val);
  }
  function get_mixed_ref() public view returns (uint8, uint256, uint8) {
    return (mixed_ref.x, mixed_ref.arr.length, mixed_ref.y);
  }
  function get_mixed_ref_arr(uint256 idx) public view returns (uint256) {
    return mixed_ref.arr[idx];
  }
}

// ====
// compileViaMlir: true
// ----
// set_packed(uint8,uint8,uint8,uint8): 0x11, 0x22, 0x33, 0x44 ->
// get_packed() -> 0x11, 0x22, 0x33, 0x44
// set_non_packed(uint256,uint256): 0xaa, 0xbb ->
// get_non_packed() -> 0xaa, 0xbb
// set_mixed(uint8,uint256,uint8): 0x11, 0xaa, 0x22 ->
// get_mixed() -> 0x11, 0xaa, 0x22
// set_nested(uint8,uint8,uint8): 0x11, 0x22, 0x33 ->
// get_nested() -> 0x11, 0x22, 0x33
// set_mixed_ref(uint8,uint8): 0xaa, 0xbb ->
// get_mixed_ref() -> 0xaa, 0, 0xbb
// push_mixed_ref(uint256): 0x111 ->
// push_mixed_ref(uint256): 0x222 ->
// get_mixed_ref() -> 0xaa, 2, 0xbb
// get_mixed_ref_arr(uint256): 0 -> 0x111
// get_mixed_ref_arr(uint256): 1 -> 0x222
