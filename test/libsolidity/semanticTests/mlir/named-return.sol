contract Test {
  uint8[33] a;

  function f_basic() public returns (uint a) { a = 42; }

  function f_default() public returns (uint a) {}

  function f_multi() public returns (uint a, uint b) { a = 1; b = 2; }

  function f_partial() public returns (uint a, uint b) { a = 7; }

  function f_bool() public returns (bool ok) { ok = true; }

  function f_cond(bool flag) public returns (uint result) {
    if (flag) { result = 10; } else { result = 20; }
  }

  function f_loop(uint n) public returns (uint sum) {
    for (uint i = 0; i < n; i++) { sum += i; }
  }

  function f_call() public returns (uint a) { a = helper(); }
  function helper() internal returns (uint) { return 5; }

  function f_explicit() public returns (uint) { return 99; }

  function f_noname(uint b) public returns (uint a) { return b; }

  function f_int_default() public returns (int256 a) {}
  function f_int_neg() public returns (int256 a) { a = -5; }

  function f_addr_default() public returns (address a) {}
  function f_addr_set() public returns (address a) { a = address(1); }

  function f_bytes1_default() public returns (bytes1 a) {}
  function f_bytes1_set() public returns (bytes1 a) { a = 0x41; }

  function f_bytes4_default() public returns (bytes4 a) {}

  function f_bytes32_default() public returns (bytes32 a) {}

  function f_early(bool flag) public returns (uint r) {
    r = 1;
    if (flag) return r;
    r = 2;
  }

  function f_early_multi(bool flag) public returns (uint a, uint b) {
    a = 10; b = 20;
    if (flag) return (a, b);
    a = 1; b = 2;
  }

  function f_bytes_default() public returns (bytes memory a) {}
  function f_bytes_set() public returns (bytes memory a) { a = "hi"; }

  function f_str_default() public returns (string memory a) {}
  function f_str_set() public returns (string memory a) { a = "hello"; }

  function f_arr_default() public returns (uint[] memory a) {}
  function f_arr_set() public returns (uint[] memory a) {
    a = new uint[](2);
    a[0] = 1; a[1] = 2;
  }

  function f_fixed_arr_default() public returns (uint[2] memory a) {}
  function f_fixed_arr_set() public returns (uint[2] memory a) {
    a[0] = 3; a[1] = 4;
  }

  function f_int_u() public returns (int256) { return -5; }
  function f_addr_u() public returns (address) { return address(1); }
  function f_bytes1_u() public returns (bytes1) { return 0x41; }

  function f_bytes_u() public returns (bytes memory) { return "hi"; }
  function f_str_u() public returns (string memory) { return "hello"; }
  function f_arr_u() public returns (uint[] memory) {
    uint[] memory a = new uint[](2);
    a[0] = 5; a[1] = 6;
    return a;
  }

  function f_multi_u() public returns (uint, bool) { return (42, true); }

  // Mixed named and unnamed return parameters.
  // a is named (assigned, falls off end), b is unnamed (explicit return value).
  function f_mixed(uint x) public returns (uint a, uint) {
    a = x + 2;
    return (a, x + 1);
  }

  // Mixed with default: a is named (zero default), b is unnamed (explicit).
  function f_mixed_default() public returns (uint a, uint) {
    return (a, 99);
  }

  function f_arr() public returns (uint8, uint8, uint8) {
    a[0] = 2;
    a[16] = 3;
    a[32] = 4;
    return (a[0], a[16], a[32]);
  }
}

// ====
// compileViaMlir: true
// ----
// f_basic() -> 42
// f_default() -> 0
// f_noname(uint256): 7 -> 7
// f_multi() -> 1, 2
// f_partial() -> 7, 0
// f_bool() -> true
// f_cond(bool): true -> 10
// f_cond(bool): false -> 20
// f_loop(uint256): 5 -> 10
// f_call() -> 5
// f_explicit() -> 99
// f_int_default() -> 0
// f_int_neg() -> -5
// f_addr_default() -> 0
// f_addr_set() -> 1
// f_bytes1_default() -> 0
// f_bytes1_set() -> left(0x41)
// f_bytes4_default() -> 0
// f_bytes32_default() -> 0
// f_early(bool): true -> 1
// f_early(bool): false -> 2
// f_early_multi(bool): true -> 10, 20
// f_early_multi(bool): false -> 1, 2
// f_bytes_default() -> 32, 0
// f_bytes_set() -> 32, 2, "hi"
// f_str_default() -> 32, 0
// f_str_set() -> 32, 5, "hello"
// f_arr_default() -> 32, 0
// f_arr_set() -> 32, 2, 1, 2
// f_fixed_arr_default() -> 0, 0
// f_fixed_arr_set() -> 3, 4
// f_int_u() -> -5
// f_addr_u() -> 1
// f_bytes1_u() -> left(0x41)
// f_bytes_u() -> 32, 2, "hi"
// f_str_u() -> 32, 5, "hello"
// f_arr_u() -> 32, 2, 5, 6
// f_multi_u() -> 42, true
// f_mixed(uint256): 5 -> 7, 6
// f_mixed(uint256): 0 -> 2, 1
// f_mixed_default() -> 0, 99
// f_arr() -> 2, 3, 4
