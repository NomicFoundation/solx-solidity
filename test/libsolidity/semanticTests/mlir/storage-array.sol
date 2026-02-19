contract C {
  // Static uint array
  uint[5] static_arr;
  function static_arr_rw() public returns (uint) {
    static_arr[0] = 1;
    return static_arr[0];
  }

  // Dynamic uint array
  uint[] dynamic_arr;
  function dynamic_arr_rw(uint a) public returns (uint) {
    if (a != 0) {
      dynamic_arr.push();
      dynamic_arr.push() = a;
      dynamic_arr.push(1 + dynamic_arr[0] + dynamic_arr[1]);
    }
    return dynamic_arr[2];
  }

  function dynamic_arr_pop() public returns (uint) {
    dynamic_arr.pop();
    return 0;
  }

  // 2D static array
  uint[3][2] static_2d;
  function static_2d_rw() public returns (uint) {
  unchecked {
    for (uint i = 0; i < 2; ++i)
      for (uint j = 0; j < 3; ++j)
        static_2d[i][j] = i*10 + j;
    return static_2d[1][2];
  }
  }

  // Dynamic array of static arrays
  uint[2][] dynamic_2d;
  function dynamic_2d_rw() public returns (uint[2][] memory) {
    dynamic_2d.push()[0] = 0x10;
    dynamic_2d[0][1] = 0x20;
    dynamic_2d.push();
    dynamic_2d[1][0] = 0x30;
    return dynamic_2d;
  }

  // Return storage array as memory
  uint[] m;
  function return_as_mem() public returns (uint[] memory) {
    m.push() = 1;
    m.push(2);
    return m;
  }

  // Packed static array
  uint8[4] packed_static;
  function set_packed_static(uint256 i, uint8 v) public {
    packed_static[i] = v;
  }
  function get_packed_static(uint256 i) public view returns (uint8) {
    return packed_static[i];
  }

  // Packed dynamic array
  uint8[] packed_dynamic;
  function push_packed_dynamic(uint8 v) public {
    packed_dynamic.push(v);
  }
  function get_packed_dynamic(uint256 i) public view returns (uint8) {
    return packed_dynamic[i];
  }
  function len_packed_dynamic() public view returns (uint256) {
    return packed_dynamic.length;
  }
}

// ====
// compileViaMlir: true
// ----
// static_arr_rw() -> 1
// dynamic_arr_pop() -> FAILURE, hex"4e487b71", 0x31
// dynamic_arr_rw(uint256): 0 -> FAILURE, hex"4e487b71", 0x32
// dynamic_arr_rw(uint256): 1 -> 2
// dynamic_arr_pop() -> 0
// static_2d_rw() -> 12
// dynamic_2d_rw() -> 0x20, 2, 0x10, 0x20, 0x30, 0
// return_as_mem() -> 0x20, 2, 1, 2
// set_packed_static(uint256,uint8): 0, 0x11 ->
// set_packed_static(uint256,uint8): 1, 0x22 ->
// set_packed_static(uint256,uint8): 2, 0x33 ->
// set_packed_static(uint256,uint8): 3, 0x44 ->
// get_packed_static(uint256): 0 -> 0x11
// get_packed_static(uint256): 1 -> 0x22
// get_packed_static(uint256): 2 -> 0x33
// get_packed_static(uint256): 3 -> 0x44
// len_packed_dynamic() -> 0
// push_packed_dynamic(uint8): 0x11 ->
// push_packed_dynamic(uint8): 0x22 ->
// push_packed_dynamic(uint8): 0x33 ->
// push_packed_dynamic(uint8): 0x44 ->
// len_packed_dynamic() -> 4
// get_packed_dynamic(uint256): 0 -> 0x11
// get_packed_dynamic(uint256): 1 -> 0x22
// get_packed_dynamic(uint256): 2 -> 0x33
// get_packed_dynamic(uint256): 3 -> 0x44
