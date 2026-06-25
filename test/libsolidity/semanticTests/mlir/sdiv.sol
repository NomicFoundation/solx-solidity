contract C {
  function div(int a, int b) public pure returns (int) {
    return a / b;
  }
  function divU(int a, int b) public pure returns (int) {
    unchecked { return a / b; }
  }
  function div8U(int8 a, int8 b) public pure returns (int8) {
    unchecked { return a / b; }
  }

  function divImm() public pure returns (int) {
    return int(-7) / 2;
  }
  function divImmCheckedOvf() public pure returns (int) {
    int x = type(int).min;
    return x / -1;
  }
  function divImmUncheckedWrap() public pure returns (int) {
    int x = type(int).min;
    unchecked { return x / -1; }
  }
}

// ====
// compileViaMlir: true
// ----
// div(int256,int256): 7, 2 -> 3
// div(int256,int256): -7, 2 -> -3
// div(int256,int256): 7, -2 -> -3
// div(int256,int256): -7, -2 -> 3
// div(int256,int256): 1, 0 -> FAILURE, hex"4e487b71", 0x12
// div(int256,int256): -57896044618658097711785492504343953926634992332820282019728792003956564819968, -1 -> FAILURE, hex"4e487b71", 0x11
// divU(int256,int256): -7, 2 -> -3
// divU(int256,int256): 1, 0 -> FAILURE, hex"4e487b71", 0x12
// divU(int256,int256): -57896044618658097711785492504343953926634992332820282019728792003956564819968, -1 -> -57896044618658097711785492504343953926634992332820282019728792003956564819968
// div8U(int8,int8): -8, 2 -> -4
// div8U(int8,int8): -128, -1 -> -128
// divImm() -> -3
// divImmCheckedOvf() -> FAILURE, hex"4e487b71", 0x11
// divImmUncheckedWrap() -> -57896044618658097711785492504343953926634992332820282019728792003956564819968
