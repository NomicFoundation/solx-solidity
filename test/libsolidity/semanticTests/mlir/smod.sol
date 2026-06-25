contract C {
  function mod(int a, int b) public pure returns (int) {
    return a % b;
  }
  function modU(int a, int b) public pure returns (int) {
    unchecked { return a % b; }
  }
  function mod8U(int8 a, int8 b) public pure returns (int8) {
    unchecked { return a % b; }
  }

  function modImm() public pure returns (int) {
    return int(-7) % 5;
  }
  function modImmMinusOne() public pure returns (int) {
    int x = type(int).min;
    return x % -1;
  }
  function modImmUncheckedMinusOne() public pure returns (int) {
    int x = type(int).min;
    unchecked { return x % -1; }
  }
}

// ====
// compileViaMlir: true
// ----
// mod(int256,int256): 7, 5 -> 2
// mod(int256,int256): -7, 5 -> -2
// mod(int256,int256): 7, -5 -> 2
// mod(int256,int256): -7, -5 -> -2
// mod(int256,int256): 1, 0 -> FAILURE, hex"4e487b71", 0x12
// mod(int256,int256): -57896044618658097711785492504343953926634992332820282019728792003956564819968, -1 -> 0
// modU(int256,int256): -7, 5 -> -2
// modU(int256,int256): 1, 0 -> FAILURE, hex"4e487b71", 0x12
// modU(int256,int256): -57896044618658097711785492504343953926634992332820282019728792003956564819968, -1 -> 0
// mod8U(int8,int8): -7, 5 -> -2
// mod8U(int8,int8): -128, -1 -> 0
// modImm() -> -2
// modImmMinusOne() -> 0
// modImmUncheckedMinusOne() -> 0
