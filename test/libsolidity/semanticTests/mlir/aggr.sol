contract C {
  mapping(address => uint) private m0;
  mapping(address => mapping(address => uint)) private m1;

  function m(address a, address b, uint c) public returns (uint) {
    m0[a] = c;
    m1[a][b] = c + 2;
    return m0[a] + m1[a][b];
  }

  function ar(uint[2] memory a) public returns (uint, uint) {
    a[0] = a[1];
    return (a[0], a[1]);
  }
  function ar2(uint[2][2] memory a) public returns (uint, uint) {
    a[0][1] = a[1][0];
    return (a[0][1], a[1][0]);
  }
  function arr() public returns (uint[2] memory) {
    uint[2] memory a;
    a[0] = 1;
    a[1] = 2;
    return a;
  }
}

// ====
// compileViaMlir: true
// ----
// m(address,address,uint256): 0, 0, 1 -> 4
// ar(uint256[2]): 1, 2 -> 2, 2
// ar2(uint256[2][2]): 1, 2, 3, 4 -> 3, 3
// arr() -> 1, 2
