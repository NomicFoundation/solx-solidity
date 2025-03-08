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
    a[1] = a[0];
    return a;
  }
  function dar(uint[] memory a) public returns (uint, uint) {
    a[0] = a[1];
    return (a[0], a[1]);
  }
  function dar2(uint[][] memory a) public returns (uint, uint) {
    a[0][1] = a[1][0];
    return (a[0][1], a[1][0]);
  }
  function darr() public returns (uint[] memory) {
    uint[] memory a;
    a = new uint[](2);
    a[0] = 1;
    a[1] = a[0];
    return a;
  }
  function darr2() public returns (uint[][] memory) {
    uint[][] memory a;
    a = new uint[][](2);
    a[0] = new uint[](2);
    a[1] = new uint[](2);
    a[0][1] = 1;
    a[1][0] = a[0][1];
    return a;
  }
}

// ====
// compileViaMlir: true
// ----
// m(address,address,uint256): 0, 0, 1 -> 4
// ar(uint256[2]): 1, 2 -> 2, 2
// ar2(uint256[2][2]): 1, 2, 3, 4 -> 3, 3
// arr() -> 1, 1
// dar2(uint256[][]): 0x20, 2, 0x40, 0xa0, 2, 3, 4, 2, 5, 6 -> 5, 5
// darr() -> 32, 2, 1, 1
// darr2() -> 0x20, 2, 0x40, 0xa0, 2, 0, 1, 2, 1, 0
