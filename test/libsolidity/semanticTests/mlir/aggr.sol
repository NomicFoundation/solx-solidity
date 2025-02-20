contract C {
  mapping(address => uint) private m0;
  mapping(address => mapping(address => uint)) private m1;

  function m(address a, address b, uint c) public returns (uint) {
    m0[a] = c;
    m1[a][b] = c + 2;
    return m0[a] + m1[a][b];
  }

  function l(uint[2] memory a) public returns (uint, uint) {
    a[0] = a[1];
    return (a[0], a[1]);
  }
  function l2(uint[2][2] memory a) public returns (uint, uint) {
    a[0][1] = a[1][0];
    return (a[0][1], a[1][0]);
  }
}

// ====
// compileViaMlir: true
// ----
// m(address,address,uint256): 0, 0, 1 -> 4
// l(uint256[2]): 1, 2 -> 2, 2
// l2(uint256[2][2]): 1, 2, 3, 4 -> 3, 3
