contract C {
  mapping(address => uint) private m0;
  mapping(address => mapping(address => uint)) private m1;

  function m(address a, address b, uint c) public returns (uint) {
    m0[a] = c;
    m1[a][b] = c + 2;
    return m0[a] + m1[a][b];
  }
}

// ====
// compileViaMlir: true
// ----
// m(address,address,uint256): 0, 0, 1 -> 4
