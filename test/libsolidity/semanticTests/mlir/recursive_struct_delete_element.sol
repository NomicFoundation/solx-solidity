// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

struct S {
  uint256 a;
  S[] b;
}

contract C {
  S s;

  function build() public {
    s.a = 1;
    s.b.push();
    s.b[0].a = 2;
    s.b[0].b.push();
    s.b[0].b[0].a = 3;
    s.b.push();
    s.b[1].a = 4;
  }

  function delElem() public {
    delete s.b[0];
  }

  function read() public view returns (uint256, uint256, uint256, uint256) {
    return (s.a, s.b.length, s.b[0].a, s.b[1].a);
  }

  function child0Len() public view returns (uint256) {
    return s.b[0].b.length;
  }
}
// ====
// compileViaMlir: true
// ----
// build() ->
// delElem() ->
// read() -> 1, 2, 0, 4
// child0Len() -> 0
