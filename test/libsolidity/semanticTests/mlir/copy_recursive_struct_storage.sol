// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

struct S {
  uint256 a;
  S[] b;
}

contract C {
  S s1;
  S s2;

  function build() public {
    s1.a = 1;
    s1.b.push();
    s1.b[0].a = 2;
    s1.b[0].b.push();
    s1.b[0].b[0].a = 3;
  }

  function copy() public {
    s2 = s1;
  }

  function readS2() public view returns (uint256, uint256, uint256) {
    return (s2.a, s2.b[0].a, s2.b[0].b[0].a);
  }

  function mutateS2() public {
    s2.a = 99;
    s2.b[0].a = 98;
    s2.b[0].b[0].a = 97;
  }

  function readS1() public view returns (uint256, uint256, uint256) {
    return (s1.a, s1.b[0].a, s1.b[0].b[0].a);
  }
}
// ====
// compileViaMlir: true
// ----
// build() ->
// copy() ->
// readS2() -> 1, 2, 3
// mutateS2() ->
// readS1() -> 1, 2, 3
// readS2() -> 99, 98, 97
