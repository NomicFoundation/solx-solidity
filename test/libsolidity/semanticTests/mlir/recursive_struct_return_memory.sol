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
  }

  function get() internal view returns (S memory) {
    return s;
  }

  function total() public view returns (uint256, uint256, uint256) {
    S memory m = get();
    return (m.a, m.b[0].a, m.b[0].b[0].a);
  }
}
// ====
// compileViaMlir: true
// ----
// build() ->
// total() -> 1, 2, 3
