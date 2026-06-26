// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

struct S {
  uint256 a;
  S[] b;
}

contract C {
  S s;

  // Builds a 3-level tree:
  //   s              (a = 1)
  //   ├─ s.b[0]      (a = 2)
  //   │  └─ b[0].b[0] (a = 3)
  //   └─ s.b[1]      (a = 4)
  function build() public {
    s.a = 1;
    s.b.push();
    s.b[0].a = 2;
    s.b[0].b.push();
    s.b[0].b[0].a = 3;
    s.b.push();
    s.b[1].a = 4;
  }

  function copyToMem()
    public
    view
    returns (uint256, uint256, uint256, uint256)
  {
    S memory m = s;
    return (m.a, m.b[0].a, m.b[0].b[0].a, m.b[1].a);
  }
}
// ====
// compileViaMlir: true
// ----
// build() ->
// copyToMem() -> 1, 2, 3, 4
