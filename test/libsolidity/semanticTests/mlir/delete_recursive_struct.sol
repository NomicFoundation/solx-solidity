// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract C {
  struct S {
    uint256 a;
    S[] b;
  }

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

  function clear() public {
    delete s;
  }

  function topA() public view returns (uint256) {
    return s.a;
  }

  function len() public view returns (uint256) {
    return s.b.length;
  }

  function childA(uint256 i) public view returns (uint256) {
    return s.b[i].a;
  }

  function grandchildA() public view returns (uint256) {
    return s.b[0].b[0].a;
  }
}
// ====
// compileViaMlir: true
// ----
// build() ->
// topA() -> 1
// len() -> 2
// childA(uint256): 0 -> 2
// childA(uint256): 1 -> 4
// grandchildA() -> 3
// clear() ->
// topA() -> 0
// len() -> 0
// storageEmpty -> 1
