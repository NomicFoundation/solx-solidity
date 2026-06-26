// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

struct S {
  string name;
  S[] kids;
}

contract C {
  S s;

  function build() public {
    s.name = "root";
    s.kids.push();
    s.kids[0].name = "child";
  }

  function copyMemLens() public view returns (uint256, uint256) {
    S memory m = s;
    return (bytes(m.name).length, bytes(m.kids[0].name).length);
  }

  function del() public {
    delete s;
  }

  function rootLen() public view returns (uint256) {
    return bytes(s.name).length;
  }
}
// ====
// compileViaMlir: true
// ----
// build() ->
// copyMemLens() -> 4, 5
// rootLen() -> 4
// del() ->
// rootLen() -> 0
