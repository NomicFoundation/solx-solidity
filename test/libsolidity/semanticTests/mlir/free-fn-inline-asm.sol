function double(uint x) pure returns (uint r) {
  assembly {
    function g(a) -> b {
      b := mul(a, 2)
    }
    r := g(x)
  }
}

contract C {
  function f() public pure returns (uint) {
    return double(21);
  }
}
// ----
// f() -> 42
