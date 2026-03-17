interface I {
  function h(uint x) external pure returns (uint);
}

contract D is I {
  function h(uint x) external pure returns (uint) { return x + 1; }
}

contract C {
  function f(uint a) internal returns (uint) { return a + 10; }
  function g(uint a) internal returns (uint) { return a + 20; }
  function m(uint a, uint b) public returns (uint) {
    function(uint) internal returns (uint) p;
    if (a == 0)
      p = f;
    else if (a == 1)
      p = g;
    return p(b);
  }

  function (uint) internal returns (uint) s0;
  function (uint) internal returns (uint) s1 = f;
  function n(bool a) public returns (uint) {
    if (a)
      return s1(0);
    return s0(0);
  }

  I t;
  function setT(address a) public { t = I(a); }
  function deploy() public { t = new D(); }

  function call(uint x) public view returns (uint) {
    function(uint) external pure returns (uint) fp = t.h;
    return fp(x);
  }

  function callGas(uint x) public view returns (uint) {
    function(uint) external pure returns (uint) fp = t.h;
    return fp{gas: 100000}(x);
  }

  function tryCall(uint x) public returns (bool, uint) {
    function(uint) external pure returns (uint) fp = t.h;
    bool success;
    uint result;
    try fp(x) returns (uint r) {
      success = true;
      result = r;
    } catch {
      success = false;
      result = 0;
    }
    return (success, result);
  }

  function tryCallGas(uint x) public returns (bool, uint) {
    function(uint) external pure returns (uint) fp = t.h;
    bool success;
    uint result;
    try fp{gas: 100000}(x) returns (uint r) {
      success = true;
      result = r;
    } catch {
      success = false;
      result = 0;
    }
    return (success, result);
  }
}

// ====
// compileViaMlir: true
// ----
// m(uint256,uint256): 0, 1 -> 11
// m(uint256,uint256): 1, 2 -> 22
// m(uint256,uint256): 2, 3 -> FAILURE, hex"4e487b71", 0x51
// n(bool): true -> 10
// n(bool): false -> FAILURE, hex"4e487b71", 0x51
// setT(address): 0 ->
// call(uint256): 41 -> FAILURE
// callGas(uint256): 41 -> FAILURE
// deploy() ->
// call(uint256): 41 -> 42
// callGas(uint256): 41 -> 42
// tryCall(uint256): 41 -> true, 42
// tryCallGas(uint256): 41 -> true, 42
