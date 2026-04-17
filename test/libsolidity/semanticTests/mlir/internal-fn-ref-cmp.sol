// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract C {
    function(uint) internal pure returns (uint) stored;

    function f(uint x) internal pure returns (uint) { return x + 10; }
    function g(uint x) internal pure returns (uint) { return x + 20; }

    function test_eq_same_fn() public pure returns (bool) {
        return f == f;
    }

    function test_neq_diff_fn() public pure returns (bool) {
        return f != g;
    }

    function test_eq_false_diff_fn() public pure returns (bool) {
        return !(f == g);
    }

    function test_eq_stored() public returns (bool) {
        stored = f;
        return stored == f;
    }

    function test_neq_stored() public returns (bool) {
        stored = f;
        return stored != g;
    }
}
// ====
// compileViaMlir: true
// ----
// test_eq_same_fn() -> true
// test_neq_diff_fn() -> true
// test_eq_false_diff_fn() -> true
// test_eq_stored() -> true
// test_neq_stored() -> true
