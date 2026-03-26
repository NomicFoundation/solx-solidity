// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression test for genMemAlloc: array literals whose element type is
// narrower than i256 must be zero-extended to i256 before MStore.
// Without the fix, the upper bits of each 32-byte memory slot were left
// uninitialized, corrupting the ABI encoding of the returned array.

contract ArrayLitNarrowElt {
    function bool_lit() public pure returns (bool[3] memory) {
        return [true, false, true];
    }

    function uint8_lit() public pure returns (uint8[4] memory) {
        return [1, 2, 3, 255];
    }

    function addr_lit() public pure returns (address[2] memory) {
        return [address(1), address(2)];
    }

    // uint256 elements with small values: no narrowing, but verifies that
    // small compile-time constants are stored correctly into 32-byte slots.
    uint[3] ad;
    function uint256_mixed() public returns (uint[3] memory) {
        ad = [7, type(uint256).max, 2];
        return ad;
    }
}
// ====
// compileViaMlir: true
// ----
// bool_lit() -> true, false, true
// uint8_lit() -> 1, 2, 3, 255
// addr_lit() -> 0x0000000000000000000000000000000000000001, 0x0000000000000000000000000000000000000002
// uint256_mixed() -> 7, 115792089237316195423570985008687907853269984665640564039457584007913129639935, 2
