// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

function add(uint a, uint b) pure returns (uint) {
    return a + b;
}

contract Test {
    function computeA() public pure returns (uint) {
        return add(3, 4);
    }

    function computeB() public pure returns (uint) {
        return add(5, 6);
    }
}

// ====
// compileViaMlir: true
// ----
// computeA() -> 7
// computeB() -> 11
