// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract C {
    struct S {
        uint256 a;
        uint256[2] b;
        uint256 c;
    }

    function readAll(S calldata s)
        external pure
        returns (uint256 a, uint256 b0, uint256 b1, uint256 c)
    {
        a = s.a;
        b0 = s.b[0];
        b1 = s.b[1];
        c = s.c;
    }

    function readC(S calldata s) external pure returns (uint256) {
        return s.c;
    }
}
// ====
// compileViaMlir: true
// ----
// readAll((uint256,uint256[2],uint256)): 42, 1, 2, 23 -> 42, 1, 2, 23
// readC((uint256,uint256[2],uint256)): 0, 100, 200, 999 -> 999
// readC((uint256,uint256[2],uint256)): 0, 0, 42, 0 -> 0
