// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract C {
    struct S {
        uint256 a;
        uint256[3] b;
        uint256[2] c;
        uint256 d;
    }

    function readAll(S calldata s)
        external pure
        returns (uint256 a, uint256 b2, uint256 c1, uint256 d)
    {
        a = s.a;
        b2 = s.b[2];
        c1 = s.c[1];
        d = s.d;
    }

    function readD(S calldata s) external pure returns (uint256) {
        return s.d;
    }
}
// ====
// compileViaMlir: true
// ----
// readAll((uint256,uint256[3],uint256[2],uint256)): 1, 10, 20, 30, 100, 200, 999 -> 1, 30, 200, 999
// readD((uint256,uint256[3],uint256[2],uint256)): 1, 10, 20, 30, 100, 200, 999 -> 999
// readD((uint256,uint256[3],uint256[2],uint256)): 0, 0, 0, 42, 0, 0, 0 -> 0
