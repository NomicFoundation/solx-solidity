// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s
// CHECK: sol.contract

pragma solidity ^0.8.0;

contract C {
    struct Y {
        uint a;
        uint b;
    }
    mapping(uint256 => Y)[] public m;
    mapping(uint256 => Y)[3] public n;
    constructor() {
        m.push();
        m.push();
        m[1][0].a = 1;
        m[1][0].b = 2;
        m[1][1].a = 3;
        m[1][1].b = 4;
        n[1][0].a = 7;
        n[1][0].b = 8;
        n[1][1].a = 9;
        n[1][1].b = 10;
    }
}
