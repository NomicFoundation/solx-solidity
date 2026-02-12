// SPDX-License-Identifier: GPL-2.0
pragma solidity >=0.0;

contract C {
    uint256 transient varTransient;
    uint256 public varStorage = 0xeeeeeeeeee;

    function foo() external {
        varTransient = 0xffffffff;
        assert(varTransient == 0xffffffff);
        delete varStorage;
        delete varTransient;
    }
}

// ====
// EVMVersion: >=cancun
// compileViaYul: true
// ----
// varStorage() -> 0xeeeeeeeeee
// foo() ->
// varStorage() -> 0
