pragma solidity >=0.0;

contract C {
    bool[3] flags = [true, true, true];
    uint256 transient temp;

    function f() external {
        temp = 0xffffffff;
        assert(temp == 0xffffffff);
        delete temp;
        delete flags;
    }

    function getFlags() external returns(bool[3] memory)
    {
        return flags;
    }
}

// ====
// EVMVersion: >=cancun
// compileViaYul: true
// ----
// getFlags() -> true, true, true
// f() ->
// getFlags() -> true, true, true
