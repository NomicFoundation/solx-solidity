contract C {
    function pair() internal pure returns (uint256, uint256) {
        return (7, 9);
    }

    function localIgnore() public pure returns (uint256) {
        (uint256 x,) = pair();
        return x;
    }

    function callIgnore() public returns (bool) {
        (bool success,) = address(this).call(abi.encodeWithSignature("localIgnore()"));
        return success;
    }
}

// ====
// compileViaMlir: true
// ----
// localIgnore() -> 7
// callIgnore() -> true
