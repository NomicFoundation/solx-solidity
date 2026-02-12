contract C {
    mapping(uint256 => uint256) m;
    uint256 transient t;

    function setAndClear() external {
        m[0] = 1;
        t = 2;
        delete m[0];
        delete t;
        assert(t == 2);
    }

    function getM() external view returns (uint256) {
        return m[0];
    }
}

// ====
// EVMVersion: >=cancun
// compileViaYul: true
// ----
// setAndClear() ->
// getM() -> 0
