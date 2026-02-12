contract C {
    struct S {
        uint256 a;
        address b;
    }

    S s;
    uint256 transient t;

    function setAndDelete() external {
        s.a = 1;
        s.b = address(0x1234);
        t = 2;
        delete s;
        delete t;
        assert(t == 2);
    }

    function getS() external view returns (uint256, address) {
        return (s.a, s.b);
    }
}
// ====
// EVMVersion: >=cancun
// compileViaYul: true
// ----
// setAndDelete() ->
// getS() -> 0, 0
