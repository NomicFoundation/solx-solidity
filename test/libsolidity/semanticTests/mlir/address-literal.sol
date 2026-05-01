contract C {
    function direct() external pure returns (address) {
        return 0x00000000000000000000000000000000000000ff;
    }

    function via_local() external pure returns (address) {
        address a = 0x00000000000000000000000000000000000000ff;
        return a;
    }
}

// ====
// compileViaMlir: true
// ----
// direct() -> 0xff
// via_local() -> 0xff
