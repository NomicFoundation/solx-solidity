contract C {
    function infinite_break() external pure returns (uint) {
        uint i = 0;
        for (;;) {
            i += 1;
            if (i == 5) {
                break;
            }
        }
        return i;
    }

    function no_step(uint n) external pure returns (uint) {
        uint i = 0;
        for (; i < n;) {
            i += 1;
        }
        return i;
    }
}

// ====
// compileViaMlir: true
// ----
// infinite_break() -> 5
// no_step(uint256): 7 -> 7
