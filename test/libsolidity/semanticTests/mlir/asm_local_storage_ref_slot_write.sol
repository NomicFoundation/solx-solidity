// Verifies that a named `S storage s` return parameter's .slot can be
// written from inline assembly.
contract C {
    struct S {
        uint256 x;
    }

    function getS(uint256 targetSlot) internal pure returns (S storage s) {
        assembly {
            s.slot := targetSlot
        }
    }

    function write(uint256 targetSlot, uint256 val) public {
        getS(targetSlot).x = val;
    }

    function read(uint256 targetSlot) public view returns (uint256) {
        return getS(targetSlot).x;
    }
}
// ====
// compileViaMlir: true
// ----
// write(uint256, uint256): 7, 99
// read(uint256): 7 -> 99
