contract C {
    function dirtyNewArrayLength(uint256 dirty) public pure returns (uint256) {
        uint8 n;
        assembly { n := dirty }

        uint256[] memory a = new uint256[](n);
        return a.length;
    }

    function dirtyArrayLiteralElement(uint256 dirty) public pure returns (uint256 r) {
        uint8 n;
        assembly { n := dirty }

        uint8[1] memory a = [n];
        assembly { r := mload(a) }
    }
}
// ====
// compileViaMlir: true
// ----
// dirtyNewArrayLength(uint256): 0x100000000000000000000000000000000000000000000000000000000000001 -> 1
// dirtyArrayLiteralElement(uint256): 0x100000000000000000000000000000000000000000000000000000000000001 -> 1
