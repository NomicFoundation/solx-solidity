contract C {
    function dirtyStart(uint256[] calldata arr, uint256 dirty, uint256 end)
        external
        pure
        returns (uint256)
    {
        uint8 start;
        assembly { start := dirty }

        uint256[] calldata s = arr[start:end];
        return s.length;
    }

    function dirtyEnd(uint256[] calldata arr, uint256 start, uint256 dirty)
        external
        pure
        returns (uint256)
    {
        uint8 end;
        assembly { end := dirty }

        uint256[] calldata s = arr[start:end];
        return s.length;
    }
}
// ====
// compileViaMlir: true
// ----
// dirtyStart(uint256[],uint256,uint256): 0x60, 0x100000000000000000000000000000000000000000000000000000000000001, 3, 4, 10, 20, 30, 40 -> 2
// dirtyEnd(uint256[],uint256,uint256): 0x60, 1, 0x100000000000000000000000000000000000000000000000000000000000003, 4, 10, 20, 30, 40 -> 2
