contract C {
    uint8[] arr;

    function test() public {
        for (uint256 i = 0; i < 33; i++)
            arr.push(uint8(i + 1));

        arr.pop();

        assembly {
            mstore(0, arr.slot)
            let dataSlot := keccak256(0, 0x20)
            // Slot 1 must be fully zeroed.
            if sload(add(dataSlot, 1)) { revert(0, 0) }
            // Slot 0 must be intact (elements 0-31 preserved).
            if iszero(sload(dataSlot)) { revert(0, 0) }
        }
        assert(arr.length == 32);
        assert(arr[0] == 1);
        assert(arr[31] == 32);
        while (arr.length > 0)
            arr.pop();
    }
}
// ====
// compileViaMlir: true
// ----
// test() ->
// storageEmpty -> 1
