contract C {
    uint72[] arr;

    function test() public {
        arr.push(0xAAAAAAAAAAAAAAAAA);
        arr.push(0xBBBBBBBBBBBBBBBBB);
        arr.push(0xCCCCCCCCCCCCCCCCC);
        arr.push(0xDDDDDDDDDDDDDDDDD);

        arr.pop();

        assembly {
            mstore(0, arr.slot)
            let dataSlot := keccak256(0, 0x20)
            // Slot 1 must be fully zeroed.
            if sload(add(dataSlot, 1)) { revert(0, 0) }
            // Slot 0 must be intact.
            if iszero(sload(dataSlot)) { revert(0, 0) }
        }
        assert(arr.length == 3);
        arr.pop();
        arr.pop();
        arr.pop();
    }
}
// ====
// compileViaMlir: true
// ----
// test() ->
// storageEmpty -> 1
