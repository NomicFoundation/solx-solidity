contract C {
    uint256[][] arr;

    function test() public {
        arr.push();
        arr[0].push(111);
        arr[0].push(222);

        arr.pop();

        assembly {
            mstore(0, arr.slot)
            let outerData := keccak256(0, 0x20)
            // Inner array length slot must be zero.
            if sload(outerData) { revert(0, 0) }
            // Inner data slots must be zero.
            mstore(0, outerData)
            let innerData := keccak256(0, 0x20)
            if sload(innerData) { revert(0, 0) }
            if sload(add(innerData, 1)) { revert(0, 0) }
        }
        assert(arr.length == 0);
    }
}
// ====
// compileViaMlir: true
// ----
// test() ->
// storageEmpty -> 1
