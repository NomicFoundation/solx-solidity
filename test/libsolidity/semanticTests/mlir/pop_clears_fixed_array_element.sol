contract C {
    uint256[3][] arr;

    function test() public {
        arr.push();
        arr[0][0] = 1; arr[0][1] = 2; arr[0][2] = 3;
        arr.push();
        arr[1][0] = 4; arr[1][1] = 5; arr[1][2] = 6;

        arr.pop();

        assembly {
            mstore(0, arr.slot)
            let dataSlot := keccak256(0, 0x20)
            if sload(add(dataSlot, 3)) { revert(0, 0) }
            if sload(add(dataSlot, 4)) { revert(0, 0) }
            if sload(add(dataSlot, 5)) { revert(0, 0) }
        }
        assert(arr.length == 1);
        assert(arr[0][0] == 1);
        assert(arr[0][1] == 2);
        assert(arr[0][2] == 3);
        arr.pop();
    }
}
// ====
// compileViaMlir: true
// ----
// test() ->
// storageEmpty -> 1
