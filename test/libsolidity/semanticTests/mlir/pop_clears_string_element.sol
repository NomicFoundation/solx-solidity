contract C {
    string[] arr;

    function test() public {
        arr.push("hello");
        arr.push("this string is definitely longer than 31b");

        arr.pop(); // removes out-of-place string (arr[1])

        assembly {
            mstore(0, arr.slot)
            let dataSlot := keccak256(0, 0x20)
            let longSlot := add(dataSlot, 1)
            // Length slot of arr[1] must be zero.
            if sload(longSlot) { revert(0, 0) }
            // First data slot of arr[1] must be zero.
            mstore(0, longSlot)
            if sload(keccak256(0, 0x20)) { revert(0, 0) }
        }
        assert(arr.length == 1);

        arr.pop(); // removes in-place string (arr[0])

        assembly {
            mstore(0, arr.slot)
            let dataSlot := keccak256(0, 0x20)
            // Slot of arr[0] must be zero.
            if sload(dataSlot) { revert(0, 0) }
        }
        assert(arr.length == 0);
    }
}
// ====
// compileViaMlir: true
// ----
// test() ->
// storageEmpty -> 1
