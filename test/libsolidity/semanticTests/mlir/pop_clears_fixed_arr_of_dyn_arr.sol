contract C {
    uint256[][3][] arr;

    function test() public {
        arr.push();
        arr[0][0].push(111);
        arr[0][1].push(222);
        arr[0][1].push(333);
        arr.pop();
        assembly {
            mstore(0, arr.slot)
            let base := keccak256(0, 0x20)
            if sload(base)           { revert(0, 0) }  // arr[0][0] length
            if sload(add(base, 1))   { revert(0, 0) }  // arr[0][1] length
            if sload(add(base, 2))   { revert(0, 0) }  // arr[0][2] length
            mstore(0, base)
            if sload(keccak256(0, 0x20)) { revert(0, 0) }  // arr[0][0][0] = 111
            mstore(0, add(base, 1))
            let d1 := keccak256(0, 0x20)
            if sload(d1)             { revert(0, 0) }  // arr[0][1][0] = 222
            if sload(add(d1, 1))     { revert(0, 0) }  // arr[0][1][1] = 333
        }
        assert(arr.length == 0);
    }
}
// ====
// compileViaMlir: true
// ----
// test() ->
// storageEmpty -> 1
