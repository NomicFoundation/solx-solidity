contract C {
    struct S {
        uint256 x;
        uint256[3] arr;
        uint256 y;
    }
    S[] sarr;

    function test() public {
        sarr.push();
        sarr[0].x = 10;
        sarr[0].arr[0] = 11;
        sarr[0].arr[1] = 12;
        sarr[0].arr[2] = 13;
        sarr[0].y = 14;

        sarr.pop();

        assembly {
            mstore(0, sarr.slot)
            let base := keccak256(0, 0x20)
            if sload(base)           { revert(0, 0) }  // x
            if sload(add(base, 1))   { revert(0, 0) }  // arr[0]
            if sload(add(base, 2))   { revert(0, 0) }  // arr[1]
            if sload(add(base, 3))   { revert(0, 0) }  // arr[2]
            if sload(add(base, 4))   { revert(0, 0) }  // y
        }
        assert(sarr.length == 0);
    }
}
// ====
// compileViaMlir: true
// ----
// test() ->
// storageEmpty -> 1
