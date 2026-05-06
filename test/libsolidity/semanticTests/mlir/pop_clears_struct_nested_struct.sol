contract C {
    struct Inner {
        uint256 v;
        bool active;
    }
    struct Outer {
        uint256 a;
        Inner inner;
        uint256 b;
    }
    Outer[] arr;

    function test() public {
        arr.push();
        arr[0].a = 1;
        arr[0].inner.v = 2;
        arr[0].inner.active = true;
        arr[0].b = 3;

        arr.pop();

        assembly {
            mstore(0, arr.slot)
            let base := keccak256(0, 0x20)
            if sload(base)           { revert(0, 0) }  // a
            if sload(add(base, 1))   { revert(0, 0) }  // inner.v
            if sload(add(base, 2))   { revert(0, 0) }  // inner.active
            if sload(add(base, 3))   { revert(0, 0) }  // b
        }
        assert(arr.length == 0);
    }
}
// ====
// compileViaMlir: true
// ----
// test() ->
// storageEmpty -> 1
