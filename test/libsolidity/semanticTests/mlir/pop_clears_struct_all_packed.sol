contract C {
    struct S {
        uint64 a;   // bits  63..0
        uint64 b;   // bits 127..64
        uint64 c;   // bits 191..128
        uint64 d;   // bits 255..192
    }
    S[] arr;

    function test() public {
        arr.push();
        arr[0].a = 1;
        arr[0].b = 2;
        arr[0].c = 3;
        arr[0].d = 4;

        arr.pop();

        assembly {
            mstore(0, arr.slot)
            let base := keccak256(0, 0x20)
            // All four fields share slot 0; one sstore must zero the whole slot.
            if sload(base) { revert(0, 0) }
        }
        assert(arr.length == 0);
    }
}
// ====
// compileViaMlir: true
// ----
// test() ->
// storageEmpty -> 1
