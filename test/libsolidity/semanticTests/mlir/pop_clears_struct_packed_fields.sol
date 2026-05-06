enum Dir { North, South }

contract C {
    struct T {
        uint256 x;   // slot 0 (32 bytes, full slot)
        bool flag;   // slot 1, byte 0
        Dir dir;     // slot 1, byte 1 — same slot as flag
    }
    T[] arr;

    function test() public {
        arr.push();
        arr[0].x = 99;
        arr[0].flag = true;
        arr[0].dir = Dir.South;

        arr.pop(); // must zero slot 0 (x) and slot 1 (flag+dir packed)

        assembly {
            mstore(0, arr.slot)
            let dataSlot := keccak256(0, 0x20)
            if sload(dataSlot) { revert(0, 0) }
            if sload(add(dataSlot, 1)) { revert(0, 0) }
        }
        assert(arr.length == 0);
    }
}
// ====
// compileViaMlir: true
// ----
// test() ->
// storageEmpty -> 1
