// Verifies the getter for a fixed-outer / dynamic-inner packed array
// (uint8[][2]): outer bounds check, inner bounds check, packed uint8 slot
// extraction (keccak of inner-array slot, index/32, byte shift & AND 0xff).
contract C {
    uint8[][2] public a;
    constructor() {
        a[1].push(3);
        a[1].push(4);
    }
}
// ====
// compileViaMlir: true
// ----
// a(uint256,uint256): 0, 0 -> FAILURE
// a(uint256,uint256): 1, 0 -> 3
// a(uint256,uint256): 1, 1 -> 4
// a(uint256,uint256): 2, 0 -> FAILURE
