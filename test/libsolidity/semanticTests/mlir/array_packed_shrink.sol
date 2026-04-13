// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;


contract PackedShrinkTest {

    bytes16[] b16;
    uint8[]   u8;
    uint16[]  u16;


    function b16_set(bytes16[] memory src) public { b16 = src; }
    function b16_len() public view returns (uint256) { return b16.length; }
    function b16_get(uint256 i) public view returns (bytes16) { return b16[i]; }

    function b16_shrink_odd(bytes16[] memory full, bytes16[] memory shortened)
        public
    {
        b16 = full;
        b16 = shortened;
    }

    function b16_shrink_odd_raw(bytes16[] memory full, bytes16[] memory shortened)
        public returns (uint256 upperHalf)
    {
        b16 = full;
        b16 = shortened;
        assembly {
            mstore(0, b16.slot)
            let slot0 := sload(keccak256(0, 0x20))
            upperHalf := shr(128, slot0)
        }
    }

    function b16_raw_upper() public view returns (uint256 upperHalf) {
        assembly {
            mstore(0, b16.slot)
            let slot0 := sload(keccak256(0, 0x20))
            upperHalf := shr(128, slot0)
        }
    }


    function u8_set(uint8[] memory src) public { u8 = src; }
    function u8_len() public view returns (uint256) { return u8.length; }
    function u8_get(uint256 i) public view returns (uint8) { return u8[i]; }

    function u8_shrink(uint8[] memory full, uint8[] memory shortened) public {
        u8 = full;
        u8 = shortened;
    }

    function u8_shrink_raw(uint8[] memory full, uint8[] memory shortened)
        public returns (uint256 rawSlot)
    {
        u8 = full;
        u8 = shortened;
        assembly {
            mstore(0, u8.slot)
            rawSlot := sload(keccak256(0, 0x20))
        }
    }


    function b16_shrink_4to2_raw(bytes16[] memory full, bytes16[] memory shortened)
        public returns (uint256 slot1)
    {
        b16 = full;
        b16 = shortened;
        assembly {
            mstore(0, b16.slot)
            let base := keccak256(0, 0x20)
            slot1 := sload(add(base, 1))
        }
    }

    function b16_shrink_4to3_raw(bytes16[] memory full, bytes16[] memory shortened)
        public returns (uint256 upperSlot1)
    {
        b16 = full;
        b16 = shortened;
        assembly {
            mstore(0, b16.slot)
            let base := keccak256(0, 0x20)
            upperSlot1 := shr(128, sload(add(base, 1)))
        }
    }


    function u16_set(uint16[] memory src) public { u16 = src; }
    function u16_len() public view returns (uint256) { return u16.length; }
    function u16_get(uint256 i) public view returns (uint16) { return u16[i]; }

    function u16_shrink(uint16[] memory full, uint16[] memory shortened) public {
        u16 = full;
        u16 = shortened;
    }

    function u16_shrink_raw(uint16[] memory full, uint16[] memory shortened)
        public returns (uint256 rawSlot)
    {
        u16 = full;
        u16 = shortened;
        assembly {
            mstore(0, u16.slot)
            rawSlot := sload(keccak256(0, 0x20))
        }
    }
}
// ====
// compileViaMlir: true
// ----
// b16_shrink_odd(bytes16[],bytes16[]): 0x40, 0xc0, 3, left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa), left(0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb), left(0xcccccccccccccccccccccccccccccccc), 1, left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) ->
// b16_len() -> 1
// b16_get(uint256): 0 -> left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)
// b16_shrink_odd_raw(bytes16[],bytes16[]): 0x40, 0xc0, 3, left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa), left(0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb), left(0xcccccccccccccccccccccccccccccccc), 1, left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) -> 0
// b16_set(bytes16[]): 0x20, 2, left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa), left(0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) ->
// b16_set(bytes16[]): 0x20, 1, left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) ->
// b16_get(uint256): 0 -> left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)
// b16_raw_upper() -> 0
// u8_shrink(uint8[],uint8[]): 0x40, 0x480, 33, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 1, 42 ->
// u8_len() -> 1
// u8_get(uint256): 0 -> 42
// u8_shrink_raw(uint8[],uint8[]): 0x40, 0x480, 33, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 1, 42 -> 42
// u8_shrink(uint8[],uint8[]): 0x40, 0x480, 33, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 0 ->
// u8_len() -> 0
// u8_shrink(uint8[],uint8[]): 0x40, 0x480, 33, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 ->
// u8_len() -> 17
// u8_get(uint256): 0 -> 1
// u8_get(uint256): 16 -> 17
// u16_shrink(uint16[],uint16[]): 0x40, 0x280, 17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1, 42 ->
// u16_len() -> 1
// u16_get(uint256): 0 -> 42
// u16_shrink(uint16[],uint16[]): 0x40, 0x280, 17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0 ->
// u16_len() -> 0
// b16_shrink_4to2_raw(bytes16[],bytes16[]): 0x40, 0xe0, 4, left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa), left(0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb), left(0xcccccccccccccccccccccccccccccccc), left(0xdddddddddddddddddddddddddddddddd), 2, left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa), left(0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) -> 0
// b16_shrink_4to3_raw(bytes16[],bytes16[]): 0x40, 0xe0, 4, left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa), left(0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb), left(0xcccccccccccccccccccccccccccccccc), left(0xdddddddddddddddddddddddddddddddd), 3, left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa), left(0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb), left(0xcccccccccccccccccccccccccccccccc) -> 0
// b16_set(bytes16[]): 0x20, 2, left(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa), left(0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) ->
// b16_set(bytes16[]): 0x20, 0 ->
// b16_len() -> 0
// b16_raw_upper() -> 0
// u8_shrink_raw(uint8[],uint8[]): 0x40, 0x460, 32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 1, 42 -> 42
// u8_shrink_raw(uint8[],uint8[]): 0x40, 0x460, 32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0 -> 0
// u8_len() -> 0
// u16_shrink(uint16[],uint16[]): 0x40, 0x260, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 42 ->
// u16_len() -> 1
// u16_get(uint256): 0 -> 42
// u16_shrink(uint16[],uint16[]): 0x40, 0x260, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0 ->
// u16_len() -> 0
// u16_shrink_raw(uint16[],uint16[]): 0x40, 0x280, 17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1, 42 -> 42
// u16_shrink_raw(uint16[],uint16[]): 0x40, 0x260, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 42 -> 42
