// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Storage-to-memory copy of S.
// Plants raw slot values via sstore, then copies to memory and asserts that
// read-side cleanup is applied: sign extension for signed ints, left-
// alignment for bytesN.
//
// S memory layout (32 bytes per field; reference types as pointer):
//   m+0x000: i72  (int72,  sign-extended)   m+0x020: u8   (uint8)
//   m+0x040: flags (pointer)                m+0x060: addrs (pointer)
//   m+0x080: i192 (int192, sign-extended)   m+0x0a0: u8b  (uint8)
//   m+0x0c0: b4   (bytes4, left-aligned)
//   m+0x0e0: i8   (int8,   sign-extended)   m+0x100: i16  (int16,  sign-extended)
//   (remaining fields i128..b19 occupy m+0x120..m+0x1c0)

struct S {
    int72        i72;
    uint8        u8;
    bool[][]     flags;
    address[2]   addrs;
    int192       i192;
    uint8        u8b;
    bytes4       b4;
    int8         i8;
    int16        i16;
    int128       i128;
    uint72       u72;
    uint128      u128;
    uint224      u224;
    bytes8       b8;
    bytes19      b19;
}

contract StructCopyStoToMem {
    S src;

    // Plant src slot 0 (i72=-1, u8=7).
    function writeSrcSlot0() external {
        assembly { sstore(src.slot, 0x0000000000000000000000000000000000000000000007ffffffffffffffffff) }
    }

    // Plant src slot 4 (i192=-1, u8b=3, b4=0xDEADBEEF, i8=-128, i16=-32000).
    function writeSrcSlot4() external {
        assembly { sstore(add(src.slot, 4), 0x830080deadbeef03ffffffffffffffffffffffffffffffffffffffffffffffff) }
    }

    // Plant src slot 5 (i128=-111 at bits 0-127, u72=99 at bits 128-199).
    function writeSrcSlot5() external {
        assembly { sstore(add(src.slot, 5), 0x00000000000000000000000000000063ffffffffffffffffffffffffffffff91) }
    }

    // Plant src slot 6 (u128=111 at bits 0-127).
    function writeSrcSlot6() external {
        assembly { sstore(add(src.slot, 6), 0x000000000000000000000000000000000000000000000000000000000000006f) }
    }

    // Plant src slot 7 (u224=5678 at bits 0-223).
    function writeSrcSlot7() external {
        assembly { sstore(add(src.slot, 7), 0x000000000000000000000000000000000000000000000000000000000000162e) }
    }

    // Plant src slot 8 (b8=0x0102030405060708 at bits 0-63, b19=0xaabb...1122 at bits 64-215).
    function writeSrcSlot8() external {
        assembly { sstore(add(src.slot, 8), 0x0000000000aabbccdd112233445566778899aabbccdd11220102030405060708) }
    }

    // Plant slot 0 with garbage in the upper bits (above u8 at bit 79) to verify
    // that genCleanupPackedStorageValue strips them on the storage→memory read path.
    function writeSrcSlot0Dirty() external {
        assembly { sstore(src.slot, 0xDEADBEEF00000000000000000000000000000000000007ffffffffffffffffff) }
    }

    // Raw memory slot for i72 (m+0x000): must be sign-extended int72.
    function i72SlotRaw() external view returns (uint256 raw) {
        S memory m = src;
        assembly { raw := mload(m) }
    }

    // Raw memory slot for i192 (m+0x080): must be sign-extended int192.
    function i192SlotRaw() external view returns (uint256 raw) {
        S memory m = src;
        assembly { raw := mload(add(m, 0x80)) }
    }

    // Raw memory slot for b4 (m+0x0c0): bytes4 must be left-aligned.
    function b4SlotRaw() external view returns (uint256 raw) {
        S memory m = src;
        assembly { raw := mload(add(m, 0xc0)) }
    }

    // Raw memory slot for i8 (m+0x0e0): must be sign-extended int8.
    function i8SlotRaw() external view returns (uint256 raw) {
        S memory m = src;
        assembly { raw := mload(add(m, 0xe0)) }
    }

    // Raw memory slot for i16 (m+0x100): must be sign-extended int16.
    function i16SlotRaw() external view returns (uint256 raw) {
        S memory m = src;
        assembly { raw := mload(add(m, 0x100)) }
    }

    // m+0x1a0: bytes8 b8=0x0102030405060708 → left-aligned, 24 trailing zero bytes.
    function b8SlotRaw() external view returns (uint256 raw) {
        S memory m = src;
        assembly { raw := mload(add(m, 0x1a0)) }
    }

    // m+0x1c0: bytes19 b19=0xaabb...1122 → left-aligned, 13 trailing zero bytes.
    function b19SlotRaw() external view returns (uint256 raw) {
        S memory m = src;
        assembly { raw := mload(add(m, 0x1c0)) }
    }

    // Full typed round-trip after storage→memory copy.
    function stoToMemTyped()
        external view
        returns (int72 i72, uint8 u8, int192 i192, uint8 u8b, bytes4 b4)
    {
        S memory m = src;
        return (m.i72, m.u8, m.i192, m.u8b, m.b4);
    }

    function stoToMemExt()
        external view
        returns (int8 i8, int16 i16, int128 i128,
                 uint72 u72, uint128 u128, uint224 u224,
                 bytes8 b8, bytes19 b19)
    {
        S memory m = src;
        return (m.i8, m.i16, m.i128, m.u72, m.u128, m.u224, m.b8, m.b19);
    }
}

// ====
// compileViaMlir: true
// ----
// writeSrcSlot0() ->
// writeSrcSlot4() ->
// writeSrcSlot5() ->
// writeSrcSlot6() ->
// writeSrcSlot7() ->
// writeSrcSlot8() ->
// i72SlotRaw() -> 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
// i192SlotRaw() -> 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
// b4SlotRaw() -> 0xdeadbeef00000000000000000000000000000000000000000000000000000000
// i8SlotRaw() -> 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff80
// i16SlotRaw() -> 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8300
// b8SlotRaw() -> 0x0102030405060708000000000000000000000000000000000000000000000000
// b19SlotRaw() -> 0xaabbccdd112233445566778899aabbccdd112200000000000000000000000000
// stoToMemTyped() -> -1, 7, -1, 3, left(0xDEADBEEF)
// stoToMemExt() -> -128, -32000, -111, 99, 111, 5678, left(0x0102030405060708), left(0xaabbccdd112233445566778899aabbccdd1122)
// writeSrcSlot0Dirty() ->
// i72SlotRaw() -> 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
// stoToMemTyped() -> -1, 7, -1, 3, left(0xDEADBEEF)
