// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Storage-to-storage copy of S.
// Writes all fields of src via writeSrc(), then copies dst = src and
// verifies that the packed slot layout is preserved exactly.
//
// S storage layout (key slots verified):
//   slot+0: i72 (bits 0-71) | u8 (bits 72-79)
//   slot+4: i192 (bits 0-191) | u8b (bits 192-199) | b4 (bits 200-231)
//           | i8 (bits 232-239) | i16 (bits 240-255)

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

contract StructCopyStoToSto {
    S src;
    S dst;

    function writeSrc() external {
        src.i72 = -1;
        src.u8 = 7;
        src.i192 = -1;
        src.u8b = 3;
        src.b4 = bytes4(hex"DEADBEEF");
        src.i8 = -128;
        src.i16 = -32000;
        src.i128 = -111;
        src.u72 = 99;
        src.u128 = 111;
        src.u224 = 5678;
        src.b8 = bytes8(hex"0102030405060708");
        src.b19 = bytes19(hex"aabbccdd112233445566778899aabbccdd1122");
    }

    function copySrcToDst() external { dst = src; }

    // Self-assignment: dst = dst must be a no-op (runtime guard: if srcAddr != dstAddr).
    function dstSelfAssign() external { dst = dst; }

    // Raw slot 0 (i72+u8 packed) after storage→storage copy.
    function readDstSlot0() external view returns (uint256 v) {
        assembly { v := sload(dst.slot) }
    }

    // Raw slot 4 (i192+u8b+b4+i8+i16 packed) after storage→storage copy.
    function readDstSlot4() external view returns (uint256 v) {
        assembly { v := sload(add(dst.slot, 4)) }
    }

    // Typed reads from dst.
    function readDstTyped()
        external view
        returns (int72 i72, uint8 u8, int192 i192, uint8 u8b, bytes4 b4)
    {
        return (dst.i72, dst.u8, dst.i192, dst.u8b, dst.b4);
    }

    function readDstExt()
        external view
        returns (int8 i8, int16 i16, int128 i128,
                 uint72 u72, uint128 u128, uint224 u224,
                 bytes8 b8, bytes19 b19)
    {
        return (dst.i8, dst.i16, dst.i128, dst.u72, dst.u128, dst.u224, dst.b8, dst.b19);
    }
}

// ====
// compileViaMlir: true
// ----
// writeSrc() ->
// copySrcToDst() ->
// readDstSlot0() -> 0x0000000000000000000000000000000000000000000007ffffffffffffffffff
// readDstSlot4() -> 0x830080deadbeef03ffffffffffffffffffffffffffffffffffffffffffffffff
// readDstTyped() -> -1, 7, -1, 3, left(0xDEADBEEF)
// readDstExt() -> -128, -32000, -111, 99, 111, 5678, left(0x0102030405060708), left(0xaabbccdd112233445566778899aabbccdd1122)
// dstSelfAssign() ->
// readDstSlot0() -> 0x0000000000000000000000000000000000000000000007ffffffffffffffffff
// readDstSlot4() -> 0x830080deadbeef03ffffffffffffffffffffffffffffffffffffffffffffffff
// readDstTyped() -> -1, 7, -1, 3, left(0xDEADBEEF)
// readDstExt() -> -128, -32000, -111, 99, 111, 5678, left(0x0102030405060708), left(0xaabbccdd112233445566778899aabbccdd1122)
