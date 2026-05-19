// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Calldata-to-storage copy of S.
// After `src = s` (calldata S → storage), verifies that the packed storage
// slot layout matches the expected bit pattern.
//
// S storage layout:
//   slot+0: i72 (int72, bits 0-71) | u8 (uint8, bits 72-79)
//   slot+1: flags outer length
//   slot+2: addrs[0]
//   slot+3: addrs[1]
//   slot+4: i192 (bits 0-191) | u8b (bits 192-199) | b4 (bits 200-231) | i8 (bits 232-239) | i16 (bits 240-255)
//   slot+5: i128 (bits 0-127) | u72 (bits 128-199)
//   slot+6: u128 (bits 0-127)
//   slot+7: u224 (bits 0-223)
//   slot+8: b8 (bits 0-63) | b19 (bits 64-215)
//
// For i72=-1 (int72), u8=7:
//   slot+0 = 0x0000000000000000000000000000000000000000000007ffffffffffffffffff
//
// For i192=-1, u8b=3, b4=0xDEADBEEF, i8=-128 (0x80), i16=-32000 (0x8300):
//   slot+4 = 0x830080deadbeef03ffffffffffffffffffffffffffffffffffffffffffffffff

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

contract StructCopyCdToStorage {
    S src;

    function storeS(S calldata s) external { src = s; }

    // Raw slot 0 (i72+u8 packed) after calldata→storage write.
    function readSrcSlot0() external view returns (uint256 v) {
        assembly { v := sload(src.slot) }
    }

    // Raw slot 4 (i192+u8b+b4+i8+i16 packed) after calldata→storage write.
    function readSrcSlot4() external view returns (uint256 v) {
        assembly { v := sload(add(src.slot, 4)) }
    }

    // Typed reads from src for additional assurance.
    function readSrcTyped()
        external view
        returns (int72 i72, uint8 u8, int192 i192, uint8 u8b, bytes4 b4)
    {
        return (src.i72, src.u8, src.i192, src.u8b, src.b4);
    }

    function readSrcExt()
        external view
        returns (int8 i8, int16 i16, int128 i128,
                 uint72 u72, uint128 u128, uint224 u224,
                 bytes8 b8, bytes19 b19)
    {
        return (src.i8, src.i16, src.i128, src.u72, src.u128, src.u224, src.b8, src.b19);
    }
}

// ====
// compileViaMlir: true
// ----
// storeS((int72,uint8,bool[][],address[2],int192,uint8,bytes4,int8,int16,int128,uint72,uint128,uint224,bytes8,bytes19)): 0x20, -1, 7, 0x200, 0, 0, -1, 3, left(0xDEADBEEF), -128, -32000, -111, 99, 111, 5678, left(0x0102030405060708), left(0xaabbccdd112233445566778899aabbccdd1122), 0 ->
// readSrcSlot0() -> 0x0000000000000000000000000000000000000000000007ffffffffffffffffff
// readSrcSlot4() -> 0x830080deadbeef03ffffffffffffffffffffffffffffffffffffffffffffffff
// readSrcTyped() -> -1, 7, -1, 3, left(0xDEADBEEF)
// readSrcExt() -> -128, -32000, -111, 99, 111, 5678, left(0x0102030405060708), left(0xaabbccdd112233445566778899aabbccdd1122)
// storeS((int72,uint8,bool[][],address[2],int192,uint8,bytes4,int8,int16,int128,uint72,uint128,uint224,bytes8,bytes19)): 0x20, 0x0000000000000000000000000000000000000000000000ffffffffffffffffff, 7, 0x200, 0, 0, -1, 3, left(0xDEADBEEF), -128, -32000, -111, 99, 111, 5678, left(0x0102030405060708), left(0xaabbccdd112233445566778899aabbccdd1122), 0 -> FAILURE
// storeS((int72,uint8,bool[][],address[2],int192,uint8,bytes4,int8,int16,int128,uint72,uint128,uint224,bytes8,bytes19)): 0x20, -1, 0x107, 0x200, 0, 0, -1, 3, left(0xDEADBEEF), -128, -32000, -111, 99, 111, 5678, left(0x0102030405060708), left(0xaabbccdd112233445566778899aabbccdd1122), 0 -> FAILURE
// storeS((int72,uint8,bool[][],address[2],int192,uint8,bytes4,int8,int16,int128,uint72,uint128,uint224,bytes8,bytes19)): 0x20, -1, 7, 0x200, 0, 0, -1, 3, 0xdeadbeef00000000000000000000000000000000000000000000000000000001, -128, -32000, -111, 99, 111, 5678, left(0x0102030405060708), left(0xaabbccdd112233445566778899aabbccdd1122), 0 -> FAILURE
// storeS((int72,uint8,bool[][],address[2],int192,uint8,bytes4,int8,int16,int128,uint72,uint128,uint224,bytes8,bytes19)): 0x20, -1, 7, 0x200, 0x0000000000000000000000010000000000000000000000000000000000000001, 0, -1, 3, left(0xDEADBEEF), -128, -32000, -111, 99, 111, 5678, left(0x0102030405060708), left(0xaabbccdd112233445566778899aabbccdd1122), 0 -> FAILURE
// storeS((int72,uint8,bool[][],address[2],int192,uint8,bytes4,int8,int16,int128,uint72,uint128,uint224,bytes8,bytes19)): 0x20, -1, 7, 0x200, 0, 0, -1, 3, left(0xDEADBEEF), -128, 0x0000000000000000000000000000000000000000000000000000000000008300, -111, 99, 111, 5678, left(0x0102030405060708), left(0xaabbccdd112233445566778899aabbccdd1122), 0 -> FAILURE
