// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ArrayCopy2D {
    // SS state vars
    uint[4][4] ss1;
    uint[4][4] ss2;

    // DS state vars
    uint[4][] ds1;
    uint[4][] ds2;

    // SD state vars
    uint[][4] sd1;
    uint[][4] sd2;

    // DD state vars
    uint[][] dd1;
    uint[][] dd2;

    // Sub-array state vars
    uint[4][] sub_ds;
    uint[][]  sub_dd;

    // Static-to-dynamic state vars
    uint[4]  std_s4;
    uint[]   std_d;
    uint[4][] std_ds;
    uint[][]  std_dd;

    // Storage → Memory: mutating storage must not affect the memory copy.
    function ss_s2m(uint[4][4] memory src) public returns (uint[4][4] memory) {
        ss1 = src;
        uint[4][4] memory m = ss1;
        ss1[0][0] = 0;
        return m;
    }

    // Storage → Storage: independent copy; mutating ss1 must not affect ss2.
    function ss_s2s(uint[4][4] memory src) public returns (uint[4][4] memory) {
        ss1 = src;
        ss2 = ss1;
        ss1[0][0] = 0;
        return ss2;
    }

    // Memory → Memory: alias — dst and src share the same memory block.
    function ss_m2m(uint[4][4] memory src) public pure returns (bool) {
        uint[4][4] memory dst = src;
        dst[0][0] = 99;
        return src[0][0] == 99;
    }

    // Memory → Storage
    function ss_m2s(uint[4][4] memory src) public returns (uint[4][4] memory) {
        ss1 = src;
        return ss1;
    }

    // CallData → Memory
    function ss_cd2m(uint[4][4] calldata src) external pure returns (uint[4][4] memory) {
        uint[4][4] memory m = src;
        return m;
    }

    // CallData → Storage
    function ss_cd2s(uint[4][4] calldata src) external returns (uint[4][4] memory) {
        ss1 = src;
        return ss1;
    }

    // Storage → Memory: mutating storage must not affect the memory copy.
    function ds_s2m(uint[4][] memory src) public returns (uint[4][] memory) {
        ds1 = src;
        uint[4][] memory m = ds1;
        ds1[0][0] = 0;
        return m;
    }

    // Storage → Storage: independent copy.
    function ds_s2s(uint[4][] memory src) public returns (uint[4][] memory) {
        ds1 = src;
        ds2 = ds1;
        ds1[0][0] = 0;
        return ds2;
    }

    // Memory → Memory: alias.
    function ds_m2m(uint[4][] memory src) public pure returns (bool) {
        uint[4][] memory dst = src;
        dst[0][0] = 99;
        return src[0][0] == 99;
    }

    // Memory → Storage
    function ds_m2s(uint[4][] memory src) public returns (uint[4][] memory) {
        ds1 = src;
        return ds1;
    }

    // CallData → Memory
    function ds_cd2m(uint[4][] calldata src) external pure returns (uint[4][] memory) {
        uint[4][] memory m = src;
        return m;
    }

    // CallData → Storage
    function ds_cd2s(uint[4][] calldata src) external returns (uint[4][] memory) {
        ds1 = src;
        return ds1;
    }

    // Storage → Memory: mutating storage must not affect the memory copy.
    function sd_s2m(uint[][4] memory src) public returns (uint[][4] memory) {
        sd1 = src;
        uint[][4] memory m = sd1;
        sd1[0][0] = 0;
        return m;
    }

    // Storage → Storage: independent copy.
    function sd_s2s(uint[][4] memory src) public returns (uint[][4] memory) {
        sd1 = src;
        sd2 = sd1;
        sd1[0][0] = 0;
        return sd2;
    }

    // Memory → Memory: alias — inner uint[] pointers are shared.
    function sd_m2m(uint[][4] memory src) public pure returns (bool) {
        uint[][4] memory dst = src;
        dst[0][0] = 99;
        return src[0][0] == 99;
    }

    // Memory → Storage
    function sd_m2s(uint[][4] memory src) public returns (uint[][4] memory) {
        sd1 = src;
        return sd1;
    }

    // CallData → Memory
    function sd_cd2m(uint[][4] calldata src) external pure returns (uint[][4] memory) {
        uint[][4] memory m = src;
        return m;
    }

    // CallData → Storage
    function sd_cd2s(uint[][4] calldata src) external returns (uint[][4] memory) {
        sd1 = src;
        return sd1;
    }

    // Storage → Memory: mutating storage must not affect the memory copy.
    function dd_s2m(uint[][] memory src) public returns (uint[][] memory) {
        dd1 = src;
        uint[][] memory m = dd1;
        dd1[0][0] = 0;
        return m;
    }

    // Storage → Storage: independent copy.
    function dd_s2s(uint[][] memory src) public returns (uint[][] memory) {
        dd1 = src;
        dd2 = dd1;
        dd1[0][0] = 0;
        return dd2;
    }

    // Memory → Memory: alias.
    function dd_m2m(uint[][] memory src) public pure returns (bool) {
        uint[][] memory dst = src;
        dst[0][0] = 99;
        return src[0][0] == 99;
    }

    // Memory → Storage
    function dd_m2s(uint[][] memory src) public returns (uint[][] memory) {
        dd1 = src;
        return dd1;
    }

    // CallData → Memory
    function dd_cd2m(uint[][] calldata src) external pure returns (uint[][] memory) {
        uint[][] memory m = src;
        return m;
    }

    // CallData → Storage
    function dd_cd2s(uint[][] calldata src) external returns (uint[][] memory) {
        dd1 = src;
        return dd1;
    }

    // uint[4] memory → DS storage[0]
    function sub_ds_m2s(uint[4] memory row) public returns (uint[4] memory) {
        sub_ds = new uint[4][](0);
        sub_ds.push();
        sub_ds[0] = row;
        return sub_ds[0];
    }

    // DS storage[1] ← DS storage[0]: independent copy.
    function sub_ds_s2s(uint[4] memory row) public returns (uint[4] memory) {
        sub_ds = new uint[4][](0);
        sub_ds.push(row);
        sub_ds.push([uint(0), 0, 0, 0]);
        sub_ds[1] = sub_ds[0];
        sub_ds[0][0] = 0;
        return sub_ds[1];
    }

    // uint[4] calldata → DS storage[0]
    function sub_ds_cd2s(uint[4] calldata row) external returns (uint[4] memory) {
        sub_ds = new uint[4][](0);
        sub_ds.push();
        sub_ds[0] = row;
        return sub_ds[0];
    }

    // DS storage[0] → memory uint[4]: storage mutation must not affect copy.
    function sub_ds_s2m(uint[4] memory row) public returns (uint[4] memory) {
        sub_ds = new uint[4][](0);
        sub_ds.push(row);
        uint[4] memory m = sub_ds[0];
        sub_ds[0][0] = 0;
        return m;
    }

    // uint[4] memory → DS memory[0]: alias.
    function sub_ds_m2m(uint[4] memory row) public pure returns (bool) {
        uint[4][] memory m = new uint[4][](1);
        m[0] = row;
        row[0] = 99;
        return m[0][0] == 99;
    }

    // DS calldata[0] → memory uint[4]
    function sub_ds_cd2m(uint[4][] calldata src) external pure returns (uint[4] memory) {
        uint[4] memory m = src[0];
        return m;
    }

    // uint[] memory → DD storage[0]
    function sub_dd_m2s(uint[] memory row) public returns (uint[] memory) {
        sub_dd = new uint[][](0);
        sub_dd.push();
        sub_dd[0] = row;
        return sub_dd[0];
    }

    // DD storage[1] ← DD storage[0]: independent copy.
    function sub_dd_s2s(uint[] memory row) public returns (uint[] memory) {
        sub_dd = new uint[][](0);
        sub_dd.push();
        sub_dd[0] = row;
        sub_dd.push();
        sub_dd[1] = sub_dd[0];
        sub_dd[0][0] = 0;
        return sub_dd[1];
    }

    // uint[] calldata → DD storage[0]
    function sub_dd_cd2s(uint[] calldata row) external returns (uint[] memory) {
        sub_dd = new uint[][](0);
        sub_dd.push();
        sub_dd[0] = row;
        return sub_dd[0];
    }

    // DD storage[0] → memory uint[]: storage mutation must not affect copy.
    function sub_dd_s2m(uint[] memory row) public returns (uint[] memory) {
        sub_dd = new uint[][](0);
        sub_dd.push();
        sub_dd[0] = row;
        uint[] memory m = sub_dd[0];
        sub_dd[0][0] = 0;
        return m;
    }

    // uint[] memory → DD memory[0]: alias.
    function sub_dd_m2m(uint[] memory row) public pure returns (bool) {
        uint[][] memory m = new uint[][](1);
        m[0] = row;
        row[0] = 99;
        return m[0][0] == 99;
    }

    // DD calldata[0] → memory uint[]
    function sub_dd_cd2m(uint[][] calldata src) external pure returns (uint[] memory) {
        uint[] memory m = src[0];
        return m;
    }

    // uint[4] storage → uint[] storage: Solidity resizes std_d to 4.
    function s2d_1d_s2s() public returns (uint[] memory) {
        std_s4 = [uint(1), 2, 3, 4];
        std_d = std_s4;
        return std_d;
    }

    // uint[4] memory → uint[] storage
    function s2d_1d_m2s(uint[4] memory src) public returns (uint[] memory) {
        std_d = src;
        return std_d;
    }

    // uint[4] calldata → uint[] storage
    function s2d_1d_cd2s(uint[4] calldata src) external returns (uint[] memory) {
        std_d = src;
        return std_d;
    }

    // uint[4] calldata → uint[4] memory (copy to memory, static type)
    function s2d_1d_cd2m(uint[4] calldata src) external pure returns (uint[4] memory) {
        uint[4] memory m = src;
        return m;
    }

    // uint[4] memory → uint[] memory: no implicit conversion; explicit loop.
    function s2d_1d_m2m_loop(uint[4] memory src) public pure returns (uint[] memory) {
        uint[] memory m = new uint[](src.length);
        for (uint i = 0; i < src.length; i++)
            m[i] = src[i];
        return m;
    }

    // DS storage row (uint[4]) → DD storage slot (uint[])
    function s2d_ds_row_s2s(uint[4] memory row) public returns (uint[] memory) {
        std_ds = new uint[4][](0);
        std_ds.push(row);
        std_dd.push();
        std_dd[0] = std_ds[0];
        return std_dd[0];
    }

    // DS memory row (uint[4]) → DD storage slot (uint[])
    function s2d_ds_row_m2s(uint[4] memory row) public returns (uint[] memory) {
        std_dd = new uint[][](0);
        std_dd.push();
        std_dd[0] = row;
        return std_dd[0];
    }

    // DS calldata row (uint[4]) → DD storage slot (uint[])
    function s2d_ds_row_cd2s(uint[4][] calldata ds) external returns (uint[] memory) {
        std_dd = new uint[][](0);
        std_dd.push();
        std_dd[0] = ds[0];
        return std_dd[0];
    }

    // DS calldata row (uint[4]) → memory uint[]
    function s2d_ds_row_cd2m(uint[4][] calldata ds) external pure returns (uint[4] memory) {
        uint[4] memory m = ds[0];
        return m;
    }

    // Storage → Storage (self-assignment): array must be unchanged.
    function ss_self(uint[4][4] memory src) public returns (uint[4][4] memory) {
        ss1 = src;
        ss1 = ss1;
        return ss1;
    }

    // Storage → Storage (self-assignment): array must be unchanged.
    function dd_self(uint[][] memory src) public returns (uint[][] memory) {
        dd1 = src;
        dd1 = dd1;
        return dd1;
    }
}
// ====
// compileViaMlir: true
// ----
// ss_s2m(uint256[4][4]): 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 -> 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
// ss_s2s(uint256[4][4]): 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 -> 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
// ss_m2m(uint256[4][4]): 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 -> true
// ss_m2s(uint256[4][4]): 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 -> 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
// ss_cd2m(uint256[4][4]): 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 -> 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
// ss_cd2s(uint256[4][4]): 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 -> 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
// ds_s2m(uint256[4][]): 0x20, 2, 1,2,3,4, 5,6,7,8 -> 0x20, 2, 1,2,3,4, 5,6,7,8
// ds_s2s(uint256[4][]): 0x20, 2, 1,2,3,4, 5,6,7,8 -> 0x20, 2, 1,2,3,4, 5,6,7,8
// ds_m2m(uint256[4][]): 0x20, 2, 1,2,3,4, 5,6,7,8 -> true
// ds_m2s(uint256[4][]): 0x20, 2, 1,2,3,4, 5,6,7,8 -> 0x20, 2, 1,2,3,4, 5,6,7,8
// ds_cd2m(uint256[4][]): 0x20, 2, 1,2,3,4, 5,6,7,8 -> 0x20, 2, 1,2,3,4, 5,6,7,8
// ds_cd2s(uint256[4][]): 0x20, 2, 1,2,3,4, 5,6,7,8 -> 0x20, 2, 1,2,3,4, 5,6,7,8
// sd_s2m(uint256[][4]): 32, 128, 256, 384, 416, 3,1,2,3, 3,4,5,6, 0, 1,7 -> 32, 128, 256, 384, 416, 3,1,2,3, 3,4,5,6, 0, 1,7
// sd_s2s(uint256[][4]): 32, 128, 256, 384, 416, 3,1,2,3, 3,4,5,6, 0, 1,7 -> 32, 128, 256, 384, 416, 3,1,2,3, 3,4,5,6, 0, 1,7
// sd_m2m(uint256[][4]): 32, 128, 256, 384, 416, 3,1,2,3, 3,4,5,6, 0, 1,7 -> true
// sd_m2s(uint256[][4]): 32, 128, 256, 384, 416, 3,1,2,3, 3,4,5,6, 0, 1,7 -> 32, 128, 256, 384, 416, 3,1,2,3, 3,4,5,6, 0, 1,7
// sd_cd2m(uint256[][4]): 32, 128, 256, 384, 416, 3,1,2,3, 3,4,5,6, 0, 1,7 -> 32, 128, 256, 384, 416, 3,1,2,3, 3,4,5,6, 0, 1,7
// sd_cd2s(uint256[][4]): 32, 128, 256, 384, 416, 3,1,2,3, 3,4,5,6, 0, 1,7 -> 32, 128, 256, 384, 416, 3,1,2,3, 3,4,5,6, 0, 1,7
// dd_s2m(uint256[][]): 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6 -> 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6
// dd_s2s(uint256[][]): 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6 -> 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6
// dd_m2m(uint256[][]): 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6 -> true
// dd_m2s(uint256[][]): 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6 -> 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6
// dd_cd2m(uint256[][]): 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6 -> 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6
// dd_cd2s(uint256[][]): 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6 -> 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6
// sub_ds_m2s(uint256[4]): 1,2,3,4 -> 1,2,3,4
// sub_ds_s2s(uint256[4]): 1,2,3,4 -> 1,2,3,4
// sub_ds_cd2s(uint256[4]): 1,2,3,4 -> 1,2,3,4
// sub_ds_s2m(uint256[4]): 1,2,3,4 -> 1,2,3,4
// sub_ds_m2m(uint256[4]): 1,2,3,4 -> true
// sub_ds_cd2m(uint256[4][]): 0x20, 2, 1,2,3,4, 5,6,7,8 -> 1,2,3,4
// sub_dd_m2s(uint256[]): 0x20, 3, 10,20,30 -> 0x20, 3, 10,20,30
// sub_dd_s2s(uint256[]): 0x20, 3, 10,20,30 -> 0x20, 3, 10,20,30
// sub_dd_cd2s(uint256[]): 0x20, 3, 10,20,30 -> 0x20, 3, 10,20,30
// sub_dd_s2m(uint256[]): 0x20, 3, 10,20,30 -> 0x20, 3, 10,20,30
// sub_dd_m2m(uint256[]): 0x20, 3, 10,20,30 -> true
// sub_dd_cd2m(uint256[][]): 32, 1, 32, 3, 10,20,30 -> 0x20, 3, 10,20,30
// s2d_1d_s2s() -> 0x20, 4, 1,2,3,4
// s2d_1d_m2s(uint256[4]): 1,2,3,4 -> 0x20, 4, 1,2,3,4
// s2d_1d_cd2s(uint256[4]): 1,2,3,4 -> 0x20, 4, 1,2,3,4
// s2d_1d_cd2m(uint256[4]): 1,2,3,4 -> 1,2,3,4
// s2d_1d_m2m_loop(uint256[4]): 1,2,3,4 -> 0x20, 4, 1,2,3,4
// s2d_ds_row_s2s(uint256[4]): 1,2,3,4 -> 0x20, 4, 1,2,3,4
// s2d_ds_row_m2s(uint256[4]): 1,2,3,4 -> 0x20, 4, 1,2,3,4
// s2d_ds_row_cd2s(uint256[4][]): 0x20, 1, 1,2,3,4 -> 0x20, 4, 1,2,3,4
// s2d_ds_row_cd2m(uint256[4][]): 0x20, 1, 1,2,3,4 -> 1,2,3,4
// ss_self(uint256[4][4]): 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 -> 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
// dd_self(uint256[][]): 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6 -> 32, 2, 64, 192, 3, 1,2,3, 3, 4,5,6
