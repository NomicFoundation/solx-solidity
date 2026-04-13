// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PackedArrayCopy {
    enum Status { Pending, Active, Done, Cancelled }

    uint8[33]   s8a;
    uint8[33]   s8b;
    uint8[]     d8a;
    uint8[]     d8b;

    uint16[17]  s16a;
    uint16[17]  s16b;
    uint16[]    d16a;
    uint16[]    d16b;

    uint32[9]   s32a;
    uint32[9]   s32b;
    uint32[]    d32a;
    uint32[]    d32b;

    uint64[5]   s64a;
    uint64[5]   s64b;
    uint64[]    d64a;
    uint64[]    d64b;

    uint128[3]  s128a;
    uint128[3]  s128b;
    uint128[]   d128a;
    uint128[]   d128b;

    uint256[4]  s256a;
    uint256[4]  s256b;
    uint256[]   d256a;
    uint256[]   d256b;

    uint72[2][2]  s72ss_a;
    uint72[2][2]  s72ss_b;
    uint72[2][]   s72ds;
    uint72[][2]   s72sd;
    uint72[][]    s72dd;
    uint8[4][2]   s8_2d;
    uint16[3][2]  s16_2d;
    uint32[3][2]  s32_2d;
    uint128[2][2] s128_2d;

    bytes3[5]  sb3s;
    bytes3[]   sb3d;
    bytes16[3] sb16s;
    bytes16[]  sb16d;
    bytes17[3] sb17s;
    bytes17[3] sb17b;
    bytes17[]  sb17d;
    bytes17[]  db17b;

    bytes3[4][2]  sb3ss;
    bytes16[2][2] sb16ss;
    bytes17[2][2] sb17ss;
    bytes3[][]    sb3dd;
    bytes3[2][]   sb3ds;
    bytes3[][2]   sb3sd;
    bytes16[][]   sb16dd;
    bytes16[2][]  sb16ds;
    bytes16[][2]  sb16sd;
    bytes17[][]   sb17dd;
    bytes17[2][]  sb17ds;
    bytes17[][2]  sb17sd;

    address[5]   sa_s;
    address[]    sa_d;
    address[3][2] sa_ss;
    address[][]   sa_dd;
    address[2][]  sa_ds;
    address[][2]  sa_sd;

    address payable[3] sp_s;
    address payable[]  sp_d;

    bool[33]    sb_s;
    bool[]      sb_d;
    bool[4][2]  sb_ss;
    bool[][]    sb_dd;
    bool[2][]   sb_ds;
    bool[][2]   sb_sd;

    Status[5]    se_s;
    Status[]     se_d;
    Status[3][2] se_ss;
    Status[][]   se_dd;
    Status[2][]  se_ds;
    Status[][2]  se_sd;

    function() external[3]   sf_s;
    function() external[]    sf_d;
    function() external[2][2] sf_ss;
    function() external[][]   sf_dd;
    function() external[2][]  sf_ds;
    function() external[][2]  sf_sd;


    function u8s_m2s(uint8[33] memory src) public returns (uint8[33] memory) {
        s8a = src;
        return s8a;
    }

    function u8s_cd2s(uint8[33] calldata src) external returns (uint8[33] memory) {
        s8a = src;
        return s8a;
    }

    function u8s_s2m(uint8[33] memory src) public returns (uint8[33] memory) {
        s8a = src;
        uint8[33] memory m = s8a;
        return m;
    }

    function u8s_s2s(uint8[33] memory src) public returns (uint8[33] memory) {
        s8a = src;
        s8b = s8a;
        s8a[0] = 0;
        return s8b;
    }


    function u8d_m2s(uint8[] memory src) public returns (uint8[] memory) {
        d8a = src;
        return d8a;
    }

    function u8d_cd2s(uint8[] calldata src) external returns (uint8[] memory) {
        d8a = src;
        return d8a;
    }

    function u8d_s2m(uint8[] memory src) public returns (uint8[] memory) {
        d8a = src;
        uint8[] memory m = d8a;
        return m;
    }

    function u8d_s2s(uint8[] memory src) public returns (uint8[] memory) {
        d8a = src;
        d8b = d8a;
        d8a[0] = 0;
        return d8b;
    }


    function u16s_m2s(uint16[17] memory src) public returns (uint16[17] memory) {
        s16a = src;
        return s16a;
    }

    function u16s_cd2s(uint16[17] calldata src) external returns (uint16[17] memory) {
        s16a = src;
        return s16a;
    }

    function u16s_s2m(uint16[17] memory src) public returns (uint16[17] memory) {
        s16a = src;
        uint16[17] memory m = s16a;
        return m;
    }

    function u16s_s2s(uint16[17] memory src) public returns (uint16[17] memory) {
        s16a = src;
        s16b = s16a;
        s16a[0] = 0;
        return s16b;
    }


    function u16d_m2s(uint16[] memory src) public returns (uint16[] memory) {
        d16a = src;
        return d16a;
    }

    function u16d_cd2s(uint16[] calldata src) external returns (uint16[] memory) {
        d16a = src;
        return d16a;
    }

    function u16d_s2m(uint16[] memory src) public returns (uint16[] memory) {
        d16a = src;
        uint16[] memory m = d16a;
        return m;
    }

    function u16d_s2s(uint16[] memory src) public returns (uint16[] memory) {
        d16a = src;
        d16b = d16a;
        d16a[0] = 0;
        return d16b;
    }


    function u32s_m2s(uint32[9] memory src) public returns (uint32[9] memory) {
        s32a = src;
        return s32a;
    }

    function u32s_cd2s(uint32[9] calldata src) external returns (uint32[9] memory) {
        s32a = src;
        return s32a;
    }

    function u32s_s2m(uint32[9] memory src) public returns (uint32[9] memory) {
        s32a = src;
        uint32[9] memory m = s32a;
        return m;
    }

    function u32s_s2s(uint32[9] memory src) public returns (uint32[9] memory) {
        s32a = src;
        s32b = s32a;
        s32a[0] = 0;
        return s32b;
    }


    function u32d_m2s(uint32[] memory src) public returns (uint32[] memory) {
        d32a = src;
        return d32a;
    }

    function u32d_cd2s(uint32[] calldata src) external returns (uint32[] memory) {
        d32a = src;
        return d32a;
    }

    function u32d_s2m(uint32[] memory src) public returns (uint32[] memory) {
        d32a = src;
        uint32[] memory m = d32a;
        return m;
    }

    function u32d_s2s(uint32[] memory src) public returns (uint32[] memory) {
        d32a = src;
        d32b = d32a;
        d32a[0] = 0;
        return d32b;
    }


    function u64s_m2s(uint64[5] memory src) public returns (uint64[5] memory) {
        s64a = src;
        return s64a;
    }

    function u64s_cd2s(uint64[5] calldata src) external returns (uint64[5] memory) {
        s64a = src;
        return s64a;
    }

    function u64s_s2m(uint64[5] memory src) public returns (uint64[5] memory) {
        s64a = src;
        uint64[5] memory m = s64a;
        return m;
    }

    function u64s_s2s(uint64[5] memory src) public returns (uint64[5] memory) {
        s64a = src;
        s64b = s64a;
        s64a[0] = 0;
        return s64b;
    }


    function u64d_m2s(uint64[] memory src) public returns (uint64[] memory) {
        d64a = src;
        return d64a;
    }

    function u64d_cd2s(uint64[] calldata src) external returns (uint64[] memory) {
        d64a = src;
        return d64a;
    }

    function u64d_s2m(uint64[] memory src) public returns (uint64[] memory) {
        d64a = src;
        uint64[] memory m = d64a;
        return m;
    }

    function u64d_s2s(uint64[] memory src) public returns (uint64[] memory) {
        d64a = src;
        d64b = d64a;
        d64a[0] = 0;
        return d64b;
    }


    function u128s_m2s(uint128[3] memory src) public returns (uint128[3] memory) {
        s128a = src;
        return s128a;
    }

    function u128s_cd2s(uint128[3] calldata src) external returns (uint128[3] memory) {
        s128a = src;
        return s128a;
    }

    function u128s_s2m(uint128[3] memory src) public returns (uint128[3] memory) {
        s128a = src;
        uint128[3] memory m = s128a;
        return m;
    }

    function u128s_s2s(uint128[3] memory src) public returns (uint128[3] memory) {
        s128a = src;
        s128b = s128a;
        s128a[0] = 0;
        return s128b;
    }


    function u128d_m2s(uint128[] memory src) public returns (uint128[] memory) {
        d128a = src;
        return d128a;
    }

    function u128d_cd2s(uint128[] calldata src) external returns (uint128[] memory) {
        d128a = src;
        return d128a;
    }

    function u128d_s2m(uint128[] memory src) public returns (uint128[] memory) {
        d128a = src;
        uint128[] memory m = d128a;
        return m;
    }

    function u128d_s2s(uint128[] memory src) public returns (uint128[] memory) {
        d128a = src;
        d128b = d128a;
        d128a[0] = 0;
        return d128b;
    }


    function u72ss(uint72[2][2] memory src) public returns (uint72[2][2] memory) {
        s72ss_a = src;
        return s72ss_a;
    }

    function u72ss_s2s(uint72[2][2] memory src) public returns (uint72[2][2] memory) {
        s72ss_a = src;
        s72ss_b = s72ss_a;
        s72ss_a[0][0] = 0;
        return s72ss_b;
    }

    function u72ds(uint72[2][] memory src) public returns (uint72[2][] memory) {
        s72ds = src;
        return s72ds;
    }

    function u72sd(uint72[][2] memory src) public returns (uint72[][2] memory) {
        s72sd = src;
        return s72sd;
    }

    function u72dd(uint72[][] memory src) public returns (uint72[][] memory) {
        s72dd = src;
        return s72dd;
    }

    function u8ss(uint8[4][2] memory src) public returns (uint8[4][2] memory) {
        s8_2d = src;
        return s8_2d;
    }

    function u16ss(uint16[3][2] memory src) public returns (uint16[3][2] memory) {
        s16_2d = src;
        return s16_2d;
    }

    function u32ss(uint32[3][2] memory src) public returns (uint32[3][2] memory) {
        s32_2d = src;
        return s32_2d;
    }

    function u128ss(uint128[2][2] memory src) public returns (uint128[2][2] memory) {
        s128_2d = src;
        return s128_2d;
    }


    function b3s_m2s(bytes3[5] memory src) public returns (bytes3[5] memory) {
        sb3s = src; return sb3s;
    }
    function b3s_s2m(bytes3[5] memory src) public returns (bytes3[5] memory) {
        sb3s = src;
        bytes3[5] memory m = sb3s;
        return m;
    }
    function b3d_m2s(bytes3[] memory src) public returns (bytes3[] memory) {
        sb3d = src; return sb3d;
    }
    function b3d_s2m(bytes3[] memory src) public returns (bytes3[] memory) {
        sb3d = src;
        bytes3[] memory m = sb3d;
        return m;
    }

    function b16s_m2s(bytes16[3] memory src) public returns (bytes16[3] memory) {
        sb16s = src; return sb16s;
    }
    function b16s_s2m(bytes16[3] memory src) public returns (bytes16[3] memory) {
        sb16s = src;
        bytes16[3] memory m = sb16s;
        return m;
    }
    function b16d_m2s(bytes16[] memory src) public returns (bytes16[] memory) {
        sb16d = src; return sb16d;
    }
    function b16d_s2m(bytes16[] memory src) public returns (bytes16[] memory) {
        sb16d = src;
        bytes16[] memory m = sb16d;
        return m;
    }

    function b17s_m2s(bytes17[3] memory src) public returns (bytes17[3] memory) {
        sb17s = src; return sb17s;
    }
    function b17s_s2m(bytes17[3] memory src) public returns (bytes17[3] memory) {
        sb17s = src;
        bytes17[3] memory m = sb17s;
        return m;
    }
    function b17d_m2s(bytes17[] memory src) public returns (bytes17[] memory) {
        sb17d = src; return sb17d;
    }
    function b17d_s2m(bytes17[] memory src) public returns (bytes17[] memory) {
        sb17d = src;
        bytes17[] memory m = sb17d;
        return m;
    }


    function b3ss(bytes3[4][2] memory src) public returns (bytes3[4][2] memory) {
        sb3ss = src; return sb3ss;
    }
    function b16ss(bytes16[2][2] memory src) public returns (bytes16[2][2] memory) {
        sb16ss = src; return sb16ss;
    }
    function b17ss(bytes17[2][2] memory src) public returns (bytes17[2][2] memory) {
        sb17ss = src; return sb17ss;
    }


    function b3ds(bytes3[2][] memory src) public returns (bytes3[2][] memory) {
        sb3ds = src; return sb3ds;
    }
    function b3sd(bytes3[][2] memory src) public returns (bytes3[][2] memory) {
        sb3sd = src; return sb3sd;
    }
    function b3dd(bytes3[][] memory src) public returns (bytes3[][] memory) {
        sb3dd = src; return sb3dd;
    }

    function b16ds(bytes16[2][] memory src) public returns (bytes16[2][] memory) {
        sb16ds = src; return sb16ds;
    }
    function b16sd(bytes16[][2] memory src) public returns (bytes16[][2] memory) {
        sb16sd = src; return sb16sd;
    }
    function b16dd(bytes16[][] memory src) public returns (bytes16[][] memory) {
        sb16dd = src; return sb16dd;
    }

    function b17ds(bytes17[2][] memory src) public returns (bytes17[2][] memory) {
        sb17ds = src; return sb17ds;
    }
    function b17sd(bytes17[][2] memory src) public returns (bytes17[][2] memory) {
        sb17sd = src; return sb17sd;
    }
    function b17dd(bytes17[][] memory src) public returns (bytes17[][] memory) {
        sb17dd = src; return sb17dd;
    }


    function u8d_shrink() public returns (uint8[] memory) {
        uint8[] memory big = new uint8[](4);
        big[0] = 1; big[1] = 2; big[2] = 3; big[3] = 4;
        d8a = big;
        uint8[] memory small = new uint8[](2);
        small[0] = 10; small[1] = 20;
        d8a = small;
        d8a.push();
        d8a.push();
        return d8a;
    }

    uint8[32] s8exact;
    function u8s_exact_slot(uint8[32] memory src)
        public returns (uint8[32] memory)
    {
        s8exact = src;
        return s8exact;
    }


    function a_s_m2s(address[5] memory src) public returns (address[5] memory) {
        sa_s = src; return sa_s;
    }
    function a_s_s2m(address[5] memory src) public returns (address[5] memory) {
        sa_s = src;
        address[5] memory m = sa_s;
        return m;
    }
    function a_d_m2s(address[] memory src) public returns (address[] memory) {
        sa_d = src; return sa_d;
    }
    function a_d_s2m(address[] memory src) public returns (address[] memory) {
        sa_d = src;
        address[] memory m = sa_d;
        return m;
    }
    function a_ss(address[3][2] memory src) public returns (address[3][2] memory) {
        sa_ss = src; return sa_ss;
    }
    function a_ds(address[2][] memory src) public returns (address[2][] memory) {
        sa_ds = src; return sa_ds;
    }
    function a_sd(address[][2] memory src) public returns (address[][2] memory) {
        sa_sd = src; return sa_sd;
    }
    function a_dd(address[][] memory src) public returns (address[][] memory) {
        sa_dd = src; return sa_dd;
    }


    function ap_s_m2s(address payable[3] memory src)
        public returns (address payable[3] memory)
    {
        sp_s = src; return sp_s;
    }
    function ap_d_m2s(address payable[] memory src)
        public returns (address payable[] memory)
    {
        sp_d = src; return sp_d;
    }


    function b_s_m2s(bool[33] memory src) public returns (bool[33] memory) {
        sb_s = src; return sb_s;
    }
    function b_s_s2m(bool[33] memory src) public returns (bool[33] memory) {
        sb_s = src;
        bool[33] memory m = sb_s;
        return m;
    }
    function b_d_m2s(bool[] memory src) public returns (bool[] memory) {
        sb_d = src; return sb_d;
    }
    function b_d_s2m(bool[] memory src) public returns (bool[] memory) {
        sb_d = src;
        bool[] memory m = sb_d;
        return m;
    }
    function b_ss(bool[4][2] memory src) public returns (bool[4][2] memory) {
        sb_ss = src; return sb_ss;
    }
    function b_ds(bool[2][] memory src) public returns (bool[2][] memory) {
        sb_ds = src; return sb_ds;
    }
    function b_sd(bool[][2] memory src) public returns (bool[][2] memory) {
        sb_sd = src; return sb_sd;
    }
    function b_dd(bool[][] memory src) public returns (bool[][] memory) {
        sb_dd = src; return sb_dd;
    }


    function e_s_m2s(Status[5] memory src) public returns (Status[5] memory) {
        se_s = src; return se_s;
    }
    function e_s_s2m(Status[5] memory src) public returns (Status[5] memory) {
        se_s = src;
        Status[5] memory m = se_s;
        return m;
    }
    function e_d_m2s(Status[] memory src) public returns (Status[] memory) {
        se_d = src; return se_d;
    }
    function e_d_s2m(Status[] memory src) public returns (Status[] memory) {
        se_d = src;
        Status[] memory m = se_d;
        return m;
    }
    function e_ss(Status[3][2] memory src) public returns (Status[3][2] memory) {
        se_ss = src; return se_ss;
    }
    function e_ds(Status[2][] memory src) public returns (Status[2][] memory) {
        se_ds = src; return se_ds;
    }
    function e_sd(Status[][2] memory src) public returns (Status[][2] memory) {
        se_sd = src; return se_sd;
    }
    function e_dd(Status[][] memory src) public returns (Status[][] memory) {
        se_dd = src; return se_dd;
    }


    function f_s_m2s(function() external[3] memory src)
        public returns (function() external[3] memory)
    {
        sf_s = src; return sf_s;
    }
    function f_s_s2m(function() external[3] memory src)
        public returns (function() external[3] memory)
    {
        sf_s = src;
        function() external[3] memory m = sf_s;
        return m;
    }
    function f_d_m2s(function() external[] memory src)
        public returns (function() external[] memory)
    {
        sf_d = src; return sf_d;
    }
    function f_d_s2m(function() external[] memory src)
        public returns (function() external[] memory)
    {
        sf_d = src;
        function() external[] memory m = sf_d;
        return m;
    }
    function f_ss(function() external[2][2] memory src)
        public returns (function() external[2][2] memory)
    {
        sf_ss = src; return sf_ss;
    }
    function f_ds(function() external[2][] memory src)
        public returns (function() external[2][] memory)
    {
        sf_ds = src; return sf_ds;
    }
    function f_sd(function() external[][2] memory src)
        public returns (function() external[][2] memory)
    {
        sf_sd = src; return sf_sd;
    }
    function f_dd(function() external[][] memory src)
        public returns (function() external[][] memory)
    {
        sf_dd = src; return sf_dd;
    }


    function b3s_cd2s(bytes3[5] calldata src) external returns (bytes3[5] memory) {
        sb3s = src; return sb3s;
    }
    function b3d_cd2s(bytes3[] calldata src) external returns (bytes3[] memory) {
        sb3d = src; return sb3d;
    }
    function b16s_cd2s(bytes16[3] calldata src) external returns (bytes16[3] memory) {
        sb16s = src; return sb16s;
    }
    function b16d_cd2s(bytes16[] calldata src) external returns (bytes16[] memory) {
        sb16d = src; return sb16d;
    }
    function b17s_cd2s(bytes17[3] calldata src) external returns (bytes17[3] memory) {
        sb17s = src; return sb17s;
    }
    function b17d_cd2s(bytes17[] calldata src) external returns (bytes17[] memory) {
        sb17d = src; return sb17d;
    }

    function b_s_cd2s(bool[33] calldata src) external returns (bool[33] memory) {
        sb_s = src; return sb_s;
    }
    function b_d_cd2s(bool[] calldata src) external returns (bool[] memory) {
        sb_d = src; return sb_d;
    }
    function e_s_cd2s(Status[5] calldata src) external returns (Status[5] memory) {
        se_s = src; return se_s;
    }
    function e_d_cd2s(Status[] calldata src) external returns (Status[] memory) {
        se_d = src; return se_d;
    }

    function u8s_cd2m(uint8[33] calldata src) external pure returns (uint8[33] memory) {
        return src;
    }
    function u8d_cd2m(uint8[] calldata src) external pure returns (uint8[] memory) {
        return src;
    }
    function u16s_cd2m(uint16[17] calldata src) external pure returns (uint16[17] memory) {
        return src;
    }
    function u16d_cd2m(uint16[] calldata src) external pure returns (uint16[] memory) {
        return src;
    }
    function u128s_cd2m(uint128[3] calldata src) external pure returns (uint128[3] memory) {
        return src;
    }
    function u128d_cd2m(uint128[] calldata src) external pure returns (uint128[] memory) {
        return src;
    }
    function b3s_cd2m(bytes3[5] calldata src) external pure returns (bytes3[5] memory) {
        return src;
    }
    function b3d_cd2m(bytes3[] calldata src) external pure returns (bytes3[] memory) {
        return src;
    }
    function b16s_cd2m(bytes16[3] calldata src) external pure returns (bytes16[3] memory) {
        return src;
    }
    function b16d_cd2m(bytes16[] calldata src) external pure returns (bytes16[] memory) {
        return src;
    }
    function b_s_cd2m(bool[33] calldata src) external pure returns (bool[33] memory) {
        return src;
    }
    function b_d_cd2m(bool[] calldata src) external pure returns (bool[] memory) {
        return src;
    }
    function e_s_cd2m(Status[5] calldata src) external pure returns (Status[5] memory) {
        return src;
    }
    function e_d_cd2m(Status[] calldata src) external pure returns (Status[] memory) {
        return src;
    }


    function u256s_m2s(uint256[4] memory src) public {
        s256a = src;
    }
    function u256s_cd2s(uint256[4] calldata src) external {
        s256a = src;
    }
    function u256s_s2s() public {
        s256b = s256a;
    }
    function u256s_s2m() public view returns (uint256[4] memory) {
        return s256a;
    }
    function u256s_get_a(uint256 i) public view returns (uint256) {
        return s256a[i];
    }
    function u256s_get_b(uint256 i) public view returns (uint256) {
        return s256b[i];
    }
    function u256s_cd2m(uint256[4] calldata src) external pure returns (uint256[4] memory) {
        return src;
    }

    function u256d_m2s(uint256[] memory src) public {
        d256a = src;
    }
    function u256d_cd2s(uint256[] calldata src) external {
        d256a = src;
    }
    function u256d_s2s() public {
        d256b = d256a;
    }
    function u256d_s2m() public view returns (uint256[] memory) {
        return d256a;
    }
    function u256d_get_a(uint256 i) public view returns (uint256) {
        return d256a[i];
    }
    function u256d_get_b(uint256 i) public view returns (uint256) {
        return d256b[i];
    }
    function u256d_cd2m(uint256[] calldata src) external pure returns (uint256[] memory) {
        return src;
    }
    function u256d_shrink(uint256[] memory full, uint256[] memory shortened) public {
        d256a = full;
        d256a = shortened;
    }
    function u256d_len_a() public view returns (uint256) {
        return d256a.length;
    }


    function b17s_get_a(uint256 i) public view returns (bytes17) {
        return sb17s[i];
    }
    function b17s_s2s() public {
        sb17b = sb17s;
    }
    function b17s_get_b(uint256 i) public view returns (bytes17) {
        return sb17b[i];
    }

    function b17d_get_a(uint256 i) public view returns (bytes17) {
        return sb17d[i];
    }
    function b17d_s2s() public {
        db17b = sb17d;
    }
    function b17d_get_b(uint256 i) public view returns (bytes17) {
        return db17b[i];
    }
    function b17d_shrink(bytes17[] memory full, bytes17[] memory shortened) public {
        sb17d = full;
        sb17d = shortened;
    }
    function b17d_len() public view returns (uint256) {
        return sb17d.length;
    }


    function b17s_cd2m(bytes17[2] calldata src) external pure returns (bytes17[2] memory) {
        return src;
    }
    function b17d_cd2m(bytes17[] calldata src) external pure returns (bytes17[] memory) {
        return src;
    }

    function a_s_cd2m(address[2] calldata src) external pure returns (address[2] memory) {
        return src;
    }
    function a_d_cd2m(address[] calldata src) external pure returns (address[] memory) {
        return src;
    }
}

// ====
// compileViaMlir: true
// ----
// u8s_m2s(uint8[33]): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 -> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
// u8s_cd2s(uint8[33]): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 -> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
// u8s_s2m(uint8[33]): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 -> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
// u8s_s2s(uint8[33]): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 -> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
// u8d_m2s(uint8[]): 0x20, 3, 11, 22, 33 -> 0x20, 3, 11, 22, 33
// u8d_cd2s(uint8[]): 0x20, 3, 11, 22, 33 -> 0x20, 3, 11, 22, 33
// u8d_s2m(uint8[]): 0x20, 3, 11, 22, 33 -> 0x20, 3, 11, 22, 33
// u8d_s2s(uint8[]): 0x20, 3, 11, 22, 33 -> 0x20, 3, 11, 22, 33
// u16s_m2s(uint16[17]): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 -> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
// u16s_cd2s(uint16[17]): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 -> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
// u16s_s2m(uint16[17]): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 -> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
// u16s_s2s(uint16[17]): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 -> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
// u16d_m2s(uint16[]): 0x20, 3, 100, 200, 300 -> 0x20, 3, 100, 200, 300
// u16d_cd2s(uint16[]): 0x20, 3, 100, 200, 300 -> 0x20, 3, 100, 200, 300
// u16d_s2m(uint16[]): 0x20, 3, 100, 200, 300 -> 0x20, 3, 100, 200, 300
// u16d_s2s(uint16[]): 0x20, 3, 100, 200, 300 -> 0x20, 3, 100, 200, 300
// u32s_m2s(uint32[9]): 1, 2, 3, 4, 5, 6, 7, 8, 9 -> 1, 2, 3, 4, 5, 6, 7, 8, 9
// u32s_cd2s(uint32[9]): 1, 2, 3, 4, 5, 6, 7, 8, 9 -> 1, 2, 3, 4, 5, 6, 7, 8, 9
// u32s_s2m(uint32[9]): 1, 2, 3, 4, 5, 6, 7, 8, 9 -> 1, 2, 3, 4, 5, 6, 7, 8, 9
// u32s_s2s(uint32[9]): 1, 2, 3, 4, 5, 6, 7, 8, 9 -> 1, 2, 3, 4, 5, 6, 7, 8, 9
// u32d_m2s(uint32[]): 0x20, 3, 0x11111111, 0x22222222, 0x33333333 -> 0x20, 3, 0x11111111, 0x22222222, 0x33333333
// u32d_cd2s(uint32[]): 0x20, 3, 0x11111111, 0x22222222, 0x33333333 -> 0x20, 3, 0x11111111, 0x22222222, 0x33333333
// u32d_s2m(uint32[]): 0x20, 3, 0x11111111, 0x22222222, 0x33333333 -> 0x20, 3, 0x11111111, 0x22222222, 0x33333333
// u32d_s2s(uint32[]): 0x20, 3, 0x11111111, 0x22222222, 0x33333333 -> 0x20, 3, 0x11111111, 0x22222222, 0x33333333
// u64s_m2s(uint64[5]): 1, 2, 3, 4, 5 -> 1, 2, 3, 4, 5
// u64s_cd2s(uint64[5]): 1, 2, 3, 4, 5 -> 1, 2, 3, 4, 5
// u64s_s2m(uint64[5]): 1, 2, 3, 4, 5 -> 1, 2, 3, 4, 5
// u64s_s2s(uint64[5]): 1, 2, 3, 4, 5 -> 1, 2, 3, 4, 5
// u64d_m2s(uint64[]): 0x20, 3, 0x1111111111111111, 0x2222222222222222, 0x3333333333333333 -> 0x20, 3, 0x1111111111111111, 0x2222222222222222, 0x3333333333333333
// u64d_cd2s(uint64[]): 0x20, 3, 0x1111111111111111, 0x2222222222222222, 0x3333333333333333 -> 0x20, 3, 0x1111111111111111, 0x2222222222222222, 0x3333333333333333
// u64d_s2m(uint64[]): 0x20, 3, 0x1111111111111111, 0x2222222222222222, 0x3333333333333333 -> 0x20, 3, 0x1111111111111111, 0x2222222222222222, 0x3333333333333333
// u64d_s2s(uint64[]): 0x20, 3, 0x1111111111111111, 0x2222222222222222, 0x3333333333333333 -> 0x20, 3, 0x1111111111111111, 0x2222222222222222, 0x3333333333333333
// u128s_m2s(uint128[3]): 1, 2, 3 -> 1, 2, 3
// u128s_cd2s(uint128[3]): 1, 2, 3 -> 1, 2, 3
// u128s_s2m(uint128[3]): 1, 2, 3 -> 1, 2, 3
// u128s_s2s(uint128[3]): 1, 2, 3 -> 1, 2, 3
// u128d_m2s(uint128[]): 0x20, 3, 1, 2, 3 -> 0x20, 3, 1, 2, 3
// u128d_cd2s(uint128[]): 0x20, 3, 1, 2, 3 -> 0x20, 3, 1, 2, 3
// u128d_s2m(uint128[]): 0x20, 3, 1, 2, 3 -> 0x20, 3, 1, 2, 3
// u128d_s2s(uint128[]): 0x20, 3, 1, 2, 3 -> 0x20, 3, 1, 2, 3
// u8d_shrink() -> 0x20, 4, 10, 20, 0, 0
// u8s_exact_slot(uint8[32]): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 -> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
// u72ss(uint72[2][2]): 1, 2, 3, 4 -> 1, 2, 3, 4
// u72ss_s2s(uint72[2][2]): 1, 2, 3, 4 -> 1, 2, 3, 4
// u72ds(uint72[2][]): 0x20, 2, 10, 20, 30, 40 -> 0x20, 2, 10, 20, 30, 40
// u72sd(uint72[][2]): 0x20, 0x40, 0xa0, 2, 100, 200, 2, 300, 400 -> 0x20, 0x40, 0xa0, 2, 100, 200, 2, 300, 400
// u72dd(uint72[][]): 0x20, 2, 0x40, 0xa0, 2, 10, 20, 2, 30, 40 -> 0x20, 2, 0x40, 0xa0, 2, 10, 20, 2, 30, 40
// u8ss(uint8[4][2]): 1, 2, 3, 4, 5, 6, 7, 8 -> 1, 2, 3, 4, 5, 6, 7, 8
// u16ss(uint16[3][2]): 100, 200, 300, 400, 500, 600 -> 100, 200, 300, 400, 500, 600
// u32ss(uint32[3][2]): 0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666 -> 0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666
// u128ss(uint128[2][2]): 1, 2, 3, 4 -> 1, 2, 3, 4
// b3s_m2s(bytes3[5]): left(0x010203), left(0x040506), left(0x070809), left(0x0a0b0c), left(0x0d0e0f) -> left(0x010203), left(0x040506), left(0x070809), left(0x0a0b0c), left(0x0d0e0f)
// b3s_s2m(bytes3[5]): left(0x010203), left(0x040506), left(0x070809), left(0x0a0b0c), left(0x0d0e0f) -> left(0x010203), left(0x040506), left(0x070809), left(0x0a0b0c), left(0x0d0e0f)
// b3d_m2s(bytes3[]): 0x20, 3, left(0xaabbcc), left(0xddeeff), left(0x112233) -> 0x20, 3, left(0xaabbcc), left(0xddeeff), left(0x112233)
// b3d_s2m(bytes3[]): 0x20, 3, left(0xaabbcc), left(0xddeeff), left(0x112233) -> 0x20, 3, left(0xaabbcc), left(0xddeeff), left(0x112233)
// b16s_m2s(bytes16[3]): left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30) -> left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30)
// b16s_s2m(bytes16[3]): left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30) -> left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30)
// b16d_m2s(bytes16[]): 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30) -> 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30)
// b16d_s2m(bytes16[]): 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30) -> 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30)
// b17s_m2s(bytes17[3]): left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031) -> left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031)
// b17s_s2m(bytes17[3]): left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031) -> left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031)
// b17d_m2s(bytes17[]): 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031) -> 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031)
// b17d_s2m(bytes17[]): 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031) -> 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031)
// b3ss(bytes3[4][2]): left(0x010203), left(0x040506), left(0x070809), left(0x0a0b0c), left(0x0d0e0f), left(0x101112), left(0x131415), left(0x161718) -> left(0x010203), left(0x040506), left(0x070809), left(0x0a0b0c), left(0x0d0e0f), left(0x101112), left(0x131415), left(0x161718)
// b16ss(bytes16[2][2]): left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30), left(0x3132333435363738393a3b3c3d3e3f40) -> left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30), left(0x3132333435363738393a3b3c3d3e3f40)
// b17ss(bytes17[2][2]): left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031), left(0x3132333435363738393a3b3c3d3e3f4041) -> left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031), left(0x3132333435363738393a3b3c3d3e3f4041)
// b3ds(bytes3[2][]): 0x20, 2, left(0x010203), left(0x040506), left(0x070809), left(0x0a0b0c) -> 0x20, 2, left(0x010203), left(0x040506), left(0x070809), left(0x0a0b0c)
// b3sd(bytes3[][2]): 0x20, 0x40, 0xa0, 2, left(0x010203), left(0x040506), 2, left(0x070809), left(0x0a0b0c) -> 0x20, 0x40, 0xa0, 2, left(0x010203), left(0x040506), 2, left(0x070809), left(0x0a0b0c)
// b3dd(bytes3[][]): 0x20, 2, 0x40, 0xa0, 2, left(0x010203), left(0x040506), 2, left(0x070809), left(0x0a0b0c) -> 0x20, 2, 0x40, 0xa0, 2, left(0x010203), left(0x040506), 2, left(0x070809), left(0x0a0b0c)
// b16ds(bytes16[2][]): 0x20, 2, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30), left(0x3132333435363738393a3b3c3d3e3f40) -> 0x20, 2, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30), left(0x3132333435363738393a3b3c3d3e3f40)
// b16sd(bytes16[][2]): 0x20, 0x40, 0xa0, 2, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), 2, left(0x2122232425262728292a2b2c2d2e2f30), left(0x3132333435363738393a3b3c3d3e3f40) -> 0x20, 0x40, 0xa0, 2, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), 2, left(0x2122232425262728292a2b2c2d2e2f30), left(0x3132333435363738393a3b3c3d3e3f40)
// b16dd(bytes16[][]): 0x20, 2, 0x40, 0xa0, 2, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), 2, left(0x2122232425262728292a2b2c2d2e2f30), left(0x3132333435363738393a3b3c3d3e3f40) -> 0x20, 2, 0x40, 0xa0, 2, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), 2, left(0x2122232425262728292a2b2c2d2e2f30), left(0x3132333435363738393a3b3c3d3e3f40)
// b17ds(bytes17[2][]): 0x20, 2, left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031), left(0x3132333435363738393a3b3c3d3e3f4041) -> 0x20, 2, left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031), left(0x3132333435363738393a3b3c3d3e3f4041)
// b17sd(bytes17[][2]): 0x20, 0x40, 0xa0, 2, left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), 2, left(0x2122232425262728292a2b2c2d2e2f3031), left(0x3132333435363738393a3b3c3d3e3f4041) -> 0x20, 0x40, 0xa0, 2, left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), 2, left(0x2122232425262728292a2b2c2d2e2f3031), left(0x3132333435363738393a3b3c3d3e3f4041)
// b17dd(bytes17[][]): 0x20, 2, 0x40, 0xa0, 2, left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), 2, left(0x2122232425262728292a2b2c2d2e2f3031), left(0x3132333435363738393a3b3c3d3e3f4041) -> 0x20, 2, 0x40, 0xa0, 2, left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), 2, left(0x2122232425262728292a2b2c2d2e2f3031), left(0x3132333435363738393a3b3c3d3e3f4041)
// a_s_m2s(address[5]): 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 0x3333333333333333333333333333333333333333, 0x4444444444444444444444444444444444444444, 0x5555555555555555555555555555555555555555 -> 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 0x3333333333333333333333333333333333333333, 0x4444444444444444444444444444444444444444, 0x5555555555555555555555555555555555555555
// a_s_s2m(address[5]): 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 0x3333333333333333333333333333333333333333, 0x4444444444444444444444444444444444444444, 0x5555555555555555555555555555555555555555 -> 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 0x3333333333333333333333333333333333333333, 0x4444444444444444444444444444444444444444, 0x5555555555555555555555555555555555555555
// a_d_m2s(address[]): 0x20, 3, 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 0x3333333333333333333333333333333333333333 -> 0x20, 3, 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 0x3333333333333333333333333333333333333333
// a_d_s2m(address[]): 0x20, 3, 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 0x3333333333333333333333333333333333333333 -> 0x20, 3, 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 0x3333333333333333333333333333333333333333
// a_ss(address[3][2]): 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 0x3333333333333333333333333333333333333333, 0x4444444444444444444444444444444444444444, 0x5555555555555555555555555555555555555555, 0x6666666666666666666666666666666666666666 -> 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 0x3333333333333333333333333333333333333333, 0x4444444444444444444444444444444444444444, 0x5555555555555555555555555555555555555555, 0x6666666666666666666666666666666666666666
// a_ds(address[2][]): 0x20, 2, 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 0x3333333333333333333333333333333333333333, 0x4444444444444444444444444444444444444444 -> 0x20, 2, 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 0x3333333333333333333333333333333333333333, 0x4444444444444444444444444444444444444444
// a_sd(address[][2]): 0x20, 0x40, 0xa0, 2, 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 2, 0x3333333333333333333333333333333333333333, 0x4444444444444444444444444444444444444444 -> 0x20, 0x40, 0xa0, 2, 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 2, 0x3333333333333333333333333333333333333333, 0x4444444444444444444444444444444444444444
// a_dd(address[][]): 0x20, 2, 0x40, 0xa0, 2, 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 2, 0x3333333333333333333333333333333333333333, 0x4444444444444444444444444444444444444444 -> 0x20, 2, 0x40, 0xa0, 2, 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222, 2, 0x3333333333333333333333333333333333333333, 0x4444444444444444444444444444444444444444
// ap_s_m2s(address[3]): 0xaaaa111111111111111111111111111111111111, 0xaaaa222222222222222222222222222222222222, 0xaaaa333333333333333333333333333333333333 -> 0xaaaa111111111111111111111111111111111111, 0xaaaa222222222222222222222222222222222222, 0xaaaa333333333333333333333333333333333333
// ap_d_m2s(address[]): 0x20, 3, 0xaaaa111111111111111111111111111111111111, 0xaaaa222222222222222222222222222222222222, 0xaaaa333333333333333333333333333333333333 -> 0x20, 3, 0xaaaa111111111111111111111111111111111111, 0xaaaa222222222222222222222222222222222222, 0xaaaa333333333333333333333333333333333333
// b_s_m2s(bool[33]): 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 -> true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true
// b_s_s2m(bool[33]): 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 -> true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true
// b_d_m2s(bool[]): 0x20, 5, 1, 0, 1, 0, 1 -> 0x20, 5, 1, 0, 1, 0, 1
// b_d_s2m(bool[]): 0x20, 5, 1, 0, 1, 0, 1 -> 0x20, 5, 1, 0, 1, 0, 1
// b_ss(bool[4][2]): 1, 0, 1, 0, 0, 1, 0, 1 -> 1, 0, 1, 0, 0, 1, 0, 1
// b_ds(bool[2][]): 0x20, 2, 1, 0, 0, 1 -> 0x20, 2, 1, 0, 0, 1
// b_sd(bool[][2]): 0x20, 0x40, 0xa0, 2, 1, 0, 2, 0, 1 -> 0x20, 0x40, 0xa0, 2, 1, 0, 2, 0, 1
// b_dd(bool[][]): 0x20, 2, 0x40, 0xa0, 2, 1, 0, 2, 0, 1 -> 0x20, 2, 0x40, 0xa0, 2, 1, 0, 2, 0, 1
// e_s_m2s(uint8[5]): 0, 1, 2, 3, 0 -> 0, 1, 2, 3, 0
// e_s_s2m(uint8[5]): 0, 1, 2, 3, 0 -> 0, 1, 2, 3, 0
// e_d_m2s(uint8[]): 0x20, 4, 0, 1, 2, 3 -> 0x20, 4, 0, 1, 2, 3
// e_d_s2m(uint8[]): 0x20, 4, 0, 1, 2, 3 -> 0x20, 4, 0, 1, 2, 3
// e_ss(uint8[3][2]): 0, 1, 2, 1, 2, 3 -> 0, 1, 2, 1, 2, 3
// e_ds(uint8[2][]): 0x20, 2, 0, 1, 2, 3 -> 0x20, 2, 0, 1, 2, 3
// e_sd(uint8[][2]): 0x20, 0x40, 0xa0, 2, 0, 1, 2, 2, 3 -> 0x20, 0x40, 0xa0, 2, 0, 1, 2, 2, 3
// e_dd(uint8[][]): 0x20, 2, 0x40, 0xa0, 2, 0, 1, 2, 2, 3 -> 0x20, 2, 0x40, 0xa0, 2, 0, 1, 2, 2, 3
// f_s_m2s(function[3]): left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd), left(0x3234567890abcdef3234567890abcdef32345678aabbccdd) -> left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd), left(0x3234567890abcdef3234567890abcdef32345678aabbccdd)
// f_s_s2m(function[3]): left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd), left(0x3234567890abcdef3234567890abcdef32345678aabbccdd) -> left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd), left(0x3234567890abcdef3234567890abcdef32345678aabbccdd)
// f_d_m2s(function[]): 0x20, 2, left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd) -> 0x20, 2, left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd)
// f_d_s2m(function[]): 0x20, 2, left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd) -> 0x20, 2, left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd)
// f_ss(function[2][2]): left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd), left(0x3234567890abcdef3234567890abcdef32345678aabbccdd), left(0x4234567890abcdef4234567890abcdef42345678aabbccdd) -> left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd), left(0x3234567890abcdef3234567890abcdef32345678aabbccdd), left(0x4234567890abcdef4234567890abcdef42345678aabbccdd)
// f_ds(function[2][]): 0x20, 2, left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd), left(0x3234567890abcdef3234567890abcdef32345678aabbccdd), left(0x4234567890abcdef4234567890abcdef42345678aabbccdd) -> 0x20, 2, left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd), left(0x3234567890abcdef3234567890abcdef32345678aabbccdd), left(0x4234567890abcdef4234567890abcdef42345678aabbccdd)
// f_sd(function[][2]): 0x20, 0x40, 0xa0, 2, left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd), 2, left(0x3234567890abcdef3234567890abcdef32345678aabbccdd), left(0x4234567890abcdef4234567890abcdef42345678aabbccdd) -> 0x20, 0x40, 0xa0, 2, left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd), 2, left(0x3234567890abcdef3234567890abcdef32345678aabbccdd), left(0x4234567890abcdef4234567890abcdef42345678aabbccdd)
// f_dd(function[][]): 0x20, 2, 0x40, 0xa0, 2, left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd), 2, left(0x3234567890abcdef3234567890abcdef32345678aabbccdd), left(0x4234567890abcdef4234567890abcdef42345678aabbccdd) -> 0x20, 2, 0x40, 0xa0, 2, left(0x1234567890abcdef1234567890abcdef12345678aabbccdd), left(0x2234567890abcdef2234567890abcdef22345678aabbccdd), 2, left(0x3234567890abcdef3234567890abcdef32345678aabbccdd), left(0x4234567890abcdef4234567890abcdef42345678aabbccdd)
// b3s_cd2s(bytes3[5]): left(0xaabbcc), left(0xddeeff), left(0x112233), left(0x445566), left(0x778899) -> left(0xaabbcc), left(0xddeeff), left(0x112233), left(0x445566), left(0x778899)
// b3d_cd2s(bytes3[]): 0x20, 3, left(0xaabbcc), left(0xddeeff), left(0x112233) -> 0x20, 3, left(0xaabbcc), left(0xddeeff), left(0x112233)
// b16s_cd2s(bytes16[3]): left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30) -> left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30)
// b16d_cd2s(bytes16[]): 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30) -> 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30)
// b17s_cd2s(bytes17[3]): left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031) -> left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031)
// b17d_cd2s(bytes17[]): 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031) -> 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f1011), left(0x1112131415161718191a1b1c1d1e1f2021), left(0x2122232425262728292a2b2c2d2e2f3031)
// b_s_cd2s(bool[33]): 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 -> true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true
// b_d_cd2s(bool[]): 0x20, 5, 1, 0, 1, 0, 1 -> 0x20, 5, 1, 0, 1, 0, 1
// e_s_cd2s(uint8[5]): 0, 1, 2, 3, 0 -> 0, 1, 2, 3, 0
// e_d_cd2s(uint8[]): 0x20, 4, 0, 1, 2, 3 -> 0x20, 4, 0, 1, 2, 3
// u8s_cd2m(uint8[33]): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 -> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
// u8d_cd2m(uint8[]): 0x20, 3, 11, 22, 33 -> 0x20, 3, 11, 22, 33
// u16s_cd2m(uint16[17]): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 -> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
// u16d_cd2m(uint16[]): 0x20, 3, 100, 200, 300 -> 0x20, 3, 100, 200, 300
// u128s_cd2m(uint128[3]): 1, 2, 3 -> 1, 2, 3
// u128d_cd2m(uint128[]): 0x20, 3, 1, 2, 3 -> 0x20, 3, 1, 2, 3
// b3s_cd2m(bytes3[5]): left(0xaabbcc), left(0xddeeff), left(0x112233), left(0x445566), left(0x778899) -> left(0xaabbcc), left(0xddeeff), left(0x112233), left(0x445566), left(0x778899)
// b3d_cd2m(bytes3[]): 0x20, 3, left(0xaabbcc), left(0xddeeff), left(0x112233) -> 0x20, 3, left(0xaabbcc), left(0xddeeff), left(0x112233)
// b16s_cd2m(bytes16[3]): left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30) -> left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30)
// b16d_cd2m(bytes16[]): 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30) -> 0x20, 3, left(0x0102030405060708090a0b0c0d0e0f10), left(0x1112131415161718191a1b1c1d1e1f20), left(0x2122232425262728292a2b2c2d2e2f30)
// b_s_cd2m(bool[33]): 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 -> true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true
// b_d_cd2m(bool[]): 0x20, 5, 1, 0, 1, 0, 1 -> 0x20, 5, 1, 0, 1, 0, 1
// e_s_cd2m(uint8[5]): 0, 1, 2, 3, 0 -> 0, 1, 2, 3, 0
// e_d_cd2m(uint8[]): 0x20, 4, 0, 1, 2, 3 -> 0x20, 4, 0, 1, 2, 3
// u256s_m2s(uint256[4]): 1, 2, 3, 4 ->
// u256s_s2m() -> 1, 2, 3, 4
// u256s_cd2s(uint256[4]): 10, 20, 30, 40 ->
// u256s_get_a(uint256): 0 -> 10
// u256s_get_a(uint256): 1 -> 20
// u256s_get_a(uint256): 2 -> 30
// u256s_get_a(uint256): 3 -> 40
// u256s_m2s(uint256[4]): 5, 6, 7, 8 ->
// u256s_s2s() ->
// u256s_get_b(uint256): 0 -> 5
// u256s_get_b(uint256): 1 -> 6
// u256s_get_b(uint256): 2 -> 7
// u256s_get_b(uint256): 3 -> 8
// u256s_cd2m(uint256[4]): 100, 200, 300, 400 -> 100, 200, 300, 400
// u256d_m2s(uint256[]): 0x20, 3, 100, 200, 300 ->
// u256d_s2m() -> 0x20, 3, 100, 200, 300
// u256d_cd2s(uint256[]): 0x20, 3, 11, 22, 33 ->
// u256d_s2s() ->
// u256d_get_b(uint256): 0 -> 11
// u256d_get_b(uint256): 1 -> 22
// u256d_get_b(uint256): 2 -> 33
// u256d_cd2m(uint256[]): 0x20, 3, 100, 200, 300 -> 0x20, 3, 100, 200, 300
// u256d_shrink(uint256[],uint256[]): 0x40, 0xc0, 3, 100, 200, 300, 1, 42 ->
// u256d_len_a() -> 1
// u256d_get_a(uint256): 0 -> 42
// u256d_shrink(uint256[],uint256[]): 0x40, 0xc0, 3, 100, 200, 300, 0 ->
// u256d_len_a() -> 0
// b17s_cd2s(bytes17[3]): left(0x112233445566778899aabbccddeeff0011), left(0x223344556677889900aabbccddeeff0022), left(0x334455667788990011aabbccddeeff0033) -> left(0x112233445566778899aabbccddeeff0011), left(0x223344556677889900aabbccddeeff0022), left(0x334455667788990011aabbccddeeff0033)
// b17s_get_a(uint256): 0 -> left(0x112233445566778899aabbccddeeff0011)
// b17s_get_a(uint256): 1 -> left(0x223344556677889900aabbccddeeff0022)
// b17s_get_a(uint256): 2 -> left(0x334455667788990011aabbccddeeff0033)
// b17s_m2s(bytes17[3]): left(0xaabbccddeeff00112233445566778899aa), left(0xbbccddee00112233445566778899aabbcc), left(0xccddee00112233445566778899aabbccdd) -> left(0xaabbccddeeff00112233445566778899aa), left(0xbbccddee00112233445566778899aabbcc), left(0xccddee00112233445566778899aabbccdd)
// b17s_s2s() ->
// b17s_get_b(uint256): 0 -> left(0xaabbccddeeff00112233445566778899aa)
// b17s_get_b(uint256): 1 -> left(0xbbccddee00112233445566778899aabbcc)
// b17s_get_b(uint256): 2 -> left(0xccddee00112233445566778899aabbccdd)
// b17d_cd2s(bytes17[]): 0x20, 3, left(0x112233445566778899aabbccddeeff0011), left(0x223344556677889900aabbccddeeff0022), left(0x334455667788990011aabbccddeeff0033) -> 0x20, 3, left(0x112233445566778899aabbccddeeff0011), left(0x223344556677889900aabbccddeeff0022), left(0x334455667788990011aabbccddeeff0033)
// b17d_s2s() ->
// b17d_get_b(uint256): 0 -> left(0x112233445566778899aabbccddeeff0011)
// b17d_get_b(uint256): 1 -> left(0x223344556677889900aabbccddeeff0022)
// b17d_get_b(uint256): 2 -> left(0x334455667788990011aabbccddeeff0033)
// b17d_shrink(bytes17[],bytes17[]): 0x40, 0xc0, 3, left(0xaabbccddeeff00112233445566778899aa), left(0xbbccddee00112233445566778899aabbcc), left(0xccddee00112233445566778899aabbccdd), 1, left(0xaabbccddeeff00112233445566778899aa) ->
// b17d_len() -> 1
// b17d_get_a(uint256): 0 -> left(0xaabbccddeeff00112233445566778899aa)
// b17d_shrink(bytes17[],bytes17[]): 0x40, 0xc0, 3, left(0xaabbccddeeff00112233445566778899aa), left(0xbbccddee00112233445566778899aabbcc), left(0xccddee00112233445566778899aabbccdd), 0 ->
// b17d_len() -> 0
// b17s_cd2m(bytes17[2]): left(0xaabbccddeeff00112233445566778899aa), left(0x112233445566778899aabbccddeeff1122) -> left(0xaabbccddeeff00112233445566778899aa), left(0x112233445566778899aabbccddeeff1122)
// b17d_cd2m(bytes17[]): 0x20, 2, left(0xaabbccddeeff00112233445566778899aa), left(0x112233445566778899aabbccddeeff1122) -> 0x20, 2, left(0xaabbccddeeff00112233445566778899aa), left(0x112233445566778899aabbccddeeff1122)
// a_s_cd2m(address[2]): 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222 -> 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222
// a_d_cd2m(address[]): 0x20, 2, 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222 -> 0x20, 2, 0x1111111111111111111111111111111111111111, 0x2222222222222222222222222222222222222222
