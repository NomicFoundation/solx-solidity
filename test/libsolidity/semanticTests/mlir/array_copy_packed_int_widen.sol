// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;


contract PackedIntWidenCopy {


    uint8[5]    s8s;
    uint160[5]  s160s;

    uint8[]     d8;
    uint160[]   d160;

    uint8[2][2]   s8_ss;
    uint160[2][2] s160_ss;

    uint8[2][]    s8_ds;
    uint160[2][]  s160_ds;

    uint8[][2]    s8_sd;
    uint160[][2]  s160_sd;

    uint8[][]     s8_dd;
    uint160[][]   s160_dd;

    uint8[5]   s8_64s;
    uint64[5]  s64_8s;

    uint8[]    d8_64;
    uint64[]   d64_8;

    uint8[2][2]  s8_64_ss;
    uint64[2][2] s64_8_ss;

    uint8[2][]   s8_64_ds;
    uint64[2][]  s64_8_ds;


    function set_1s(uint8[5] calldata src) public { s8s = src; }
    function copy_1s() public { s160s = s8s; }
    function get_1s(uint256 i) public view returns (uint160) { return s160s[i]; }


    function set_1d(uint8[] calldata src) public { d8 = src; }
    function copy_1d() public { d160 = d8; }
    function get_1d(uint256 i) public view returns (uint160) { return d160[i]; }


    function set_2ss() public {
        s8_ss[0][0] = 1; s8_ss[0][1] = 2;
        s8_ss[1][0] = 3; s8_ss[1][1] = 4;
    }
    function copy_2ss() public { s160_ss = s8_ss; }
    function get_2ss(uint256 i, uint256 j) public view returns (uint160) { return s160_ss[i][j]; }


    function set_2ds() public {
        delete s8_ds;
        s8_ds.push();
        s8_ds.push();
        s8_ds[0][0] = 5; s8_ds[0][1] = 6;
        s8_ds[1][0] = 7; s8_ds[1][1] = 8;
    }
    function copy_2ds() public { s160_ds = s8_ds; }
    function get_2ds(uint256 i, uint256 j) public view returns (uint160) { return s160_ds[i][j]; }


    function set_2sd() public {
        delete s8_sd[0];
        delete s8_sd[1];
        s8_sd[0].push(9);  s8_sd[0].push(10);
        s8_sd[1].push(11); s8_sd[1].push(12);
    }
    function copy_2sd() public { s160_sd = s8_sd; }
    function get_2sd(uint256 i, uint256 j) public view returns (uint160) { return s160_sd[i][j]; }


    function set_2dd() public {
        delete s8_dd;
        s8_dd.push();
        s8_dd.push();
        s8_dd[0].push(13); s8_dd[0].push(14);
        s8_dd[1].push(15); s8_dd[1].push(16);
    }
    function copy_2dd() public { s160_dd = s8_dd; }
    function get_2dd(uint256 i, uint256 j) public view returns (uint160) { return s160_dd[i][j]; }


    function set_64s(uint8[5] calldata src) public {
        s8_64s = src;
    }
    function copy_64s() public {
        s64_8s = s8_64s;
    }
    function get_64s(uint256 i) public view returns (uint64) {
        return s64_8s[i];
    }

    function set_64d(uint8[] calldata src) public {
        d8_64 = src;
    }
    function copy_64d() public {
        d64_8 = d8_64;
    }
    function get_64d(uint256 i) public view returns (uint64) {
        return d64_8[i];
    }

    function set_64ss() public {
        s8_64_ss[0][0] = 10; s8_64_ss[0][1] = 20;
        s8_64_ss[1][0] = 30; s8_64_ss[1][1] = 40;
    }
    function copy_64ss() public {
        s64_8_ss = s8_64_ss;
    }
    function get_64ss(uint256 i, uint256 j) public view returns (uint64) {
        return s64_8_ss[i][j];
    }

    function set_64ds() public {
        delete s8_64_ds;
        s8_64_ds.push();
        s8_64_ds.push();
        s8_64_ds[0][0] = 50; s8_64_ds[0][1] = 60;
        s8_64_ds[1][0] = 70; s8_64_ds[1][1] = 80;
    }
    function copy_64ds() public {
        s64_8_ds = s8_64_ds;
    }
    function get_64ds(uint256 i, uint256 j) public view returns (uint64) {
        return s64_8_ds[i][j];
    }
}
// ====
// compileViaMlir: true
// ----
// set_1s(uint8[5]): 10, 20, 30, 40, 50 ->
// copy_1s() ->
// get_1s(uint256): 0 -> 10
// get_1s(uint256): 1 -> 20
// get_1s(uint256): 2 -> 30
// get_1s(uint256): 3 -> 40
// get_1s(uint256): 4 -> 50
// set_1d(uint8[]): 0x20, 5, 1, 2, 3, 4, 5 ->
// copy_1d() ->
// get_1d(uint256): 0 -> 1
// get_1d(uint256): 4 -> 5
// set_2ss() ->
// copy_2ss() ->
// get_2ss(uint256,uint256): 0, 0 -> 1
// get_2ss(uint256,uint256): 0, 1 -> 2
// get_2ss(uint256,uint256): 1, 0 -> 3
// get_2ss(uint256,uint256): 1, 1 -> 4
// set_2ds() ->
// copy_2ds() ->
// get_2ds(uint256,uint256): 0, 0 -> 5
// get_2ds(uint256,uint256): 0, 1 -> 6
// get_2ds(uint256,uint256): 1, 0 -> 7
// get_2ds(uint256,uint256): 1, 1 -> 8
// set_2sd() ->
// copy_2sd() ->
// get_2sd(uint256,uint256): 0, 0 -> 9
// get_2sd(uint256,uint256): 0, 1 -> 10
// get_2sd(uint256,uint256): 1, 0 -> 11
// get_2sd(uint256,uint256): 1, 1 -> 12
// set_2dd() ->
// copy_2dd() ->
// get_2dd(uint256,uint256): 0, 0 -> 13
// get_2dd(uint256,uint256): 0, 1 -> 14
// get_2dd(uint256,uint256): 1, 0 -> 15
// get_2dd(uint256,uint256): 1, 1 -> 16
// set_64s(uint8[5]): 10, 20, 30, 40, 50 ->
// copy_64s() ->
// get_64s(uint256): 0 -> 10
// get_64s(uint256): 1 -> 20
// get_64s(uint256): 2 -> 30
// get_64s(uint256): 3 -> 40
// get_64s(uint256): 4 -> 50
// set_64d(uint8[]): 0x20, 5, 1, 2, 3, 4, 5 ->
// copy_64d() ->
// get_64d(uint256): 0 -> 1
// get_64d(uint256): 2 -> 3
// get_64d(uint256): 4 -> 5
// set_64ss() ->
// copy_64ss() ->
// get_64ss(uint256,uint256): 0, 0 -> 10
// get_64ss(uint256,uint256): 0, 1 -> 20
// get_64ss(uint256,uint256): 1, 0 -> 30
// get_64ss(uint256,uint256): 1, 1 -> 40
// set_64ds() ->
// copy_64ds() ->
// get_64ds(uint256,uint256): 0, 0 -> 50
// get_64ds(uint256,uint256): 0, 1 -> 60
// get_64ds(uint256,uint256): 1, 0 -> 70
// get_64ds(uint256,uint256): 1, 1 -> 80
