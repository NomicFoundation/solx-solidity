// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PackedBytesWidenCopy {
    bytes9[4]   s9s;
    bytes17[4]  s17s;

    bytes9[]    d9;
    bytes17[]   d17;

    bytes9[2][2]   s9_ss;
    bytes17[2][2]  s17_ss;

    bytes9[2][]    s9_ds;
    bytes17[2][]   s17_ds;

    bytes9[][2]    s9_sd;
    bytes17[][2]   s17_sd;

    bytes9[][]     s9_dd;
    bytes17[][]    s17_dd;

    bytes3[5]  s3_to8_s;
    bytes8[5]  s8_from3_s;

    bytes3[]   d3_to8;
    bytes8[]   d8_from3;

    bytes3[2][2]  s3_to8_ss;
    bytes8[2][2]  s8_from3_ss;

    bytes3[2][]   s3_to8_ds;
    bytes8[2][]   s8_from3_ds;


    function set_1s(bytes9[4] calldata src) public { s9s = src; }
    function copy_1s() public { s17s = s9s; }
    function get_1s(uint256 i) public view returns (bytes17) { return s17s[i]; }


    function set_1d(bytes9[] calldata src) public { d9 = src; }
    function copy_1d() public { d17 = d9; }
    function get_1d(uint256 i) public view returns (bytes17) { return d17[i]; }


    function set_2ss(bytes9 v00, bytes9 v01, bytes9 v10, bytes9 v11) public {
        s9_ss[0][0] = v00; s9_ss[0][1] = v01;
        s9_ss[1][0] = v10; s9_ss[1][1] = v11;
    }
    function copy_2ss() public { s17_ss = s9_ss; }
    function get_2ss(uint256 i, uint256 j) public view returns (bytes17) { return s17_ss[i][j]; }


    function set_2ds(bytes9 v00, bytes9 v01, bytes9 v10, bytes9 v11) public {
        delete s9_ds;
        s9_ds.push();
        s9_ds.push();
        s9_ds[0][0] = v00; s9_ds[0][1] = v01;
        s9_ds[1][0] = v10; s9_ds[1][1] = v11;
    }
    function copy_2ds() public { s17_ds = s9_ds; }
    function get_2ds(uint256 i, uint256 j) public view returns (bytes17) { return s17_ds[i][j]; }


    function set_2sd(bytes9 v00, bytes9 v01, bytes9 v10, bytes9 v11) public {
        delete s9_sd[0];
        delete s9_sd[1];
        s9_sd[0].push(v00); s9_sd[0].push(v01);
        s9_sd[1].push(v10); s9_sd[1].push(v11);
    }
    function copy_2sd() public { s17_sd = s9_sd; }
    function get_2sd(uint256 i, uint256 j) public view returns (bytes17) { return s17_sd[i][j]; }


    function set_2dd(bytes9 v00, bytes9 v01, bytes9 v10, bytes9 v11) public {
        delete s9_dd;
        s9_dd.push();
        s9_dd.push();
        s9_dd[0].push(v00); s9_dd[0].push(v01);
        s9_dd[1].push(v10); s9_dd[1].push(v11);
    }
    function copy_2dd() public { s17_dd = s9_dd; }
    function get_2dd(uint256 i, uint256 j) public view returns (bytes17) { return s17_dd[i][j]; }


    function set_b3b8_s(bytes3[5] calldata src) public {
        s3_to8_s = src;
    }
    function copy_b3b8_s() public {
        s8_from3_s = s3_to8_s;
    }
    function get_b3b8_s(uint256 i) public view returns (bytes8) {
        return s8_from3_s[i];
    }

    function set_b3b8_d(bytes3[] calldata src) public {
        d3_to8 = src;
    }
    function copy_b3b8_d() public {
        d8_from3 = d3_to8;
    }
    function get_b3b8_d(uint256 i) public view returns (bytes8) {
        return d8_from3[i];
    }

    function set_b3b8_ss(bytes3 v00, bytes3 v01, bytes3 v10, bytes3 v11) public {
        s3_to8_ss[0][0] = v00; s3_to8_ss[0][1] = v01;
        s3_to8_ss[1][0] = v10; s3_to8_ss[1][1] = v11;
    }
    function copy_b3b8_ss() public {
        s8_from3_ss = s3_to8_ss;
    }
    function get_b3b8_ss(uint256 i, uint256 j) public view returns (bytes8) {
        return s8_from3_ss[i][j];
    }

    function set_b3b8_ds(bytes3 v00, bytes3 v01, bytes3 v10, bytes3 v11) public {
        delete s3_to8_ds;
        s3_to8_ds.push();
        s3_to8_ds.push();
        s3_to8_ds[0][0] = v00; s3_to8_ds[0][1] = v01;
        s3_to8_ds[1][0] = v10; s3_to8_ds[1][1] = v11;
    }
    function copy_b3b8_ds() public {
        s8_from3_ds = s3_to8_ds;
    }
    function get_b3b8_ds(uint256 i, uint256 j) public view returns (bytes8) {
        return s8_from3_ds[i][j];
    }
}
// ====
// compileViaMlir: true
// ----
// set_1s(bytes9[4]): left(0x01), left(0x02), left(0x03), left(0x04) ->
// copy_1s() ->
// get_1s(uint256): 0 -> left(0x01)
// get_1s(uint256): 1 -> left(0x02)
// get_1s(uint256): 2 -> left(0x03)
// get_1s(uint256): 3 -> left(0x04)
// set_1d(bytes9[]): 0x20, 5, left(0x11), left(0x22), left(0x33), left(0x44), left(0x55) ->
// copy_1d() ->
// get_1d(uint256): 0 -> left(0x11)
// get_1d(uint256): 2 -> left(0x33)
// get_1d(uint256): 4 -> left(0x55)
// set_2ss(bytes9,bytes9,bytes9,bytes9): left(0x0a), left(0x0b), left(0x0c), left(0x0d) ->
// copy_2ss() ->
// get_2ss(uint256,uint256): 0, 0 -> left(0x0a)
// get_2ss(uint256,uint256): 0, 1 -> left(0x0b)
// get_2ss(uint256,uint256): 1, 0 -> left(0x0c)
// get_2ss(uint256,uint256): 1, 1 -> left(0x0d)
// set_2ds(bytes9,bytes9,bytes9,bytes9): left(0x10), left(0x20), left(0x30), left(0x40) ->
// copy_2ds() ->
// get_2ds(uint256,uint256): 0, 0 -> left(0x10)
// get_2ds(uint256,uint256): 0, 1 -> left(0x20)
// get_2ds(uint256,uint256): 1, 0 -> left(0x30)
// get_2ds(uint256,uint256): 1, 1 -> left(0x40)
// set_2sd(bytes9,bytes9,bytes9,bytes9): left(0x50), left(0x60), left(0x70), left(0x80) ->
// copy_2sd() ->
// get_2sd(uint256,uint256): 0, 0 -> left(0x50)
// get_2sd(uint256,uint256): 0, 1 -> left(0x60)
// get_2sd(uint256,uint256): 1, 0 -> left(0x70)
// get_2sd(uint256,uint256): 1, 1 -> left(0x80)
// set_2dd(bytes9,bytes9,bytes9,bytes9): left(0x90), left(0xa0), left(0xb0), left(0xc0) ->
// copy_2dd() ->
// get_2dd(uint256,uint256): 0, 0 -> left(0x90)
// get_2dd(uint256,uint256): 0, 1 -> left(0xa0)
// get_2dd(uint256,uint256): 1, 0 -> left(0xb0)
// get_2dd(uint256,uint256): 1, 1 -> left(0xc0)
// set_b3b8_s(bytes3[5]): left(0xaabbcc), left(0xddeeff), left(0x112233), left(0x445566), left(0x778899) ->
// copy_b3b8_s() ->
// get_b3b8_s(uint256): 0 -> left(0xaabbcc0000000000)
// get_b3b8_s(uint256): 1 -> left(0xddeeff0000000000)
// get_b3b8_s(uint256): 2 -> left(0x1122330000000000)
// get_b3b8_s(uint256): 4 -> left(0x7788990000000000)
// set_b3b8_d(bytes3[]): 0x20, 4, left(0xaabbcc), left(0xddeeff), left(0x112233), left(0x445566) ->
// copy_b3b8_d() ->
// get_b3b8_d(uint256): 0 -> left(0xaabbcc0000000000)
// get_b3b8_d(uint256): 3 -> left(0x4455660000000000)
// set_b3b8_ss(bytes3,bytes3,bytes3,bytes3): left(0x010203), left(0x040506), left(0x070809), left(0x0a0b0c) ->
// copy_b3b8_ss() ->
// get_b3b8_ss(uint256,uint256): 0, 0 -> left(0x0102030000000000)
// get_b3b8_ss(uint256,uint256): 0, 1 -> left(0x0405060000000000)
// get_b3b8_ss(uint256,uint256): 1, 0 -> left(0x0708090000000000)
// get_b3b8_ss(uint256,uint256): 1, 1 -> left(0x0a0b0c0000000000)
// set_b3b8_ds(bytes3,bytes3,bytes3,bytes3): left(0x111213), left(0x141516), left(0x171819), left(0x1a1b1c) ->
// copy_b3b8_ds() ->
// get_b3b8_ds(uint256,uint256): 0, 0 -> left(0x1112130000000000)
// get_b3b8_ds(uint256,uint256): 0, 1 -> left(0x1415160000000000)
// get_b3b8_ds(uint256,uint256): 1, 0 -> left(0x1718190000000000)
// get_b3b8_ds(uint256,uint256): 1, 1 -> left(0x1a1b1c0000000000)
