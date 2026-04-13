// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;


contract WidenTest {


    bytes16[]    sb16d;
    bytes16[4]   sb16s;
    bytes16[][]  sb16dd;
    bytes16[2][] sb16ds;

    uint16[]    su16d;
    uint16[3]   su16s;
    uint16[][]  su16dd;
    uint16[2][] su16ds;


    function cd_b16d(bytes15[] calldata src) public {
        sb16d = src;
    }
    function cd_b16s(bytes15[4] calldata src) public {
        sb16s = src;
    }
    function cd_b16dd(bytes15[][] calldata src) public {
        sb16dd = src;
    }
    function cd_b16ds(bytes15[2][] calldata src) public {
        sb16ds = src;
    }

    function cd_u16d(uint8[] calldata src) public {
        su16d = src;
    }
    function cd_u16s(uint8[3] calldata src) public {
        su16s = src;
    }
    function cd_u16dd(uint8[][] calldata src) public {
        su16dd = src;
    }
    function cd_u16ds(uint8[2][] calldata src) public {
        su16ds = src;
    }


    function m_b16d(bytes15[] memory src) public {
        sb16d = src;
    }
    function m_u16d(uint8[] memory src) public {
        su16d = src;
    }


    function get_b16d(uint256 i) public view returns (bytes16) {
        return sb16d[i];
    }
    function get_b16s(uint256 i) public view returns (bytes16) {
        return sb16s[i];
    }
    function get_b16dd(uint256 i, uint256 j) public view returns (bytes16) {
        return sb16dd[i][j];
    }
    function get_b16ds(uint256 i, uint256 j) public view returns (bytes16) {
        return sb16ds[i][j];
    }

    function get_u16d(uint256 i) public view returns (uint16) {
        return su16d[i];
    }
    function get_u16s(uint256 i) public view returns (uint16) {
        return su16s[i];
    }
    function get_u16dd(uint256 i, uint256 j) public view returns (uint16) {
        return su16dd[i][j];
    }
    function get_u16ds(uint256 i, uint256 j) public view returns (uint16) {
        return su16ds[i][j];
    }
}
// ====
// compileViaMlir: true
// ----
// cd_b16d(bytes15[]): 0x20, 2, left(0xffffffffffffffffffffffffffffff), left(0x010203040506070809101112131415) ->
// get_b16d(uint256): 0 -> left(0xffffffffffffffffffffffffffffff00)
// get_b16d(uint256): 1 -> left(0x01020304050607080910111213141500)
// cd_b16s(bytes15[4]): left(0xffffffffffffffffffffffffffffff), left(0x010203040506070809101112131415), left(0xaabbccddeeff001122334455667788), left(0x00) ->
// get_b16s(uint256): 0 -> left(0xffffffffffffffffffffffffffffff00)
// get_b16s(uint256): 1 -> left(0x01020304050607080910111213141500)
// get_b16s(uint256): 2 -> left(0xaabbccddeeff00112233445566778800)
// get_b16s(uint256): 3 -> left(0x00)
// cd_b16dd(bytes15[][]): 0x20, 2, 0x40, 0xa0, 2, left(0xffffffffffffffffffffffffffffff), left(0x010203040506070809101112131415), 2, left(0xaabbccddeeff001122334455667788), left(0x00) ->
// get_b16dd(uint256,uint256): 0, 0 -> left(0xffffffffffffffffffffffffffffff00)
// get_b16dd(uint256,uint256): 0, 1 -> left(0x01020304050607080910111213141500)
// get_b16dd(uint256,uint256): 1, 0 -> left(0xaabbccddeeff00112233445566778800)
// get_b16dd(uint256,uint256): 1, 1 -> left(0x00)
// cd_b16ds(bytes15[2][]): 0x20, 2, left(0xffffffffffffffffffffffffffffff), left(0x010203040506070809101112131415), left(0xaabbccddeeff001122334455667788), left(0x00) ->
// get_b16ds(uint256,uint256): 0, 0 -> left(0xffffffffffffffffffffffffffffff00)
// get_b16ds(uint256,uint256): 0, 1 -> left(0x01020304050607080910111213141500)
// get_b16ds(uint256,uint256): 1, 0 -> left(0xaabbccddeeff00112233445566778800)
// get_b16ds(uint256,uint256): 1, 1 -> left(0x00)
// cd_u16d(uint8[]): 0x20, 3, 10, 20, 255 ->
// get_u16d(uint256): 0 -> 10
// get_u16d(uint256): 1 -> 20
// get_u16d(uint256): 2 -> 255
// cd_u16s(uint8[3]): 10, 20, 255 ->
// get_u16s(uint256): 0 -> 10
// get_u16s(uint256): 1 -> 20
// get_u16s(uint256): 2 -> 255
// cd_u16dd(uint8[][]): 0x20, 2, 0x40, 0xa0, 2, 1, 2, 2, 10, 20 ->
// get_u16dd(uint256,uint256): 0, 0 -> 1
// get_u16dd(uint256,uint256): 0, 1 -> 2
// get_u16dd(uint256,uint256): 1, 0 -> 10
// get_u16dd(uint256,uint256): 1, 1 -> 20
// cd_u16ds(uint8[2][]): 0x20, 2, 1, 2, 10, 20 ->
// get_u16ds(uint256,uint256): 0, 0 -> 1
// get_u16ds(uint256,uint256): 0, 1 -> 2
// get_u16ds(uint256,uint256): 1, 0 -> 10
// get_u16ds(uint256,uint256): 1, 1 -> 20
// m_b16d(bytes15[]): 0x20, 2, left(0xffffffffffffffffffffffffffffff), left(0x010203040506070809101112131415) ->
// get_b16d(uint256): 0 -> left(0xffffffffffffffffffffffffffffff00)
// get_b16d(uint256): 1 -> left(0x01020304050607080910111213141500)
// m_u16d(uint8[]): 0x20, 3, 10, 20, 255 ->
// get_u16d(uint256): 0 -> 10
// get_u16d(uint256): 1 -> 20
// get_u16d(uint256): 2 -> 255
