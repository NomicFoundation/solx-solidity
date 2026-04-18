// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ChainedAssign {
    uint u1;
    uint u2;
    uint u3;
    int  i1;
    int  i2;
    bool b1;
    bool b2;
    bytes4  bz4_1;
    bytes4  bz4_2;
    bytes32 bz32_1;
    bytes32 bz32_2;
    address a1;
    address a2;

    uint[3] fa1;
    uint[3] fa2;
    uint[2][3] fa2d1;
    uint[2][3] fa2d2;

    uint[]    da1;
    uint[]    da2;
    uint[2][] da2d1;
    uint[2][] da2d2;

    function chain_uint3(uint v) public returns (uint, uint, uint) {
        u1 = u2 = u3 = v;
        return (u1, u2, u3);
    }

    function chain_int2(int v) public returns (int, int) {
        i1 = i2 = v;
        return (i1, i2);
    }

    function chain_bool2(bool v) public returns (bool, bool) {
        b1 = b2 = v;
        return (b1, b2);
    }

    function chain_bz4(bytes4 v) public returns (bytes4, bytes4) {
        bz4_1 = bz4_2 = v;
        return (bz4_1, bz4_2);
    }

    function chain_bz32(bytes32 v) public returns (bytes32, bytes32) {
        bz32_1 = bz32_2 = v;
        return (bz32_1, bz32_2);
    }

    function chain_addr2(address v) public returns (address, address) {
        a1 = a2 = v;
        return (a1, a2);
    }

    function chain_local_uint3(uint v) public pure returns (uint, uint, uint) {
        uint la1;
        uint la2;
        uint la3;
        la1 = la2 = la3 = v;
        return (la1, la2, la3);
    }

    function chain_local_int2(int v) public pure returns (int, int) {
        int la1;
        int la2;
        la1 = la2 = v;
        return (la1, la2);
    }

    function chain_local_bool2(bool v) public pure returns (bool, bool) {
        bool la1;
        bool la2;
        la1 = la2 = v;
        return (la1, la2);
    }

    function chain_compound(uint v) public returns (uint, uint) {
        u1 = 10;
        u2 = 10;
        u1 = u2 += v;
        return (u1, u2);
    }

    function chain_fa(uint[3] memory src) public returns (uint[3] memory, uint[3] memory) {
        fa1 = fa2 = src;
        return (fa1, fa2);
    }

    function chain_fa_indep(uint[3] memory src) public returns (bool) {
        fa1 = fa2 = src;
        fa2[0] = 0;
        return fa1[0] == src[0];
    }

    function chain_fa2d(uint[2][3] memory src)
        public
        returns (uint[2][3] memory, uint[2][3] memory)
    {
        fa2d1 = fa2d2 = src;
        return (fa2d1, fa2d2);
    }

    function chain_fa2d_indep(uint[2][3] memory src) public returns (bool) {
        fa2d1 = fa2d2 = src;
        fa2d2[0][0] = 0;
        return fa2d1[0][0] == src[0][0];
    }

    function chain_da(uint[] memory src) public returns (uint[] memory) {
        da1 = da2 = src;
        return da1;
    }

    function get_da2() public view returns (uint[] memory) {
        return da2;
    }

    function chain_da_indep(uint[] memory src) public returns (bool) {
        da1 = da2 = src;
        da2[0] = 0;
        return da1[0] == src[0];
    }

    function chain_da2d(uint[2][] memory src) public returns (uint[2][] memory) {
        da2d1 = da2d2 = src;
        return da2d1;
    }

    function get_da2d2() public view returns (uint[2][] memory) {
        return da2d2;
    }

    function chain_da2d_indep(uint[2][] memory src) public returns (bool) {
        da2d1 = da2d2 = src;
        da2d2[0][0] = 0;
        return da2d1[0][0] == src[0][0];
    }

    function _add1(uint x) internal pure returns (uint) { return x + 1; }

    function chain_fnref_local(uint v) public pure returns (uint, uint) {
        function(uint) internal pure returns (uint) fp1;
        function(uint) internal pure returns (uint) fp2;
        fp1 = fp2 = _add1;
        return (fp1(v), fp2(v));
    }
}
// ====
// compileViaMlir: true
// ----
// chain_uint3(uint256): 42 -> 42, 42, 42
// chain_int2(int256): -7 -> -7, -7
// chain_bool2(bool): true -> true, true
// chain_bz4(bytes4): 0xdeadbeef00000000000000000000000000000000000000000000000000000000 -> 0xdeadbeef00000000000000000000000000000000000000000000000000000000, 0xdeadbeef00000000000000000000000000000000000000000000000000000000
// chain_bz32(bytes32): 0x1234000000000000000000000000000000000000000000000000000000000000 -> 0x1234000000000000000000000000000000000000000000000000000000000000, 0x1234000000000000000000000000000000000000000000000000000000000000
// chain_addr2(address): 0x1234567890123456789012345678901234567890 -> 0x1234567890123456789012345678901234567890, 0x1234567890123456789012345678901234567890
// chain_local_uint3(uint256): 99 -> 99, 99, 99
// chain_local_int2(int256): -1 -> -1, -1
// chain_local_bool2(bool): false -> false, false
// chain_compound(uint256): 5 -> 15, 15
// chain_fa(uint256[3]): 10, 20, 30 -> 10, 20, 30, 10, 20, 30
// chain_fa_indep(uint256[3]): 10, 20, 30 -> true
// chain_fa2d(uint256[2][3]): 1, 2, 3, 4, 5, 6 -> 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6
// chain_fa2d_indep(uint256[2][3]): 1, 2, 3, 4, 5, 6 -> true
// chain_da(uint256[]): 0x20, 3, 1, 2, 3 -> 0x20, 3, 1, 2, 3
// get_da2() -> 0x20, 3, 1, 2, 3
// chain_da_indep(uint256[]): 0x20, 3, 7, 8, 9 -> true
// chain_da2d(uint256[2][]): 0x20, 2, 10, 20, 30, 40 -> 0x20, 2, 10, 20, 30, 40
// get_da2d2() -> 0x20, 2, 10, 20, 30, 40
// chain_da2d_indep(uint256[2][]): 0x20, 2, 10, 20, 30, 40 -> true
// chain_fnref_local(uint256): 5 -> 6, 6
