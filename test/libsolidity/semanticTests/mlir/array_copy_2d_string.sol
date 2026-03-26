// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression tests for the hasDynamicallySizedElt fix in GepOpLowering,
// DataLocCastOpLowering, and genCopy.
//
// The bug: isDynamicallySized(string[N]) returned false because string[N] is a
// *static* array type, even though it contains dynamically-sized strings.
// As a result, the fat-pointer reconstruction for calldata elements was skipped.
//
// The fix: replace isDynamicallySized with hasDynamicallySizedElt, which
// recurses into the element type and returns true for string[N].
//
// These tests specifically cover the case where the element type is string[N]
// or bytes[N] (both map to StringType in the Sol dialect):
//   string[2][3]  — static outer, static inner of strings
//   string[2][]   — dynamic outer, static inner of strings
//   bytes[2][3]   — static outer, static inner of bytes
//   bytes[2][]    — dynamic outer, static inner of bytes

contract ArrayCopy2DString {
    string[2][3] ss_st;
    string[2][]  ds_st;
    bytes[2][3]  bs_st;
    bytes[2][]   bd_st;

    // static outer (3), static inner (2 strings) — calldata → memory
    function ss_cd2m(string[2][3] calldata src) external pure returns (string[2][3] memory) {
        return src;
    }

    // static outer (3), static inner (2 strings) — calldata → storage → memory
    function ss_cd2s(string[2][3] calldata src) external returns (string[2][3] memory) {
        ss_st = src;
        return ss_st;
    }

    // dynamic outer, static inner (2 strings) — calldata → memory
    function ds_cd2m(string[2][] calldata src) external pure returns (string[2][] memory) {
        return src;
    }

    // dynamic outer, static inner (2 strings) — calldata → storage → memory
    function ds_cd2s(string[2][] calldata src) external returns (string[2][] memory) {
        ds_st = src;
        return ds_st;
    }

    // Access a single string[2] element from a dynamic outer array
    function ds_elt(string[2][] calldata src, uint i) external pure returns (string[2] memory) {
        return src[i];
    }

    // Assign a string[2] calldata value into slot i of the static outer storage array
    function ss_elt2s(string[2] calldata src, uint i) external returns (string[2] memory) {
        ss_st[i] = src;
        return ss_st[i];
    }

    // Assign a string[2] calldata value into slot 0 of the dynamic outer storage array
    function ds_elt2s(string[2] calldata src) external returns (string[2] memory) {
        if (ds_st.length == 0) ds_st.push();
        ds_st[0] = src;
        return ds_st[0];
    }

    // static outer (3), static inner (2 bytes) — calldata → memory
    function bs_cd2m(bytes[2][3] calldata src) external pure returns (bytes[2][3] memory) {
        return src;
    }

    // static outer (3), static inner (2 bytes) — calldata → storage → memory
    function bs_cd2s(bytes[2][3] calldata src) external returns (bytes[2][3] memory) {
        bs_st = src;
        return bs_st;
    }

    // dynamic outer, static inner (2 bytes) — calldata → memory
    function bd_cd2m(bytes[2][] calldata src) external pure returns (bytes[2][] memory) {
        return src;
    }

    // dynamic outer, static inner (2 bytes) — calldata → storage → memory
    function bd_cd2s(bytes[2][] calldata src) external returns (bytes[2][] memory) {
        bd_st = src;
        return bd_st;
    }

    // Access a single bytes[2] element from a dynamic outer array
    function bd_elt(bytes[2][] calldata src, uint i) external pure returns (bytes[2] memory) {
        return src[i];
    }

    // Assign a bytes[2] calldata value into slot i of the static outer storage array
    function bs_elt2s(bytes[2] calldata src, uint i) external returns (bytes[2] memory) {
        bs_st[i] = src;
        return bs_st[i];
    }

    // Assign a bytes[2] calldata value into slot 0 of the dynamic outer storage array
    function bd_elt2s(bytes[2] calldata src) external returns (bytes[2] memory) {
        if (bd_st.length == 0) bd_st.push();
        bd_st[0] = src;
        return bd_st[0];
    }
}
// ====
// compileViaMlir: true
// ----
// ss_cd2m(string[2][3]): 0x20, 96, 288, 480, 64, 128, 2, "aa", 2, "bb", 64, 128, 2, "cc", 2, "dd", 64, 128, 2, "ee", 2, "ff" -> 0x20, 96, 288, 480, 64, 128, 2, "aa", 2, "bb", 64, 128, 2, "cc", 2, "dd", 64, 128, 2, "ee", 2, "ff"
// ss_cd2s(string[2][3]): 0x20, 96, 288, 480, 64, 128, 2, "aa", 2, "bb", 64, 128, 2, "cc", 2, "dd", 64, 128, 2, "ee", 2, "ff" -> 0x20, 96, 288, 480, 64, 128, 2, "aa", 2, "bb", 64, 128, 2, "cc", 2, "dd", 64, 128, 2, "ee", 2, "ff"
// ds_cd2m(string[2][]): 0x20, 2, 64, 256, 64, 128, 2, "aa", 2, "bb", 64, 128, 2, "cc", 2, "dd" -> 0x20, 2, 64, 256, 64, 128, 2, "aa", 2, "bb", 64, 128, 2, "cc", 2, "dd"
// ds_cd2s(string[2][]): 0x20, 2, 64, 256, 64, 128, 2, "aa", 2, "bb", 64, 128, 2, "cc", 2, "dd" -> 0x20, 2, 64, 256, 64, 128, 2, "aa", 2, "bb", 64, 128, 2, "cc", 2, "dd"
// ds_elt(string[2][], uint256): 0x40, 0, 2, 64, 256, 64, 128, 2, "aa", 2, "bb", 64, 128, 2, "cc", 2, "dd" -> 0x20, 64, 128, 2, "aa", 2, "bb"
// ds_elt(string[2][], uint256): 0x40, 1, 2, 64, 256, 64, 128, 2, "aa", 2, "bb", 64, 128, 2, "cc", 2, "dd" -> 0x20, 64, 128, 2, "cc", 2, "dd"
// bs_cd2m(bytes[2][3]): 0x20, 96, 288, 480, 64, 128, 2, "ab", 2, "cd", 64, 128, 2, "ef", 2, "gh", 64, 128, 2, "ij", 2, "kl" -> 0x20, 96, 288, 480, 64, 128, 2, "ab", 2, "cd", 64, 128, 2, "ef", 2, "gh", 64, 128, 2, "ij", 2, "kl"
// bs_cd2s(bytes[2][3]): 0x20, 96, 288, 480, 64, 128, 2, "ab", 2, "cd", 64, 128, 2, "ef", 2, "gh", 64, 128, 2, "ij", 2, "kl" -> 0x20, 96, 288, 480, 64, 128, 2, "ab", 2, "cd", 64, 128, 2, "ef", 2, "gh", 64, 128, 2, "ij", 2, "kl"
// bd_cd2m(bytes[2][]): 0x20, 2, 64, 256, 64, 128, 2, "ab", 2, "cd", 64, 128, 2, "ef", 2, "gh" -> 0x20, 2, 64, 256, 64, 128, 2, "ab", 2, "cd", 64, 128, 2, "ef", 2, "gh"
// bd_cd2s(bytes[2][]): 0x20, 2, 64, 256, 64, 128, 2, "ab", 2, "cd", 64, 128, 2, "ef", 2, "gh" -> 0x20, 2, 64, 256, 64, 128, 2, "ab", 2, "cd", 64, 128, 2, "ef", 2, "gh"
// bd_elt(bytes[2][], uint256): 0x40, 0, 2, 64, 256, 64, 128, 2, "ab", 2, "cd", 64, 128, 2, "ef", 2, "gh" -> 0x20, 64, 128, 2, "ab", 2, "cd"
// bd_elt(bytes[2][], uint256): 0x40, 1, 2, 64, 256, 64, 128, 2, "ab", 2, "cd", 64, 128, 2, "ef", 2, "gh" -> 0x20, 64, 128, 2, "ef", 2, "gh"
// ss_elt2s(string[2], uint256): 0x40, 0, 64, 128, 2, "aa", 2, "bb" -> 0x20, 64, 128, 2, "aa", 2, "bb"
// ss_elt2s(string[2], uint256): 0x40, 2, 64, 128, 2, "cc", 2, "dd" -> 0x20, 64, 128, 2, "cc", 2, "dd"
// ds_elt2s(string[2]): 0x20, 64, 128, 2, "ee", 2, "ff" -> 0x20, 64, 128, 2, "ee", 2, "ff"
// bs_elt2s(bytes[2], uint256): 0x40, 0, 64, 128, 2, "ab", 2, "cd" -> 0x20, 64, 128, 2, "ab", 2, "cd"
// bs_elt2s(bytes[2], uint256): 0x40, 2, 64, 128, 2, "ef", 2, "gh" -> 0x20, 64, 128, 2, "ef", 2, "gh"
// bd_elt2s(bytes[2]): 0x20, 64, 128, 2, "ij", 2, "kl" -> 0x20, 64, 128, 2, "ij", 2, "kl"
