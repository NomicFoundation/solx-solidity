contract C {
    bytes s;
    bytes s2;
    bytes se;

    constructor(bytes memory a, bytes memory a2, bytes memory a3) {
        s = a;
        s2 = a2;
        se = a3;
    }
    function f(bytes memory a) public returns (bytes memory) {
        return bytes.concat(a, s);
    }
    function g(bytes memory a) public returns (bytes memory) {
        return bytes.concat(a, s2);
    }
    function h(bytes memory a) public returns (bytes memory) {
        return bytes.concat(a, s);
    }
    function j(bytes memory a) public returns (bytes memory) {
        bytes storage ref = s;
        return bytes.concat(a, ref, s);
    }
    function k(bytes memory a, string memory b) public returns (bytes memory) {
        return bytes.concat(a, bytes(b));
    }
    function strParam(string memory a) public returns (bytes memory) {
        return bytes.concat(bytes(a), s);
    }
}
// ====
// compileViaMlir: true
// ----
// constructor(): 0x60, 0xa0, 0x100, 5, "bcdef", 34, "abcdefghabcdefghabcdefghabcdefgh", "ab", 0 ->
// f(bytes): 0x20, 32, "abcdabcdabcdabcdabcdabcdabcdabcd" -> 0x20, 37, "abcdabcdabcdabcdabcdabcdabcdabcd", "bcdef"
// g(bytes): 0x20, 32, "abcdabcdabcdabcdabcdabcdabcdabcd" -> 0x20, 66, "abcdabcdabcdabcdabcdabcdabcdabcd", "abcdefghabcdefghabcdefghabcdefgh", "ab"
// h(bytes): 0x20, 32, "abcdabcdabcdabcdabcdabcdabcdabcd" -> 0x20, 37, "abcdabcdabcdabcdabcdabcdabcdabcd", "bcdef"
// j(bytes): 0x20, 32, "abcdabcdabcdabcdabcdabcdabcdabcd" -> 0x20, 42, "abcdabcdabcdabcdabcdabcdabcdabcd", "bcdefbcdef"
// k(bytes,string): 0x40, 0x80, 32, "abcdabcdabcdabcdabcdabcdabcdabcd", 5, "bcdef" -> 0x20, 37, "abcdabcdabcdabcdabcdabcdabcdabcd", "bcdef"
// strParam(string): 0x20, 32, "abcdabcdabcdabcdabcdabcdabcdabcd" -> 0x20, 37, "abcdabcdabcdabcdabcdabcdabcdabcd", "bcdef"
