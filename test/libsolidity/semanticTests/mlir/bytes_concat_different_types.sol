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
    function fixedBytesParam(bytes16 b1, bytes15 b2, bytes31 b3) public returns (
        bytes memory,
        bytes memory,
        bytes memory,
        bytes memory
    ) {
        return (
            bytes.concat(b1, b2),
            bytes.concat(b1, b3),
            bytes.concat(b1, s),
            bytes.concat(b1, s)
        );
    }
    function fixedBytesParam2(bytes calldata c, bytes6 b1, bytes6 b2) public returns (bytes memory, bytes memory) {
        return (
            bytes.concat(s, b1, c),
            bytes.concat(b1, c, b2)
        );
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
// fixedBytesParam(bytes16,bytes15,bytes31):
//  "aabbccddeeffgghh",
//  "abcdefghabcdefg",
//  "0123456789012345678901234567890" ->
//  0x80, 0xc0, 0x120, 0x160,
//  31, "aabbccddeeffgghhabcdefghabcdefg",
//  47, "aabbccddeeffgghh0123456789012345", "678901234567890",
//  21, "aabbccddeeffgghhbcdef",
//  21, "aabbccddeeffgghhbcdef"
// fixedBytesParam2(bytes,bytes6,bytes6): 0x60, left(0x010203040506), left(0x0708090A0B0C), 20, left(0x1011121314151617181920212223242526272829) ->
//   0x40, 0x80,
//   31, left(0x62636465660102030405061011121314151617181920212223242526272829),
//   32, 0x01020304050610111213141516171819202122232425262728290708090A0B0C
// fixedBytesParam2(bytes,bytes6,bytes6): 0x60, left(0x01), left(0x02), 5, left(0x03) ->
//   0x40, 0x80,
//   16, left(0x6263646566010000000000030000000000),
//   17, left(0x010000000000030000000002000000000000)
