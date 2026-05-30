// Verifies that the auto-generated public getter for a mapping-to-struct
// includes value types (uint256, int256, uint8, bool, address, bytes32,
// string, bytes) and excludes arrays (dynamic and fixed).
// Return type: (uint256,int256,uint8,bool,address,bytes32,string,bytes,uint256)
contract C {
    struct S {
        uint256    a;
        int256     isigned;
        uint8      small;
        bool       flag;
        address    addr;
        bytes32    b32;
        uint256[]  dynArr;
        string     s;
        bytes      bdata;
        uint256[2] fixArr;
        uint256    b;
    }

    mapping(uint256 => S) public m;

    function setA(uint256 key, uint256 val) public { m[key].a = val; }
    function setIsigned(uint256 key, int256 val) public { m[key].isigned = val; }
    function setSmall(uint256 key, uint8 val) public { m[key].small = val; }
    function setFlag(uint256 key, bool val) public { m[key].flag = val; }
    function setAddr(uint256 key, address val) public { m[key].addr = val; }
    function setB32(uint256 key, bytes32 val) public { m[key].b32 = val; }
    function setB(uint256 key, uint256 val) public { m[key].b = val; }
    function setStringHello(uint256 key) public { m[key].s = "hello"; }
}
// ====
// compileViaMlir: true
// ----
// m(uint256): 1 -> 0, 0, 0, 0, 0, 0, 0x120, 0x140, 0, 0, 0
// setA(uint256,uint256): 1, 42
// setIsigned(uint256,int256): 1, -7
// setSmall(uint256,uint8): 1, 255
// setFlag(uint256,bool): 1, 1
// setAddr(uint256,address): 1, 0x1234
// setB32(uint256,bytes32): 1, 0x41
// setB(uint256,uint256): 1, 99
// m(uint256): 1 -> 42, -7, 255, 1, 0x1234, 0x41, 0x120, 0x140, 99, 0, 0
// setStringHello(uint256): 1
// m(uint256): 1 -> 42, -7, 255, 1, 0x1234, 0x41, 0x120, 0x160, 99, 5, "hello", 0
