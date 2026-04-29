contract C {
    uint256[4] s;
    uint256[] d;

    function init() public {
        s[0] = 11;
        s[1] = 22;
        s[2] = 33;
        s[3] = 44;
        d.push(11);
        d.push(22);
        d.push(33);
        d.push(44);
    }

    function memoryArray(uint256 dirty) public pure returns (uint256) {
        uint8 i;
        assembly { i := dirty }
        uint256[4] memory a;
        a[0] = 11;
        a[1] = 22;
        a[2] = 33;
        a[3] = 44;
        return a[i];
    }

    function memoryBytes(uint256 dirty) public pure returns (bytes1) {
        uint8 i;
        assembly { i := dirty }
        bytes memory b = new bytes(4);
        b[0] = 0x11;
        b[1] = 0x22;
        b[2] = 0x33;
        b[3] = 0x44;
        return b[i];
    }

    function storageStatic(uint256 dirty) public view returns (uint256) {
        uint8 i;
        assembly { i := dirty }
        return s[i];
    }

    function storageDynamic(uint256 dirty) public view returns (uint256) {
        uint8 i;
        assembly { i := dirty }
        return d[i];
    }
}
// ====
// compileViaMlir: true
// ----
// init() ->
// memoryArray(uint256): 0x100000000000000000000000000000000000000000000000000000000000001 -> 22
// memoryBytes(uint256): 0x100000000000000000000000000000000000000000000000000000000000001 -> left(0x22)
// storageStatic(uint256): 0x100000000000000000000000000000000000000000000000000000000000001 -> 22
// storageDynamic(uint256): 0x100000000000000000000000000000000000000000000000000000000000001 -> 22
