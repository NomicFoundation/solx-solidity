contract C {
    mapping(address => uint) addressMap;
    mapping(uint8 => uint) uintMap;

    function addressKey() public returns (uint) {
        address key;
        assembly {
            key := not(0)
        }
        addressMap[key] = 7;
        return addressMap[address(type(uint160).max)];
    }

    function uintKey() public returns (uint) {
        uint8 key;
        assembly {
            key := not(0)
        }
        uintMap[key] = 9;
        return uintMap[255];
    }
}

// ====
// compileViaMlir: true
// ----
// addressKey() -> 7
// uintKey() -> 9
