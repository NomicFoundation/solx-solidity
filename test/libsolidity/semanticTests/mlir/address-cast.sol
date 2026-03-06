contract C {
    function to_payable(address a) external pure returns (address payable) {
        return payable(a);
    }

    function from_payable(address payable a) external pure returns (address) {
        return address(a);
    }

    function from_uint(uint256 x) external pure returns (address) {
        return address(uint160(x));
    }

    function eq(address a, address b) external pure returns (bool) {
        return a == b;
    }

    function to_bytes20(address a) external pure returns (bytes20) {
        return bytes20(a);
    }

    function from_bytes20(bytes20 b) external pure returns (address) {
        return address(b);
    }
}

// ====
// compileViaMlir: true
// ----
// to_payable(address): 1 -> 1
// from_payable(address): 1 -> 1
// from_uint(uint256): 0x10000000000000000000000000000000000000001 -> 1
// eq(address,address): 1, 1 -> true
// eq(address,address): 1, 0x2 -> false
// from_bytes20(bytes20): left(0x01) -> 0x0100000000000000000000000000000000000000
