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

    function to_contract(address a) external pure returns (C) {
        return C(a);
    }

    function from_contract(C c) external pure returns (address) {
        return address(c);
    }

    function this_to_address() public returns (address) {
      return address(this);
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
// to_contract(address): 1 -> 1
// from_contract(address): 1 -> 1
// this_to_address() -> 0xc06afe3a8444fc0004668591e8306bfb9968e79e
