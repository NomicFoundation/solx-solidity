contract Failer {
    constructor() {
        revert();
    }
}

contract C {
    function create() public {
        new Failer();
    }

    function create2() public {
        new Failer{salt: bytes32(uint256(0x10))}();
    }
}

// ====
// compileViaMlir: true
// ----
// create() -> FAILURE
// create2() -> FAILURE
