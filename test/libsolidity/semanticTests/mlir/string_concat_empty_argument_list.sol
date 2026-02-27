contract C {
    function f() public returns (string memory) {
        return string.concat();
    }
}

// ====
// compileViaMlir: true
// ----
// f() -> 0x20, 0
