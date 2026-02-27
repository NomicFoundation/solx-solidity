contract C {
    function f() public returns (bytes memory) {
        return bytes.concat();
    }
}

// ====
// compileViaMlir: true
// ----
// f() -> 0x20, 0
