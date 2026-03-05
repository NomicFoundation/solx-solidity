contract C {
    function f() public returns (uint) {
        return block.gaslimit;
    }
}
// ====
// compileViaMlir: true
// ----
// f() -> 20000000
