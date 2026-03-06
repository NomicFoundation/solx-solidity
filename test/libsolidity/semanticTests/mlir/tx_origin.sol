contract C {
    function f() public returns (address) {
        return tx.origin;
    }
}
// ====
// compileViaMlir: true
// ----
// f() -> 0x9292929292929292929292929292929292929292
