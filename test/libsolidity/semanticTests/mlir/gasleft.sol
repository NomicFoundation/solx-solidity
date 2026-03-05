contract C {
    function f() public returns (bool) {
        return gasleft() > 0;
    }
}
// ====
// compileViaMlir: true
// ----
// f() -> true
