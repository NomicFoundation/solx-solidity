contract C {
    function f() public returns (bytes4) {
        return msg.sig;
    }
    function g() public returns (bytes4) {
        return msg.sig;
    }
}
// ====
// compileViaMlir: true
// ----
// f() -> left(0x26121ff0)
// g() -> left(0xe2179b8e)
