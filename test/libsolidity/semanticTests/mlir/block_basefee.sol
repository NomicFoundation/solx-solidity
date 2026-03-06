contract C {
    function f() public view returns (uint) {
        return block.basefee;
    }
}
// ====
// compileViaMlir: true
// EVMVersion: >=london
// ----
// f() -> 7
