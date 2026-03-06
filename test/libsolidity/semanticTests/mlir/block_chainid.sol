contract C {
    function f() public returns (uint) {
        return block.chainid;
    }
}
// ====
// compileViaMlir: true
// EVMVersion: >=istanbul
// ----
// f() -> 1
