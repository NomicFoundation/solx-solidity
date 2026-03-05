contract C {
    function f() public returns (uint) {
        return block.difficulty;
    }
}
// ====
// compileViaMlir: true
// EVMVersion: <paris
// ----
// f() -> 200000000
