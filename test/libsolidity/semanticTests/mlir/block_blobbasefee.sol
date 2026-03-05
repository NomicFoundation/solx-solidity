contract C {
    function f() public view returns (uint) {
        return block.blobbasefee;
    }
}
// ====
// compileViaMlir: true
// EVMVersion: >=cancun
// ----
// f() -> 1
