contract C {
    function f() public returns (uint) {
        return block.difficulty;
    }
}
// ====
// compileViaMlir: true
// EVMVersion: >=paris
// ----
// f() -> 0xa86c2e601b6c44eb4848f7d23d9df3113fbcac42041c49cbed5000cb4f118777
