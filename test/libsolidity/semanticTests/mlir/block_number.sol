contract C {
    constructor() {}
    function f() public returns (uint) {
        return block.number;
    }
}
// ====
// compileViaMlir: true
// ----
// constructor()
// f() -> 2
// f() -> 3
