contract C {
    function f() public returns (address payable) {
        return block.coinbase;
    }
}
// ====
// compileViaMlir: true
// ----
// f() -> 0x7878787878787878787878787878787878787878
