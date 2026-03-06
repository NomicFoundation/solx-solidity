contract C {
    function f() public payable returns (uint) {
        return msg.value;
    }
}
// ====
// compileViaMlir: true
// ----
// f() -> 0
// f(), 12 ether -> 12000000000000000000
