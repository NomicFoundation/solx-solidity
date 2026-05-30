// Verifies that out-of-bounds access via an auto-generated public getter emits
// bare revert(0, 0) rather than Panic(0x32), matching both the old codegen
// (ExpressionCompiler::appendStateVariableAccessor) and the via-IR path
// (IRGenerator::generateGetter).
contract C {
    uint256[] public dynArr;
    uint256[3] public fixArr;
    constructor() {
        dynArr.push(10);
        dynArr.push(20);
    }
}
// ====
// compileViaMlir: true
// ----
// dynArr(uint256): 0 -> 10
// dynArr(uint256): 1 -> 20
// dynArr(uint256): 2 -> FAILURE
// fixArr(uint256): 0 -> 0
// fixArr(uint256): 2 -> 0
// fixArr(uint256): 3 -> FAILURE
