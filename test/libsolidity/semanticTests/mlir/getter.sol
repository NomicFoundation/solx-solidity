contract C {
  uint public i;
  uint[2][2] public i2;
  string public str;
  struct S {
    uint i;
    uint8 i8;
  }
  S public s;
  mapping(address => uint) public m;
  mapping(address => mapping(uint => uint)) public m2;

  uint public constant ci = 7;

  constructor () {
    i = 1;
    i2[0][0] = 2;
    s.i = 3;
    // FIXME:
    // llvm/lib/Target/EVM/EVMStackModel.cpp:107: void llvm::EVMStackModel::processMI(const MachineInstr &): Assertion
    // `Opc != EVM::STACK_LOAD && Opc != EVM::STACK_STORE && "Unexpected stack memory instruction"' failed.
	  // s.i8 = 4;
    m[address(0)] = 5;
    m2[address(0)][0] = 6;
  }
}

// ====
// compileViaMlir: true
// ----
// i() -> 1
// i2(uint256,uint256): 0, 0 -> 2
// s() -> 3, 0
// m(address): 0 -> 5
// m2(address,uint256): 0, 0 -> 6
// ci() -> 7
