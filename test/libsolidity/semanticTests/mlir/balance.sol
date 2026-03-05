contract D {
  constructor() payable {}
}

contract C {
  function selfbalance() public payable returns (uint) {
    return address(this).balance;
  }

  function balance() public payable returns (uint) {
    D d = new D{value: 7 wei}();
    return address(d).balance;
  }
}

// ====
// compileViaMlir: true
// ----
// selfbalance(), 23 wei -> 23
// balance() -> 7
