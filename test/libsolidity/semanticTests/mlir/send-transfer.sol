contract Receiver {
  receive() external payable {}
}

contract Rejector {
  receive() external payable {
    revert();
  }
}

contract C {
  constructor() payable {}

  function sendOk() public returns (bool) {
    Receiver r = new Receiver();
    return payable(r).send(1 wei);
  }

  function sendFail() public returns (bool) {
    Rejector r = new Rejector();
    return payable(r).send(1 wei);
  }

  function transferOk() public {
    Receiver r = new Receiver();
    payable(r).transfer(1 wei);
  }

  function transferFail() public {
    Rejector r = new Rejector();
    payable(r).transfer(1 wei);
  }
}

// ====
// compileViaMlir: true
// ----
// constructor(), 10 wei ->
// sendOk() -> true
// sendFail() -> false
// transferOk() ->
// transferFail() -> FAILURE
