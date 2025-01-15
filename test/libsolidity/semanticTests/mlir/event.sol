contract C {
  event E(address indexed a, uint b);
  function f(address a, uint b) public {
    emit E(a, b);
  }
}

// ====
// compileViaMlir: true
// ----
// f(address,uint256): 1, 2 ->
// ~ emit E(address,uint256): #0x01, 0x02
