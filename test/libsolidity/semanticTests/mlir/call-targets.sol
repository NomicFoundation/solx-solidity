contract C {
  uint public value = 11;

  struct S {
    function() external view returns (uint) fn;
  }

  function g() public view returns (uint) {
    return 7;
  }

  function directGetter() public view returns (uint) {
    return this.value();
  }

  function getterPointer() public view returns (uint) {
    function() external view returns (uint) fp = this.value;
    return fp();
  }

  function getterStruct() public view returns (uint) {
    S memory s;
    s.fn = this.value;
    return s.fn();
  }

  function functionStruct() public view returns (uint) {
    S memory s;
    s.fn = this.g;
    return s.fn();
  }
}

// ====
// compileViaMlir: true
// ----
// directGetter() -> 11
// getterPointer() -> 11
// getterStruct() -> 11
// functionStruct() -> 7
