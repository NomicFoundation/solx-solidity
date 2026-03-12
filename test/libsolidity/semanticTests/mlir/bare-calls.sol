contract C {
  function inc(uint256 v) external pure returns (uint256) {
    return v + 1;
  }

  function call(uint256 v) external returns (bool, uint256) {
    (bool ok, bytes memory data) =
        address(this).call(abi.encodeWithSignature("inc(uint256)", v));
    if (ok == false)
      return (false, 0);
    return (true, abi.decode(data, (uint256)));
  }

  function call_options(uint256 v) external returns (bool, uint256) {
    (bool ok, bytes memory data) =
        address(this).call{gas: 100000, value: 0}(
          abi.encodeWithSignature("inc(uint256)", v)
        );
    if (ok == false)
      return (false, 0);
    return (true, abi.decode(data, (uint256)));
  }

  function call_fail(uint256 v) external returns (bool, uint256) {
    (bool ok, bytes memory data) =
        address(this).call(abi.encodeWithSignature("missing(uint256)", v));
    if (ok == false)
      return (false, 0);
    return (true, abi.decode(data, (uint256)));
  }

  function delegatecall(uint256 v) external returns (bool, uint256) {
    (bool ok, bytes memory data) =
        address(this).delegatecall(
          abi.encodeWithSignature("inc(uint256)", v)
        );
    if (ok == false)
      return (false, 0);
    return (true, abi.decode(data, (uint256)));
  }

  function delegatecall_options(uint256 v) external returns (bool, uint256) {
    (bool ok, bytes memory data) =
        address(this).delegatecall{gas: 100000}(
          abi.encodeWithSignature("inc(uint256)", v)
        );
    if (ok == false)
      return (false, 0);
    return (true, abi.decode(data, (uint256)));
  }

  function delegatecall_fail(uint256 v) external returns (bool, uint256) {
    (bool ok, bytes memory data) =
        address(this).delegatecall(
          abi.encodeWithSignature("missing(uint256)", v)
        );
    if (ok == false)
      return (false, 0);
    return (true, abi.decode(data, (uint256)));
  }

  function staticcall(uint256 v) external view returns (bool, uint256) {
    (bool ok, bytes memory data) =
        address(this).staticcall(
          abi.encodeWithSignature("inc(uint256)", v)
        );
    if (ok == false)
      return (false, 0);
    return (true, abi.decode(data, (uint256)));
  }

  function staticcall_options(uint256 v) external view returns (bool, uint256) {
    (bool ok, bytes memory data) =
        address(this).staticcall{gas: 100000}(
          abi.encodeWithSignature("inc(uint256)", v)
        );
    if (ok == false)
      return (false, 0);
    return (true, abi.decode(data, (uint256)));
  }

  function staticcall_fail(uint256 v) external view returns (bool, uint256) {
    (bool ok, bytes memory data) =
        address(this).staticcall(
          abi.encodeWithSignature("missing(uint256)", v)
        );
    if (ok == false)
      return (false, 0);
    return (true, abi.decode(data, (uint256)));
  }
}

// ====
// compileViaMlir: true
// ----
// call(uint256): 5 -> true, 6
// call_options(uint256): 8 -> true, 9
// call_fail(uint256): 11 -> false, 0
// delegatecall(uint256): 9 -> true, 10
// delegatecall_options(uint256): 2 -> true, 3
// delegatecall_fail(uint256): 4 -> false, 0
// staticcall(uint256): 1 -> true, 2
// staticcall_options(uint256): 7 -> true, 8
// staticcall_fail(uint256): 10 -> false, 0
