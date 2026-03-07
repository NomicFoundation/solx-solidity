contract Test {
  // Single named return, fall off end
  function f_basic() public returns (uint a) { a = 42; }

  // Default zero-initialization (no assignment)
  function f_default() public returns (uint a) {}

  // Multiple named returns, fall off end
  function f_multi() public returns (uint a, uint b) { a = 1; b = 2; }

  // Partial assignment: b stays at default zero
  function f_partial() public returns (uint a, uint b) { a = 7; }

  // Named bool return
  function f_bool() public returns (bool ok) { ok = true; }

  // Named return with conditional: both branches assign
  function f_cond(bool flag) public returns (uint result) {
    if (flag) { result = 10; } else { result = 20; }
  }

  // Named return accumulates in a loop
  function f_loop(uint n) public returns (uint sum) {
    for (uint i = 0; i < n; i++) { sum += i; }
  }

  // Named return updated via a helper call
  function f_call() public returns (uint a) { a = helper(); }
  function helper() internal returns (uint) { return 5; }

  // Explicit return value (non-named) still works
  function f_explicit() public returns (uint) { return 99; }

  function f_noname(uint b) public returns (uint a) { return b; }
}

// ====
// compileViaMlir: true
// ----
// f_basic() -> 42
// f_default() -> 0
// f_noname(uint256): 7 -> 7
// f_multi() -> 1, 2
// f_partial() -> 7, 0
// f_bool() -> true
// f_cond(bool): 1 -> 10
// f_cond(bool): 0 -> 20
// f_loop(uint256): 5 -> 10
// f_call() -> 5
// f_explicit() -> 99
