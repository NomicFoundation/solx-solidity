contract C {
  uint x;

  function f(bool a, bool b, bool c) public returns (bool) {
    return a && (b && c);
  }

  function g(bool a, bool b, bool c) public returns (bool) {
    return a || (b || c);
  }

  function s(bool v) internal returns (bool) {
    x++;
    return v;
  }

  function h(bool a, bool b) public returns (bool, uint) {
    x = 0;
    return (a && s(b), x);
  }

  function i(bool a, bool b) public returns (bool, uint) {
    x = 0;
    return (a || s(b), x);
  }
}

// ====
// compileViaMlir: true
// ----
// f(bool,bool,bool): 0, 0, 0 -> false
// f(bool,bool,bool): 1, 0, 1 -> false
// f(bool,bool,bool): 1, 1, 0 -> false
// f(bool,bool,bool): 1, 1, 1 -> true
// g(bool,bool,bool): 0, 0, 0 -> false
// g(bool,bool,bool): 1, 0, 0 -> true
// g(bool,bool,bool): 0, 1, 0 -> true
// g(bool,bool,bool): 0, 0, 1 -> true
// h(bool,bool): 0, 1 -> false, 0
// h(bool,bool): 1, 0 -> false, 1
// h(bool,bool): 1, 1 -> true, 1
// i(bool,bool): 1, 0 -> true, 0
// i(bool,bool): 0, 1 -> true, 1
// i(bool,bool): 0, 0 -> false, 1
