contract C {
  string s;
  mapping(string => uint) m;

  function f() public returns (uint) {
    s = "hello";
    m["hello"] = 42;
    return m[s];
  }
}
// ====
// compileViaMlir: true
// ----
// f() -> 42
