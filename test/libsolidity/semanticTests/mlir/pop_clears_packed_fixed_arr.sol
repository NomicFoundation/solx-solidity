contract C {
    uint72[2][] arr;

    function test() public {
        arr.push();
        arr[0][0] = 42;
        arr[0][1] = 99;
        arr.pop();
        assert(arr.length == 0);
    }
}
// ====
// compileViaMlir: true
// ----
// test() ->
// storageEmpty -> 1
