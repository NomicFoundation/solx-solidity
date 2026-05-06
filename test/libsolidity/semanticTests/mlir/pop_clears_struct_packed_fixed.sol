contract C {
    struct S {
        uint256 x;
        uint72[2] packed;
    }

    S[] arr;

    function test() public {
        arr.push();
        arr[0].x = 123;
        arr[0].packed[0] = 42;
        arr[0].packed[1] = 99;
        arr.pop();
        assert(arr.length == 0);
    }
}
// ====
// compileViaMlir: true
// ----
// test() ->
// storageEmpty -> 1
