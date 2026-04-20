struct Data {
    uint value;
}

contract A {
    function get() public pure returns (Data memory) {
        return Data(5);
    }
}

contract B {
    uint x = 10;
    uint y = 10;

    modifier updateStorage() {
        A a = new A();
        x = a.get().value;
        _;
        y = a.get().value;
    }

    function test() public updateStorage returns (uint, uint) {
        return (x, y);
    }
}

// ====
// compileViaMlir: true
// ----
// test() -> 5, 10
