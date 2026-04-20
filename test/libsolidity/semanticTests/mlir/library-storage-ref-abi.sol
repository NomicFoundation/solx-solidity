library L {
    struct S {
        uint x;
        uint y;
    }

    function f(uint[] storage r, S storage s)
        external
        view
        returns (uint, uint, uint, uint)
    {
        return (r[0], r[1], s.x, s.y);
    }
}

contract C {
    L.S s;
    uint[] r;

    constructor() {
        r.push(1);
        r.push(2);
        s.x = 7;
        s.y = 12;
    }

    function g() external view returns (uint, uint, uint, uint) {
        return L.f(r, s);
    }
}

// ====
// compileViaMlir: true
// ----
// library: L
// g() -> 1, 2, 7, 12
