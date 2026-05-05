contract C {
  enum En { A, B, C }
  type Udvt is uint64;
  struct S { uint x; bool y; bytes32 z; }

  // Indexed: value types
  event Ibool(bool indexed);
  event Iint(int8 indexed, uint256 indexed);
  event Iaddr(address indexed);
  event Ibytes(bytes1 indexed, bytes32 indexed);
  event Ienum(En indexed);
  event Iudvt(Udvt indexed);
  event Icontract(C indexed);
  event Ifnptr(function() external indexed);

  // Indexed: reference types
  event Istr(string indexed);
  event Ibts(bytes indexed);
  event Idynarr(uint[] indexed);
  event Ifixarr(uint[3] indexed);

  // Non-indexed: full ABI tuple
  event N(
    bool, int256, uint256, address,
    bytes1, bytes32, En, Udvt,
    string, bytes, uint[], uint[3], S
  );

  // Anonymous (4 indexed)
  event Anon(uint indexed, uint indexed, uint indexed, uint indexed) anonymous;

  // Mixed indexed + non-indexed
  event Mixed(uint indexed, string s, uint[] arr);

  function f() public {
    emit Ibool(true);
    emit Iint(-1, 0xff);
    emit Iaddr(address(0xabc));
    emit Ibytes(0x01, 0xff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00);
    emit Ienum(En.B);
    emit Iudvt(Udvt.wrap(0xcafe));
    emit Icontract(this);
    emit Ifnptr(this.f);

    emit Istr("hello");
    emit Ibts(hex"deadbeef");

    uint[] memory a = new uint[](3);
    a[0] = 1; a[1] = 2; a[2] = 3;
    emit Idynarr(a);

    uint[3] memory b = [uint(10), 20, 30];
    emit Ifixarr(b);

    emit N(
      true, -1, 0xff, address(0x123),
      0x42, 0xff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00, En.C, Udvt.wrap(0xfeed),
      "data", hex"01020304", a, b, S(100, true, 0xff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00)
    );

    emit Anon(0x11, 0x22, 0x33, 0x44);
    emit Mixed(7, "key", a);

    // Keccak edge cases: empty / multi-word reference args.
    emit Istr("");
    emit Ibts(hex"");
    uint[] memory empty = new uint[](0);
    emit Idynarr(empty);
    emit Istr("ABCDEFGHIJKLMNOPQRSTUVWXYZ-this-string-is-longer-than-32-bytes");
    emit Ibts(hex"deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef00");
  }
}

// ====
// compileViaMlir: true
// ----
// f() ->
// ~ emit Ibool(bool): #0x01
// ~ emit Iint(int8,uint256): #0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff, #0xff
// ~ emit Iaddr(address): #0x0abc
// ~ emit Ibytes(bytes1,bytes32): #0x0100000000000000000000000000000000000000000000000000000000000000, #0xff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00
// ~ emit Ienum(uint8): #0x01
// ~ emit Iudvt(uint64): #0xcafe
// ~ emit Icontract(address): #0xc06afe3a8444fc0004668591e8306bfb9968e79e
// ~ emit Ifnptr(function): #0xc06afe3a8444fc0004668591e8306bfb9968e79e26121ff00000000000000000
// ~ emit Istr(string): #0x1c8aff950685c2ed4bc3174f3472287b56d9517b9c948127319a09a7a36deac8
// ~ emit Ibts(bytes): #0xd4fd4e189132273036449fc9e11198c739161b4c0116a9a2dccdfa1c492006f1
// ~ emit Idynarr(uint256[]): #0x6e0c627900b24bd432fe7b1f713f1b0744091a646a9fe4a65a18dfed21f2949c
// ~ emit Ifixarr(uint256[3]): #0x993a17e836e3d65fac6232390ad969f667a21cb52cf3dc1bcd207d28a216c944
// ~ emit N(bool,int256,uint256,address,bytes1,bytes32,uint8,uint64,string,bytes,uint256[],uint256[3],(uint256,bool,bytes32)): true, 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff, 0xff, 0x0123, "B", 0xff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00, 0x02, 0xfeed, 0x0220, 0x0260, 0x02a0, 0x0a, 0x14, 0x1e, 0x64, 0x01, 0xff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00, 0x04, "data", 0x04, 0x0102030400000000000000000000000000000000000000000000000000000000, 0x03, 0x01, 0x02, 0x03
// ~ emit <anonymous>: #0x11, #0x22, #0x33, #0x44
// ~ emit Mixed(uint256,string,uint256[]): #0x07, 0x40, 0x80, 0x03, "key", 0x03, 0x01, 0x02, 0x03
// ~ emit Istr(string): #0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
// ~ emit Ibts(bytes): #0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
// ~ emit Idynarr(uint256[]): #0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
// ~ emit Istr(string): #0xd40a805ab6f9f68c502048cc3806684dab75031166a4c09a259df77aad1c1121
// ~ emit Ibts(bytes): #0xeaacc4b644af7f3e1b9345c8a687198b505eeeafa66dc79201a9b856e7a29125
