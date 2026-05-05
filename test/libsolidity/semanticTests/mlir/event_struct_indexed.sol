contract C {
  struct S15 { string s; uint x; }
  struct S2 { uint a; uint b; }
  struct S16 { S2 inner; uint y; }
  struct S17 { uint[] a; }
  struct S18 { string[] a; }
  struct S19 { function() external f; uint x; }
  struct SStr { string s; }
  struct SInner { string[] ss; }
  struct SOuter { SInner i; }

  event E15(S15 indexed);
  event E16(S16 indexed);
  event E17(S17 indexed);
  event E18(S18 indexed);
  event E19(S19 indexed);
  event E20(uint[][] indexed);
  event E21(string[] indexed);
  event E22(S2[] indexed);
  event E23(S2[3] indexed);
  event EStrPlain(string indexed);
  event EStrWrapped(SStr indexed);
  event EBytesNested(bytes[2][2] indexed);
  event EDeep(SOuter indexed);

  function dummy() external {}

  function f15() public {
    S15 memory s = S15("hello", 0x42);
    emit E15(s);
  }

  function f16() public {
    S16 memory s = S16(S2(0x11, 0x22), 0x33);
    emit E16(s);
  }

  function f17() public {
    uint[] memory a = new uint[](3);
    a[0] = 1; a[1] = 2; a[2] = 3;
    S17 memory s = S17(a);
    emit E17(s);
  }

  function f18() public {
    string[] memory a = new string[](2);
    a[0] = "foo";
    a[1] = "bar";
    S18 memory s = S18(a);
    emit E18(s);
  }

  function f19() public {
    S19 memory s = S19(this.dummy, 0x77);
    emit E19(s);
  }

  function f20() public {
    uint[][] memory a = new uint[][](2);
    a[0] = new uint[](2);
    a[0][0] = 1; a[0][1] = 2;
    a[1] = new uint[](1);
    a[1][0] = 3;
    emit E20(a);
  }

  function f21() public {
    string[] memory a = new string[](2);
    a[0] = "x";
    a[1] = "yz";
    emit E21(a);
  }

  function f22() public {
    S2[] memory a = new S2[](2);
    a[0] = S2(1, 2);
    a[1] = S2(3, 4);
    emit E22(a);
  }

  function f23() public {
    S2[3] memory a;
    a[0] = S2(10, 20);
    a[1] = S2(30, 40);
    a[2] = S2(50, 60);
    emit E23(a);
  }

  // String roundup: outer indexed `string` vs indexed `struct { string }`
  // at lengths 31, 32, 33 - exercises packed-vs-tuple alignment in the
  // unified ABI encoder.
  function f24a() public {
    string memory s = "0123456789012345678901234567890";
    emit EStrPlain(s);
    emit EStrWrapped(SStr(s));
  }

  function f24b() public {
    string memory s = "01234567890123456789012345678901";
    emit EStrPlain(s);
    emit EStrWrapped(SStr(s));
  }

  function f24c() public {
    string memory s = "012345678901234567890123456789012";
    emit EStrPlain(s);
    emit EStrWrapped(SStr(s));
  }

  // Indexed nested dynamic array: aggregate-element packed array.
  function f25() public {
    uint256[][] memory a = new uint256[][](3);
    a[0] = new uint256[](2);
    a[0][0] = 0x11;
    a[0][1] = 0x22;
    a[1] = new uint256[](0);
    a[2] = new uint256[](1);
    a[2][0] = 0x33;
    emit E20(a);
  }

  // Indexed nested fixed array of bytes: bytes[2][2].
  function f26() public {
    bytes[2][2] memory a;
    a[0][0] = hex"01";
    a[0][1] = hex"020304";
    a[1][0] = hex"";
    a[1][1] = hex"101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f3031";
    emit EBytesNested(a);
  }

  // Indexed deeply nested struct: struct{struct{string[]}}.
  function f27() public {
    SOuter memory o;
    o.i.ss = new string[](3);
    o.i.ss[0] = "short";
    o.i.ss[1] = "012345678901234567890123456789012";
    o.i.ss[2] = "";
    emit EDeep(o);
  }
}
// ====
// compileViaMlir: true
// ----
// f15() ->
// ~ emit E15((string,uint256)): #0xcf20d6a60643c9496dc62a5064a295b33d705bd34f77756b984806fc5d5c6205
// f16() ->
// ~ emit E16(((uint256,uint256),uint256)): #0x7f654b5c8bf6519cddb680bf8bf2f6fc0b22e04163af6d4ac782a35c35847278
// f17() ->
// ~ emit E17((uint256[])): #0x6e0c627900b24bd432fe7b1f713f1b0744091a646a9fe4a65a18dfed21f2949c
// f18() ->
// ~ emit E18((string[])): #0x7657c71556b01ae610890a804c5dfd8ad52d690cfe3843e503f58f9a25fdcb82
// f19() ->
// ~ emit E19((function,uint256)): #0x1a7fba8ccad075483b9b0bfba1052965bb5c823a088e9420b3e5b6e861f56041
// f20() ->
// ~ emit E20(uint256[][]): #0x6e0c627900b24bd432fe7b1f713f1b0744091a646a9fe4a65a18dfed21f2949c
// f21() ->
// ~ emit E21(string[]): #0x39ef1bda3b822ebb76142b91a5e3faebff2f17652f45c3ca8d8b16ea1dd25602
// f22() ->
// ~ emit E22((uint256,uint256)[]): #0x392791df626408017a264f53fde61065d5a93a32b60171df9d8a46afdf82992d
// f23() ->
// ~ emit E23((uint256,uint256)[3]): #0x6eede4f9626e1263dc1f48607c070349c2f9a71d55d303c4a26bcb43b8e89bd8
// f24a() ->
// ~ emit EStrPlain(string): #0x37580b044dcb770403c020672ee7f35dfc1ca488c71bdd63e186a83ad9422a85
// ~ emit EStrWrapped((string)): #0xb73c2f908dffa19741346b63013e33c1ad4f877b9c13663b1b374fd03a213343
// f24b() ->
// ~ emit EStrPlain(string): #0xb6f6f0bb62127422171f029ef8588af3a45d58989134675112c2acc78dd16078
// ~ emit EStrWrapped((string)): #0xb6f6f0bb62127422171f029ef8588af3a45d58989134675112c2acc78dd16078
// f24c() ->
// ~ emit EStrPlain(string): #0x66faf70c9e8400306ad3d944112e8a3aae89dc039cfbfe97512fc09bc7c07383
// ~ emit EStrWrapped((string)): #0x0d3d200c7592f7bf7d43466f4d378e36c8f7bf20086adcd82b988f91ea6fb984
// f25() ->
// ~ emit E20(uint256[][]): #0x7f654b5c8bf6519cddb680bf8bf2f6fc0b22e04163af6d4ac782a35c35847278
// f26() ->
// ~ emit EBytesNested(bytes[2][2]): #0x90d79d24e406a2590d86a799298cd0131e4207698f19b6d91d7112354fa6f3f0
// f27() ->
// ~ emit EDeep(((string[]))): #0xb09826b1753b8aa137004ce2a1ee6ef23a72452a384db46bd34579888ce36ede
