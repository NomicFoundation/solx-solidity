contract C {
  struct Big {
    uint256 id;
    uint256 code;
    string note;
    uint256[2] fixedArr;
    uint256[] dynArr;
  }

  struct Nested {
    uint256 salt;
    Big payload;
    uint256 extra;
  }

  struct StorageFlat {
    uint256 a;
    uint128 b;
    string note;
  }

  struct StorageNested {
    uint256 left;
    StorageFlat inner;
    uint256 right;
  }

  struct StoragePacked {
    uint128 a;
    uint64 b;
    bool c;
    uint256 d;
  }

  struct S {
    uint256 a;
    uint256 b;
  }

  Big storageBig;
  Nested storageBigNested;
  StorageFlat storageFlat;
  StorageNested storageNested;
  StoragePacked storagePacked;

  constructor() {
    storageFlat.a = 9;
    storageFlat.b = 10;
    storageFlat.note = "stor";

    storageNested.left = 100;
    storageNested.inner.a = 201;
    storageNested.inner.b = 202;
    storageNested.inner.note = "in";
    storageNested.right = 300;

    storagePacked.a = 1;
    storagePacked.b = 2;
    storagePacked.c = true;
    storagePacked.d = 3;

    storageBig.id = 7;
    storageBig.code = 8;
    storageBig.note = "abc";
    storageBig.fixedArr[0] = 41;
    storageBig.fixedArr[1] = 42;
    storageBig.dynArr.push(11);
    storageBig.dynArr.push(12);
    storageBig.dynArr.push(13);

    storageBigNested.salt = 100;
    storageBigNested.payload.id = 5;
    storageBigNested.payload.code = 6;
    storageBigNested.payload.note = "xy";
    storageBigNested.payload.fixedArr[0] = 21;
    storageBigNested.payload.fixedArr[1] = 22;
    storageBigNested.payload.dynArr.push(31);
    storageBigNested.payload.dynArr.push(32);
    storageBigNested.extra = 200;
  }

  function mem_big(Big memory s) public pure returns (bytes memory) {
    return abi.encode(s);
  }

  function mem_nested(Nested memory s) public pure returns (bytes memory) {
    return abi.encode(s);
  }

  function cd_big(Big calldata s) public pure returns (bytes memory) {
    return abi.encode(s);
  }

  function cd_nested(Nested calldata s) public pure returns (bytes memory) {
    return abi.encode(s);
  }

  function enc_struct(S[2] calldata x) public pure returns (bytes memory) {
    return abi.encode(x);
  }

  function storage_big() public view returns (bytes memory) {
    return abi.encode(storageBig);
  }

  function storage_big_nested() public view returns (bytes memory) {
    return abi.encode(storageBigNested);
  }

  function storage_flat() public view returns (bytes memory) {
    return abi.encode(storageFlat);
  }

  function storage_nested() public view returns (bytes memory) {
    return abi.encode(storageNested);
  }

  function storage_packed() public view returns (bytes memory) {
    return abi.encode(storagePacked);
  }
}

// ====
// compileViaMlir: true
// ----
// constructor() ->
// mem_big((uint256,uint256,string,uint256[2],uint256[])): 0x20, 7, 8, 0xc0, 41, 42, 0x100, 3, "abc", 3, 11, 12, 13 -> 0x20, 0x1a0, 0x20, 7, 8, 0xc0, 41, 42, 0x100, 3, "abc", 3, 11, 12, 13
// cd_big((uint256,uint256,string,uint256[2],uint256[])): 0x20, 7, 8, 0xc0, 41, 42, 0x100, 3, "abc", 3, 11, 12, 13 -> 0x20, 0x1a0, 0x20, 7, 8, 0xc0, 41, 42, 0x100, 3, "abc", 3, 11, 12, 13
// mem_nested((uint256,(uint256,uint256,string,uint256[2],uint256[]),uint256)): 0x20, 100, 0x60, 200, 5, 6, 0xc0, 21, 22, 0x100, 2, "xy", 2, 31, 32 -> 0x20, 0x1e0, 0x20, 100, 0x60, 200, 5, 6, 0xc0, 21, 22, 0x100, 2, "xy", 2, 31, 32
// cd_nested((uint256,(uint256,uint256,string,uint256[2],uint256[]),uint256)): 0x20, 100, 0x60, 200, 5, 6, 0xc0, 21, 22, 0x100, 2, "xy", 2, 31, 32 -> 0x20, 0x1e0, 0x20, 100, 0x60, 200, 5, 6, 0xc0, 21, 22, 0x100, 2, "xy", 2, 31, 32
// enc_struct((uint256,uint256)[2]): 11, 22, 33, 44 -> 0x20, 0x80, 11, 22, 33, 44
// storage_big() -> 0x20, 0x1a0, 0x20, 7, 8, 0xc0, 41, 42, 0x100, 3, "abc", 3, 11, 12, 13
// storage_big_nested() -> 0x20, 0x1e0, 0x20, 100, 0x60, 200, 5, 6, 0xc0, 21, 22, 0x100, 2, "xy", 2, 31, 32
// storage_flat() -> 0x20, 0xc0, 0x20, 9, 10, 0x60, 4, "stor"
// storage_nested() -> 0x20, 288, 0x20, 0x64, 0x60, 300, 0xc9, 0xca, 0x60, 2, 47687202278368593055453199545051370742183790376672955679702151372887807754240
// storage_packed() -> 0x20, 0x80, 1, 2, 1, 3
