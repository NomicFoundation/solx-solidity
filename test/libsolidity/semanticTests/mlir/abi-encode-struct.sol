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

  // FIXME: Enable storage encoding tests with Big/Nested once storage-array
  // encoding is fixed.
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
// storage_flat() -> 0x20, 0xc0, 0x20, 9, 10, 0x60, 4, "stor"
// storage_nested() -> 0x20, 288, 0x20, 0x64, 0x60, 300, 0xc9, 0xca, 0x60, 2, 47687202278368593055453199545051370742183790376672955679702151372887807754240
// storage_packed() -> 0x20, 0x80, 1, 2, 1, 3
