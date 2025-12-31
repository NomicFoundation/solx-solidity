contract C {
  function keccak(bytes memory data) public returns (bytes32) {
    return keccak256(data);
  }

  function sha(bytes memory data) public returns (bytes32) {
    return sha256(data);
  }

  function ripemd(bytes memory data) public returns (bytes20) {
    return ripemd160(data);
  }

  function recover(bytes32 hash, uint8 v, bytes32 r, bytes32 s) public returns (address) {
    return ecrecover(hash, v, r, s);
  }
}

// ====
// compileViaMlir: true
// ----
// keccak(bytes): 32, 32, "abcdbcdecdefdefgefghfghighijhijk" -> 0x4b50e45e85ca4a0a9c089890faf83098c75b04fe0e0f9c5488effd1643711033
// sha(bytes): 32, 32, hex"993dab3dd91f5c6dc28e17439be475478f5635c92a56e17e82349d3fb2f16619" -> 0xfc91f256e276707af00133a3fea5eadc371b4904a429bce31579ef507f942eeb
// ripemd(bytes): 32, 32, hex"0000000000000000000000000000000000000000000000000000000000000004" -> 0x1b0f3c404d12075c68c938f9f60ebea4f74941a0000000000000000000000000
// recover(bytes32,uint8,bytes32,bytes32): hex"47173285a8d7341e5e972fc677286384f802f8ef42a5ec5f03bbfa254cb01fad", hex"000000000000000000000000000000000000000000000000000000000000001c", hex"debaaa0cddb321b2dcaaf846d39605de7b97e77ba6106587855b9106cb104215", hex"61a22d94fa8b8a687ff9c911c844d1c016d1a685a9166858f9c7c1bc85128aca" -> 0x0000000000000000000000008743523d96a1b2cbe0c6909653a56da18ed484af
