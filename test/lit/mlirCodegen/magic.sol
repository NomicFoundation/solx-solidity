// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

function msgSender() returns (address) {
  return msg.sender;
}

function addr() returns (address) {
  return address(0);
}

function encode(uint ui, uint8 ui8, int32 si32) returns (bytes memory) {
  return abi.encode(ui, ui8, si32);
}

function decode(bytes memory a) returns (uint, uint8, int32) {
  return abi.decode(a, (uint, uint8, int32));
}

function encode_packed(uint24 x, uint96 y, uint136 z) returns (bytes memory) {
  return abi.encodePacked(x, y, z);
}

function encode_addr(address a) returns (bytes memory) {
  return abi.encode(a);
}

function decode_addr(bytes memory a) returns (address) {
  return abi.decode(a, (address));
}

function encode_packed_addr(address a) returns (bytes memory) {
  return abi.encodePacked(a);
}

function roundtrip_addr(address a) returns (address) {
  return abi.decode(abi.encode(a), (address));
}

function roundtrip_addr_tuple(address a, uint256 x) returns (address, uint256) {
  return abi.decode(abi.encode(a, x), (address, uint256));
}

function roundtrip_addr_payable(address a) returns (address) {
  return abi.decode(abi.encode(payable(a)), (address));
}

contract CC {
  function foo() external pure {}
}

function encode_contract(CC c) returns (bytes memory) {
  return abi.encode(c);
}

function encode_packed_contract(CC c) returns (bytes memory) {
  return abi.encodePacked(c);
}

function decode_contract(bytes memory data) returns (CC) {
  return abi.decode(data, (CC));
}

function roundtrip_contract(CC c) returns (CC) {
  return abi.decode(abi.encode(c), (CC));
}

function decode_contract_tuple(bytes memory data) returns (CC, uint256) {
  return abi.decode(data, (CC, uint256));
}

function encode_selector(bytes4 sel, uint256 x) returns (bytes memory) {
  return abi.encodeWithSelector(sel, x);
}

function encode_signature_literal() returns (bytes memory) {
  return abi.encodeWithSignature("bar(uint256)");
}

function encode_signature_runtime(string memory sig) returns (bytes memory) {
  return abi.encodeWithSignature(sig);
}

function encode_signature_runtime_calldata(string calldata sig) returns (bytes memory) {
  return abi.encodeWithSignature(sig);
}

interface I {
  function foo(uint256) external;

  function bar(uint256, uint8) external;
}

function encode_call_non_tuple(uint256 x) returns (bytes memory) {
  return abi.encodeCall(I.foo, x);
}

function encode_call_tuple(uint256 x, uint8 y) returns (bytes memory) {
  return abi.encodeCall(I.bar, (x, y));
}

function encode_fnptr(function(uint) external pure returns (uint) fp) pure returns (bytes memory) {
  return abi.encode(fp);
}

function encode_packed_fnptr(function(uint) external pure returns (uint) fp) pure returns (bytes memory) {
  return abi.encodePacked(fp);
}

function encode_packed_fnptr_array(function(uint) external pure returns (uint)[] memory arr) pure returns (bytes memory) {
  return abi.encodePacked(arr);
}

contract StorageBytes {
  bytes str;
  bytes str2;
  string sig;

  function ep_bytes_storage(bytes calldata a) public returns (bytes memory) {
    str = a;
    return abi.encodePacked(str);
  }

  function ep_bytes_concat_storage(bytes calldata a, bytes calldata b) public returns (bytes memory) {
    str = a;
    str2 = b;
    return abi.encodePacked(str, str2);
  }

  function ews_bytes_bytes_storage(bytes4 sel, bytes memory data) public returns (bytes memory) {
    str = data;
    return abi.encodeWithSelector(sel, str);
  }

  function ewsig_runtime_storage(string memory runtimeSig, uint256 x) public returns (bytes memory) {
    sig = runtimeSig;
    return abi.encodeWithSignature(sig, x);
  }
}

contract Magic {
  function callMsgSender() public returns (address) { return msgSender(); }
  function callAddr() public returns (address) { return addr(); }
  function callEncode(uint ui, uint8 ui8, int32 si32) public returns (bytes memory) { return encode(ui, ui8, si32); }
  function callDecode(bytes memory a) public returns (uint, uint8, int32) { return decode(a); }
  function callEncodePacked(uint24 x, uint96 y, uint136 z) public returns (bytes memory) { return encode_packed(x, y, z); }
  function callEncodeAddr(address a) public returns (bytes memory) { return encode_addr(a); }
  function callDecodeAddr(bytes memory a) public returns (address) { return decode_addr(a); }
  function callEncodePackedAddr(address a) public returns (bytes memory) { return encode_packed_addr(a); }
  function callRoundtripAddr(address a) public returns (address) { return roundtrip_addr(a); }
  function callRoundtripAddrTuple(address a, uint256 x) public returns (address, uint256) { return roundtrip_addr_tuple(a, x); }
  function callRoundtripAddrPayable(address a) public returns (address) { return roundtrip_addr_payable(a); }
  function callEncodeContract(CC c) public returns (bytes memory) { return encode_contract(c); }
  function callEncodePackedContract(CC c) public returns (bytes memory) { return encode_packed_contract(c); }
  function callDecodeContract(bytes memory data) public returns (CC) { return decode_contract(data); }
  function callRoundtripContract(CC c) public returns (CC) { return roundtrip_contract(c); }
  function callDecodeContractTuple(bytes memory data) public returns (CC, uint256) { return decode_contract_tuple(data); }
  function callEncodeSelector(bytes4 sel, uint256 x) public returns (bytes memory) { return encode_selector(sel, x); }
  function callEncodeSignatureLiteral() public returns (bytes memory) { return encode_signature_literal(); }
  function callEncodeSignatureRuntime(string memory sig) public returns (bytes memory) { return encode_signature_runtime(sig); }
  function callEncodeSignatureRuntimeCalldata(string calldata sig) public returns (bytes memory) { return encode_signature_runtime_calldata(sig); }
  function callEncodeCallNonTuple(uint256 x) public returns (bytes memory) { return encode_call_non_tuple(x); }
  function callEncodeCallTuple(uint256 x, uint8 y) public returns (bytes memory) { return encode_call_tuple(x, y); }
  function callEncodeFnptr(function(uint) external pure returns (uint) fp) public returns (bytes memory) { return encode_fnptr(fp); }
  function callEncodePackedFnptr(function(uint) external pure returns (uint) fp) public returns (bytes memory) { return encode_packed_fnptr(fp); }
  function callEncodePackedFnptrArray(function(uint) external pure returns (uint)[] memory arr) public returns (bytes memory) { return encode_packed_fnptr_array(arr); }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #Constructor = #sol<FunctionKind Constructor>
// CHECK-NEXT: #Contract = #sol<ContractKind Contract>
// CHECK-NEXT: #Default = #sol<RevertStrings Default>
// CHECK-NEXT: #NonPayable = #sol<StateMutability NonPayable>
// CHECK-NEXT: #Osaka = #sol<EvmVersion Osaka>
// CHECK-NEXT: #Pure = #sol<StateMutability Pure>
// CHECK-NEXT: module attributes {llvm.data_layout = "E-p:256:256-i256:256:256-S256-a:256:256", llvm.target_triple = "evm-unknown-unknown", sol.evm_version = #Osaka, sol.revert_strings = #Default} {
// CHECK-NEXT:   sol.contract @CC_196 {
// CHECK-NEXT:     sol.func @CC_196() attributes {kind = #Constructor, orig_fn_type = () -> (), state_mutability = #NonPayable} {
// CHECK-NEXT:       sol.return loc(#loc1)
// CHECK-NEXT:     } loc(#loc1)
// CHECK-NEXT:     sol.func @foo_195() attributes {id = 195 : i64, orig_fn_type = () -> (), selector = -1030204040 : i32, state_mutability = #Pure} {
// CHECK-NEXT:       sol.return loc(#loc2)
// CHECK-NEXT:     } loc(#loc2)
// CHECK-NEXT:   } {kind = #Contract} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:46:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:47:2)
// CHECK-NEXT: #Constructor = #sol<FunctionKind Constructor>
// CHECK-NEXT: #Contract = #sol<ContractKind Contract>
// CHECK-NEXT: #Default = #sol<RevertStrings Default>
// CHECK-NEXT: #NonPayable = #sol<StateMutability NonPayable>
// CHECK-NEXT: #Osaka = #sol<EvmVersion Osaka>
// CHECK-NEXT: #loc6 = loc({{.*}}:117:28)
// CHECK-NEXT: #loc12 = loc({{.*}}:122:35)
// CHECK-NEXT: #loc13 = loc({{.*}}:122:53)
// CHECK-NEXT: #loc21 = loc({{.*}}:128:35)
// CHECK-NEXT: #loc22 = loc({{.*}}:128:47)
// CHECK-NEXT: #loc29 = loc({{.*}}:133:33)
// CHECK-NEXT: #loc30 = loc({{.*}}:133:59)
// CHECK-NEXT: module attributes {llvm.data_layout = "E-p:256:256-i256:256:256-S256-a:256:256", llvm.target_triple = "evm-unknown-unknown", sol.evm_version = #Osaka, sol.revert_strings = #Default} {
// CHECK-NEXT:   sol.contract @StorageBytes_526 {
// CHECK-NEXT:     sol.state_var @str_440 slot 0 offset 0 : !sol.string<Storage> loc(#loc2)
// CHECK-NEXT:     sol.state_var @str2_442 slot 1 offset 0 : !sol.string<Storage> loc(#loc3)
// CHECK-NEXT:     sol.state_var @sig_444 slot 2 offset 0 : !sol.string<Storage> loc(#loc4)
// CHECK-NEXT:     sol.func @StorageBytes_526() attributes {kind = #Constructor, orig_fn_type = () -> (), state_mutability = #NonPayable} {
// CHECK-NEXT:       sol.return loc(#loc1)
// CHECK-NEXT:     } loc(#loc1)
// CHECK-NEXT:     sol.func @ep_bytes_storage_461(%arg0: !sol.string<CallData> loc({{.*}}:117:28)) -> !sol.string<Memory> attributes {id = 461 : i64, orig_fn_type = (!sol.string<CallData>) -> !sol.string<Memory>, selector = -1796405943 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<CallData>, Stack> loc(#loc6)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<CallData>, !sol.ptr<!sol.string<CallData>, Stack> loc(#loc6)
// CHECK-NEXT:       %1 = sol.addr_of @str_440 : !sol.string<Storage> loc(#loc2)
// CHECK-NEXT:       %2 = sol.load %0 : !sol.ptr<!sol.string<CallData>, Stack>, !sol.string<CallData> loc(#loc7)
// CHECK-NEXT:       sol.copy %2, %1 : !sol.string<CallData>, !sol.string<Storage> loc(#loc8)
// CHECK-NEXT:       %3 = sol.addr_of @str_440 : !sol.string<Storage> loc(#loc2)
// CHECK-NEXT:       %4 = sol.encode %3 :  !sol.string<Storage> : !sol.string<Memory> {packed} loc(#loc9)
// CHECK-NEXT:       sol.return %4 : !sol.string<Memory> loc(#loc10)
// CHECK-NEXT:     } loc(#loc5)
// CHECK-NEXT:     sol.func @ep_bytes_concat_storage_485(%arg0: !sol.string<CallData> loc({{.*}}:122:35), %arg1: !sol.string<CallData> loc({{.*}}:122:53)) -> !sol.string<Memory> attributes {id = 485 : i64, orig_fn_type = (!sol.string<CallData>, !sol.string<CallData>) -> !sol.string<Memory>, selector = 1136153535 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<CallData>, Stack> loc(#loc12)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<CallData>, !sol.ptr<!sol.string<CallData>, Stack> loc(#loc12)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<!sol.string<CallData>, Stack> loc(#loc13)
// CHECK-NEXT:       sol.store %arg1, %1 : !sol.string<CallData>, !sol.ptr<!sol.string<CallData>, Stack> loc(#loc13)
// CHECK-NEXT:       %2 = sol.addr_of @str_440 : !sol.string<Storage> loc(#loc2)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<!sol.string<CallData>, Stack>, !sol.string<CallData> loc(#loc14)
// CHECK-NEXT:       sol.copy %3, %2 : !sol.string<CallData>, !sol.string<Storage> loc(#loc15)
// CHECK-NEXT:       %4 = sol.addr_of @str2_442 : !sol.string<Storage> loc(#loc3)
// CHECK-NEXT:       %5 = sol.load %1 : !sol.ptr<!sol.string<CallData>, Stack>, !sol.string<CallData> loc(#loc16)
// CHECK-NEXT:       sol.copy %5, %4 : !sol.string<CallData>, !sol.string<Storage> loc(#loc17)
// CHECK-NEXT:       %6 = sol.addr_of @str_440 : !sol.string<Storage> loc(#loc2)
// CHECK-NEXT:       %7 = sol.addr_of @str2_442 : !sol.string<Storage> loc(#loc3)
// CHECK-NEXT:       %8 = sol.encode %6, %7 :  !sol.string<Storage>, !sol.string<Storage> : !sol.string<Memory> {packed} loc(#loc18)
// CHECK-NEXT:       sol.return %8 : !sol.string<Memory> loc(#loc19)
// CHECK-NEXT:     } loc(#loc11)
// CHECK-NEXT:     sol.func @ews_bytes_bytes_storage_505(%arg0: !sol.fixedbytes<4> loc({{.*}}:128:35), %arg1: !sol.string<Memory> loc({{.*}}:128:47)) -> !sol.string<Memory> attributes {id = 505 : i64, orig_fn_type = (!sol.fixedbytes<4>, !sol.string<Memory>) -> !sol.string<Memory>, selector = -1336816964 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.fixedbytes<4>, Stack> loc(#loc21)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.fixedbytes<4>, !sol.ptr<!sol.fixedbytes<4>, Stack> loc(#loc21)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc22)
// CHECK-NEXT:       sol.store %arg1, %1 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc22)
// CHECK-NEXT:       %2 = sol.addr_of @str_440 : !sol.string<Storage> loc(#loc2)
// CHECK-NEXT:       %3 = sol.load %1 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc23)
// CHECK-NEXT:       sol.copy %3, %2 : !sol.string<Memory>, !sol.string<Storage> loc(#loc24)
// CHECK-NEXT:       %4 = sol.load %0 : !sol.ptr<!sol.fixedbytes<4>, Stack>, !sol.fixedbytes<4> loc(#loc25)
// CHECK-NEXT:       %5 = sol.addr_of @str_440 : !sol.string<Storage> loc(#loc2)
// CHECK-NEXT:       %6 = sol.encode selector(%4) %5 : !sol.fixedbytes<4> !sol.string<Storage> : !sol.string<Memory> loc(#loc26)
// CHECK-NEXT:       sol.return %6 : !sol.string<Memory> loc(#loc27)
// CHECK-NEXT:     } loc(#loc20)
// CHECK-NEXT:     sol.func @ewsig_runtime_storage_525(%arg0: !sol.string<Memory> loc({{.*}}:133:33), %arg1: ui256 loc({{.*}}:133:59)) -> !sol.string<Memory> attributes {id = 525 : i64, orig_fn_type = (!sol.string<Memory>, ui256) -> !sol.string<Memory>, selector = 331104600 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc29)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc29)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc30)
// CHECK-NEXT:       sol.store %arg1, %1 : ui256, !sol.ptr<ui256, Stack> loc(#loc30)
// CHECK-NEXT:       %2 = sol.addr_of @sig_444 : !sol.string<Storage> loc(#loc4)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc31)
// CHECK-NEXT:       sol.copy %3, %2 : !sol.string<Memory>, !sol.string<Storage> loc(#loc32)
// CHECK-NEXT:       %4 = sol.addr_of @sig_444 : !sol.string<Storage> loc(#loc4)
// CHECK-NEXT:       %5 = sol.data_loc_cast %4 : !sol.string<Storage>, !sol.string<Memory> loc(#loc4)
// CHECK-NEXT:       %6 = "sol.keccak256"(%5) : (!sol.string<Memory>) -> !sol.fixedbytes<32> loc(#loc33)
// CHECK-NEXT:       %7 = sol.bytes_cast %6 : !sol.fixedbytes<32> to !sol.fixedbytes<4> loc(#loc33)
// CHECK-NEXT:       %8 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc34)
// CHECK-NEXT:       %9 = sol.encode selector(%7) %8 : !sol.fixedbytes<4> ui256 : !sol.string<Memory> loc(#loc33)
// CHECK-NEXT:       sol.return %9 : !sol.string<Memory> loc(#loc35)
// CHECK-NEXT:     } loc(#loc28)
// CHECK-NEXT:   } {kind = #Contract} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:112:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:113:2)
// CHECK-NEXT: #loc3 = loc({{.*}}:114:2)
// CHECK-NEXT: #loc4 = loc({{.*}}:115:2)
// CHECK-NEXT: #loc5 = loc({{.*}}:117:2)
// CHECK-NEXT: #loc7 = loc({{.*}}:118:10)
// CHECK-NEXT: #loc8 = loc({{.*}}:118:4)
// CHECK-NEXT: #loc9 = loc({{.*}}:119:11)
// CHECK-NEXT: #loc10 = loc({{.*}}:119:4)
// CHECK-NEXT: #loc11 = loc({{.*}}:122:2)
// CHECK-NEXT: #loc14 = loc({{.*}}:123:10)
// CHECK-NEXT: #loc15 = loc({{.*}}:123:4)
// CHECK-NEXT: #loc16 = loc({{.*}}:124:11)
// CHECK-NEXT: #loc17 = loc({{.*}}:124:4)
// CHECK-NEXT: #loc18 = loc({{.*}}:125:11)
// CHECK-NEXT: #loc19 = loc({{.*}}:125:4)
// CHECK-NEXT: #loc20 = loc({{.*}}:128:2)
// CHECK-NEXT: #loc23 = loc({{.*}}:129:10)
// CHECK-NEXT: #loc24 = loc({{.*}}:129:4)
// CHECK-NEXT: #loc25 = loc({{.*}}:130:34)
// CHECK-NEXT: #loc26 = loc({{.*}}:130:11)
// CHECK-NEXT: #loc27 = loc({{.*}}:130:4)
// CHECK-NEXT: #loc28 = loc({{.*}}:133:2)
// CHECK-NEXT: #loc31 = loc({{.*}}:134:10)
// CHECK-NEXT: #loc32 = loc({{.*}}:134:4)
// CHECK-NEXT: #loc33 = loc({{.*}}:135:11)
// CHECK-NEXT: #loc34 = loc({{.*}}:135:40)
// CHECK-NEXT: #loc35 = loc({{.*}}:135:4)
// CHECK-NEXT: #Constructor = #sol<FunctionKind Constructor>
// CHECK-NEXT: #Contract = #sol<ContractKind Contract>
// CHECK-NEXT: #Default = #sol<RevertStrings Default>
// CHECK-NEXT: #NonPayable = #sol<StateMutability NonPayable>
// CHECK-NEXT: #Osaka = #sol<EvmVersion Osaka>
// CHECK-NEXT: #Pure = #sol<StateMutability Pure>
// CHECK-NEXT: #loc15 = loc({{.*}}:10:16)
// CHECK-NEXT: #loc16 = loc({{.*}}:10:25)
// CHECK-NEXT: #loc17 = loc({{.*}}:10:36)
// CHECK-NEXT: #loc24 = loc({{.*}}:142:22)
// CHECK-NEXT: #loc25 = loc({{.*}}:142:31)
// CHECK-NEXT: #loc26 = loc({{.*}}:142:42)
// CHECK-NEXT: #loc33 = loc({{.*}}:14:16)
// CHECK-NEXT: #loc38 = loc({{.*}}:143:22)
// CHECK-NEXT: #loc43 = loc({{.*}}:18:23)
// CHECK-NEXT: #loc44 = loc({{.*}}:18:33)
// CHECK-NEXT: #loc45 = loc({{.*}}:18:43)
// CHECK-NEXT: #loc52 = loc({{.*}}:144:28)
// CHECK-NEXT: #loc53 = loc({{.*}}:144:38)
// CHECK-NEXT: #loc54 = loc({{.*}}:144:48)
// CHECK-NEXT: #loc61 = loc({{.*}}:22:21)
// CHECK-NEXT: #loc66 = loc({{.*}}:145:26)
// CHECK-NEXT: #loc71 = loc({{.*}}:26:21)
// CHECK-NEXT: #loc76 = loc({{.*}}:146:26)
// CHECK-NEXT: #loc81 = loc({{.*}}:30:28)
// CHECK-NEXT: #loc86 = loc({{.*}}:147:32)
// CHECK-NEXT: #loc91 = loc({{.*}}:34:24)
// CHECK-NEXT: #loc97 = loc({{.*}}:148:29)
// CHECK-NEXT: #loc102 = loc({{.*}}:38:30)
// CHECK-NEXT: #loc103 = loc({{.*}}:38:41)
// CHECK-NEXT: #loc110 = loc({{.*}}:149:34)
// CHECK-NEXT: #loc111 = loc({{.*}}:149:45)
// CHECK-NEXT: #loc117 = loc({{.*}}:42:32)
// CHECK-NEXT: #loc123 = loc({{.*}}:150:36)
// CHECK-NEXT: #loc128 = loc({{.*}}:50:25)
// CHECK-NEXT: #loc133 = loc({{.*}}:151:30)
// CHECK-NEXT: #loc138 = loc({{.*}}:54:32)
// CHECK-NEXT: #loc143 = loc({{.*}}:152:36)
// CHECK-NEXT: #loc148 = loc({{.*}}:58:25)
// CHECK-NEXT: #loc153 = loc({{.*}}:153:30)
// CHECK-NEXT: #loc158 = loc({{.*}}:62:28)
// CHECK-NEXT: #loc164 = loc({{.*}}:154:33)
// CHECK-NEXT: #loc169 = loc({{.*}}:66:31)
// CHECK-NEXT: #loc174 = loc({{.*}}:155:35)
// CHECK-NEXT: #loc179 = loc({{.*}}:70:25)
// CHECK-NEXT: #loc180 = loc({{.*}}:70:37)
// CHECK-NEXT: #loc186 = loc({{.*}}:156:30)
// CHECK-NEXT: #loc187 = loc({{.*}}:156:42)
// CHECK-NEXT: #loc199 = loc({{.*}}:78:34)
// CHECK-NEXT: #loc204 = loc({{.*}}:158:38)
// CHECK-NEXT: #loc209 = loc({{.*}}:82:43)
// CHECK-NEXT: #loc214 = loc({{.*}}:159:46)
// CHECK-NEXT: #loc219 = loc({{.*}}:92:31)
// CHECK-NEXT: #loc224 = loc({{.*}}:160:34)
// CHECK-NEXT: #loc229 = loc({{.*}}:96:27)
// CHECK-NEXT: #loc230 = loc({{.*}}:96:38)
// CHECK-NEXT: #loc236 = loc({{.*}}:161:31)
// CHECK-NEXT: #loc237 = loc({{.*}}:161:42)
// CHECK-NEXT: #loc243 = loc({{.*}}:100:22)
// CHECK-NEXT: #loc248 = loc({{.*}}:162:27)
// CHECK-NEXT: #loc253 = loc({{.*}}:104:29)
// CHECK-NEXT: #loc258 = loc({{.*}}:163:33)
// CHECK-NEXT: #loc263 = loc({{.*}}:108:35)
// CHECK-NEXT: #loc268 = loc({{.*}}:164:38)
// CHECK-NEXT: module attributes {llvm.data_layout = "E-p:256:256-i256:256:256-S256-a:256:256", llvm.target_triple = "evm-unknown-unknown", sol.evm_version = #Osaka, sol.revert_strings = #Default} {
// CHECK-NEXT:   sol.contract @Magic_872 {
// CHECK-NEXT:     sol.func @Magic_872() attributes {kind = #Constructor, orig_fn_type = () -> (), state_mutability = #NonPayable} {
// CHECK-NEXT:       sol.return loc(#loc1)
// CHECK-NEXT:     } loc(#loc1)
// CHECK-NEXT:     sol.func @msgSender_9() -> !sol.address attributes {id = 9 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.caller : !sol.address loc(#loc3)
// CHECK-NEXT:       sol.return %0 : !sol.address loc(#loc4)
// CHECK-NEXT:     } loc(#loc2)
// CHECK-NEXT:     sol.func @callMsgSender_535() -> !sol.address attributes {id = 535 : i64, orig_fn_type = () -> !sol.address, selector = 1795483785 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.call @msgSender_9() : () -> !sol.address loc(#loc6)
// CHECK-NEXT:       sol.return %0 : !sol.address loc(#loc7)
// CHECK-NEXT:     } loc(#loc5)
// CHECK-NEXT:     sol.func @addr_20() -> !sol.address attributes {id = 20 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %c0_ui8 = sol.constant 0 : ui8 loc(#loc9)
// CHECK-NEXT:       %0 = sol.cast %c0_ui8 : ui8 to ui160 loc(#loc9)
// CHECK-NEXT:       %1 = sol.address_cast %0 : ui160 to !sol.address loc(#loc9)
// CHECK-NEXT:       sol.return %1 : !sol.address loc(#loc10)
// CHECK-NEXT:     } loc(#loc8)
// CHECK-NEXT:     sol.func @callAddr_544() -> !sol.address attributes {id = 544 : i64, orig_fn_type = () -> !sol.address, selector = -1409954593 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.call @addr_20() : () -> !sol.address loc(#loc12)
// CHECK-NEXT:       sol.return %0 : !sol.address loc(#loc13)
// CHECK-NEXT:     } loc(#loc11)
// CHECK-NEXT:     sol.func @encode_39(%arg0: ui256 loc({{.*}}:10:16), %arg1: ui8 loc({{.*}}:10:25), %arg2: si32 loc({{.*}}:10:36)) -> !sol.string<Memory> attributes {id = 39 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc15)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc15)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui8, Stack> loc(#loc16)
// CHECK-NEXT:       sol.store %arg1, %1 : ui8, !sol.ptr<ui8, Stack> loc(#loc16)
// CHECK-NEXT:       %2 = sol.alloca : !sol.ptr<si32, Stack> loc(#loc17)
// CHECK-NEXT:       sol.store %arg2, %2 : si32, !sol.ptr<si32, Stack> loc(#loc17)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc18)
// CHECK-NEXT:       %4 = sol.load %1 : !sol.ptr<ui8, Stack>, ui8 loc(#loc19)
// CHECK-NEXT:       %5 = sol.load %2 : !sol.ptr<si32, Stack>, si32 loc(#loc20)
// CHECK-NEXT:       %6 = sol.encode %3, %4, %5 :  ui256, ui8, si32 : !sol.string<Memory> loc(#loc21)
// CHECK-NEXT:       sol.return %6 : !sol.string<Memory> loc(#loc22)
// CHECK-NEXT:     } loc(#loc14)
// CHECK-NEXT:     sol.func @callEncode_562(%arg0: ui256 loc({{.*}}:142:22), %arg1: ui8 loc({{.*}}:142:31), %arg2: si32 loc({{.*}}:142:42)) -> !sol.string<Memory> attributes {id = 562 : i64, orig_fn_type = (ui256, ui8, si32) -> !sol.string<Memory>, selector = -883344432 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc24)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc24)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui8, Stack> loc(#loc25)
// CHECK-NEXT:       sol.store %arg1, %1 : ui8, !sol.ptr<ui8, Stack> loc(#loc25)
// CHECK-NEXT:       %2 = sol.alloca : !sol.ptr<si32, Stack> loc(#loc26)
// CHECK-NEXT:       sol.store %arg2, %2 : si32, !sol.ptr<si32, Stack> loc(#loc26)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc27)
// CHECK-NEXT:       %4 = sol.load %1 : !sol.ptr<ui8, Stack>, ui8 loc(#loc28)
// CHECK-NEXT:       %5 = sol.load %2 : !sol.ptr<si32, Stack>, si32 loc(#loc29)
// CHECK-NEXT:       %6 = sol.call @encode_39(%3, %4, %5) : (ui256, ui8, si32) -> !sol.string<Memory> loc(#loc30)
// CHECK-NEXT:       sol.return %6 : !sol.string<Memory> loc(#loc31)
// CHECK-NEXT:     } loc(#loc23)
// CHECK-NEXT:     sol.func @decode_63(%arg0: !sol.string<Memory> loc({{.*}}:14:16)) -> (ui256, ui8, si32) attributes {id = 63 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc33)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc33)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc34)
// CHECK-NEXT:       %2:3 = sol.decode %1 : !sol.string<Memory> -> ui256, ui8, si32 loc(#loc35)
// CHECK-NEXT:       sol.return %2#0, %2#1, %2#2 : ui256, ui8, si32 loc(#loc36)
// CHECK-NEXT:     } loc(#loc32)
// CHECK-NEXT:     sol.func @callDecode_578(%arg0: !sol.string<Memory> loc({{.*}}:143:22)) -> (ui256, ui8, si32) attributes {id = 578 : i64, orig_fn_type = (!sol.string<Memory>) -> (ui256, ui8, si32), selector = -237943393 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc38)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc38)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc39)
// CHECK-NEXT:       %2:3 = sol.call @decode_63(%1) : (!sol.string<Memory>) -> (ui256, ui8, si32) loc(#loc40)
// CHECK-NEXT:       sol.return %2#0, %2#1, %2#2 : ui256, ui8, si32 loc(#loc41)
// CHECK-NEXT:     } loc(#loc37)
// CHECK-NEXT:     sol.func @encode_packed_82(%arg0: ui24 loc({{.*}}:18:23), %arg1: ui96 loc({{.*}}:18:33), %arg2: ui136 loc({{.*}}:18:43)) -> !sol.string<Memory> attributes {id = 82 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui24, Stack> loc(#loc43)
// CHECK-NEXT:       sol.store %arg0, %0 : ui24, !sol.ptr<ui24, Stack> loc(#loc43)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui96, Stack> loc(#loc44)
// CHECK-NEXT:       sol.store %arg1, %1 : ui96, !sol.ptr<ui96, Stack> loc(#loc44)
// CHECK-NEXT:       %2 = sol.alloca : !sol.ptr<ui136, Stack> loc(#loc45)
// CHECK-NEXT:       sol.store %arg2, %2 : ui136, !sol.ptr<ui136, Stack> loc(#loc45)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<ui24, Stack>, ui24 loc(#loc46)
// CHECK-NEXT:       %4 = sol.load %1 : !sol.ptr<ui96, Stack>, ui96 loc(#loc47)
// CHECK-NEXT:       %5 = sol.load %2 : !sol.ptr<ui136, Stack>, ui136 loc(#loc48)
// CHECK-NEXT:       %6 = sol.encode %3, %4, %5 :  ui24, ui96, ui136 : !sol.string<Memory> {packed} loc(#loc49)
// CHECK-NEXT:       sol.return %6 : !sol.string<Memory> loc(#loc50)
// CHECK-NEXT:     } loc(#loc42)
// CHECK-NEXT:     sol.func @callEncodePacked_596(%arg0: ui24 loc({{.*}}:144:28), %arg1: ui96 loc({{.*}}:144:38), %arg2: ui136 loc({{.*}}:144:48)) -> !sol.string<Memory> attributes {id = 596 : i64, orig_fn_type = (ui24, ui96, ui136) -> !sol.string<Memory>, selector = -695057225 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui24, Stack> loc(#loc52)
// CHECK-NEXT:       sol.store %arg0, %0 : ui24, !sol.ptr<ui24, Stack> loc(#loc52)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui96, Stack> loc(#loc53)
// CHECK-NEXT:       sol.store %arg1, %1 : ui96, !sol.ptr<ui96, Stack> loc(#loc53)
// CHECK-NEXT:       %2 = sol.alloca : !sol.ptr<ui136, Stack> loc(#loc54)
// CHECK-NEXT:       sol.store %arg2, %2 : ui136, !sol.ptr<ui136, Stack> loc(#loc54)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<ui24, Stack>, ui24 loc(#loc55)
// CHECK-NEXT:       %4 = sol.load %1 : !sol.ptr<ui96, Stack>, ui96 loc(#loc56)
// CHECK-NEXT:       %5 = sol.load %2 : !sol.ptr<ui136, Stack>, ui136 loc(#loc57)
// CHECK-NEXT:       %6 = sol.call @encode_packed_82(%3, %4, %5) : (ui24, ui96, ui136) -> !sol.string<Memory> loc(#loc58)
// CHECK-NEXT:       sol.return %6 : !sol.string<Memory> loc(#loc59)
// CHECK-NEXT:     } loc(#loc51)
// CHECK-NEXT:     sol.func @encode_addr_95(%arg0: !sol.address loc({{.*}}:22:21)) -> !sol.string<Memory> attributes {id = 95 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.address, Stack> loc(#loc61)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.address, !sol.ptr<!sol.address, Stack> loc(#loc61)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.address, Stack>, !sol.address loc(#loc62)
// CHECK-NEXT:       %2 = sol.encode %1 :  !sol.address : !sol.string<Memory> loc(#loc63)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc64)
// CHECK-NEXT:     } loc(#loc60)
// CHECK-NEXT:     sol.func @callEncodeAddr_608(%arg0: !sol.address loc({{.*}}:145:26)) -> !sol.string<Memory> attributes {id = 608 : i64, orig_fn_type = (!sol.address) -> !sol.string<Memory>, selector = 1199621205 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.address, Stack> loc(#loc66)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.address, !sol.ptr<!sol.address, Stack> loc(#loc66)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.address, Stack>, !sol.address loc(#loc67)
// CHECK-NEXT:       %2 = sol.call @encode_addr_95(%1) : (!sol.address) -> !sol.string<Memory> loc(#loc68)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc69)
// CHECK-NEXT:     } loc(#loc65)
// CHECK-NEXT:     sol.func @decode_addr_111(%arg0: !sol.string<Memory> loc({{.*}}:26:21)) -> !sol.address attributes {id = 111 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc71)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc71)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc72)
// CHECK-NEXT:       %2 = sol.decode %1 : !sol.string<Memory> -> !sol.address<payable> loc(#loc73)
// CHECK-NEXT:       %3 = sol.address_cast %2 : !sol.address<payable> to !sol.address loc(#loc73)
// CHECK-NEXT:       sol.return %3 : !sol.address loc(#loc74)
// CHECK-NEXT:     } loc(#loc70)
// CHECK-NEXT:     sol.func @callDecodeAddr_620(%arg0: !sol.string<Memory> loc({{.*}}:146:26)) -> !sol.address attributes {id = 620 : i64, orig_fn_type = (!sol.string<Memory>) -> !sol.address, selector = 546619949 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc76)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc76)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc77)
// CHECK-NEXT:       %2 = sol.call @decode_addr_111(%1) : (!sol.string<Memory>) -> !sol.address loc(#loc78)
// CHECK-NEXT:       sol.return %2 : !sol.address loc(#loc79)
// CHECK-NEXT:     } loc(#loc75)
// CHECK-NEXT:     sol.func @encode_packed_addr_124(%arg0: !sol.address loc({{.*}}:30:28)) -> !sol.string<Memory> attributes {id = 124 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.address, Stack> loc(#loc81)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.address, !sol.ptr<!sol.address, Stack> loc(#loc81)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.address, Stack>, !sol.address loc(#loc82)
// CHECK-NEXT:       %2 = sol.encode %1 :  !sol.address : !sol.string<Memory> {packed} loc(#loc83)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc84)
// CHECK-NEXT:     } loc(#loc80)
// CHECK-NEXT:     sol.func @callEncodePackedAddr_632(%arg0: !sol.address loc({{.*}}:147:32)) -> !sol.string<Memory> attributes {id = 632 : i64, orig_fn_type = (!sol.address) -> !sol.string<Memory>, selector = -1733599413 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.address, Stack> loc(#loc86)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.address, !sol.ptr<!sol.address, Stack> loc(#loc86)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.address, Stack>, !sol.address loc(#loc87)
// CHECK-NEXT:       %2 = sol.call @encode_packed_addr_124(%1) : (!sol.address) -> !sol.string<Memory> loc(#loc88)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc89)
// CHECK-NEXT:     } loc(#loc85)
// CHECK-NEXT:     sol.func @roundtrip_addr_143(%arg0: !sol.address loc({{.*}}:34:24)) -> !sol.address attributes {id = 143 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.address, Stack> loc(#loc91)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.address, !sol.ptr<!sol.address, Stack> loc(#loc91)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.address, Stack>, !sol.address loc(#loc92)
// CHECK-NEXT:       %2 = sol.encode %1 :  !sol.address : !sol.string<Memory> loc(#loc93)
// CHECK-NEXT:       %3 = sol.decode %2 : !sol.string<Memory> -> !sol.address<payable> loc(#loc94)
// CHECK-NEXT:       %4 = sol.address_cast %3 : !sol.address<payable> to !sol.address loc(#loc94)
// CHECK-NEXT:       sol.return %4 : !sol.address loc(#loc95)
// CHECK-NEXT:     } loc(#loc90)
// CHECK-NEXT:     sol.func @callRoundtripAddr_644(%arg0: !sol.address loc({{.*}}:148:29)) -> !sol.address attributes {id = 644 : i64, orig_fn_type = (!sol.address) -> !sol.address, selector = -136273165 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.address, Stack> loc(#loc97)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.address, !sol.ptr<!sol.address, Stack> loc(#loc97)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.address, Stack>, !sol.address loc(#loc98)
// CHECK-NEXT:       %2 = sol.call @roundtrip_addr_143(%1) : (!sol.address) -> !sol.address loc(#loc99)
// CHECK-NEXT:       sol.return %2 : !sol.address loc(#loc100)
// CHECK-NEXT:     } loc(#loc96)
// CHECK-NEXT:     sol.func @roundtrip_addr_tuple_169(%arg0: !sol.address loc({{.*}}:38:30), %arg1: ui256 loc({{.*}}:38:41)) -> (!sol.address, ui256) attributes {id = 169 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.address, Stack> loc(#loc102)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.address, !sol.ptr<!sol.address, Stack> loc(#loc102)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc103)
// CHECK-NEXT:       sol.store %arg1, %1 : ui256, !sol.ptr<ui256, Stack> loc(#loc103)
// CHECK-NEXT:       %2 = sol.load %0 : !sol.ptr<!sol.address, Stack>, !sol.address loc(#loc104)
// CHECK-NEXT:       %3 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc105)
// CHECK-NEXT:       %4 = sol.encode %2, %3 :  !sol.address, ui256 : !sol.string<Memory> loc(#loc106)
// CHECK-NEXT:       %5:2 = sol.decode %4 : !sol.string<Memory> -> !sol.address<payable>, ui256 loc(#loc107)
// CHECK-NEXT:       %6 = sol.address_cast %5#0 : !sol.address<payable> to !sol.address loc(#loc107)
// CHECK-NEXT:       sol.return %6, %5#1 : !sol.address, ui256 loc(#loc108)
// CHECK-NEXT:     } loc(#loc101)
// CHECK-NEXT:     sol.func @callRoundtripAddrTuple_661(%arg0: !sol.address loc({{.*}}:149:34), %arg1: ui256 loc({{.*}}:149:45)) -> (!sol.address, ui256) attributes {id = 661 : i64, orig_fn_type = (!sol.address, ui256) -> (!sol.address, ui256), selector = -608869620 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.address, Stack> loc(#loc110)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.address, !sol.ptr<!sol.address, Stack> loc(#loc110)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc111)
// CHECK-NEXT:       sol.store %arg1, %1 : ui256, !sol.ptr<ui256, Stack> loc(#loc111)
// CHECK-NEXT:       %2 = sol.load %0 : !sol.ptr<!sol.address, Stack>, !sol.address loc(#loc112)
// CHECK-NEXT:       %3 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc113)
// CHECK-NEXT:       %4:2 = sol.call @roundtrip_addr_tuple_169(%2, %3) : (!sol.address, ui256) -> (!sol.address, ui256) loc(#loc114)
// CHECK-NEXT:       sol.return %4#0, %4#1 : !sol.address, ui256 loc(#loc115)
// CHECK-NEXT:     } loc(#loc109)
// CHECK-NEXT:     sol.func @roundtrip_addr_payable_191(%arg0: !sol.address loc({{.*}}:42:32)) -> !sol.address attributes {id = 191 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.address, Stack> loc(#loc117)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.address, !sol.ptr<!sol.address, Stack> loc(#loc117)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.address, Stack>, !sol.address loc(#loc118)
// CHECK-NEXT:       %2 = sol.address_cast %1 : !sol.address to !sol.address<payable> loc(#loc118)
// CHECK-NEXT:       %3 = sol.encode %2 :  !sol.address<payable> : !sol.string<Memory> loc(#loc119)
// CHECK-NEXT:       %4 = sol.decode %3 : !sol.string<Memory> -> !sol.address<payable> loc(#loc120)
// CHECK-NEXT:       %5 = sol.address_cast %4 : !sol.address<payable> to !sol.address loc(#loc120)
// CHECK-NEXT:       sol.return %5 : !sol.address loc(#loc121)
// CHECK-NEXT:     } loc(#loc116)
// CHECK-NEXT:     sol.func @callRoundtripAddrPayable_673(%arg0: !sol.address loc({{.*}}:150:36)) -> !sol.address attributes {id = 673 : i64, orig_fn_type = (!sol.address) -> !sol.address, selector = -2078142653 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.address, Stack> loc(#loc123)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.address, !sol.ptr<!sol.address, Stack> loc(#loc123)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.address, Stack>, !sol.address loc(#loc124)
// CHECK-NEXT:       %2 = sol.call @roundtrip_addr_payable_191(%1) : (!sol.address) -> !sol.address loc(#loc125)
// CHECK-NEXT:       sol.return %2 : !sol.address loc(#loc126)
// CHECK-NEXT:     } loc(#loc122)
// CHECK-NEXT:     sol.func @encode_contract_210(%arg0: !sol.contract<"CC_196"> loc({{.*}}:50:25)) -> !sol.string<Memory> attributes {id = 210 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.contract<"CC_196">, Stack> loc(#loc128)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.contract<"CC_196">, !sol.ptr<!sol.contract<"CC_196">, Stack> loc(#loc128)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.contract<"CC_196">, Stack>, !sol.contract<"CC_196"> loc(#loc129)
// CHECK-NEXT:       %2 = sol.encode %1 :  !sol.contract<"CC_196"> : !sol.string<Memory> loc(#loc130)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc131)
// CHECK-NEXT:     } loc(#loc127)
// CHECK-NEXT:     sol.func @callEncodeContract_686(%arg0: !sol.contract<"CC_196"> loc({{.*}}:151:30)) -> !sol.string<Memory> attributes {id = 686 : i64, orig_fn_type = (!sol.contract<"CC_196">) -> !sol.string<Memory>, selector = -1250242159 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.contract<"CC_196">, Stack> loc(#loc133)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.contract<"CC_196">, !sol.ptr<!sol.contract<"CC_196">, Stack> loc(#loc133)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.contract<"CC_196">, Stack>, !sol.contract<"CC_196"> loc(#loc134)
// CHECK-NEXT:       %2 = sol.call @encode_contract_210(%1) : (!sol.contract<"CC_196">) -> !sol.string<Memory> loc(#loc135)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc136)
// CHECK-NEXT:     } loc(#loc132)
// CHECK-NEXT:     sol.func @encode_packed_contract_224(%arg0: !sol.contract<"CC_196"> loc({{.*}}:54:32)) -> !sol.string<Memory> attributes {id = 224 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.contract<"CC_196">, Stack> loc(#loc138)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.contract<"CC_196">, !sol.ptr<!sol.contract<"CC_196">, Stack> loc(#loc138)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.contract<"CC_196">, Stack>, !sol.contract<"CC_196"> loc(#loc139)
// CHECK-NEXT:       %2 = sol.encode %1 :  !sol.contract<"CC_196"> : !sol.string<Memory> {packed} loc(#loc140)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc141)
// CHECK-NEXT:     } loc(#loc137)
// CHECK-NEXT:     sol.func @callEncodePackedContract_699(%arg0: !sol.contract<"CC_196"> loc({{.*}}:152:36)) -> !sol.string<Memory> attributes {id = 699 : i64, orig_fn_type = (!sol.contract<"CC_196">) -> !sol.string<Memory>, selector = -2066139394 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.contract<"CC_196">, Stack> loc(#loc143)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.contract<"CC_196">, !sol.ptr<!sol.contract<"CC_196">, Stack> loc(#loc143)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.contract<"CC_196">, Stack>, !sol.contract<"CC_196"> loc(#loc144)
// CHECK-NEXT:       %2 = sol.call @encode_packed_contract_224(%1) : (!sol.contract<"CC_196">) -> !sol.string<Memory> loc(#loc145)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc146)
// CHECK-NEXT:     } loc(#loc142)
// CHECK-NEXT:     sol.func @decode_contract_240(%arg0: !sol.string<Memory> loc({{.*}}:58:25)) -> !sol.contract<"CC_196"> attributes {id = 240 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc148)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc148)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc149)
// CHECK-NEXT:       %2 = sol.decode %1 : !sol.string<Memory> -> !sol.contract<"CC_196"> loc(#loc150)
// CHECK-NEXT:       sol.return %2 : !sol.contract<"CC_196"> loc(#loc151)
// CHECK-NEXT:     } loc(#loc147)
// CHECK-NEXT:     sol.func @callDecodeContract_712(%arg0: !sol.string<Memory> loc({{.*}}:153:30)) -> !sol.contract<"CC_196"> attributes {id = 712 : i64, orig_fn_type = (!sol.string<Memory>) -> !sol.contract<"CC_196">, selector = -444657899 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc153)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc153)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc154)
// CHECK-NEXT:       %2 = sol.call @decode_contract_240(%1) : (!sol.string<Memory>) -> !sol.contract<"CC_196"> loc(#loc155)
// CHECK-NEXT:       sol.return %2 : !sol.contract<"CC_196"> loc(#loc156)
// CHECK-NEXT:     } loc(#loc152)
// CHECK-NEXT:     sol.func @roundtrip_contract_260(%arg0: !sol.contract<"CC_196"> loc({{.*}}:62:28)) -> !sol.contract<"CC_196"> attributes {id = 260 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.contract<"CC_196">, Stack> loc(#loc158)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.contract<"CC_196">, !sol.ptr<!sol.contract<"CC_196">, Stack> loc(#loc158)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.contract<"CC_196">, Stack>, !sol.contract<"CC_196"> loc(#loc159)
// CHECK-NEXT:       %2 = sol.encode %1 :  !sol.contract<"CC_196"> : !sol.string<Memory> loc(#loc160)
// CHECK-NEXT:       %3 = sol.decode %2 : !sol.string<Memory> -> !sol.contract<"CC_196"> loc(#loc161)
// CHECK-NEXT:       sol.return %3 : !sol.contract<"CC_196"> loc(#loc162)
// CHECK-NEXT:     } loc(#loc157)
// CHECK-NEXT:     sol.func @callRoundtripContract_726(%arg0: !sol.contract<"CC_196"> loc({{.*}}:154:33)) -> !sol.contract<"CC_196"> attributes {id = 726 : i64, orig_fn_type = (!sol.contract<"CC_196">) -> !sol.contract<"CC_196">, selector = -805142616 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.contract<"CC_196">, Stack> loc(#loc164)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.contract<"CC_196">, !sol.ptr<!sol.contract<"CC_196">, Stack> loc(#loc164)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.contract<"CC_196">, Stack>, !sol.contract<"CC_196"> loc(#loc165)
// CHECK-NEXT:       %2 = sol.call @roundtrip_contract_260(%1) : (!sol.contract<"CC_196">) -> !sol.contract<"CC_196"> loc(#loc166)
// CHECK-NEXT:       sol.return %2 : !sol.contract<"CC_196"> loc(#loc167)
// CHECK-NEXT:     } loc(#loc163)
// CHECK-NEXT:     sol.func @decode_contract_tuple_280(%arg0: !sol.string<Memory> loc({{.*}}:66:31)) -> (!sol.contract<"CC_196">, ui256) attributes {id = 280 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc169)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc169)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc170)
// CHECK-NEXT:       %2:2 = sol.decode %1 : !sol.string<Memory> -> !sol.contract<"CC_196">, ui256 loc(#loc171)
// CHECK-NEXT:       sol.return %2#0, %2#1 : !sol.contract<"CC_196">, ui256 loc(#loc172)
// CHECK-NEXT:     } loc(#loc168)
// CHECK-NEXT:     sol.func @callDecodeContractTuple_741(%arg0: !sol.string<Memory> loc({{.*}}:155:35)) -> (!sol.contract<"CC_196">, ui256) attributes {id = 741 : i64, orig_fn_type = (!sol.string<Memory>) -> (!sol.contract<"CC_196">, ui256), selector = -1340098699 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc174)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc174)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc175)
// CHECK-NEXT:       %2:2 = sol.call @decode_contract_tuple_280(%1) : (!sol.string<Memory>) -> (!sol.contract<"CC_196">, ui256) loc(#loc176)
// CHECK-NEXT:       sol.return %2#0, %2#1 : !sol.contract<"CC_196">, ui256 loc(#loc177)
// CHECK-NEXT:     } loc(#loc173)
// CHECK-NEXT:     sol.func @encode_selector_296(%arg0: !sol.fixedbytes<4> loc({{.*}}:70:25), %arg1: ui256 loc({{.*}}:70:37)) -> !sol.string<Memory> attributes {id = 296 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.fixedbytes<4>, Stack> loc(#loc179)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.fixedbytes<4>, !sol.ptr<!sol.fixedbytes<4>, Stack> loc(#loc179)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc180)
// CHECK-NEXT:       sol.store %arg1, %1 : ui256, !sol.ptr<ui256, Stack> loc(#loc180)
// CHECK-NEXT:       %2 = sol.load %0 : !sol.ptr<!sol.fixedbytes<4>, Stack>, !sol.fixedbytes<4> loc(#loc181)
// CHECK-NEXT:       %3 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc182)
// CHECK-NEXT:       %4 = sol.encode selector(%2) %3 : !sol.fixedbytes<4> ui256 : !sol.string<Memory> loc(#loc183)
// CHECK-NEXT:       sol.return %4 : !sol.string<Memory> loc(#loc184)
// CHECK-NEXT:     } loc(#loc178)
// CHECK-NEXT:     sol.func @callEncodeSelector_756(%arg0: !sol.fixedbytes<4> loc({{.*}}:156:30), %arg1: ui256 loc({{.*}}:156:42)) -> !sol.string<Memory> attributes {id = 756 : i64, orig_fn_type = (!sol.fixedbytes<4>, ui256) -> !sol.string<Memory>, selector = -1198008956 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.fixedbytes<4>, Stack> loc(#loc186)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.fixedbytes<4>, !sol.ptr<!sol.fixedbytes<4>, Stack> loc(#loc186)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc187)
// CHECK-NEXT:       sol.store %arg1, %1 : ui256, !sol.ptr<ui256, Stack> loc(#loc187)
// CHECK-NEXT:       %2 = sol.load %0 : !sol.ptr<!sol.fixedbytes<4>, Stack>, !sol.fixedbytes<4> loc(#loc188)
// CHECK-NEXT:       %3 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc189)
// CHECK-NEXT:       %4 = sol.call @encode_selector_296(%2, %3) : (!sol.fixedbytes<4>, ui256) -> !sol.string<Memory> loc(#loc190)
// CHECK-NEXT:       sol.return %4 : !sol.string<Memory> loc(#loc191)
// CHECK-NEXT:     } loc(#loc185)
// CHECK-NEXT:     sol.func @encode_signature_literal_307() -> !sol.string<Memory> attributes {id = 307 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %c69443890_ui32 = sol.constant 69443890 : ui32 loc(#loc193)
// CHECK-NEXT:       %0 = sol.bytes_cast %c69443890_ui32 : ui32 to !sol.fixedbytes<4> loc(#loc193)
// CHECK-NEXT:       %1 = sol.encode selector(%0) : !sol.fixedbytes<4>  : !sol.string<Memory> loc(#loc193)
// CHECK-NEXT:       sol.return %1 : !sol.string<Memory> loc(#loc194)
// CHECK-NEXT:     } loc(#loc192)
// CHECK-NEXT:     sol.func @callEncodeSignatureLiteral_765() -> !sol.string<Memory> attributes {id = 765 : i64, orig_fn_type = () -> !sol.string<Memory>, selector = 245136495 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.call @encode_signature_literal_307() : () -> !sol.string<Memory> loc(#loc196)
// CHECK-NEXT:       sol.return %0 : !sol.string<Memory> loc(#loc197)
// CHECK-NEXT:     } loc(#loc195)
// CHECK-NEXT:     sol.func @encode_signature_runtime_320(%arg0: !sol.string<Memory> loc({{.*}}:78:34)) -> !sol.string<Memory> attributes {id = 320 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc199)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc199)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc200)
// CHECK-NEXT:       %2 = "sol.keccak256"(%1) : (!sol.string<Memory>) -> !sol.fixedbytes<32> loc(#loc201)
// CHECK-NEXT:       %3 = sol.bytes_cast %2 : !sol.fixedbytes<32> to !sol.fixedbytes<4> loc(#loc201)
// CHECK-NEXT:       %4 = sol.encode selector(%3) : !sol.fixedbytes<4>  : !sol.string<Memory> loc(#loc201)
// CHECK-NEXT:       sol.return %4 : !sol.string<Memory> loc(#loc202)
// CHECK-NEXT:     } loc(#loc198)
// CHECK-NEXT:     sol.func @callEncodeSignatureRuntime_777(%arg0: !sol.string<Memory> loc({{.*}}:158:38)) -> !sol.string<Memory> attributes {id = 777 : i64, orig_fn_type = (!sol.string<Memory>) -> !sol.string<Memory>, selector = 1006883156 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc204)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc204)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc205)
// CHECK-NEXT:       %2 = sol.call @encode_signature_runtime_320(%1) : (!sol.string<Memory>) -> !sol.string<Memory> loc(#loc206)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc207)
// CHECK-NEXT:     } loc(#loc203)
// CHECK-NEXT:     sol.func @encode_signature_runtime_calldata_333(%arg0: !sol.string<CallData> loc({{.*}}:82:43)) -> !sol.string<Memory> attributes {id = 333 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<CallData>, Stack> loc(#loc209)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<CallData>, !sol.ptr<!sol.string<CallData>, Stack> loc(#loc209)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<CallData>, Stack>, !sol.string<CallData> loc(#loc210)
// CHECK-NEXT:       %2 = sol.data_loc_cast %1 : !sol.string<CallData>, !sol.string<Memory> loc(#loc210)
// CHECK-NEXT:       %3 = "sol.keccak256"(%2) : (!sol.string<Memory>) -> !sol.fixedbytes<32> loc(#loc211)
// CHECK-NEXT:       %4 = sol.bytes_cast %3 : !sol.fixedbytes<32> to !sol.fixedbytes<4> loc(#loc211)
// CHECK-NEXT:       %5 = sol.encode selector(%4) : !sol.fixedbytes<4>  : !sol.string<Memory> loc(#loc211)
// CHECK-NEXT:       sol.return %5 : !sol.string<Memory> loc(#loc212)
// CHECK-NEXT:     } loc(#loc208)
// CHECK-NEXT:     sol.func @callEncodeSignatureRuntimeCalldata_789(%arg0: !sol.string<CallData> loc({{.*}}:159:46)) -> !sol.string<Memory> attributes {id = 789 : i64, orig_fn_type = (!sol.string<CallData>) -> !sol.string<Memory>, selector = -2039013801 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<CallData>, Stack> loc(#loc214)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<CallData>, !sol.ptr<!sol.string<CallData>, Stack> loc(#loc214)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<CallData>, Stack>, !sol.string<CallData> loc(#loc215)
// CHECK-NEXT:       %2 = sol.call @encode_signature_runtime_calldata_333(%1) : (!sol.string<CallData>) -> !sol.string<Memory> loc(#loc216)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc217)
// CHECK-NEXT:     } loc(#loc213)
// CHECK-NEXT:     sol.func @encode_call_non_tuple_361(%arg0: ui256 loc({{.*}}:92:31)) -> !sol.string<Memory> attributes {id = 361 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc219)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc219)
// CHECK-NEXT:       %c801029432_ui32 = sol.constant 801029432 : ui32 loc(#loc220)
// CHECK-NEXT:       %1 = sol.bytes_cast %c801029432_ui32 : ui32 to !sol.fixedbytes<4> loc(#loc220)
// CHECK-NEXT:       %2 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc221)
// CHECK-NEXT:       %3 = sol.encode selector(%1) %2 : !sol.fixedbytes<4> ui256 : !sol.string<Memory> loc(#loc220)
// CHECK-NEXT:       sol.return %3 : !sol.string<Memory> loc(#loc222)
// CHECK-NEXT:     } loc(#loc218)
// CHECK-NEXT:     sol.func @callEncodeCallNonTuple_801(%arg0: ui256 loc({{.*}}:160:34)) -> !sol.string<Memory> attributes {id = 801 : i64, orig_fn_type = (ui256) -> !sol.string<Memory>, selector = -1809608251 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc224)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc224)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc225)
// CHECK-NEXT:       %2 = sol.call @encode_call_non_tuple_361(%1) : (ui256) -> !sol.string<Memory> loc(#loc226)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc227)
// CHECK-NEXT:     } loc(#loc223)
// CHECK-NEXT:     sol.func @encode_call_tuple_380(%arg0: ui256 loc({{.*}}:96:27), %arg1: ui8 loc({{.*}}:96:38)) -> !sol.string<Memory> attributes {id = 380 : i64, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc229)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc229)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui8, Stack> loc(#loc230)
// CHECK-NEXT:       sol.store %arg1, %1 : ui8, !sol.ptr<ui8, Stack> loc(#loc230)
// CHECK-NEXT:       %c105187873_ui32 = sol.constant 105187873 : ui32 loc(#loc231)
// CHECK-NEXT:       %2 = sol.bytes_cast %c105187873_ui32 : ui32 to !sol.fixedbytes<4> loc(#loc231)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc232)
// CHECK-NEXT:       %4 = sol.load %1 : !sol.ptr<ui8, Stack>, ui8 loc(#loc233)
// CHECK-NEXT:       %5 = sol.encode selector(%2) %3, %4 : !sol.fixedbytes<4> ui256, ui8 : !sol.string<Memory> loc(#loc231)
// CHECK-NEXT:       sol.return %5 : !sol.string<Memory> loc(#loc234)
// CHECK-NEXT:     } loc(#loc228)
// CHECK-NEXT:     sol.func @callEncodeCallTuple_816(%arg0: ui256 loc({{.*}}:161:31), %arg1: ui8 loc({{.*}}:161:42)) -> !sol.string<Memory> attributes {id = 816 : i64, orig_fn_type = (ui256, ui8) -> !sol.string<Memory>, selector = -947751044 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc236)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc236)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui8, Stack> loc(#loc237)
// CHECK-NEXT:       sol.store %arg1, %1 : ui8, !sol.ptr<ui8, Stack> loc(#loc237)
// CHECK-NEXT:       %2 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc238)
// CHECK-NEXT:       %3 = sol.load %1 : !sol.ptr<ui8, Stack>, ui8 loc(#loc239)
// CHECK-NEXT:       %4 = sol.call @encode_call_tuple_380(%2, %3) : (ui256, ui8) -> !sol.string<Memory> loc(#loc240)
// CHECK-NEXT:       sol.return %4 : !sol.string<Memory> loc(#loc241)
// CHECK-NEXT:     } loc(#loc235)
// CHECK-NEXT:     sol.func @encode_fnptr_399(%arg0: !sol.ext_func_ref<(ui256) -> ui256> loc({{.*}}:100:22)) -> !sol.string<Memory> attributes {id = 399 : i64, state_mutability = #Pure} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.ext_func_ref<(ui256) -> ui256>, Stack> loc(#loc243)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.ext_func_ref<(ui256) -> ui256>, !sol.ptr<!sol.ext_func_ref<(ui256) -> ui256>, Stack> loc(#loc243)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.ext_func_ref<(ui256) -> ui256>, Stack>, !sol.ext_func_ref<(ui256) -> ui256> loc(#loc244)
// CHECK-NEXT:       %2 = sol.encode %1 :  !sol.ext_func_ref<(ui256) -> ui256> : !sol.string<Memory> loc(#loc245)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc246)
// CHECK-NEXT:     } loc(#loc242)
// CHECK-NEXT:     sol.func @callEncodeFnptr_834(%arg0: !sol.ext_func_ref<(ui256) -> ui256> loc({{.*}}:162:27)) -> !sol.string<Memory> attributes {id = 834 : i64, orig_fn_type = (!sol.ext_func_ref<(ui256) -> ui256>) -> !sol.string<Memory>, selector = 1242102221 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.ext_func_ref<(ui256) -> ui256>, Stack> loc(#loc248)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.ext_func_ref<(ui256) -> ui256>, !sol.ptr<!sol.ext_func_ref<(ui256) -> ui256>, Stack> loc(#loc248)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.ext_func_ref<(ui256) -> ui256>, Stack>, !sol.ext_func_ref<(ui256) -> ui256> loc(#loc249)
// CHECK-NEXT:       %2 = sol.call @encode_fnptr_399(%1) : (!sol.ext_func_ref<(ui256) -> ui256>) -> !sol.string<Memory> loc(#loc250)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc251)
// CHECK-NEXT:     } loc(#loc247)
// CHECK-NEXT:     sol.func @encode_packed_fnptr_418(%arg0: !sol.ext_func_ref<(ui256) -> ui256> loc({{.*}}:104:29)) -> !sol.string<Memory> attributes {id = 418 : i64, state_mutability = #Pure} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.ext_func_ref<(ui256) -> ui256>, Stack> loc(#loc253)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.ext_func_ref<(ui256) -> ui256>, !sol.ptr<!sol.ext_func_ref<(ui256) -> ui256>, Stack> loc(#loc253)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.ext_func_ref<(ui256) -> ui256>, Stack>, !sol.ext_func_ref<(ui256) -> ui256> loc(#loc254)
// CHECK-NEXT:       %2 = sol.encode %1 :  !sol.ext_func_ref<(ui256) -> ui256> : !sol.string<Memory> {packed} loc(#loc255)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc256)
// CHECK-NEXT:     } loc(#loc252)
// CHECK-NEXT:     sol.func @callEncodePackedFnptr_852(%arg0: !sol.ext_func_ref<(ui256) -> ui256> loc({{.*}}:163:33)) -> !sol.string<Memory> attributes {id = 852 : i64, orig_fn_type = (!sol.ext_func_ref<(ui256) -> ui256>) -> !sol.string<Memory>, selector = -1943122258 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.ext_func_ref<(ui256) -> ui256>, Stack> loc(#loc258)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.ext_func_ref<(ui256) -> ui256>, !sol.ptr<!sol.ext_func_ref<(ui256) -> ui256>, Stack> loc(#loc258)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.ext_func_ref<(ui256) -> ui256>, Stack>, !sol.ext_func_ref<(ui256) -> ui256> loc(#loc259)
// CHECK-NEXT:       %2 = sol.call @encode_packed_fnptr_418(%1) : (!sol.ext_func_ref<(ui256) -> ui256>) -> !sol.string<Memory> loc(#loc260)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc261)
// CHECK-NEXT:     } loc(#loc257)
// CHECK-NEXT:     sol.func @encode_packed_fnptr_array_438(%arg0: !sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory> loc({{.*}}:108:35)) -> !sol.string<Memory> attributes {id = 438 : i64, state_mutability = #Pure} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory>, Stack> loc(#loc263)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory>, !sol.ptr<!sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory>, Stack> loc(#loc263)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory>, Stack>, !sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory> loc(#loc264)
// CHECK-NEXT:       %2 = sol.encode %1 :  !sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory> : !sol.string<Memory> {packed} loc(#loc265)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc266)
// CHECK-NEXT:     } loc(#loc262)
// CHECK-NEXT:     sol.func @callEncodePackedFnptrArray_871(%arg0: !sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory> loc({{.*}}:164:38)) -> !sol.string<Memory> attributes {id = 871 : i64, orig_fn_type = (!sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory>) -> !sol.string<Memory>, selector = 618167190 : i32, state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory>, Stack> loc(#loc268)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory>, !sol.ptr<!sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory>, Stack> loc(#loc268)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory>, Stack>, !sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory> loc(#loc269)
// CHECK-NEXT:       %2 = sol.call @encode_packed_fnptr_array_438(%1) : (!sol.array<? x !sol.ext_func_ref<(ui256) -> ui256>, Memory>) -> !sol.string<Memory> loc(#loc270)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc271)
// CHECK-NEXT:     } loc(#loc267)
// CHECK-NEXT:   } {kind = #Contract} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:139:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:2:0)
// CHECK-NEXT: #loc3 = loc({{.*}}:3:9)
// CHECK-NEXT: #loc4 = loc({{.*}}:3:2)
// CHECK-NEXT: #loc5 = loc({{.*}}:140:2)
// CHECK-NEXT: #loc6 = loc({{.*}}:140:61)
// CHECK-NEXT: #loc7 = loc({{.*}}:140:54)
// CHECK-NEXT: #loc8 = loc({{.*}}:6:0)
// CHECK-NEXT: #loc9 = loc({{.*}}:7:17)
// CHECK-NEXT: #loc10 = loc({{.*}}:7:2)
// CHECK-NEXT: #loc11 = loc({{.*}}:141:2)
// CHECK-NEXT: #loc12 = loc({{.*}}:141:56)
// CHECK-NEXT: #loc13 = loc({{.*}}:141:49)
// CHECK-NEXT: #loc14 = loc({{.*}}:10:0)
// CHECK-NEXT: #loc18 = loc({{.*}}:11:20)
// CHECK-NEXT: #loc19 = loc({{.*}}:11:24)
// CHECK-NEXT: #loc20 = loc({{.*}}:11:29)
// CHECK-NEXT: #loc21 = loc({{.*}}:11:9)
// CHECK-NEXT: #loc22 = loc({{.*}}:11:2)
// CHECK-NEXT: #loc23 = loc({{.*}}:142:2)
// CHECK-NEXT: #loc27 = loc({{.*}}:142:100)
// CHECK-NEXT: #loc28 = loc({{.*}}:142:104)
// CHECK-NEXT: #loc29 = loc({{.*}}:142:109)
// CHECK-NEXT: #loc30 = loc({{.*}}:142:93)
// CHECK-NEXT: #loc31 = loc({{.*}}:142:86)
// CHECK-NEXT: #loc32 = loc({{.*}}:14:0)
// CHECK-NEXT: #loc34 = loc({{.*}}:15:20)
// CHECK-NEXT: #loc35 = loc({{.*}}:15:9)
// CHECK-NEXT: #loc36 = loc({{.*}}:15:2)
// CHECK-NEXT: #loc37 = loc({{.*}}:143:2)
// CHECK-NEXT: #loc39 = loc({{.*}}:143:90)
// CHECK-NEXT: #loc40 = loc({{.*}}:143:83)
// CHECK-NEXT: #loc41 = loc({{.*}}:143:76)
// CHECK-NEXT: #loc42 = loc({{.*}}:18:0)
// CHECK-NEXT: #loc46 = loc({{.*}}:19:26)
// CHECK-NEXT: #loc47 = loc({{.*}}:19:29)
// CHECK-NEXT: #loc48 = loc({{.*}}:19:32)
// CHECK-NEXT: #loc49 = loc({{.*}}:19:9)
// CHECK-NEXT: #loc50 = loc({{.*}}:19:2)
// CHECK-NEXT: #loc51 = loc({{.*}}:144:2)
// CHECK-NEXT: #loc55 = loc({{.*}}:144:112)
// CHECK-NEXT: #loc56 = loc({{.*}}:144:115)
// CHECK-NEXT: #loc57 = loc({{.*}}:144:118)
// CHECK-NEXT: #loc58 = loc({{.*}}:144:98)
// CHECK-NEXT: #loc59 = loc({{.*}}:144:91)
// CHECK-NEXT: #loc60 = loc({{.*}}:22:0)
// CHECK-NEXT: #loc62 = loc({{.*}}:23:20)
// CHECK-NEXT: #loc63 = loc({{.*}}:23:9)
// CHECK-NEXT: #loc64 = loc({{.*}}:23:2)
// CHECK-NEXT: #loc65 = loc({{.*}}:145:2)
// CHECK-NEXT: #loc67 = loc({{.*}}:145:88)
// CHECK-NEXT: #loc68 = loc({{.*}}:145:76)
// CHECK-NEXT: #loc69 = loc({{.*}}:145:69)
// CHECK-NEXT: #loc70 = loc({{.*}}:26:0)
// CHECK-NEXT: #loc72 = loc({{.*}}:27:20)
// CHECK-NEXT: #loc73 = loc({{.*}}:27:9)
// CHECK-NEXT: #loc74 = loc({{.*}}:27:2)
// CHECK-NEXT: #loc75 = loc({{.*}}:146:2)
// CHECK-NEXT: #loc77 = loc({{.*}}:146:88)
// CHECK-NEXT: #loc78 = loc({{.*}}:146:76)
// CHECK-NEXT: #loc79 = loc({{.*}}:146:69)
// CHECK-NEXT: #loc80 = loc({{.*}}:30:0)
// CHECK-NEXT: #loc82 = loc({{.*}}:31:26)
// CHECK-NEXT: #loc83 = loc({{.*}}:31:9)
// CHECK-NEXT: #loc84 = loc({{.*}}:31:2)
// CHECK-NEXT: #loc85 = loc({{.*}}:147:2)
// CHECK-NEXT: #loc87 = loc({{.*}}:147:101)
// CHECK-NEXT: #loc88 = loc({{.*}}:147:82)
// CHECK-NEXT: #loc89 = loc({{.*}}:147:75)
// CHECK-NEXT: #loc90 = loc({{.*}}:34:0)
// CHECK-NEXT: #loc92 = loc({{.*}}:35:31)
// CHECK-NEXT: #loc93 = loc({{.*}}:35:20)
// CHECK-NEXT: #loc94 = loc({{.*}}:35:9)
// CHECK-NEXT: #loc95 = loc({{.*}}:35:2)
// CHECK-NEXT: #loc96 = loc({{.*}}:148:2)
// CHECK-NEXT: #loc98 = loc({{.*}}:148:89)
// CHECK-NEXT: #loc99 = loc({{.*}}:148:74)
// CHECK-NEXT: #loc100 = loc({{.*}}:148:67)
// CHECK-NEXT: #loc101 = loc({{.*}}:38:0)
// CHECK-NEXT: #loc104 = loc({{.*}}:39:31)
// CHECK-NEXT: #loc105 = loc({{.*}}:39:34)
// CHECK-NEXT: #loc106 = loc({{.*}}:39:20)
// CHECK-NEXT: #loc107 = loc({{.*}}:39:9)
// CHECK-NEXT: #loc108 = loc({{.*}}:39:2)
// CHECK-NEXT: #loc109 = loc({{.*}}:149:2)
// CHECK-NEXT: #loc112 = loc({{.*}}:149:120)
// CHECK-NEXT: #loc113 = loc({{.*}}:149:123)
// CHECK-NEXT: #loc114 = loc({{.*}}:149:99)
// CHECK-NEXT: #loc115 = loc({{.*}}:149:92)
// CHECK-NEXT: #loc116 = loc({{.*}}:42:0)
// CHECK-NEXT: #loc118 = loc({{.*}}:43:39)
// CHECK-NEXT: #loc119 = loc({{.*}}:43:20)
// CHECK-NEXT: #loc120 = loc({{.*}}:43:9)
// CHECK-NEXT: #loc121 = loc({{.*}}:43:2)
// CHECK-NEXT: #loc122 = loc({{.*}}:150:2)
// CHECK-NEXT: #loc124 = loc({{.*}}:150:104)
// CHECK-NEXT: #loc125 = loc({{.*}}:150:81)
// CHECK-NEXT: #loc126 = loc({{.*}}:150:74)
// CHECK-NEXT: #loc127 = loc({{.*}}:50:0)
// CHECK-NEXT: #loc129 = loc({{.*}}:51:20)
// CHECK-NEXT: #loc130 = loc({{.*}}:51:9)
// CHECK-NEXT: #loc131 = loc({{.*}}:51:2)
// CHECK-NEXT: #loc132 = loc({{.*}}:151:2)
// CHECK-NEXT: #loc134 = loc({{.*}}:151:91)
// CHECK-NEXT: #loc135 = loc({{.*}}:151:75)
// CHECK-NEXT: #loc136 = loc({{.*}}:151:68)
// CHECK-NEXT: #loc137 = loc({{.*}}:54:0)
// CHECK-NEXT: #loc139 = loc({{.*}}:55:26)
// CHECK-NEXT: #loc140 = loc({{.*}}:55:9)
// CHECK-NEXT: #loc141 = loc({{.*}}:55:2)
// CHECK-NEXT: #loc142 = loc({{.*}}:152:2)
// CHECK-NEXT: #loc144 = loc({{.*}}:152:104)
// CHECK-NEXT: #loc145 = loc({{.*}}:152:81)
// CHECK-NEXT: #loc146 = loc({{.*}}:152:74)
// CHECK-NEXT: #loc147 = loc({{.*}}:58:0)
// CHECK-NEXT: #loc149 = loc({{.*}}:59:20)
// CHECK-NEXT: #loc150 = loc({{.*}}:59:9)
// CHECK-NEXT: #loc151 = loc({{.*}}:59:2)
// CHECK-NEXT: #loc152 = loc({{.*}}:153:2)
// CHECK-NEXT: #loc154 = loc({{.*}}:153:94)
// CHECK-NEXT: #loc155 = loc({{.*}}:153:78)
// CHECK-NEXT: #loc156 = loc({{.*}}:153:71)
// CHECK-NEXT: #loc157 = loc({{.*}}:62:0)
// CHECK-NEXT: #loc159 = loc({{.*}}:63:31)
// CHECK-NEXT: #loc160 = loc({{.*}}:63:20)
// CHECK-NEXT: #loc161 = loc({{.*}}:63:9)
// CHECK-NEXT: #loc162 = loc({{.*}}:63:2)
// CHECK-NEXT: #loc163 = loc({{.*}}:154:2)
// CHECK-NEXT: #loc165 = loc({{.*}}:154:87)
// CHECK-NEXT: #loc166 = loc({{.*}}:154:68)
// CHECK-NEXT: #loc167 = loc({{.*}}:154:61)
// CHECK-NEXT: #loc168 = loc({{.*}}:66:0)
// CHECK-NEXT: #loc170 = loc({{.*}}:67:20)
// CHECK-NEXT: #loc171 = loc({{.*}}:67:9)
// CHECK-NEXT: #loc172 = loc({{.*}}:67:2)
// CHECK-NEXT: #loc173 = loc({{.*}}:155:2)
// CHECK-NEXT: #loc175 = loc({{.*}}:155:114)
// CHECK-NEXT: #loc176 = loc({{.*}}:155:92)
// CHECK-NEXT: #loc177 = loc({{.*}}:155:85)
// CHECK-NEXT: #loc178 = loc({{.*}}:70:0)
// CHECK-NEXT: #loc181 = loc({{.*}}:71:32)
// CHECK-NEXT: #loc182 = loc({{.*}}:71:37)
// CHECK-NEXT: #loc183 = loc({{.*}}:71:9)
// CHECK-NEXT: #loc184 = loc({{.*}}:71:2)
// CHECK-NEXT: #loc185 = loc({{.*}}:156:2)
// CHECK-NEXT: #loc188 = loc({{.*}}:156:108)
// CHECK-NEXT: #loc189 = loc({{.*}}:156:113)
// CHECK-NEXT: #loc190 = loc({{.*}}:156:92)
// CHECK-NEXT: #loc191 = loc({{.*}}:156:85)
// CHECK-NEXT: #loc192 = loc({{.*}}:74:0)
// CHECK-NEXT: #loc193 = loc({{.*}}:75:9)
// CHECK-NEXT: #loc194 = loc({{.*}}:75:2)
// CHECK-NEXT: #loc195 = loc({{.*}}:157:2)
// CHECK-NEXT: #loc196 = loc({{.*}}:157:79)
// CHECK-NEXT: #loc197 = loc({{.*}}:157:72)
// CHECK-NEXT: #loc198 = loc({{.*}}:78:0)
// CHECK-NEXT: #loc200 = loc({{.*}}:79:33)
// CHECK-NEXT: #loc201 = loc({{.*}}:79:9)
// CHECK-NEXT: #loc202 = loc({{.*}}:79:2)
// CHECK-NEXT: #loc203 = loc({{.*}}:158:2)
// CHECK-NEXT: #loc205 = loc({{.*}}:158:121)
// CHECK-NEXT: #loc206 = loc({{.*}}:158:96)
// CHECK-NEXT: #loc207 = loc({{.*}}:158:89)
// CHECK-NEXT: #loc208 = loc({{.*}}:82:0)
// CHECK-NEXT: #loc210 = loc({{.*}}:83:33)
// CHECK-NEXT: #loc211 = loc({{.*}}:83:9)
// CHECK-NEXT: #loc212 = loc({{.*}}:83:2)
// CHECK-NEXT: #loc213 = loc({{.*}}:159:2)
// CHECK-NEXT: #loc215 = loc({{.*}}:159:140)
// CHECK-NEXT: #loc216 = loc({{.*}}:159:106)
// CHECK-NEXT: #loc217 = loc({{.*}}:159:99)
// CHECK-NEXT: #loc218 = loc({{.*}}:92:0)
// CHECK-NEXT: #loc220 = loc({{.*}}:93:9)
// CHECK-NEXT: #loc221 = loc({{.*}}:93:31)
// CHECK-NEXT: #loc222 = loc({{.*}}:93:2)
// CHECK-NEXT: #loc223 = loc({{.*}}:160:2)
// CHECK-NEXT: #loc225 = loc({{.*}}:160:106)
// CHECK-NEXT: #loc226 = loc({{.*}}:160:84)
// CHECK-NEXT: #loc227 = loc({{.*}}:160:77)
// CHECK-NEXT: #loc228 = loc({{.*}}:96:0)
// CHECK-NEXT: #loc231 = loc({{.*}}:97:9)
// CHECK-NEXT: #loc232 = loc({{.*}}:97:32)
// CHECK-NEXT: #loc233 = loc({{.*}}:97:35)
// CHECK-NEXT: #loc234 = loc({{.*}}:97:2)
// CHECK-NEXT: #loc235 = loc({{.*}}:161:2)
// CHECK-NEXT: #loc238 = loc({{.*}}:161:108)
// CHECK-NEXT: #loc239 = loc({{.*}}:161:111)
// CHECK-NEXT: #loc240 = loc({{.*}}:161:90)
// CHECK-NEXT: #loc241 = loc({{.*}}:161:83)
// CHECK-NEXT: #loc242 = loc({{.*}}:100:0)
// CHECK-NEXT: #loc244 = loc({{.*}}:101:20)
// CHECK-NEXT: #loc245 = loc({{.*}}:101:9)
// CHECK-NEXT: #loc246 = loc({{.*}}:101:2)
// CHECK-NEXT: #loc247 = loc({{.*}}:162:2)
// CHECK-NEXT: #loc249 = loc({{.*}}:162:127)
// CHECK-NEXT: #loc250 = loc({{.*}}:162:114)
// CHECK-NEXT: #loc251 = loc({{.*}}:162:107)
// CHECK-NEXT: #loc252 = loc({{.*}}:104:0)
// CHECK-NEXT: #loc254 = loc({{.*}}:105:26)
// CHECK-NEXT: #loc255 = loc({{.*}}:105:9)
// CHECK-NEXT: #loc256 = loc({{.*}}:105:2)
// CHECK-NEXT: #loc257 = loc({{.*}}:163:2)
// CHECK-NEXT: #loc259 = loc({{.*}}:163:140)
// CHECK-NEXT: #loc260 = loc({{.*}}:163:120)
// CHECK-NEXT: #loc261 = loc({{.*}}:163:113)
// CHECK-NEXT: #loc262 = loc({{.*}}:108:0)
// CHECK-NEXT: #loc264 = loc({{.*}}:109:26)
// CHECK-NEXT: #loc265 = loc({{.*}}:109:9)
// CHECK-NEXT: #loc266 = loc({{.*}}:109:2)
// CHECK-NEXT: #loc267 = loc({{.*}}:164:2)
// CHECK-NEXT: #loc269 = loc({{.*}}:164:161)
// CHECK-NEXT: #loc270 = loc({{.*}}:164:135)
// CHECK-NEXT: #loc271 = loc({{.*}}:164:128)
// CHECK-EMPTY:
