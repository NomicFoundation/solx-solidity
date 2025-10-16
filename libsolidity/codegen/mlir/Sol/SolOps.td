// This file is part of solidity.

// solidity is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// solidity is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with solidity.  If not, see <http://www.gnu.org/licenses/>.

// SPDX-License-Identifier: GPL-3.0

//
// Sol dialect operations
//

#ifndef MLIR_SOL_SOLOPS_TD
#define MLIR_SOL_SOLOPS_TD

include "SolBase.td"
include "SolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

// TODO: Add summary and description to all the op definitions.
// TODO: Impl SymbolUserOpInterface's.

// TODO: Implement verifiers for all the ops.
//
// Most of the high level sol dialect ops (excluding sol.func + related ops, yul
// specific ops) should verify that their types (ssa argument types and
// TypeAttr) are high level ((un)signed int types, sol dialect types etc.).

// FIXME: Can we have a custom lowering of unrealized_conversion_cast instead?
// ReconcileUnrealizedCasts conversion won't work here.
def Sol_ConvCastOp : Sol_Op<"conv_cast"> {
  let summary = "Like unrealized_conversion_cast; Represents the remapped value";

  let arguments = (ins AnyType:$inp);
  let results = (outs AnyType:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `->` type($out)";
}

def Sol_ConstantOp : Op<Sol_Dialect, "constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     AllTypesMatch<["value", "result"]>]
>{
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

def Sol_CastOp : Sol_Op<"cast",
  [Pure, DeclareOpInterfaceMethods<CastOpInterface>]
>{
  let arguments = (ins Sol_Int:$inp);
  let results = (outs Sol_Int:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `to` type($out)";
  let hasFolder = 1;
}

// TODO: folder
def Sol_BytesCastOp : Sol_Op<"bytes_cast",
  [Pure, DeclareOpInterfaceMethods<CastOpInterface>]
>{
  let arguments = (ins AnyTypeOf<[Sol_BytesType, Sol_Int]>:$inp);
  let results = (outs AnyTypeOf<[Sol_BytesType, Sol_Int]>:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `to` type($out)";
}

// TODO: Add folders for the c0 case.

// Checked arith ops are not speculatable, but unchecked are (including div and
// mod because of evm semantics).

def Sol_CAddOp : Sol_IntArithOp<"cadd", [Commutative]>;
def Sol_CSubOp : Sol_IntArithOp<"csub">;
def Sol_CMulOp : Sol_IntArithOp<"cmul", [Commutative]>;
def Sol_CDivOp : Sol_IntArithOp<"cdiv">;

def Sol_AddOp : Sol_UncheckedIntArithOp<"add", [Commutative]> {
  // TODO: folder, canonicalizer
}
def Sol_SubOp : Sol_UncheckedIntArithOp<"sub"> {
  // TODO: folder, canonicalizer
}
def Sol_MulOp : Sol_UncheckedIntArithOp<"mul", [Commutative]> {
  // TODO: folder, canonicalizer
}
def Sol_DivOp : Sol_UncheckedIntArithOp<"div"> {
  // TODO: folder, canonicalizer
}
def Sol_ModOp : Sol_UncheckedIntArithOp<"mod"> {
  // TODO: folder, canonicalizer
}

def Sol_CmpOp : Sol_Op<"cmp", [Pure, SameTypeOperands]> {
  let arguments = (ins Sol_CmpPredicateAttr:$predicate, Sol_Int:$lhs,
                       Sol_Int:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
  // TODO: folder, canonicalizer
}

def Sol_AllocaOp : Sol_Op<"alloca"> {
  let arguments = (ins TypeAttr:$alloc_type);
  let results = (outs Res<Sol_PtrType, "",
                          [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);

  let builders = [OpBuilder<(ins "Type":$allocType), [{
    $_state.addAttribute(getAllocTypeAttrName($_state.name),
                         mlir::TypeAttr::get(allocType));
    $_state.addTypes(allocType);
  }]>];
  let skipDefaultBuilders = 1;

  let hasCustomAssemblyFormat = 1;
}

def Sol_MallocOp : Sol_Op<"malloc"> {
  let arguments = (ins UnitAttr:$zero_init, Optional<Sol_Int>:$size);
  let results = (outs Sol_AggType:$addr);

  // FIXME: Generated builders should include a version that defaults $size to
  // {}

  // FIXME: Comma after `type($size)`
  let assemblyFormat = [{
    ($size^)? (`zero_init` $zero_init^)? attr-dict `:`
      type($size) type($addr)
  }];
}

def Sol_GetCallDataOp : Sol_Op<"get_calldata"> {
  let results = (outs Sol_AggType:$addr);
  let assemblyFormat = [{ attr-dict `:` type($addr) }];

  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes(sol::StringType::get($_builder.getContext(),
                                            sol::DataLocation::CallData));
    }]>
  ];
  let skipDefaultBuilders = 1;
}

def Sol_PushOp : Sol_Op<"push"> {
  let arguments = (ins AnyType:$inp);
  // FIXME: AnyType -> Sol_PtrTy causes the asm printer to miss the prefix!
  let results = (outs AnyType:$addr);

  let builders = [
    OpBuilder<(ins "Value":$inp)>
  ];

  let assemblyFormat = "$inp attr-dict `:` type($inp) `->` type($addr)";
}

def Sol_PopOp : Sol_Op<"pop"> {
  let arguments = (ins AnyType:$inp);
  let assemblyFormat = "$inp attr-dict `:` type($inp)";
}

def Sol_ArrayLitOp : Sol_Op<"array_lit"> {
  let arguments = (ins Variadic<AnyType>:$ins);
  // FIXME: AnyType -> Sol_PtrTy causes the asm printer to miss the prefix!
  let results = (outs AnyType:$addr);
  let assemblyFormat = "$ins attr-dict `:` `(` type($ins) `)` `->` type($addr)";
}

def Sol_StateVarOp : Sol_Op<"state_var", [Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);

  let assemblyFormat = "$sym_name attr-dict `:` $type";
}

def Sol_ImmutableOp : Sol_Op<"immutable", [Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);

  let assemblyFormat = "$sym_name attr-dict `:` $type";
}

def Sol_AddrOfOp : Sol_Op<"addr_of", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$var);
  // FIXME: AnyType -> Sol_PtrTy causes the asm printer to miss the prefix!
  let results = (outs AnyType:$addr);

  let assemblyFormat = "$var attr-dict `:` type($addr)";
}

def Sol_GepOp : Sol_Op<"gep"> {
  let arguments = (ins AnyType:$baseAddr, Sol_Int:$idx);
  let results = (outs Sol_RefType:$addr);

  let builders = [
    OpBuilder<(ins "Value":$baseAddr, "Value":$idx)>
  ];

  // TODO: Verifier:
  // - Base address type is an array or a struct type.
  // - Element type should not be a pointer type.
  // - Index should be within bounds for static arrays and structs.
  // - Struct index should be an i64.

  let assemblyFormat = [{
    $baseAddr `,` $idx attr-dict `:` type($baseAddr) `,` type($idx)
      `,` type($addr)
  }];
}

def Sol_MapOp : Sol_Op<"map"> {
  let arguments = (ins AnyType:$mapping, AnyType:$key);
  let results = (outs AnyType:$addr);

  let assemblyFormat = [{
    $mapping `,` $key attr-dict `:` type($mapping) `,` type($key) `,`
                                    type($addr)
  }];
}

def Sol_CopyOp : Sol_Op<"copy"> {
  let arguments = (ins AnyType:$src, AnyType:$dst);

  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `,` type($dst)
  }];
}

// TODO: Simplify the asm format by inferring element type of Sol_RefType.

def Sol_LoadOp : Sol_Op<"load", [
  AlwaysSpeculatable,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
>{
  let arguments = (ins Sol_RefType:$addr);
  let results = (outs AnyType:$out);

  let builders = [
    OpBuilder<(ins "Value":$addr), [{
      build($_builder, $_state, getEltType(addr.getType()), addr);
    }]>
  ];

  let assemblyFormat = "$addr attr-dict `:` type($addr) `,` type($out)";
}

def Sol_LoadImmutableOp : Sol_Op<"load_immutable", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs AnyType:$val);

  let assemblyFormat = "$name attr-dict `:` type($val)";
}

def Sol_LibAddrOp : Sol_Op<"lib_addr", [Pure]> {
  let arguments = (ins StrAttr:$name);
  let results = (outs UI256:$val); // TODO: s/UI256/address

  let assemblyFormat = "$name attr-dict `:` type($val)";
}

def Sol_StoreOp : Sol_Op<"store", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
>{
  let arguments = (ins AnyType:$val, Sol_RefType:$addr);

  let assemblyFormat = [{
    $val `,` $addr attr-dict `:` type($val) `,` type($addr)
  }];
}

def Sol_DataLocCastOp : Sol_Op<"data_loc_cast"> {
  let arguments = (ins AnyType:$inp);
  let results = (outs AnyType:$out);

  let assemblyFormat = [{ $inp attr-dict `:` type($inp) `,` type($out) }];
}

def Sol_LengthOp : Sol_Op<"length"> {
  let arguments = (ins AnyType:$inp);
  let results = (outs UI256:$len);

  let assemblyFormat = [{ $inp attr-dict `:` type($inp) }];
}

def Sol_EncodeOp : Sol_Op<"encode"> {
  let arguments = (ins Variadic<AnyType>:$ins);
  // FIXME: AnyType -> Sol_StringType causes the asm printer to miss the prefix!
  let results = (outs AnyType:$res);
  let assemblyFormat = "($ins^ `:` type($ins))? `:` type($res) attr-dict";
}

def Sol_DecodeOp : Sol_Op<"decode"> {
  let arguments = (ins AnyType:$addr);
  let results = (outs Variadic<AnyType>:$outs);
  let assemblyFormat = "$addr `:` type($addr) `->` type($outs) attr-dict";
}

def Sol_EmitOp : Sol_Op<"emit"> {
  let arguments = (ins Variadic<AnyType>:$args, I8Attr:$indexedArgsCount,
                       OptionalAttr<StrAttr>:$signature);

  let builders = [
    OpBuilder<(ins "std::vector<Value>":$indexedArgs,
                    CArg<"std::vector<Value>", "{}">:$nonIndexedArgs,
                    CArg<"std::optional<std::string>",
                         "std::nullopt">:$signature), [{
      $_state.addOperands(indexedArgs);
      $_state.addOperands(nonIndexedArgs);

      IntegerAttr indexedArgsCountAttr =
          $_builder.getIntegerAttr($_builder.getIntegerType(8),
                                   indexedArgs.size());
      $_state.addAttribute(getIndexedArgsCountAttrName($_state.name),
                           indexedArgsCountAttr);

      if (signature)
        $_state.addAttribute(getSignatureAttrName($_state.name),
                             $_builder.getStringAttr(*signature));
    }]>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    OperandRange getIndexedArgs() {
      return OperandRange(getOperands().begin(), getOperands().begin() +
                          getIndexedArgsCount());
    }
    OperandRange getNonIndexedArgs() {
      return OperandRange(getOperands().begin() + getIndexedArgsCount(),
                          getOperands().end());
    }
  }];

  // TODO: Verifier: ($indexedArgsCount <= 3)

  let hasCustomAssemblyFormat = 1;
}

def Sol_RequireOp : Sol_Op<"require"> {
  // FIXME: $msg should be an OptionalAttr, but it doesn't reflect in the
  // generated build API.
  let arguments = (ins I1:$cond, DefaultValuedStrAttr<StrAttr, "">:$msg);

  let assemblyFormat = "$cond `,` $msg attr-dict";
}

// TODO: Implement verifiers for sol.break and sol.continue that checks if it
// has an ancestor loop op.

def Sol_BreakOp : Sol_Op<"break", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

def Sol_ContinueOp : Sol_Op<"continue", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

def Sol_ConditionOp : Sol_Op<"condition", [
  Terminator,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
                            ["getSuccessorRegions"]>]
> {
  let arguments = (ins I1:$condition);

  let assemblyFormat = "$condition attr-dict";
}

def Sol_YieldOp : Sol_Op<"yield", [ReturnLike, Terminator,
  ParentOneOf<["IfOp", "SwitchOp", "TryOp", "WhileOp", "DoWhileOp", "ForOp"]>]
> {
  let arguments = (ins Variadic<AnyType>:$ins);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "($ins^ `:` type($ins))? attr-dict";
}

def Sol_IfOp : Sol_Op<"if",
  [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
   AutomaticAllocationScope, NoRegionArguments]
> {
  let arguments = (ins I1:$cond);
  let regions = (region MinSizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let assemblyFormat = "$cond $thenRegion `else` $elseRegion attr-dict";
}

def Sol_SwitchOp : Sol_Op<"switch", [
  AutomaticAllocationScope, NoRegionArguments,
  DeclareOpInterfaceMethods<RegionBranchOpInterface,
                            ["getRegionInvocationBounds"]>
]> {
  let arguments = (ins AnyInteger:$arg, AnyIntElementsAttr:$cases);
  let regions = (region MinSizedRegion<1>:$defaultRegion,
                        VariadicRegion<MinSizedRegion<1>>:$caseRegions);

  let hasCustomAssemblyFormat = 1;
}

class Sol_WhileOpBase<string mnemonic> : Sol_Op<mnemonic, [
  Sol_LoopOpInterface, NoRegionArguments, AutomaticAllocationScope
]>;

def Sol_WhileOp : Sol_WhileOpBase<"while"> {
  let regions = (region SizedRegion<1>:$cond, MinSizedRegion<1>:$body);

  let assemblyFormat = "$cond `do` $body attr-dict";
}

def Sol_DoWhileOp : Sol_WhileOpBase<"do"> {
  let regions = (region MinSizedRegion<1>:$body, SizedRegion<1>:$cond);

  let assemblyFormat = "$body `while` $cond attr-dict";

  let extraClassDeclaration = [{
    Region &getEntry() { return getBody(); }
  }];
}

def Sol_ForOp : Sol_Op<"for", [Sol_LoopOpInterface, NoRegionArguments,
                               AutomaticAllocationScope]> {
  let regions = (region SizedRegion<1>:$cond, MinSizedRegion<1>:$body,
                        SizedRegion<1>:$step);

  let assemblyFormat = [{
    `cond` $cond
    `body` $body
    `step` $step
    attr-dict
  }];

  let extraClassDeclaration = [{
    mlir::Region *maybeGetStep() { return &getStep(); }
  }];
}

def Sol_TryOp : Sol_Op<"try",
  [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
   AutomaticAllocationScope]
> {
  let regions = (region AnyRegion:$successRegion, AnyRegion:$panicRegion,
                        AnyRegion:$errorRegion, AnyRegion:$fallbackRegion);
  let arguments = (ins I1:$status);

  let assemblyFormat = [{
    $status $successRegion `panic` $panicRegion `error` $errorRegion
      `fallback` $fallbackRegion attr-dict
  }];
}

// TODO: We need a trait like "HasAncestor<ModifierOp>".
def Sol_PlaceholderOp : Sol_Op<"placeholder"> {
  let assemblyFormat = "attr-dict";
}

def Sol_ModifierCallBlkOp : Sol_Op<"modifier_call_blk",
                                   [SingleBlock, NoTerminator,
                                    IsolatedFromAbove]> {
  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins), [{ $_state.addRegion()->emplaceBlock(); }]>
  ];
  let skipDefaultBuilders = 1;

  let assemblyFormat = "$bodyRegion attr-dict";
}

// TODO: Modifiers doesn't return anything and it can't be external. Should we
// replace the FunctionOpInterface here with something more appropriate?
def Sol_ModifierOp : Sol_Op<"modifier",
                            [AutomaticAllocationScope, FunctionOpInterface,
                             CallableOpInterface, IsolatedFromAbove, Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type), [{
      $_state.addRegion();
      $_state.addAttribute(getFunctionTypeAttrName($_state.name),
                           TypeAttr::get(type));
      $_state.addAttribute(SymbolTable::getSymbolAttrName(),
                           $_builder.getStringAttr(name));
    }]>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    Region *getCallableRegion() { return &getBody(); }

    ArrayRef<Type> getCallableResults() {
      return getFunctionType().getResults();
    }
    ArrayAttr getCallableArgAttrs() { return getArgAttrs().value_or(nullptr); }

    ArrayAttr getCallableResAttrs() { return getResAttrs().value_or(nullptr); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface
    //===------------------------------------------------------------------===//

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//
// FIXME: What's wrong with using func.func and related ops? We could schedule a
// custom conversion to do solidity specific things before the func dialect
// conversion.
//

def CallOp : Sol_Op<"call", [CallOpInterface]> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute(getCalleeAttrName($_state.name),
                           SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$resTys,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute(getCalleeAttrName($_state.name), callee);
      $_state.addTypes(resTys);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$resTys,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), resTys, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$resTys,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            resTys, operands);
    }]>];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallOpInterface
    //===------------------------------------------------------------------===//
    operand_range getArgOperands() {
      return {operand_begin(), operand_end()};
    }
    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def Sol_ThisOp : Sol_Op<"this"> {
  let results = (outs UI256:$addr); // TODO: s/UI256/address

  let assemblyFormat = "attr-dict";
}

def Sol_ExtCallOp : Sol_Op<"ext_call", [CallOpInterface]> {
  // TODO: s/UI256/address in $addr
  let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$ins,
                       UI256:$addr, UI256:$gas, UI256:$val, UnitAttr:$try_call,
                       UnitAttr:$static_call, UnitAttr:$delegate_call,
                       I32Attr:$selector,
                       TypeAttrOf<FunctionType>:$callee_type);
  let results = (outs I1:$status, Variadic<AnyType>);

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallOpInterface
    //===------------------------------------------------------------------===//
    operand_range getArgOperands() {
      return {operand_begin() + 1, operand_end()};
    }
    MutableOperandRange getArgOperandsMutable() {
      return MutableOperandRange(getOperation(), /*start=*/1,
          /*length=*/getNumOperands() - 1);
    }
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    $callee `(` $ins `)` `at` $addr `gas` $gas `value` $val attr-dict `:`
      functional-type($ins, results)
  }];
}

def Sol_ReturnOp : Sol_Op<"return", [Pure, Terminator]> {
  // TODO: Verifier should check if a func op is an ancestor. ParentOneOf
  // doesn't work for this.
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Sol_FuncOp : Sol_Op<"func", [AutomaticAllocationScope, CallableOpInterface,
                                 FunctionOpInterface, IsolatedFromAbove,
                                 Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       OptionalAttr<I32Attr>:$selector,
                       OptionalAttr<TypeAttrOf<FunctionType>>:$orig_fn_type,
                       UnitAttr:$runtime,
                       OptionalAttr<Sol_FunctionKindAttr>:$kind,
                       OptionalAttr<Sol_StateMutabilityAttr>:$state_mutability,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>,
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      "StateMutability":$stateMutability,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    // Copied from func.func.
    FuncOp clone(IRMapping &mapper);
    FuncOp clone();
    void cloneInto(FuncOp dest, IRMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    ArrayRef<Type> getCallableResults() {
      return getFunctionType().getResults();
    }
    ArrayAttr getCallableArgAttrs() { return getArgAttrs().value_or(nullptr); }

    ArrayAttr getCallableResAttrs() { return getResAttrs().value_or(nullptr); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface
    //===------------------------------------------------------------------===//

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Sol_ContractOp : Sol_Op<"contract", [Symbol, SymbolTable, SingleBlock,
                                         IsolatedFromAbove, NoTerminator]> {
  let arguments = (ins SymbolNameAttr:$sym_name, Sol_ContractKindAttr:$kind);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "$sym_name $bodyRegion attr-dict";
}

//
// TODO? Move the rest of the ops to the yul dialect?
//

def Sol_UpdFreePtrOp : Sol_Op<"upd_free_ptr"> {
  let arguments = (ins I256:$size);
  let results = (outs I256:$out);
  let assemblyFormat = "$size attr-dict";
}

def Sol_ObjectOp : Sol_Op<"object", [Symbol, SymbolTable, NoTerminator,
                                     IsolatedFromAbove]> {
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins CArg<"StringRef", "{}">:$name),[{
      $_state.addRegion()->emplaceBlock();
      $_state.addAttribute(getSymNameAttrName($_state.name),
                           $_builder.getStringAttr(name));
    }]>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    Block *getEntryBlock() { return &getBody().front(); }
  }];

  let assemblyFormat = "$sym_name $body attr-dict";
}

//
// We could auto-generate the following evm builtin ops from
// evmasm::InstructionInfo. But I don't think it's worth the effort.
//

def Sol_CallerOp : Sol_Op<"caller", [Pure]> {
  let summary = "Represents the `caller` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_GasOp : Sol_Op<"gas", [Pure]> {
  let summary = "Represents the `gas` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_AddressOp : Sol_Op<"address", [Pure]> {
  let summary = "Represents the `address` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_BuiltinCallOp : Sol_Op<"builtin_call"> {
  let summary = "Represents the `call` call in yul";
  let arguments = (ins I256:$gas, I256:$address, I256:$value, I256:$inpOffset,
                       I256:$inpSize, I256:$outOffset, I256:$outSize);
  let results = (outs I256:$status);
  let assemblyFormat = [{
    $gas `,` $address `,` $value `,` $inpOffset `,` $inpSize `,` $outOffset `,`
    $outSize attr-dict
  }];
}

def Sol_StaticCallOp : Sol_Op<"static_call"> {
  let summary = "Represents the `staticcall` call in yul";
  let arguments = (ins I256:$gas, I256:$address, I256:$inpOffset,
                       I256:$inpSize, I256:$outOffset, I256:$outSize);
  let results = (outs I256:$out);
  let assemblyFormat = [{
    $gas `,` $address `,` $inpOffset `,` $inpSize `,` $outOffset `,`
    $outSize attr-dict
  }];
}

def Sol_DelegateCallOp : Sol_Op<"delegate_call"> {
  let summary = "Represents the `delegatecall` call in yul";
  let arguments = (ins I256:$gas, I256:$address, I256:$inpOffset,
                       I256:$inpSize, I256:$outOffset, I256:$outSize);
  let results = (outs I256:$out);
  let assemblyFormat = [{
    $gas `,` $address `,` $inpOffset `,` $inpSize `,` $outOffset `,`
    $outSize attr-dict
  }];
}

def Sol_BuiltinRetOp : Sol_Op<"builtin_ret"> {
  let summary = "Represents the `return` call in yul";
  let arguments = (ins I256:$addr, I256:$size);
  let assemblyFormat = "$addr `,` $size attr-dict";
}

def Sol_RevertOp : Sol_Op<"revert"> {
  let summary = "Represents the `revert` call in yul";
  // TODO: Support revert strings.
  let arguments = (ins I256:$addr, I256:$size);
  let assemblyFormat = "$addr `,` $size attr-dict";
}

def Sol_StopOp : Sol_Op<"stop"> {
  let summary = "Represents the `stop` call in yul";
  let assemblyFormat = "attr-dict";
}

def Sol_MLoadOp : Sol_Op<"mload"> {
  let summary = "Represents the `mload` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Sol_MStoreOp : Sol_Op<"mstore"> {
  let summary = "Represents the `mstore` call in yul";
  let arguments = (ins I256:$addr, I256:$val);
  let assemblyFormat = "$addr `,` $val attr-dict";
}

def Sol_MStore8Op : Sol_Op<"mstore8"> {
  let summary = "Represents the `mstore8` call in yul";
  let arguments = (ins I256:$addr, I256:$val);
  let assemblyFormat = "$addr `,` $val attr-dict";
}

def Sol_ByteOp : Sol_Op<"byte"> {
  let summary = "Represents the `byte` call in yul";
  let arguments = (ins I256:$idx, I256:$val);
  let results = (outs I256:$out);
  let assemblyFormat = "$idx `,` $val attr-dict";
}

def Sol_MCopyOp : Sol_Op<"mcopy"> {
  let summary = "Represents the `mcopy` call in yul";
  let arguments = (ins I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
}

def Sol_MSizeOp : Sol_Op<"msize", [Pure]> {
  let summary = "Represents the `msize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_CallValOp : Sol_Op<"callvalue", [Pure]> {
  let summary = "Represents the `callvalue` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_CallDataLoadOp : Sol_Op<"calldataload"> {
  let summary = "Represents the `calldataload` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Sol_CallDataSizeOp : Sol_Op<"calldatasize", [Pure]> {
  let summary = "Represents the `calldatasize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_CallDataCopyOp : Sol_Op<"calldatacopy"> {
  let summary = "Represents the `calldatacopy` call in yul";
  let arguments = (ins I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
}

def Sol_ReturnDataSizeOp : Sol_Op<"returndatasize", [Pure]> {
  let summary = "Represents the `returndatasize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_ReturnDataCopyOp : Sol_Op<"returndatacopy"> {
  let summary = "Represents the `returndatacopy` call in yul";
  let arguments = (ins I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
}

def Sol_SLoadOp : Sol_Op<"sload"> {
  let summary = "Represents the `sload` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Sol_SStoreOp : Sol_Op<"sstore"> {
  let summary = "Represents the `sstore` call in yul";
  let arguments = (ins I256:$addr, I256:$val);
  let assemblyFormat = "$addr `,` $val attr-dict";
}

// TODO: Support symbolic references to objects outside the current symbol table
// (including ones outside the translation unit) using SymbolRefAttr instead of
// FlatSymbolRefAttr

def Sol_DataOffsetOp : Sol_Op<"dataoffset", [Pure]> {
  let summary = "Represents the `dataoffset` call in yul";
  let arguments = (ins FlatSymbolRefAttr:$obj);
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_DataSizeOp : Sol_Op<"datasize", [Pure]> {
  let summary = "Represents the `datasize` call in yul";
  let arguments = (ins FlatSymbolRefAttr:$obj);
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_CodeSizeOp : Sol_Op<"codesize", [Pure]> {
  let summary = "Represents the `codesize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_CodeCopyOp : Sol_Op<"codecopy"> {
  let summary = "Represents the `codecopy` call in yul";
  let arguments = (ins I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
}

def Sol_ExtCodeSizeOp : Sol_Op<"extcodesize", [Pure]> {
  let summary = "Represents the `extcodesize` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

// TODO: Is this `ConstantLike`? Adding it causes "expected ConstantLike op to
// be foldable" assert fail (probably due to the missing let hasFolder = 1).
def Sol_MemGuardOp : Sol_Op<"memoryguard", [Pure]> {
  let summary = "Represents the `memoryguard` call in yul";
  let arguments = (ins I256Attr:$size);
  let results = (outs I256:$out);
  let assemblyFormat = "$size attr-dict";
}

def Sol_Keccak256Op : Sol_Op<"keccak256"> {
  // Note that the keccak256 yul builtin has side effects.
  let summary = "Represents the `keccak256` call in yul";
  let arguments = (ins I256:$addr, I256:$size);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr `,` $size attr-dict";
}

def Sol_LogOp : Sol_Op<"log"> {
  let summary = "Represents the `log*` calls in yul";
  let arguments = (ins I256:$addr, I256:$size, Variadic<I256>:$topics);
  // FIXME: Trailing spaces with empty $topics.
  let assemblyFormat = "$addr `,` $size oilist(`topics` `(` $topics `)` ) attr-dict";
}

def Sol_SetImmutableOp : Sol_Op<"setimmutable"> {
  let summary = "Represents the `setimmutable` call in yul";
  let arguments = (ins I256:$addr, StrAttr:$name, I256:$val);
  let assemblyFormat = "$addr `,` $name `,` $val attr-dict";
}

def Sol_LoadImmutable2Op : Sol_Op<"loadimmutable"> {
  let summary = "Represents the `loadimmutable` call in yul";
  let arguments = (ins StrAttr:$name);
  let results = (outs I256:$out);
  let assemblyFormat = "$name attr-dict";
}

def Sol_LinkerSymbolOp : Sol_Op<"linkersymbol"> {
  let summary = "Represents the `linkersymbol` call in yul";
  let arguments = (ins StrAttr:$name);
  let results = (outs I256:$out);
  let assemblyFormat = "$name attr-dict";
}

#endif // MLIR_SOL_SOLOPS_TD
