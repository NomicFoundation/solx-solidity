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

#include "Sol.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::sol;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // (Copied from arith dialect)

  auto type = getType();
  if (auto intCst = dyn_cast<IntegerAttr>(getValue())) {
    auto intType = dyn_cast<IntegerType>(type);

    // Sugar i1 constants with 'true' and 'false'.
    if (intType && intType.getWidth() == 1)
      return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));

    // Otherwise, build a complex name with the value and type.
    SmallString<32> specialNameBuffer;
    llvm::raw_svector_ostream specialName(specialNameBuffer);
    specialName << 'c' << intCst.getValue();
    if (intType)
      specialName << '_' << type;
    setNameFn(getResult(), specialName.str());
  } else {
    setNameFn(getResult(), "cst");
  }
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  auto intTy = cast<IntegerType>(getType());
  return constFoldCastOp<IntegerAttr, IntegerAttr>(
      adaptor.getOperands(), getType(),
      [&](const APInt &val, bool &castStatus) {
        return intTy.isSigned() ? val.sext(intTy.getWidth())
                                : val.zext(intTy.getWidth());
      });
}

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  assert(inputs.size() == 1 && outputs.size() == 1);
  return isa<IntegerType>(inputs.front()) && isa<IntegerType>(outputs.front());
}

bool BytesCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  assert(inputs.size() == 1 && outputs.size() == 1);
  if (auto inpIntTy = dyn_cast<IntegerType>(inputs.front())) {
    auto outBytesTy = cast<BytesType>(outputs.front());
    return inpIntTy.getWidth() == outBytesTy.getSize() * 8;
  }
  return cast<BytesType>(inputs.front()).getSize() * 8 ==
         cast<IntegerType>(outputs.front()).getWidth();
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

/// Parses an allocation op.
///
///   ssa-id = alloc-op ssa-use? attr-dict `:` type
///
static ParseResult parseAllocationOp(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::UnresolvedOperand sizeOperand;
  auto parseRes = parser.parseOptionalOperand(sizeOperand);
  if (parseRes.has_value() && succeeded(parseRes.value())) {
    if (parser.resolveOperand(sizeOperand,
                              parser.getBuilder().getIntegerType(256),
                              result.operands))
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseColon())
    return failure();

  Type allocType;
  if (parser.parseType(allocType))
    return failure();
  result.addAttribute("alloc_type", TypeAttr::get(allocType));
  result.addTypes(allocType);

  return success();
}

static void printAllocationOp(Operation *op, OpAsmPrinter &p, Value size = {}) {
  if (size)
    p << ' ' << size;
  p.printOptionalAttrDict(op->getAttrs(), {"alloc_type"});
  p << " : " << op->getResultTypes()[0];
}

ParseResult AllocaOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAllocationOp(parser, result);
}

void AllocaOp::print(OpAsmPrinter &p) { printAllocationOp(*this, p); }

//===----------------------------------------------------------------------===//
// PushOp
//===----------------------------------------------------------------------===//

void PushOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value inp) {
  Type eltTy = getEltType(inp.getType());
  Type resTy = sol::PointerType::get(odsBuilder.getContext(), eltTy,
                                     sol::DataLocation::Storage);

  // Don't generate pointers to reference types in storage.
  if (sol::isNonPtrRefType(eltTy))
    resTy = eltTy;

  build(odsBuilder, odsState, resTy, inp);
}

//===----------------------------------------------------------------------===//
// GepOp
//===----------------------------------------------------------------------===//

void GepOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                  Value baseAddr, Value idx) {
  Type baseAddrTy = baseAddr.getType();

  Type eltTy;
  if (auto structTy = dyn_cast<StructType>(baseAddrTy)) {
    // TODO: Ideally, the index should be an integral attribute in this case.
    auto idxAttr = cast<ConstantOp>(idx.getDefiningOp()).getValue();
    auto idxVal = cast<IntegerAttr>(idxAttr).getUInt();
    eltTy = getEltType(structTy, idxVal);
  } else {
    eltTy = getEltType(baseAddrTy);
  }

  Type resTy = sol::PointerType::get(odsBuilder.getContext(), eltTy,
                                     getDataLocation(baseAddrTy));

  // Don't generate pointers to reference types in storage.
  if (auto eltArrTy = dyn_cast<ArrayType>(eltTy)) {
    if (eltArrTy.getDataLocation() == sol::DataLocation::Storage)
      resTy = eltTy;
  }

  build(odsBuilder, odsState, resTy, baseAddr, idx);
}

//===----------------------------------------------------------------------===//
// EmitOp
//===----------------------------------------------------------------------===//

/// Parses an emit op.
///
///   emit-op signature? (`indexed` `=` `[` indexed-args `]`)?
///           (`non-indexed` `=` `[` non-indexed-args `]`)?
///           attr-dict `:` type(args)
///
ParseResult EmitOp::parse(OpAsmParser &p, OperationState &result) {
  auto b = p.getBuilder();

  // Parse the signature.
  std::string signature;
  if (!p.parseOptionalString(&signature))
    result.addAttribute("signature", b.getStringAttr(signature));

  SmallVector<OpAsmParser::UnresolvedOperand> unresolvedOpnds;

  // Parse indexed args and add the attribute to track its count.
  if (!p.parseOptionalKeyword("indexed")) {
    if (p.parseEqual())
      return failure();

    if (p.parseOperandList(unresolvedOpnds, OpAsmParser::Delimiter::Square))
      return failure();
    assert(unresolvedOpnds.size() <= UINT8_MAX);
  }
  result.addAttribute("indexedArgsCount",
                      b.getI8IntegerAttr(unresolvedOpnds.size()));

  // Parse non-indexed args.
  if (!p.parseOptionalKeyword("non_indexed")) {
    if (p.parseEqual())
      return failure();

    if (p.parseOperandList(unresolvedOpnds, OpAsmParser::Delimiter::Square))
      return failure();
  }

  // Parse other attributes and the type list.
  SmallVector<Type> opndTys;
  if (p.parseOptionalAttrDict(result.attributes) ||
      p.parseOptionalColonTypeList(opndTys))
    return failure();

  // Resolve all the indexed and non-indexed args.
  if (p.resolveOperands(unresolvedOpnds, opndTys, p.getNameLoc(),
                        result.operands))
    return failure();

  return success();
}

void EmitOp::print(OpAsmPrinter &p) {
  if (auto signature = getSignatureAttr()) {
    p << ' ' << signature;
  }

  if (getIndexedArgsCount()) {
    p << " indexed = [";
    p.printOperands(getIndexedArgs());
    p << ']';
  }

  if (!getNonIndexedArgs().empty()) {
    p << " non_indexed = [";
    p.printOperands(getNonIndexedArgs());
    p << ']';
  }

  p.printOptionalAttrDict((*this)->getAttrs(),
                          {"indexedArgsCount", "signature"});
  if (!getArgs().empty())
    p << " : " << getArgs().getTypes();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void IfOp::getSuccessorRegions(std::optional<unsigned> index,
                               ArrayRef<Attribute> operands,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // The "then" and the "else" region branch back to the parent operation.
  if (index) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // Don't consider the else region if it is empty.
  Region *elseRegion = &this->getElseRegion();
  if (elseRegion->empty())
    elseRegion = nullptr;

  // Otherwise, the successor is dependent on the condition.
  // bool condition;
  // if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
  //   assert(0 && "not implemented");
  // condition = condAttr.getValue().isOneValue();
  // Add the successor regions using the condition.
  // regions.push_back(RegionSuccessor(condition ? &thenRegion() :
  // elseRegion));
  // return;
  // }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getThenRegion()));
  // If the else region does not exist, it is not a viable successor.
  if (elseRegion)
    regions.push_back(RegionSuccessor(elseRegion));
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

void SwitchOp::getSuccessorRegions(std::optional<unsigned> index,
                                   ArrayRef<Attribute> operands,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  // All "case" regions branch back to the parent op.
  if (index) {
    regions.push_back(RegionSuccessor());
    return;
  }

  llvm::copy(getRegions(), std::back_inserter(regions));
}

void SwitchOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  auto operandValue = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!operandValue) {
    // All regions are invoked at most once.
    bounds.append(getNumRegions(), InvocationBounds(/*lb=*/0, /*ub=*/1));
    return;
  }

  unsigned liveIndex = getNumRegions() - 1;
  const auto it = llvm::find(getCases(), operandValue.getValue());
  if (it != getCases().end())
    liveIndex = std::distance(getCases().begin(), it);
  for (unsigned i = 0, e = getNumRegions(); i < e; ++i)
    bounds.emplace_back(/*lb=*/0, /*ub=*/i == liveIndex);
}

void SwitchOp::print(OpAsmPrinter &p) {
  p << ' ' << getArg() << " : " << getArg().getType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/getCasesAttrName().getValue());

  for (auto [val, region] : llvm::zip(getCases(), getCaseRegions())) {
    p.printNewline();
    p << "case " << val << ' ';
    p.printRegion(region, /*printEntryBlockArgs=*/false);
  }

  p.printNewline();
  p << "default ";
  p.printRegion(getDefaultRegion(), /*printEntryBlockArgs=*/false);
}

ParseResult SwitchOp::parse(OpAsmParser &p, OperationState &result) {
  // Parse arg and its type.
  OpAsmParser::UnresolvedOperand arg;
  IntegerType argTy;
  if (succeeded(p.parseOperand(arg))) {
    if (p.parseColon())
      return failure();
    if (p.parseType(argTy))
      return failure();
    if (p.resolveOperand(arg, argTy, result.operands))
      return failure();
  }

  // Parse case regions.
  SmallVector<APInt> caseVals;
  while (succeeded(p.parseOptionalKeyword("case"))) {
    APInt value(argTy.getWidth(), 0);
    Region *region = result.addRegion();
    if (p.parseInteger(value) || p.parseRegion(*region))
      return failure();
    caseVals.push_back(value);
  }
  auto caseValsAttr = DenseIntElementsAttr::get(
      RankedTensorType::get(static_cast<int64_t>(caseVals.size()), argTy),
      caseVals);
  result.addAttribute(getCasesAttrName(result.name), caseValsAttr);

  // Parse default region.
  if (p.parseKeyword("default"))
    return failure();
  Region *defRegion = result.addRegion();
  if (p.parseRegion(*defRegion))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

MutableOperandRange
ConditionOp::getMutableSuccessorOperands(std::optional<unsigned> index) {
  // No values are yielded to the successor region.
  return MutableOperandRange(getOperation(), 0, 0);
}

//===----------------------------------------------------------------------===//
// LoopOpInterface
//===----------------------------------------------------------------------===//

void LoopOpInterface::getLoopOpSuccessorRegions(
    LoopOpInterface op, std::optional<unsigned> index,
    SmallVectorImpl<RegionSuccessor> &regions) {
  auto getRegionOrNull = [&](std::optional<unsigned> index,
                             Operation *op) -> Region * {
    if (!index)
      return nullptr;
    return &op->getRegion(*index);
  };

  // Branching to first region: go to condition or body (do-while).
  if (!index) {
    regions.emplace_back(&op.getEntry(), op.getEntry().getArguments());
  }
  // Branching from condition: go to body or exit.
  else if (&op.getCond() == getRegionOrNull(index, op)) {
    regions.emplace_back(RegionSuccessor(op->getResults()));
    regions.emplace_back(&op.getBody(), op.getBody().getArguments());
  }
  // Branching from body: go to step (for) or condition.
  else if (&op.getBody() == getRegionOrNull(index, op)) {
    // FIXME: Should we consider break/continue statements here?
    auto *afterBody = (op.maybeGetStep() ? op.maybeGetStep() : &op.getCond());
    regions.emplace_back(afterBody, afterBody->getArguments());
  }
  // Branching from step: go to condition.
  else if (op.maybeGetStep() == getRegionOrNull(index, op)) {
    regions.emplace_back(&op.getCond(), op.getCond().getArguments());
  } else {
    llvm_unreachable("unexpected branch origin");
  }
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

void WhileOp::getSuccessorRegions(std::optional<unsigned> index,
                                  ArrayRef<Attribute> operands,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  LoopOpInterface::getLoopOpSuccessorRegions(*this, index, regions);
}

//===----------------------------------------------------------------------===//
// DoWhileOp
//===----------------------------------------------------------------------===//

void DoWhileOp::getSuccessorRegions(std::optional<unsigned> index,
                                    ArrayRef<Attribute> operands,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  LoopOpInterface::getLoopOpSuccessorRegions(*this, index, regions);
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void ForOp::getSuccessorRegions(std::optional<unsigned> index,
                                ArrayRef<Attribute> operands,
                                SmallVectorImpl<RegionSuccessor> &regions) {
  LoopOpInterface::getLoopOpSuccessorRegions(*this, index, regions);
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

void TryOp::getSuccessorRegions(std::optional<unsigned> index,
                                ArrayRef<Attribute> operands,
                                SmallVectorImpl<RegionSuccessor> &regions) {
  // All regions branch back to the parent op.
  if (index) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // TryOp can branch to any non-empty region.
  for (Region *region : getRegions()) {
    if (region->empty())
      regions.push_back(region);
  }
}

//===----------------------------------------------------------------------===//
// ModifierOp
//===----------------------------------------------------------------------===//

ParseResult ModifierOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void ModifierOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addRegion();

  state.addAttribute(getSymNameAttrName(state.name),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs,
      /*resultAttrs=*/{}, getArgAttrsAttrName(state.name),
      getResAttrsAttrName(state.name));
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, StateMutability stateMutability,
                   ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  build(builder, state, name, type, attrs, argAttrs);
  state.addAttribute(
      getStateMutabilityAttrName(state.name),
      StateMutabilityAttr::get(builder.getContext(), stateMutability));
}

void FuncOp::cloneInto(FuncOp dest, IRMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<StringAttr, Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});
  for (const auto &attr : (*this)->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});

  auto newAttrs = llvm::to_vector(llvm::map_range(
      newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
        return NamedAttribute(attrPair.first, attrPair.second);
      }));
  dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

FuncOp FuncOp::clone(IRMapping &mapper) {
  // Create the new function.
  FuncOp newFunc = cast<FuncOp>(getOperation()->cloneWithoutRegions());

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  if (!isExternal()) {
    FunctionType oldType = getFunctionType();

    unsigned oldNumArgs = oldType.getNumInputs();
    SmallVector<Type, 4> newInputs;
    newInputs.reserve(oldNumArgs);
    for (unsigned i = 0; i != oldNumArgs; ++i)
      if (!mapper.contains(getArgument(i)))
        newInputs.push_back(oldType.getInput(i));

    /// If any of the arguments were dropped, update the type and drop any
    /// necessary argument attributes.
    if (newInputs.size() != oldNumArgs) {
      newFunc.setType(FunctionType::get(oldType.getContext(), newInputs,
                                        oldType.getResults()));

      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        SmallVector<Attribute> newArgAttrs;
        newArgAttrs.reserve(newInputs.size());
        for (unsigned i = 0; i != oldNumArgs; ++i)
          if (!mapper.contains(getArgument(i)))
            newArgAttrs.push_back(argAttrs[i]);
        newFunc.setAllArgAttrs(newArgAttrs);
      }
    }
  }

  /// Clone the current function into the new one and return it.
  cloneInto(newFunc, mapper);
  return newFunc;
}

FuncOp FuncOp::clone() {
  IRMapping mapper;
  return clone(mapper);
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

#define GET_OP_CLASSES
#include "Sol/SolOps.cpp.inc"
