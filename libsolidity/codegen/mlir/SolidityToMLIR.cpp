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
// Solidity to MLIR pass
//

#include "libevmasm/GasMeter.h"
#include "liblangutil/CharStream.h"
#include "liblangutil/EVMVersion.h"
#include "liblangutil/Exceptions.h"
#include "liblangutil/SourceLocation.h"
#include "libsolidity/ast/AST.h"
#include "libsolidity/ast/ASTEnums.h"
#include "libsolidity/ast/ASTForward.h"
#include "libsolidity/ast/ASTUtils.h"
#include "libsolidity/ast/TypeProvider.h"
#include "libsolidity/ast/Types.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Target/EVM/Util.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libsolidity/interface/CompilerStack.h"
#include "libsolutil/CommonIO.h"
#include "libsolutil/FunctionSelector.h"
#include "libsolutil/Keccak256.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Sol/Sol.h"
#include "mlir/Dialect/Sol/Utils.h"
#include "mlir/Dialect/Yul/Yul.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "range/v3/view/zip.hpp"
#include "llvm-c/Core.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/ThreadPool.h"
#include <functional>
#include <mutex>
#include <string>

using namespace solidity::langutil;
using namespace solidity::frontend;
using namespace solidity::mlirgen;

namespace solidity::frontend {

class SolidityToMLIRPass {
public:
  explicit SolidityToMLIRPass(mlir::MLIRContext &ctx, EVMVersion evmVersion,
                              RevertStrings revertStrings, bool genUnkLoc)
      : b(&ctx), evmVersion(evmVersion), revertStrings(revertStrings),
        genUnkLoc(genUnkLoc) {}

  /// Lowers the free functions in the source unit.
  void lowerFreeFuncs(SourceUnit const &);

  /// Lowers the contract.
  void lower(ContractDefinition const &);

  /// Initializes (or resets) the module and the insertion-point.
  void init(std::shared_ptr<CharStream> s) {
    stream = std::move(s);
    mod = mlir::ModuleOp::create(b.getUnknownLoc());
    mod->setAttr("sol.evm_version",
                 mlir::sol::EvmVersionAttr::get(
                     b.getContext(), *mlir::sol::symbolizeEvmVersion(
                                         evmVersion.getVersionAsInt())));
    mod->setAttr("sol.revert_strings",
                 mlir::sol::RevertStringsAttr::get(
                     b.getContext(),
                     static_cast<mlir::sol::RevertStrings>(revertStrings)));
    mod->setAttr("llvm.target_triple", b.getStringAttr("evm-unknown-unknown"));
    mod->setAttr("llvm.data_layout",
                 b.getStringAttr("E-p:256:256-i256:256:256-S256-a:256:256"));
    b.setInsertionPointToEnd(mod.getBody());
  }

  /// Returns the ModuleOp
  mlir::ModuleOp getModule() { return mod; }

private:
  mlir::OpBuilder b;
  std::shared_ptr<CharStream> stream;
  EVMVersion evmVersion;
  RevertStrings revertStrings;
  mlir::ModuleOp mod;

  /// The contract being lowered.
  ContractDefinition const *currContract = nullptr;

  /// Maps a local variable to its address.
  std::map<VariableDeclaration const *, mlir::Value> localVarAddrMap;

  /// Maps an interface function or state variable getter to its selector.
  std::map<Declaration const *, util::FixedHash<4>> selectorMap;

  /// True if the current block is unchecked.
  bool inUnchecked = false;

  /// Tracks if the codegen is generating a constructor.
  bool inCtor = false;

  /// Tracks struct types currently being converted to detect recursive structs.
  mlir::DenseSet<StructType const *> structsInProgress;

  /// Forces generated locations to be unknown. FIXME: This is to avoid the slow
  /// translatePositionToLineColumn
  bool genUnkLoc;

  /// Returns the mlir location for the solidity source location `loc`
  mlir::Location getLoc(SourceLocation const &loc) {
    if (genUnkLoc)
      return b.getUnknownLoc();
    // TODO: Cache the translatePositionToLineColumn results. (Ideally, the
    // lexer + parser should record this instead-of/along-with the existing
    // linear offset)
    //
    // FIXME: Track loc.end as well.
    LineColumn lineCol = stream->translatePositionToLineColumn(loc.start);
    return mlir::FileLineColLoc::get(b.getStringAttr(stream->name()),
                                     lineCol.line, lineCol.column);
  }

  mlir::Location getLoc(ASTNode const &ast) { return getLoc(ast.location()); }

  /// Returns the corresponding mlir type for the solidity type `ty`.
  mlir::Type getType(Type const *ty, bool indirectFn = true);

  /// Recursive worker for getType(). `identifiedStructsInProgress` tracks
  /// identified (recursive) Sol struct types whose body is currently being
  /// built, keyed by the uniqued Sol type, so a self-reference resolves to the
  /// same in-progress instance and the cycle terminates. Every recursive call
  /// within the worker must thread this set through. The public getType()
  /// overload above seeds a fresh one per top-level conversion.
  mlir::Type
  getType(Type const *ty, bool indirectFn,
          mlir::DenseSet<mlir::sol::StructType> &identifiedStructsInProgress);

  /// Tracks the address of the local variable.
  void trackLocalVarAddr(VariableDeclaration const &decl, mlir::Value addr) {
    localVarAddrMap[&decl] = addr;
  }

  /// Returns the address of the local variable.
  mlir::Value getLocalVarAddr(VariableDeclaration const &decl) {
    auto it = localVarAddrMap.find(&decl);
    assert(it != localVarAddrMap.end());
    return it->second;
  }

  /// Returns the mangled name of the declaration composed of its name and its
  /// AST ID.
  std::string getMangledName(Declaration const &decl) {
    return decl.name() + "_" + std::to_string(decl.id());
  }

  /// Returns the contract definition from a metatype member access expression.
  ContractDefinition const &getMetaTypeContract(Type const *memberAccTy) {
    auto const *magicTy = dynamic_cast<MagicType const *>(memberAccTy);
    assert(magicTy && "Expected magic type for metatype member access");
    Type const *argTy = magicTy->typeArgument();
    assert(argTy && "Expected metatype argument for metatype member access");

    auto const *contractTy = dynamic_cast<ContractType const *>(argTy);
    assert(contractTy && "Expected contract metatype argument");
    assert(!contractTy->isSuper() &&
           "Expected non-super contract type for metatype member access");
    return contractTy->contractDefinition();
  }

  /// Returns the symbol and selector for an externally callable contract
  /// member. Public state-variable getters are lowered as synthesized
  /// functions, so they need their generated getter symbol here.
  bool getContractMemberExternalCalleeInfo(Declaration const *decl,
                                           std::string &symbol,
                                           uint32_t &selector) {
    if (auto const *fn = dynamic_cast<FunctionDefinition const *>(decl)) {
      symbol = getMangledName(*fn);
      selector = FunctionType(*fn).externalIdentifier().convert_to<uint32_t>();
      return true;
    }

    auto const *var = dynamic_cast<VariableDeclaration const *>(decl);
    if (!var || !var->isStateVariable() || !var->isPartOfExternalInterface())
      return false;

    symbol = "get_" + getMangledName(*var);
    selector = FunctionType(*var).externalIdentifier().convert_to<uint32_t>();
    return true;
  }

  /// Returns the lvalue reference for a variable declaration.
  mlir::Value genLValRef(VariableDeclaration const &var) {
    // Constants (contract-level and file-level) have no storage or memory
    // home. Each reference re-evaluates the initializer.
    if (var.isConstant())
      return genRValExpr(*var.value(), getType(var.type()));
    if (var.isStateVariable())
      return genStateVarRef(var, inCtor);
    return getLocalVarAddr(var);
  }

  mlir::Value genStateVarRef(VariableDeclaration const &var,
                             bool inCreationContext) {
    auto currContr =
        b.getBlock()->getParentOp()->getParentOfType<mlir::sol::ContractOp>();
    assert(currContr);
    assert(!var.isConstant() && "Constants are handled in genLValRef");

    if (var.immutable()) {
      auto immOp =
          currContr.lookupSymbol<mlir::sol::ImmutableOp>(getMangledName(var));
      assert(immOp);
      assert(!mlir::sol::isNonPtrRefType(immOp.getType()));
      if (!inCreationContext)
        return b.create<mlir::sol::LoadImmutableOp>(
            immOp.getLoc(), immOp.getType(), immOp.getName());
      mlir::Type addrTy = mlir::sol::PointerType::get(
          b.getContext(), immOp.getType(), mlir::sol::DataLocation::Immutable);
      return b.create<mlir::sol::AddrOfOp>(immOp.getLoc(), addrTy,
                                           immOp.getName());
    }

    auto stateVarOp =
        currContr.lookupSymbol<mlir::sol::StateVarOp>(getMangledName(var));
    assert(stateVarOp);
    mlir::sol::DataLocation dataLoc = stateVarOp.getTransient()
                                          ? mlir::sol::DataLocation::Transient
                                          : mlir::sol::DataLocation::Storage;
    mlir::Type addrTy;
    if (mlir::sol::isNonPtrRefType(stateVarOp.getType()))
      addrTy = stateVarOp.getType();
    else
      addrTy = mlir::sol::PointerType::get(b.getContext(), stateVarOp.getType(),
                                           dataLoc);
    return b.create<mlir::sol::AddrOfOp>(stateVarOp.getLoc(), addrTy,
                                         stateVarOp.getName());
  }

  mlir::sol::FuncOp genGetter(VariableDeclaration const &stateVar) {
    assert(stateVar.isStateVariable());
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    mlir::Location loc = getLoc(stateVar);

    // Create the function.
    auto astFnTy = FunctionType(stateVar);
    auto fnTy =
        cast<mlir::FunctionType>(getType(&astFnTy, /*indirectFn=*/false));
    auto fn = b.create<mlir::sol::FuncOp>(
        loc, "get_" + getMangledName(stateVar), fnTy);
    assert(selectorMap.find(&stateVar) != selectorMap.end());
    fn.setSelectorAttr(
        b.getIntegerAttr(b.getIntegerType(32),
                         mlir::APInt(32, selectorMap[&stateVar].hex(), 16)));
    fn.setOrigFnTypeAttr(mlir::TypeAttr::get(fnTy));
    fn.setStateMutability(mlir::sol::StateMutability::NonPayable);

    mlir::Block *entryBlk = b.createBlock(&fn.getRegion());
    b.setInsertionPointToStart(entryBlk);

    // Load the state variable.
    mlir::Value stateVarLd;
    if (stateVar.isConstant()) {
      stateVarLd = genRValExpr(*stateVar.value(), getType(stateVar.type()));
    } else {
      mlir::Value stateVarRef =
          genStateVarRef(stateVar, /*inCreationContext=*/false);
      stateVarLd = genRValExpr(stateVarRef, stateVarRef.getLoc());
    }

    // Expands a storage struct into a flat tuple of returnable values,
    // mirroring FunctionType(VariableDeclaration) in Types.cpp:
    //   - mappings and non-byte arrays are excluded
    //   - all reference-type members (strings, nested structs) are cast
    //     from Storage to Memory
    // The exclusion of arrays is intentionally one level deep (outermost struct
    // only). This mirrors a historical asymmetry in Solidity's getter
    // generation: before ABIEncoderV2, the outermost struct had to be flattened
    // into primitives, so arrays were excluded because they cannot be
    // individually returned without an index. Nested structs are returned as
    // atomic StructType<Memory> values (ABIEncoderV2 tuples), so the ABI
    // encoder encodes all their members — including arrays — without any
    // filtering.
    // TODO: #139, packed members that share a storage slot (e.g.
    // uint8/bool/address in the same slot) each emit an independent sload,
    // there is no caching of the slot word across members. The redundant sloads
    // are eliminated by LLVM CSE during IR optimisation and do not appear in
    // the final bytecode.
    auto expandStructForReturn =
        [&](mlir::sol::StructType structTy,
            mlir::Value structVal) -> mlir::SmallVector<mlir::Value, 4> {
      mlir::SmallVector<mlir::Value, 4> tuple;
      for (auto [idx, memTy] : llvm::enumerate(structTy.getMemberTypes())) {
        if (isa<mlir::sol::MappingType>(memTy) ||
            isa<mlir::sol::ArrayType>(memTy))
          continue;
        auto gep = b.create<mlir::sol::GepOp>(
            loc, structVal, genUnsignedConst(idx, /*numBits=*/64, loc));
        mlir::Value memberVal = genRValExpr(gep, loc);
        if (mlir::sol::isNonPtrRefType(memberVal.getType()))
          memberVal = genCast(memberVal, toMemoryType(memberVal.getType()));
        tuple.push_back(memberVal);
      }
      return tuple;
    };

    // Array type
    if (isa<mlir::sol::ArrayType>(stateVarLd.getType())) {
      mlir::Value ret = stateVarLd;
      for (auto inpTy : fnTy.getInputs()) {
        mlir::BlockArgument blkArg = entryBlk->addArgument(inpTy, loc);
        if (isa<mlir::sol::ArrayType>(ret.getType())) {
          auto gep = b.create<mlir::sol::GepOp>(loc, ret, blkArg);
          gep.setNoPanicBoundsAttr(b.getUnitAttr());
          ret = genRValExpr(gep, loc);
        } else {
          // `ret` is no longer an ArrayType but the loop still has parameters
          // left. FunctionType(VariableDeclaration) only generates extra getter
          // parameters beyond array indices when the element type is a mapping;
          // scalars, structs, strings, and fixed-bytes each contribute at most
          // one parameter per array dimension and never leave residual params
          // after all GepOps are consumed. So reaching this branch guarantees
          // ret.getType() is MappingType, the cast<> is an assertion of that
          // invariant, not a blind guess.
          auto mapTy = cast<mlir::sol::MappingType>(ret.getType());
          mlir::Type addrTy = mapTy.getValType();
          if (!mlir::sol::isNonPtrRefType(mapTy.getValType()))
            addrTy =
                mlir::sol::PointerType::get(b.getContext(), mapTy.getValType(),
                                            mlir::sol::DataLocation::Storage);
          auto map = b.create<mlir::sol::MapOp>(loc, addrTy, ret, blkArg);
          ret = genRValExpr(map, loc);
        }
      }
      if (auto structTy = mlir::dyn_cast<mlir::sol::StructType>(ret.getType()))
        b.create<mlir::sol::ReturnOp>(loc,
                                      expandStructForReturn(structTy, ret));
      else {
        if (mlir::sol::isNonPtrRefType(ret.getType()))
          ret = genCast(ret, toMemoryType(ret.getType()));
        b.create<mlir::sol::ReturnOp>(loc, ret);
      }

      // Mapping type
    } else if (auto mappingTy =
                   dyn_cast<mlir::sol::MappingType>(stateVarLd.getType())) {
      mlir::Value lastMap = stateVarLd;
      for (auto inpTy : fnTy.getInputs()) {
        mlir::BlockArgument blkArg = entryBlk->addArgument(inpTy, loc);
        if (isa<mlir::sol::ArrayType>(lastMap.getType())) {
          // After resolving a mapping value that is an array type, switch to
          // GepOp for each remaining (array-index) parameter.
          auto gep = b.create<mlir::sol::GepOp>(loc, lastMap, blkArg);
          gep.setNoPanicBoundsAttr(b.getUnitAttr());
          lastMap = genRValExpr(gep, loc);
        } else {
          auto lastMapTy = cast<mlir::sol::MappingType>(lastMap.getType());
          mlir::Type addrTy = lastMapTy.getValType();
          if (!mlir::sol::isNonPtrRefType(lastMapTy.getValType()))
            addrTy = mlir::sol::PointerType::get(
                b.getContext(), lastMapTy.getValType(),
                mlir::sol::DataLocation::Storage);
          auto map = b.create<mlir::sol::MapOp>(loc, addrTy, lastMap, blkArg);
          lastMap = genRValExpr(map, loc);
        }
      }
      // If the final mapped value is a struct, expand its members individually
      // so the return types match the flattened scalar return signature.
      mlir::Value finalVal = genRValExpr(lastMap, loc);
      if (auto structTy =
              mlir::dyn_cast<mlir::sol::StructType>(finalVal.getType()))
        b.create<mlir::sol::ReturnOp>(
            loc, expandStructForReturn(structTy, finalVal));
      else {
        // mapping(K => string/bytes) public: cast storage string to memory.
        if (mlir::sol::isNonPtrRefType(finalVal.getType()))
          finalVal = genCast(finalVal, toMemoryType(finalVal.getType()));
        b.create<mlir::sol::ReturnOp>(loc, finalVal);
      }

      // Struct type
    } else if (auto structTy =
                   dyn_cast<mlir::sol::StructType>(stateVarLd.getType())) {
      b.create<mlir::sol::ReturnOp>(
          loc, expandStructForReturn(structTy, stateVarLd));

      // Scalar, string, bytes etc.
      // For string/bytes public state variables, stateVarLd is
      // StringType<Storage> (a non-ptr ref type that genRValExpr doesn't
      // dereference). Cast to Memory so the ABI encoder reads the string data
      // out of storage instead of treating the raw storage slot number as a
      // memory pointer.
    } else {
      mlir::Value ret = stateVarLd;
      if (mlir::sol::isNonPtrRefType(ret.getType()))
        ret = genCast(ret, toMemoryType(ret.getType()));
      b.create<mlir::sol::ReturnOp>(loc, ret);
    }

    return fn;
  }

  /// Returns the Solidity default value of the given type.
  mlir::Value genDefaultVal(mlir::Type ty, mlir::Location loc);

  /// Generates the ir to default-initialize the allocation.
  void genDefaultVal(mlir::sol::AllocaOp addr);

  /// Generates the side-effect of `delete addr`: zeros the value at any
  /// pointer (storage, memory, or stack).
  void genDeleteExpr(mlir::Value addr, mlir::Location loc);

  /// Returns the Memory-location variant of a reference type, recursively
  /// converting nested array element types, struct member types, and
  /// StringType from any data location to Memory. Non-reference types
  /// (scalars, address, fixedbytes, etc.) are returned unchanged.
  mlir::Type toMemoryType(mlir::Type ty) const {
    mlir::DenseSet<mlir::sol::StructType> identifiedStructsInProgress;
    return toMemoryType(ty, identifiedStructsInProgress);
  }

  /// Recursive worker for toMemoryType(). `identifiedStructsInProgress` tracks
  /// identified (recursive) Sol struct types whose body is currently being
  /// rebuilt in the Memory location, keyed by the uniqued Sol type rather than
  /// the Solidity `StructType` pointer, so a self-reference (e.g. the `S
  /// memory` element reached through an `S[] memory` member) resolves to the
  /// same in-progress instance and the cycle terminates. Every recursive call
  /// must thread this set through. The public overload above seeds a fresh one.
  mlir::Type toMemoryType(mlir::Type ty,
                          mlir::DenseSet<mlir::sol::StructType>
                              &identifiedStructsInProgress) const {
    if (auto arrTy = mlir::dyn_cast<mlir::sol::ArrayType>(ty))
      return mlir::sol::ArrayType::get(
          b.getContext(), arrTy.getSizeOpt(),
          toMemoryType(arrTy.getEltType(), identifiedStructsInProgress),
          mlir::sol::DataLocation::Memory);
    if (auto structTy = mlir::dyn_cast<mlir::sol::StructType>(ty)) {
      // Identified (recursive) structs must stay identified: building a literal
      // memory struct would recurse forever on the self-reference. Re-fetch the
      // identified type by name in the memory location (idempotent when the
      // input is already memory) and break the cycle via the in-progress set.
      if (structTy.isIdentified()) {
        auto memTy = mlir::sol::StructType::getIdentified(
            b.getContext(), structTy.getName(), mlir::sol::DataLocation::Memory);
        if (!memTy.isOpaque() ||
            !identifiedStructsInProgress.insert(memTy).second)
          return memTy;
        llvm::SmallVector<mlir::Type> memMemberTys;
        for (auto mt : structTy.getMemberTypes())
          memMemberTys.push_back(toMemoryType(mt, identifiedStructsInProgress));
        identifiedStructsInProgress.erase(memTy);
        bool bodySet = mlir::succeeded(memTy.setBody(memMemberTys));
        (void)bodySet;
        assert(bodySet && "conflicting body for identified struct");
        return memTy;
      }
      llvm::SmallVector<mlir::Type> memMemberTys;
      for (auto mt : structTy.getMemberTypes())
        memMemberTys.push_back(toMemoryType(mt, identifiedStructsInProgress));
      return mlir::sol::StructType::get(b.getContext(), memMemberTys,
                                        mlir::sol::DataLocation::Memory);
    }
    if (mlir::isa<mlir::sol::StringType>(ty))
      return mlir::sol::StringType::get(b.getContext(),
                                        mlir::sol::DataLocation::Memory);
    return ty;
  }

  /// Generates a integral constant op.
  mlir::Value genUnsignedConst(uint64_t val, unsigned numBits,
                               mlir::Location loc) {
    return b.create<mlir::sol::ConstantOp>(
        loc,
        b.getIntegerAttr(b.getIntegerType(numBits, /*isSigned=*/false), val));
  }

  /// Returns a compile-time selector for a function expression when available.
  mlir::Value genCompileTimeFunctionSelector(Expression const &fnExpr,
                                             FunctionType const &fnTy,
                                             mlir::Location loc,
                                             bool stateVarGetterOnly = false);

  /// Extracts the runtime selector from an external function pointer.
  mlir::Value genRuntimeFunctionSelector(Expression const &fnExpr,
                                         FunctionType const &fnTy,
                                         mlir::Location loc);

  /// Generates type cast expression.
  mlir::Value genCast(mlir::Value val, mlir::Type dstTy);

  /// Returns the mlir expression for the literal.
  mlir::Value genExpr(Literal const &lit);

  /// Returns the mlir expression for the identifier in an l-value context.
  mlir::Value genExpr(Identifier const &ident);

  /// Returns the mlir expression for the index access in an l-value context.
  mlir::Value genExpr(IndexAccess const &idxAcc);

  /// Returns the mlir expression for the index range access (array slice).
  mlir::Value genExpr(IndexRangeAccess const &idxRangeAcc);

  /// Returns the mlir expression for the member access in an r-value context.
  mlir::Value genExpr(MemberAccess const &memberAcc);

  /// Returns the mlir expression for the binary operation.
  mlir::Value genBinExpr(Token op, mlir::Value lhs, mlir::Value rhs,
                         mlir::Location loc);

  /// Returns the mlir expression for the unary operation.
  mlir::Value genExpr(UnaryOperation const &unaryOp);

  /// Returns the mlir expression for the binary operation.
  mlir::Value genExpr(BinaryOperation const &binOp);

  /// Returns the mlir expressions for the conditional (ternary) operation.
  mlir::SmallVector<mlir::Value> genExprs(Conditional const &cond);

  /// Returns the mlir expression for the call.
  mlir::SmallVector<mlir::Value> genExprs(FunctionCall const &call);

  /// Returns the mlir expression for the tuple.
  mlir::SmallVector<mlir::Value> genExprs(TupleExpression const &tuple);

  /// Address, gas, and value extracted from a low-level call site.
  struct LowLevelCallInfo {
    Expression const *callExpr;
    MemberAccess const *memberAcc;
    mlir::Value addr;
    mlir::Value gas;
    mlir::Value value;
  };

  /// `true` if `memAcc`'s base is a library type-expression and the callee is
  /// reached via a delegate call (e.g. `Lib.f(...)`).
  bool isDirectLibraryMemberCallBase(MemberAccess const &memAcc,
                                     FunctionType const &calleeTy);

  /// Parses gas/value call-options and the base receiver out of `call`. Used
  /// by both external and bare-call lowerings.
  LowLevelCallInfo parseLowLevelCallInfo(FunctionCall const &call,
                                         FunctionType const &calleeTy);

  /// Returns the gas value to be passed to a low-level call - the caller-set
  /// `gas` if provided, otherwise GASLEFT minus the cost the EVM charges the
  /// caller for the call instruction (on EVM versions that don't overcharge).
  mlir::Value materializeCallGas(mlir::Value gas, u256 const &gasNeededByCaller,
                                 mlir::Location loc);

  /// Status and results of a high-level external call.
  struct ExternalCallResult {
    mlir::Value status;
    mlir::SmallVector<mlir::Value> results;
  };
  ExternalCallResult genExternalCall(FunctionCall const &call);

  // We can't completely rely on ExpressionAnnotation::isLValue here since the
  // TypeChecker doesn't, for instance, tag RHS expression of an assignment as
  // an r-value.

  /// Returns the mlir expression in an l-value context.
  mlir::Value genLValExpr(Expression const &expr);
  mlir::SmallVector<mlir::Value> genLValExprs(Expression const &expr);

  /// Returns the mlir expression in an r-value context and optionally casts it
  /// to the corresponding mlir type of `resTy`.
  mlir::Value genRValExpr(Expression const &expr,
                          std::optional<mlir::Type> resTy = std::nullopt);
  mlir::Value genRValExpr(mlir::Value val, mlir::Location loc,
                          std::optional<mlir::Type> resTy = std::nullopt);
  mlir::SmallVector<mlir::Value> genRValExprs(Expression const &expr,
                                              mlir::TypeRange resTys = {});

  /// Generates an ir that assigns `rhs` to `lhs`.
  void genAssign(mlir::Value lhs, mlir::Value rhs, mlir::Location loc);

  /// Lowers the expression statement.
  void lower(ExpressionStatement const &);

  /// Lowers the emit statement.
  void lower(EmitStatement const &);

  /// Lowers the revert statement.
  void lower(RevertStatement const &);

  /// Lowers the break statement.
  void lower(Break const &);

  /// Lowers the continue statement.
  void lower(Continue const &);

  /// Lowers the placeholder statement.
  void lower(PlaceholderStatement const &);

  /// Lowers the return statement.
  void lower(Return const &);

  /// Lowers the assignment statement. Returns the LHS addresses.
  mlir::SmallVector<mlir::Value> lower(Assignment const &);

  /// Lowers the variable declaration statement.
  void lower(VariableDeclarationStatement const &);

  /// Lowers the expression of a statement for its side effects, discarding
  /// the values. Tuple statements are discarded per component.
  void genDiscardedExpr(Expression const &expr);

  /// Lowers the if-then-else statement.
  void lower(IfStatement const &);

  /// Lowers the while/do-while statement.
  void lower(WhileStatement const &);

  /// Lowers the for statement.
  void lower(ForStatement const &);

  /// Lowers the try statement.
  void lower(TryStatement const &);

  /// Lowers the inline asm statement.
  void lower(InlineAssembly const &);

  /// Lower the statement.
  void lower(Statement const &);

  /// Lowers the block.
  void lower(Block const &);

  /// Lowers the modifier definition.
  void lower(ModifierDefinition const &);

  /// Lowers the function definition.
  mlir::sol::FuncOp lower(FunctionDefinition const &);

  /// The constructor chain of the contract being lowered, most-derived
  /// first: {contract, its constructor}. The head's constructor is null when
  /// it is synthesized. Contracts without constructors do not appear
  /// (except as the head).
  std::vector<
      std::pair<ContractDefinition const *, FunctionDefinition const *>>
      ctorChain;

  /// The provider of each base constructor's argument list: the contract
  /// whose inheritance specifier or constructor modifier supplies the
  /// arguments. Argument expressions can only reference the provider's
  /// constructor parameters, so they must be lowered in the provider's
  /// constructor and the values threaded down the chain (via-ir style).
  std::map<FunctionDefinition const *, ContractDefinition const *>
      baseCtorArgProviders;

  /// Base-constructor argument values available in the constructor-chain
  /// function currently being lowered, keyed by target constructor: values
  /// received as threaded parameters plus values evaluated here.
  std::map<FunctionDefinition const *, mlir::SmallVector<mlir::Value>>
      pendingBaseCtorArgs;

  /// Returns the position of the constructor of `cont` in ctorChain.
  size_t ctorChainPos(ContractDefinition const &cont) {
    for (size_t i = 0; i < ctorChain.size(); ++i)
      if (ctorChain[i].first == &cont)
        return i;
    llvm_unreachable("Contract not in the constructor chain");
  }

  /// Returns the target constructors whose pending argument values the
  /// constructor at chain position `pos` receives as threaded parameters:
  /// deeper constructors whose provider sits strictly above `pos` in the
  /// chain.
  mlir::SmallVector<FunctionDefinition const *>
  pendingBaseCtorTargets(size_t pos) {
    mlir::SmallVector<FunctionDefinition const *> targets;
    for (size_t j = pos + 1; j < ctorChain.size(); ++j) {
      FunctionDefinition const *target = ctorChain[j].second;
      auto provider = baseCtorArgProviders.find(target);
      if (provider == baseCtorArgProviders.end())
        continue;
      // Providers without a constructor of their own are not in the chain;
      // their argument expressions cannot reference constructor parameters
      // and are lowered at the call to the target instead.
      bool providerInChain = false;
      size_t providerPos = 0;
      for (size_t p = 0; p < ctorChain.size(); ++p)
        if (ctorChain[p].first == provider->second) {
          providerInChain = true;
          providerPos = p;
          break;
        }
      if (providerInChain && providerPos < pos)
        targets.push_back(target);
    }
    return targets;
  }

  /// Generates the call to the next constructor in `currContract`'s
  /// linearization from `curCont`'s constructor: lowers every argument list
  /// provided by `curCont`, then passes the next constructor its arguments
  /// along with the pending values for deeper constructors.
  void genBaseCtorCall(ContractDefinition const &curCont,
                       FunctionDefinition const &nextCtor, mlir::Location loc);

  /// Emits a free or library function into the nearest enclosing contract
  /// scope on demand, if it hasn't been emitted there yet. A no-op when \p fn
  /// doesn't require cross-scope emission (e.g. the callee is already in the
  /// same library, or no contract scope is active).
  void lowerFreeOrLibFuncIfAbsent(FunctionDefinition const &fn);
};

} // namespace solidity::frontend

/// Returns the mlir::sol::DataLocation of the type
static mlir::sol::DataLocation getDataLocation(ReferenceType const *ty) {
  switch (ty->location()) {
  case DataLocation::CallData:
    return mlir::sol::DataLocation::CallData;
  case DataLocation::Storage:
    return mlir::sol::DataLocation::Storage;
  case DataLocation::Memory:
    return mlir::sol::DataLocation::Memory;
  case DataLocation::Transient:
    llvm_unreachable("NYI");
  }
}

mlir::Type SolidityToMLIRPass::getType(Type const *ty, bool indirectFn) {
  mlir::DenseSet<mlir::sol::StructType> identifiedStructsInProgress;
  return getType(ty, indirectFn, identifiedStructsInProgress);
}

mlir::Type SolidityToMLIRPass::getType(
    Type const *ty, bool indirectFn,
    mlir::DenseSet<mlir::sol::StructType> &identifiedStructsInProgress) {
  switch (ty->category()) {
  case Type::Category::Bool:
    return b.getIntegerType(/*width=*/1);

  case Type::Category::Integer: {
    const auto *intTy = static_cast<IntegerType const *>(ty);
    return b.getIntegerType(intTy->numBits(), intTy->isSigned());
  }
  case Type::Category::Enum: {
    const auto *enumTy = static_cast<EnumType const *>(ty);
    return mlir::sol::EnumType::get(b.getContext(), enumTy->maxValue());
  }
  case Type::Category::RationalNumber: {
    const auto *ratNumTy = static_cast<RationalNumberType const *>(ty);
    if (ratNumTy->isFractional())
      llvm_unreachable("NYI: Fractional type");
    const IntegerType *intTy = ratNumTy->integerType();
    return b.getIntegerType(intTy->numBits(), intTy->isSigned());
  }
  case Type::Category::Address: {
    const auto *addrTy = static_cast<AddressType const *>(ty);
    // Preserve FE payability in the Sol address type.
    return mlir::sol::AddressType::get(
        b.getContext(), addrTy->stateMutability() >= StateMutability::Payable);
  }

  case Type::Category::FixedBytes: {
    const auto *fixedBytesTy = static_cast<FixedBytesType const *>(ty);
    return mlir::sol::FixedBytesType::get(b.getContext(),
                                          fixedBytesTy->numBytes());
  }
  case Type::Category::Mapping: {
    auto *mappingTy = static_cast<MappingType const *>(ty);
    return mlir::sol::MappingType::get(
        b.getContext(),
        getType(mappingTy->keyType(), /*indirectFn=*/true,
                identifiedStructsInProgress),
        getType(mappingTy->valueType(), /*indirectFn=*/true,
                identifiedStructsInProgress));
  }
  case Type::Category::Array: {
    // Array or string type
    const auto *arrTy = static_cast<ArrayType const *>(ty);
    if (arrTy->isByteArrayOrString())
      return mlir::sol::StringType::get(b.getContext(), getDataLocation(arrTy));
    mlir::Type eltTy = getType(arrTy->baseType(), /*indirectFn=*/true,
                               identifiedStructsInProgress);

    std::optional<llvm::APInt> size =
        arrTy->isDynamicallySized()
            ? /*size=*/std::nullopt
            : std::optional<llvm::APInt>(
                  mlirgen::getAPInt(arrTy->length(), 256));
    return mlir::sol::ArrayType::get(b.getContext(), size, eltTy,
                                     getDataLocation(arrTy));
  }
  case Type::Category::ArraySlice: {
    const auto *sliceTy = static_cast<ArraySliceType const *>(ty);
    ArrayType const &arrTy = sliceTy->arrayType();
    if (arrTy.isByteArrayOrString())
      return mlir::sol::StringType::get(b.getContext(),
                                        getDataLocation(&arrTy));

    mlir::Type eltTy = getType(arrTy.baseType(), /*indirectFn=*/true,
                               identifiedStructsInProgress);
    return mlir::sol::ArrayType::get(b.getContext(), /*size=*/std::nullopt,
                                     eltTy, getDataLocation(&arrTy));
  }
  case Type::Category::Struct: {
    const auto *structTy = static_cast<StructType const *>(ty);
    mlir::sol::DataLocation loc = getDataLocation(structTy);

    // Self-referential structs (e.g. `struct S { S[] arr; }`) are represented
    // as *identified* Sol structs: created opaque and keyed by a unique name,
    // so a self-reference encountered while building the body resolves to the
    // same instance. The body (and storage layout) is then filled in once.
    if (structTy->structDefinition().annotation().recursive.value_or(false)) {
      // The name is the declaration name + AST id (like getMangledName for
      // functions): location-independent and pointer-independent. Data
      // location is already a uniquing key of the identified type, so encoding
      // it in the name (as Solidity's type identifier does) would be redundant
      // — and would let independently derived keys for the same struct
      // disagree, e.g. toMemoryType() re-fetching a storage struct's name in
      // the Memory location. Pointer-ness is modeled by sol.ptr<>, not by the
      // struct type, so it stays out of the name too: one identified type per
      // (struct declaration, location).
      StructDefinition const &structDef = structTy->structDefinition();
      std::string name =
          structDef.name() + "_" + std::to_string(structDef.id());
      auto structMlirTy =
          mlir::sol::StructType::getIdentified(b.getContext(), name, loc);
      // Already built, or a self-reference while still building: the (possibly
      // opaque) identified type is exactly what we want to hand back. The
      // in-progress set is keyed by the uniqued Sol type so the cycle is
      // detected regardless of which Solidity instance triggered the recursion.
      if (!structMlirTy.isOpaque() ||
          !identifiedStructsInProgress.insert(structMlirTy).second)
        return structMlirTy;
      std::vector<mlir::Type> memberTys;
      for (const auto &mem : structTy->nativeMembers(nullptr))
        memberTys.push_back(getType(mem.type, /*indirectFn=*/true,
                                    identifiedStructsInProgress));
      identifiedStructsInProgress.erase(structMlirTy);
      bool bodySet = mlir::succeeded(structMlirTy.setBody(memberTys));
      (void)bodySet;
      assert(bodySet && "conflicting body for identified struct");
      return structMlirTy;
    }

    if (!structsInProgress.insert(structTy).second)
      llvm_unreachable("unexpected recursive non-identified struct type");
    std::vector<mlir::Type> memberTys;
    for (const auto &mem : structTy->nativeMembers(nullptr))
      memberTys.push_back(
          getType(mem.type, /*indirectFn=*/true, identifiedStructsInProgress));
    structsInProgress.erase(structTy);
    return mlir::sol::StructType::get(b.getContext(), memberTys, loc);
  }
  case Type::Category::Function: {
    const auto *fnTy = static_cast<FunctionType const *>(ty);
    std::vector<mlir::Type> inTys, outTys;

    inTys.reserve(fnTy->parameterTypes().size());
    for (Type const *inTy : fnTy->parameterTypes())
      inTys.push_back(
          getType(inTy, /*indirectFn=*/true, identifiedStructsInProgress));

    outTys.reserve(fnTy->returnParameterTypes().size());
    for (Type const *outTy : fnTy->returnParameterTypes())
      outTys.push_back(
          getType(outTy, /*indirectFn=*/true, identifiedStructsInProgress));

    mlir::FunctionType mlirFnTy = b.getFunctionType(inTys, outTys);
    if (indirectFn) {
      if (fnTy->kind() == FunctionType::Kind::External)
        return mlir::sol::ExtFuncRefType::get(b.getContext(), mlirFnTy);
      return mlir::sol::FuncRefType::get(b.getContext(), mlirFnTy);
    }
    return mlirFnTy;
  }
  case Type::Category::Contract: {
    const auto *contTy = static_cast<ContractType const *>(ty);
    return mlir::sol::ContractType::get(
        b.getContext(), getMangledName(contTy->contractDefinition()),
        contTy->isPayable());
  }
  case Type::Category::UserDefinedValueType: {
    const auto *userDefTy = static_cast<UserDefinedValueType const *>(ty);
    return getType(&userDefTy->underlyingType(), indirectFn,
                   identifiedStructsInProgress);
  }
  default:
    break;
  }

  llvm_unreachable("NYI");
}

void SolidityToMLIRPass::lowerFreeOrLibFuncIfAbsent(
    FunctionDefinition const &fn) {
  bool calleeInLib =
      fn.annotation().contract && fn.annotation().contract->isLibrary();
  // A library callee is already part of the module when it belongs to the
  // library currently being compiled. Functions of a foreign library (e.g. a
  // cross-library call while compiling a library's own module) must be
  // emitted on demand.
  bool calleeForeign = fn.annotation().contract != currContract;
  bool freeCallee = !fn.annotation().contract;
  if (!((calleeInLib && calleeForeign) || (freeCallee && currContract)))
    return;

  auto *symTableOp = mlir::SymbolTable::getNearestSymbolTable(
      b.getInsertionBlock()->getParentOp());
  assert(symTableOp);
  if (mlir::SymbolTable::lookupSymbolIn(symTableOp, getMangledName(fn)))
    return;

  mlir::OpBuilder::InsertionGuard insertGuard(b);
  auto *parentOp = b.getInsertionBlock()->getParentOp();
  if (!mlir::isa<mlir::sol::FuncOp>(parentOp))
    parentOp = parentOp->getParentOfType<mlir::sol::FuncOp>();
  assert(parentOp);
  b.setInsertionPoint(parentOp);
  lower(fn);
}

mlir::Value SolidityToMLIRPass::genExpr(Identifier const &id) {
  Declaration const *decl = id.annotation().referencedDeclaration;

  if (MagicVariableDeclaration const *magicVar =
          dynamic_cast<MagicVariableDeclaration const *>(decl)) {
    switch (magicVar->type()->category()) {
    case Type::Category::Contract: {
      assert(id.name() == "this");
      assert(currContract && "'this' must be emitted in contract context");
      auto thisTy = mlir::sol::ContractType::get(
          b.getContext(), getMangledName(*currContract),
          ContractType(*currContract).isPayable());
      return b.create<mlir::sol::ThisOp>(getLoc(id), thisTy);
    }
    case Type::Category::TypeType:
      // `super` is a pure lookup handle with no runtime representation.
      if (id.name() == "super")
        return {};
      break;
    default:
      break;
    }
    llvm_unreachable("NYI");
  }

  if (const auto *var = dynamic_cast<VariableDeclaration const *>(decl))
    return genLValRef(*var);

  if (const auto *contr = dynamic_cast<ContractDefinition const *>(decl)) {
    if (contr->isLibrary())
      return b.create<mlir::sol::LibAddrOp>(
          getLoc(id), mlir::sol::AddressType::get(b.getContext(), false),
          contr->fullyQualifiedName());
    // A bare contract type name is a pure compile-time handle.
    return {};
  }

  // Type-declaration references (struct, enum and user-defined value type
  // names) are pure compile-time handles with no runtime representation.
  if (dynamic_cast<StructDefinition const *>(decl) ||
      dynamic_cast<EnumDefinition const *>(decl) ||
      dynamic_cast<UserDefinedValueTypeDefinition const *>(decl))
    return {};

  if (const auto *fn = dynamic_cast<FunctionDefinition const *>(decl)) {
    // Virtual functions referenced as values resolve against the most-derived
    // contract, exactly like called references do.
    if (fn->virtualSemantics() && currContract)
      fn = &fn->resolveVirtual(*currContract);
    // When a free/library function is referenced as a value (function pointer),
    // the call-site handler never fires. Emit it on demand so
    // FuncConstantOpLowering can resolve the symbol.
    lowerFreeOrLibFuncIfAbsent(*fn);
    return b.create<mlir::sol::FuncConstantOp>(getLoc(id), getType(fn->type()),
                                               getMangledName(*fn));
  }

  // Module aliases (`import "..." as M`) have no runtime representation.
  if (dynamic_cast<ImportDirective const *>(decl))
    return {};

  llvm_unreachable("NYI");
}

mlir::Value SolidityToMLIRPass::genDefaultVal(mlir::Type ty,
                                              mlir::Location loc) {
  return mlir::sol::genDefaultVal(b, ty, loc);
}

void SolidityToMLIRPass::genDefaultVal(mlir::sol::AllocaOp addr) {
  mlir::Location loc = addr.getLoc();
  auto pointeeTy =
      mlir::cast<mlir::sol::PointerType>(addr.getType()).getPointeeType();
  auto val = genDefaultVal(pointeeTy, loc);
  b.create<mlir::sol::StoreOp>(loc, val, addr);
}

void SolidityToMLIRPass::genDeleteExpr(mlir::Value addr, mlir::Location loc) {
  mlir::Type addrTy = addr.getType();

  // Reference types (arrays, structs, strings).
  if (mlir::sol::isNonPtrRefType(addrTy)) {
    // Storage references clear in place via sol.delete, which recursively
    // zeroes every occupied slot. Unlike the allocate-empty-and-copy approach
    // below, this never attempts to copy uncopyable members (e.g. mappings) and
    // matches the legacy codegen's `delete` semantics.
    if (mlir::sol::getDataLocation(addrTy) ==
        mlir::sol::DataLocation::Storage) {
      b.create<mlir::sol::DeleteOp>(loc, addr);
      return;
    }

    // Non-storage references (e.g. memory): zero-init by allocating a memory
    // version and copying it via genAssign which emits sol.copy.
    genAssign(addr, genDefaultVal(toMemoryType(addrTy), loc), loc);
    return;
  }

  // Value types and pointer-wrapped reference types.
  mlir::Type pointeeTy =
      mlir::cast<mlir::sol::PointerType>(addrTy).getPointeeType();

  // The type checker rejects `delete` on every storage-pointer lvalue (local
  // bindings, internal-function `storage` parameters, etc.), and member/element
  // deletes are dereferenced to a bare NonPtrRefType before reaching here, so
  // no valid Solidity source routes a storage-pointer-to-reference lvalue here.
  assert(!(mlir::sol::isNonPtrRefType(pointeeTy) &&
           mlir::sol::getDataLocation(pointeeTy) ==
               mlir::sol::DataLocation::Storage) &&
         "delete on storage-pointer-to-reference lvalue is unreachable");

  if (!mlir::sol::isScalar(pointeeTy)) {
    genAssign(addr, genDefaultVal(toMemoryType(pointeeTy), loc), loc);
    return;
  }

  // Value types: zero and store directly.
  auto val = genDefaultVal(pointeeTy, loc);
  b.create<mlir::sol::StoreOp>(loc, val, addr);
}

mlir::Value SolidityToMLIRPass::genCast(mlir::Value val, mlir::Type dstTy) {
  mlir::Location loc = val.getLoc();
  mlir::Type srcTy = val.getType();

  // Don't cast if we're casting to the same type.
  if (srcTy == dstTy)
    return val;

  if (mlir::isa<mlir::sol::ContractType>(srcTy) &&
      mlir::isa<mlir::sol::ContractType>(dstTy))
    return b.create<mlir::sol::ContractCastOp>(loc, dstTy, val);

  // Address casts, including non-payable <-> payable address typing and
  // bytes20 <-> address conversions.
  if (mlir::isa<mlir::sol::AddressType>(srcTy) ||
      mlir::isa<mlir::sol::AddressType>(dstTy)) {
    auto isUint160Ty = [](mlir::Type ty) {
      auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty);
      return intTy && intTy.getWidth() == 160 && intTy.isUnsigned();
    };

    // AddressCastOp only accepts integer<->address through uint160.
    // Normalize wider/narrower integer sources to uint160 first.
    if (mlir::isa<mlir::sol::AddressType>(dstTy) &&
        mlir::isa<mlir::IntegerType>(srcTy) && !isUint160Ty(srcTy))
      val = genCast(val, b.getIntegerType(/*width=*/160, /*isSigned=*/false));

    if (mlir::isa<mlir::sol::AddressType>(srcTy) &&
        mlir::isa<mlir::IntegerType>(dstTy))
      assert(isUint160Ty(dstTy) &&
             "Address casts to integer must target uint160");

    return b.create<mlir::sol::AddressCastOp>(loc, dstTy, val);
  }

  if (mlir::isa<mlir::sol::ByteType, mlir::sol::FixedBytesType>(srcTy) ||
      mlir::isa<mlir::sol::ByteType, mlir::sol::FixedBytesType>(dstTy)) {

    auto materializeStringLiteralAsBytesInt =
        [&](mlir::sol::StringLitOp litOp, unsigned dstBytes) -> mlir::Value {
      llvm::StringRef lit = litOp.getValue();
      assert(static_cast<unsigned>(lit.size()) <= dstBytes &&
             "string literal does not fit destination bytes type");

      unsigned width = dstBytes * 8;
      llvm::APInt litInt(width, /*val=*/0, /*isSigned=*/false);

      // Build a big-endian byte sequence in the low bits first.
      for (unsigned char c : lit.bytes()) {
        litInt = litInt.shl(8);
        litInt |= llvm::APInt(width, c);
      }

      // Then shift to Solidity fixed-bytes layout (left-aligned in 256-bit
      // slot semantics, i.e. zero-padding on the right).
      if (dstBytes > lit.size())
        litInt = litInt.shl(8 * (dstBytes - lit.size()));

      mlir::Type intTy = b.getIntegerType(width, /*isSigned=*/false);
      return b.create<mlir::sol::ConstantOp>(loc,
                                             b.getIntegerAttr(intTy, litInt));
    };

    // String literal to fixed-bytes conversion is a compile-time conversion.
    // Materialize an integer constant matching the destination byte
    // width and then reuse the regular int->bytes cast path.
    auto litOp = val.getDefiningOp<mlir::sol::StringLitOp>();
    if (auto dstBytesTy = mlir::dyn_cast<mlir::sol::FixedBytesType>(dstTy)) {
      if (litOp) {
        val = materializeStringLiteralAsBytesInt(litOp, dstBytesTy.getSize());
      } else if (mlir::isa<mlir::sol::StringType>(srcTy)) {
        return b.create<mlir::sol::DynBytesToFixedBytesOp>(loc, dstTy, val);
      }
    } else if (mlir::isa<mlir::sol::ByteType>(dstTy) && litOp) {
      val = materializeStringLiteralAsBytesInt(litOp, /*dstBytes=*/1);
    }
    // bytes_cast requires the integer operand width to match the destination
    // byte count exactly; widen / narrow the operand to bridge the difference.
    if (auto inpIntTy = mlir::dyn_cast<mlir::IntegerType>(val.getType())) {
      unsigned dstWidth = 0;
      if (auto dstBytesTy = mlir::dyn_cast<mlir::sol::FixedBytesType>(dstTy))
        dstWidth = dstBytesTy.getSize() * 8;
      else if (mlir::isa<mlir::sol::ByteType>(dstTy))
        dstWidth = 8;
      if (dstWidth != 0 && inpIntTy.getWidth() != dstWidth) {
        mlir::Type dstIntTy = b.getIntegerType(dstWidth, inpIntTy.isSigned());
        val = b.create<mlir::sol::CastOp>(loc, dstIntTy, val);
      }
    }
    return b.create<mlir::sol::BytesCastOp>(loc, dstTy, val);
  }

  // Enum casts can validate and therefore use a dedicated op in both
  // directions.
  if (mlir::isa<mlir::sol::EnumType>(srcTy) ||
      mlir::isa<mlir::sol::EnumType>(dstTy))
    return b.create<mlir::sol::EnumCastOp>(loc, dstTy, val);

  // Casting to integer type.
  if (mlir::isa<mlir::IntegerType>(dstTy))
    return b.create<mlir::sol::CastOp>(loc, dstTy, val);

  // Casting between reference types (excluding pointer types).
  if (mlir::sol::isNonPtrRefType(dstTy)) {
    assert(mlir::sol::isNonPtrRefType(srcTy));
    return b.create<mlir::sol::DataLocCastOp>(loc, dstTy, val);
  }

  llvm_unreachable("NYI or invalid cast");
}

mlir::Value SolidityToMLIRPass::genExpr(Literal const &lit) {
  mlir::Location loc = getLoc(lit);
  Type const *ty = lit.annotation().type;

  // Bool literal
  if (dynamic_cast<BoolType const *>(ty))
    return b.create<mlir::sol::ConstantOp>(
        loc, b.getBoolAttr(lit.token() == Token::TrueLiteral));

  // Rational number literal
  if (const auto *ratNumTy = dynamic_cast<RationalNumberType const *>(ty)) {
    if (ratNumTy->isFractional())
      llvm_unreachable("NYI: Fractional literal");

    auto *intTy = ratNumTy->integerType();
    u256 val = ty->literalValue(nullptr);
    // TODO: Is there a faster way to convert boost::multiprecision::number to
    // llvm::APInt?
    return b.create<mlir::sol::ConstantOp>(
        loc,
        b.getIntegerAttr(getType(ty), llvm::APInt(intTy->numBits(), val.str(),
                                                  /*radix=*/10)));
  }

  // String/bytes literal
  if (ty->category() == Type::Category::StringLiteral) {
    auto litTy = mlir::sol::StringType::get(b.getContext(),
                                            mlir::sol::DataLocation::Memory);
    return b.create<mlir::sol::StringLitOp>(loc, litTy,
                                            b.getStringAttr(lit.value()));
  }

  // Address literal: emit a 160-bit integer constant and cast it to the
  // sol address type, mirroring how `address(uint160(<n>))` already lowers.
  if (dynamic_cast<AddressType const *>(ty)) {
    u256 val = ty->literalValue(&lit);
    auto uint160Ty = b.getIntegerType(/*width=*/160, /*isSigned=*/false);
    mlir::Value intConst = b.create<mlir::sol::ConstantOp>(
        loc, b.getIntegerAttr(uint160Ty, getAPInt(val, 160)));
    return genCast(intConst, getType(ty));
  }

  llvm_unreachable("NYI: Literal");
}

mlir::Value SolidityToMLIRPass::genBinExpr(Token op, mlir::Value lhs,
                                           mlir::Value rhs,
                                           mlir::Location loc) {
  switch (op) {
  case Token::Add:
    if (inUnchecked)
      return b.create<mlir::sol::AddOp>(loc, lhs, rhs);
    else
      return b.create<mlir::sol::CAddOp>(loc, lhs, rhs);
  case Token::Sub:
    if (inUnchecked)
      return b.create<mlir::sol::SubOp>(loc, lhs, rhs);
    else
      return b.create<mlir::sol::CSubOp>(loc, lhs, rhs);
  case Token::Mul:
    if (inUnchecked)
      return b.create<mlir::sol::MulOp>(loc, lhs, rhs);
    else
      return b.create<mlir::sol::CMulOp>(loc, lhs, rhs);
  case Token::Div:
    if (inUnchecked)
      return b.create<mlir::sol::DivOp>(loc, lhs, rhs);
    else
      return b.create<mlir::sol::CDivOp>(loc, lhs, rhs);
  case Token::Mod:
    return b.create<mlir::sol::ModOp>(loc, lhs, rhs);
  case Token::Exp:
    if (inUnchecked)
      return b.create<mlir::sol::ExpOp>(loc, lhs.getType(), lhs, rhs);
    else
      return b.create<mlir::sol::CExpOp>(loc, lhs.getType(), lhs, rhs);
    break;
  case Token::BitAnd:
    return b.create<mlir::sol::AndOp>(loc, lhs, rhs);
  case Token::BitOr:
    return b.create<mlir::sol::OrOp>(loc, lhs, rhs);
  case Token::BitXor:
    return b.create<mlir::sol::XorOp>(loc, lhs, rhs);
  case Token::SHL:
  case Token::SAR: {
    // rhs stays as its mobile integer type; Sol_BitwiseShiftOp allows
    // independent lhs and rhs types (AllTypesMatch only links lhs and result).
    return op == Token::SHL
               ? static_cast<mlir::Value>(
                     b.create<mlir::sol::ShlOp>(loc, lhs.getType(), lhs, rhs))
               : static_cast<mlir::Value>(
                     b.create<mlir::sol::ShrOp>(loc, lhs.getType(), lhs, rhs));
  }
  case Token::Equal:
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::eq, lhs,
                                      rhs);
  case Token::NotEqual:
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::ne, lhs,
                                      rhs);
  case Token::LessThan:
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::lt, lhs,
                                      rhs);
  case Token::LessThanOrEqual:
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::le, lhs,
                                      rhs);
  case Token::GreaterThan:
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::gt, lhs,
                                      rhs);
  case Token::GreaterThanOrEqual:
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::ge, lhs,
                                      rhs);
  default:
    break;
  }
  llvm_unreachable("NYI: Binary operator");
}

mlir::Value SolidityToMLIRPass::genExpr(UnaryOperation const &unaryOp) {
  mlir::Location loc = getLoc(unaryOp);

  if (FunctionDefinition const *fn =
          *unaryOp.annotation().userDefinedFunction) {
    // Operator functions are free functions that may be referenced only
    // through the operator, so emit the callee on demand.
    lowerFreeOrLibFuncIfAbsent(*fn);
    mlir::Value arg = genRValExpr(unaryOp.subExpression(),
                                  getType(fn->parameters()[0]->type()));
    mlir::Type resTy = getType(fn->returnParameters()[0]->type());
    return b
        .create<mlir::sol::CallOp>(loc, getMangledName(*fn),
                                   mlir::TypeRange{resTy},
                                   mlir::ValueRange{arg})
        ->getResult(0);
  }

  // 'delete x' is a statement-level side-effect with no result value.  Its
  // annotation type is a TupleType / void that getType() does not handle, so
  // short-circuit here before the getType() call below.
  if (unaryOp.getOperator() == Token::Delete) {
    genDeleteExpr(genLValExpr(unaryOp.subExpression()), loc);
    return {};
  }

  Type const *ty = unaryOp.annotation().type;
  mlir::Type mlirTy = getType(ty);

  // Negative constant
  if (ty->category() == Type::Category::RationalNumber) {
    auto intTy = mlir::cast<mlir::IntegerType>(mlirTy);
    u256 val = ty->literalValue(nullptr);
    return b.create<mlir::sol::ConstantOp>(
        loc, b.getIntegerAttr(intTy, getAPInt(val, intTy.getWidth())));
  }

  switch (unaryOp.getOperator()) {
  // Increment and decrement
  case Token::Inc:
  case Token::Dec: {
    mlir::Value lValExpr = genLValExpr(unaryOp.subExpression());
    mlir::Value rValExpr = genRValExpr(lValExpr, lValExpr.getLoc());
    mlir::Value one =
        b.create<mlir::sol::ConstantOp>(loc, b.getIntegerAttr(mlirTy, 1));
    mlir::Value newVal = genBinExpr(
        unaryOp.getOperator() == Token::Inc ? Token::Add : Token::Sub, rValExpr,
        one, loc);
    b.create<mlir::sol::StoreOp>(loc, newVal, lValExpr);
    return unaryOp.isPrefixOperation() ? newVal : rValExpr;
  }
  // Negation
  case Token::Sub: {
    mlir::Value expr = genRValExpr(unaryOp.subExpression());
    mlir::Value zero =
        b.create<mlir::sol::ConstantOp>(loc, b.getIntegerAttr(mlirTy, 0));
    return genBinExpr(Token::Sub, zero, expr, loc);
  }
  // Logical not
  case Token::Not: {
    mlir::Value expr = genRValExpr(unaryOp.subExpression());
    mlir::Value zero = b.create<mlir::sol::ConstantOp>(
        loc, b.getIntegerAttr(expr.getType(), 0));
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::eq, expr,
                                      zero);
  }
  // Bitwise not (~x == x ^ -1)
  case Token::BitNot: {
    mlir::Value expr = genRValExpr(unaryOp.subExpression());
    return b.create<mlir::sol::NotOp>(loc, expr);
  }
  default:
    break;
  }

  llvm_unreachable("NYI");
}

mlir::Value SolidityToMLIRPass::genExpr(BinaryOperation const &binOp) {
  if (FunctionDefinition const *fn = *binOp.annotation().userDefinedFunction) {
    auto loc = getLoc(binOp);
    // Operator functions are free functions that may be referenced only
    // through the operator, so emit the callee on demand.
    lowerFreeOrLibFuncIfAbsent(*fn);
    mlir::Value lhs = genRValExpr(binOp.leftExpression(),
                                  getType(fn->parameters()[0]->type()));
    mlir::Value rhs = genRValExpr(binOp.rightExpression(),
                                  getType(fn->parameters()[1]->type()));
    mlir::Type resTy = getType(fn->returnParameters()[0]->type());
    return b
        .create<mlir::sol::CallOp>(loc, getMangledName(*fn),
                                   mlir::TypeRange{resTy},
                                   mlir::ValueRange{lhs, rhs})
        ->getResult(0);
  }

  mlir::Type argTy = getType(binOp.annotation().commonType);
  auto loc = getLoc(binOp);
  BuilderExt bExt(b, loc);

  // Both operands are compile-time rational constants AND the result is itself
  // a rational number (arithmetic ops).
  if (binOp.annotation().type->category() == Type::Category::RationalNumber) {
    auto intTy = mlir::cast<mlir::IntegerType>(argTy);
    u256 val = binOp.annotation().commonType->literalValue(nullptr);
    return b.create<mlir::sol::ConstantOp>(
        loc, b.getIntegerAttr(intTy, getAPInt(val, intTy.getWidth())));
  }

  mlir::Value lhs = genRValExpr(binOp.leftExpression(), argTy);

  // Handle logical operators that can short-circuit.
  //
  // We generate `if` ops for the short-circuiting and an alloca op to track
  // the final value.
  //
  // TODO: We won't need the alloca for the short-circuting codegen if the
  // `if` ops can yield values.
  if (binOp.getOperator() == Token::And) {
    mlir::Type allocTy = mlir::sol::PointerType::get(
        b.getContext(), argTy, mlir::sol::DataLocation::Stack);
    auto alloca = b.create<mlir::sol::AllocaOp>(loc, allocTy);
    b.create<mlir::sol::StoreOp>(loc, bExt.genBool(false), alloca);

    auto ifOp = b.create<mlir::sol::IfOp>(loc, lhs);
    auto resPt = b.saveInsertionPoint();
    b.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
    mlir::Value rhs = genRValExpr(binOp.rightExpression(), argTy);
    b.create<mlir::sol::StoreOp>(loc, rhs, alloca);
    b.create<mlir::sol::YieldOp>(loc);
    b.restoreInsertionPoint(resPt);

    return b.create<mlir::sol::LoadOp>(loc, alloca);
  }
  if (binOp.getOperator() == Token::Or) {
    mlir::Type allocTy = mlir::sol::PointerType::get(
        b.getContext(), argTy, mlir::sol::DataLocation::Stack);
    auto alloca = b.create<mlir::sol::AllocaOp>(loc, allocTy);
    b.create<mlir::sol::StoreOp>(loc, lhs, alloca);

    auto ifOp = b.create<mlir::sol::IfOp>(loc, lhs);
    auto resPt = b.saveInsertionPoint();
    b.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
    b.create<mlir::sol::YieldOp>(loc);
    b.setInsertionPointToStart(&ifOp.getElseRegion().emplaceBlock());
    mlir::Value rhs = genRValExpr(binOp.rightExpression(), argTy);
    b.create<mlir::sol::StoreOp>(loc, rhs, alloca);
    b.create<mlir::sol::YieldOp>(loc);
    b.restoreInsertionPoint(resPt);

    return b.create<mlir::sol::LoadOp>(loc, alloca);
  }

  mlir::Type rhsTy = argTy;
  if (binOp.getOperator() == Token::Exp ||
      TokenTraits::isShiftOp(binOp.getOperator())) {
    Type const *rightTargetType =
        binOp.rightExpression().annotation().type->mobileType();
    assert(rightTargetType && "Expected right operand to have a mobile type");
    rhsTy = getType(rightTargetType);
  }

  mlir::Value rhs = genRValExpr(binOp.rightExpression(), rhsTy);

  return genBinExpr(binOp.getOperator(), lhs, rhs, loc);
}

mlir::SmallVector<mlir::Value>
SolidityToMLIRPass::genExprs(Conditional const &cond) {
  mlir::Location loc = getLoc(cond);
  mlir::Value condVal = genRValExpr(cond.condition());

  // Types with no runtime representation (e.g. a module in `(c ? M : M).D`):
  // evaluate the arms for their side effects only.
  if (cond.annotation().type->sizeOnStack() == 0) {
    auto ifOp = b.create<mlir::sol::IfOp>(loc, condVal);
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
    genDiscardedExpr(cond.trueExpression());
    b.create<mlir::sol::YieldOp>(loc);
    b.setInsertionPointToStart(&ifOp.getElseRegion().emplaceBlock());
    genDiscardedExpr(cond.falseExpression());
    b.create<mlir::sol::YieldOp>(loc);
    return {};
  }

  // Get result types - could be single type or tuple.
  mlir::SmallVector<mlir::Type> resTys;
  if (TupleType const *tupleTy =
          dynamic_cast<TupleType const *>(cond.annotation().type)) {
    for (const Type *astTy : tupleTy->components())
      resTys.push_back(getType(astTy));
  } else {
    resTys.push_back(getType(cond.annotation().type));
  }

  mlir::SmallVector<mlir::sol::AllocaOp> allocas;
  for (mlir::Type resTy : resTys) {
    mlir::Type allocTy = mlir::sol::PointerType::get(
        b.getContext(), resTy, mlir::sol::DataLocation::Stack);
    allocas.push_back(b.create<mlir::sol::AllocaOp>(loc, allocTy));
  }

  auto storeResults = [&](Expression const &expr) {
    mlir::SmallVector<mlir::Value> values = genRValExprs(expr, resTys);
    for (auto [value, alloca] : ranges::views::zip(values, allocas))
      b.create<mlir::sol::StoreOp>(loc, value, alloca);
  };

  auto ifOp = b.create<mlir::sol::IfOp>(loc, condVal);
  {
    mlir::OpBuilder::InsertionGuard guard(b);

    b.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
    storeResults(cond.trueExpression());
    b.create<mlir::sol::YieldOp>(loc);

    b.setInsertionPointToStart(&ifOp.getElseRegion().emplaceBlock());
    storeResults(cond.falseExpression());
    b.create<mlir::sol::YieldOp>(loc);
  }

  mlir::SmallVector<mlir::Value> results;
  results.reserve(allocas.size());
  for (mlir::sol::AllocaOp alloca : allocas)
    results.push_back(b.create<mlir::sol::LoadOp>(loc, alloca));
  return results;
}

mlir::Value SolidityToMLIRPass::genExpr(IndexAccess const &idxAcc) {
  mlir::Location loc = getLoc(idxAcc);

  // Type expressions like `s[7][]` are pure compile-time handles: array-type
  // lengths are compile-time constants, so there is nothing to evaluate.
  if (dynamic_cast<TypeType const *>(idxAcc.annotation().type))
    return {};

  mlir::Value baseExpr = genRValExpr(idxAcc.baseExpression());

  // Mapping
  if (auto mappingTy =
          mlir::dyn_cast<mlir::sol::MappingType>(baseExpr.getType())) {
    // Convert the key to the declared key type before hashing, like the old
    // codegen does (e.g. bytesN keys hash left-aligned, narrow signed keys
    // sign-extend).
    auto const *astMappingTy = dynamic_cast<MappingType const *>(
        idxAcc.baseExpression().annotation().type);
    assert(astMappingTy);
    mlir::Value idxExpr = genRValExpr(*idxAcc.indexExpression(),
                                      getType(astMappingTy->keyType()));
    mlir::Type addrTy;
    if (mlir::sol::isNonPtrRefType(mappingTy.getValType()))
      addrTy = mappingTy.getValType();
    else
      addrTy =
          mlir::sol::PointerType::get(b.getContext(), mappingTy.getValType(),
                                      mlir::sol::DataLocation::Storage);
    return b.create<mlir::sol::MapOp>(loc, addrTy, baseExpr, idxExpr);
  }

  mlir::Value idxExpr = genRValExpr(*idxAcc.indexExpression());

  // Bytes/array indexing
  if (mlir::isa<mlir::sol::ArrayType>(baseExpr.getType()) ||
      mlir::isa<mlir::sol::StringType>(baseExpr.getType()))
    return b.create<mlir::sol::GepOp>(loc, baseExpr, idxExpr);

  // Fixed-bytes indexing (`bytesN[i]`): yields the i-th byte as a `bytes1`.
  if (mlir::isa<mlir::sol::FixedBytesType>(baseExpr.getType())) {
    mlir::Type bytes1Ty =
        mlir::sol::FixedBytesType::get(b.getContext(), /*size=*/1);
    return b.create<mlir::sol::FixedBytesIndexOp>(loc, bytes1Ty, baseExpr,
                                                  idxExpr);
  }

  llvm_unreachable("Invalid IndexAccess");
}

mlir::Value SolidityToMLIRPass::genExpr(IndexRangeAccess const &idxRangeAcc) {
  mlir::Location loc = getLoc(idxRangeAcc);

  mlir::Value baseExpr = genRValExpr(idxRangeAcc.baseExpression());
  mlir::Type resTy = getType(idxRangeAcc.annotation().type);

  // Start defaults to 0 if not provided.
  mlir::Value startExpr;
  if (idxRangeAcc.startExpression())
    startExpr = genRValExpr(*idxRangeAcc.startExpression());
  else
    startExpr = b.create<mlir::sol::ConstantOp>(
        loc, b.getIntegerAttr(b.getIntegerType(256, /*isSigned=*/false), 0));

  // End defaults to array length if not provided.
  mlir::Value endExpr;
  if (idxRangeAcc.endExpression())
    endExpr = genRValExpr(*idxRangeAcc.endExpression());
  else
    endExpr = b.create<mlir::sol::LengthOp>(loc, baseExpr);

  return b.create<mlir::sol::SliceOp>(loc, resTy, baseExpr, startExpr, endExpr);
}

mlir::Value SolidityToMLIRPass::genCompileTimeFunctionSelector(
    Expression const &fnExpr, FunctionType const &fnTy, mlir::Location loc,
    bool stateVarGetterOnly) {
  auto genExprSideEffects = [&](Expression const &expr) {
    // Contract type names in expressions like 'C.f' as well as error and
    // event references (e.g. 'E.selector') are pure declaration handles, not
    // runtime values, and should not be lowered.
    if (auto const *id = dynamic_cast<Identifier const *>(&expr)) {
      Declaration const *decl = id->annotation().referencedDeclaration;
      if (dynamic_cast<ContractDefinition const *>(decl) ||
          dynamic_cast<ErrorDefinition const *>(decl) ||
          dynamic_cast<EventDefinition const *>(decl))
        return;
      // 'this' has no side effects on its own.
      if (id->name() == "this")
        return;
    }

    // The selector itself is lowered from declaration metadata, but the
    // base expression can still have side effects. Example:
    // 'get().f.selector' must evaluate 'get()'.
    (void)genLValExpr(expr);
  };

  // Call options attached to an uncalled function reference (e.g.
  // `this.g{gas: 42}.selector`) are evaluated for their side effects. The
  // selector comes from the underlying reference.
  Expression const *unwrappedFnExpr = &fnExpr;
  while (auto const *opts =
             dynamic_cast<FunctionCallOptions const *>(unwrappedFnExpr)) {
    for (ASTPointer<Expression const> const &optExpr : opts->options())
      (void)genRValExpr(*optExpr);
    unwrappedFnExpr = &opts->expression();
  }

  // Preserve side effects of the base expression in forms like 'expr.f'.
  if (auto const *fnMember =
          dynamic_cast<MemberAccess const *>(unwrappedFnExpr))
    genExprSideEffects(fnMember->expression());
  else
    genExprSideEffects(*unwrappedFnExpr);

  // Event selectors are the full 32-byte topic hash, unlike the 4-byte
  // function/error selector.
  if (fnTy.kind() == FunctionType::Kind::Event) {
    u256 topicHash(util::h256::Arith(util::keccak256(fnTy.externalSignature())));
    auto ui256Ty = b.getIntegerType(256, /*isSigned=*/false);
    return b.create<mlir::sol::ConstantOp>(
        loc, b.getIntegerAttr(
                 ui256Ty, llvm::APInt(256, topicHash.str(), /*radix=*/10)));
  }

  if (fnTy.hasDeclaration()) {
    auto selector = fnTy.externalIdentifier().convert_to<uint32_t>();
    return genUnsignedConst(selector, /*numBits=*/32, loc);
  }

  auto genSelectorFromDecl = [&](Declaration const *decl) -> mlir::Value {
    if (auto const *fn = dynamic_cast<FunctionDefinition const *>(decl)) {
      auto selector =
          FunctionType(*fn).externalIdentifier().convert_to<uint32_t>();
      return genUnsignedConst(selector, /*numBits=*/32, loc);
    }

    if (auto const *var = dynamic_cast<VariableDeclaration const *>(decl)) {
      // In strict mode (used by abi.encodeCall), only state-variable getter
      // declarations are treated as compile-time selector sources.
      // Runtime function-pointer values, e.g.:
      //  fp = cond ? this.f : this.g;
      //  abi.encodeCall(fp, (...));
      // require runtime selector extraction, which is NYI.
      if (stateVarGetterOnly && (!var->isStateVariable() || !var->isPublic()))
        return {};
      auto selector =
          FunctionType(*var).externalIdentifier().convert_to<uint32_t>();
      return genUnsignedConst(selector, /*numBits=*/32, loc);
    }

    return {};
  };

  // Handle unresolved declaration in cases like 'this.f'.
  if (auto const *fnMember =
          dynamic_cast<MemberAccess const *>(unwrappedFnExpr))
    return genSelectorFromDecl(fnMember->annotation().referencedDeclaration);

  return {};
}

mlir::Value SolidityToMLIRPass::genRuntimeFunctionSelector(
    Expression const &fnExpr, FunctionType const &fnTy, mlir::Location loc) {
  assert(fnTy.kind() == FunctionType::Kind::External &&
         "Expected external function pointer");
  mlir::Type bytes4Ty =
      mlir::sol::FixedBytesType::get(b.getContext(), /*size=*/4);
  mlir::Type fnRefTy = getType(&fnTy);

  // TODO: We should be able to get the selector directly here instead of
  // extracting it from the ExtFuncRefType. If we can do that, this op is
  // not needed at all.
  return b.create<mlir::sol::ExtFuncSelectorOp>(loc, bytes4Ty,
                                                genRValExpr(fnExpr, fnRefTy));
}

mlir::Value SolidityToMLIRPass::genExpr(MemberAccess const &memberAcc) {
  mlir::Location loc = getLoc(memberAcc);

  const Type *memberAccTy = memberAcc.expression().annotation().type;
  const ASTString &memberName = memberAcc.memberName();
  switch (memberAccTy->category()) {
  case Type::Category::Magic:
    if (memberName == "sender")
      return b.create<mlir::sol::CallerOp>(
          loc, getType(memberAcc.annotation().type));
    if (memberName == "data")
      return b.create<mlir::sol::GetCallDataOp>(loc);
    if (memberName == "creationCode" || memberName == "runtimeCode") {
      ContractDefinition const &contract = getMetaTypeContract(memberAccTy);
      std::string objName = getMangledName(contract);
      if (memberName == "runtimeCode")
        objName += "_deployed";

      return b.create<mlir::sol::ObjectCodeOp>(
          loc, getType(memberAcc.annotation().type), b.getStringAttr(objName));
    }
    if (memberName == "interfaceId") {
      ContractDefinition const &contract = getMetaTypeContract(memberAccTy);

      auto ui32Ty = b.getIntegerType(32, /*isSigned=*/false);
      auto id = b.create<mlir::sol::ConstantOp>(
          loc,
          b.getIntegerAttr(ui32Ty, llvm::APInt(32, contract.interfaceId())));
      return genCast(id, getType(memberAcc.annotation().type));
    }
    if (memberName == "name") {
      ContractDefinition const &contract = getMetaTypeContract(memberAccTy);
      return b.create<mlir::sol::StringLitOp>(
          loc, getType(memberAcc.annotation().type),
          b.getStringAttr(contract.name()));
    }
    if (memberName == "min" || memberName == "max") {
      auto const *magicTy = dynamic_cast<MagicType const *>(memberAccTy);
      assert(magicTy && "Expected magic type for min/max member access");
      Type const *argTy = magicTy->typeArgument();
      assert(argTy && "Expected metatype argument for min/max member access");

      if (auto const *intTy = dynamic_cast<IntegerType const *>(argTy)) {
        bool isMin = memberName == "min";
        unsigned width = intTy->numBits();
        llvm::APInt bound;
        if (intTy->isSigned())
          bound = isMin ? llvm::APInt::getSignedMinValue(width)
                        : llvm::APInt::getSignedMaxValue(width);
        else
          bound = isMin ? llvm::APInt::getZero(width)
                        : llvm::APInt::getMaxValue(width);

        mlir::Type mlirIntTy =
            b.getIntegerType(width, /*isSigned=*/intTy->isSigned());
        return b.create<mlir::sol::ConstantOp>(
            loc, b.getIntegerAttr(mlirIntTy, bound));
      }

      if (auto const *enumTy = dynamic_cast<EnumType const *>(argTy)) {
        unsigned enumBound =
            memberName == "min" ? enumTy->minValue() : enumTy->maxValue();
        auto ui256Ty = b.getIntegerType(256, /*isSigned=*/false);
        return b.create<mlir::sol::ConstantOp>(
            loc, b.getIntegerAttr(ui256Ty, llvm::APInt(256, enumBound)));
      }

      llvm_unreachable("min/max requested on unexpected metatype argument");
    }
    if (memberName == "origin")
      return b.create<mlir::sol::OriginOp>(
          loc, getType(memberAcc.annotation().type));
    if (memberName == "gasprice")
      return b.create<mlir::sol::GasPriceOp>(loc);
    if (memberName == "value")
      return b.create<mlir::sol::CallValueOp>(loc);
    if (memberName == "sig") {
      mlir::Type bytes4Ty =
          mlir::sol::FixedBytesType::get(b.getContext(), /*size=*/4);
      return b.create<mlir::sol::SigOp>(loc, bytes4Ty);
    }
    if (memberName == "basefee")
      return b.create<mlir::sol::BaseFeeOp>(loc);
    if (memberName == "blobbasefee")
      return b.create<mlir::sol::BlobBaseFeeOp>(loc);
    if (memberName == "chainid")
      return b.create<mlir::sol::ChainIdOp>(loc);
    if (memberName == "coinbase")
      return b.create<mlir::sol::CoinbaseOp>(
          loc, getType(memberAcc.annotation().type));
    if (memberName == "difficulty")
      return b.create<mlir::sol::DifficultyOp>(loc);
    if (memberName == "gaslimit")
      return b.create<mlir::sol::GasLimitOp>(loc);
    if (memberName == "number")
      return b.create<mlir::sol::BlockNumberOp>(loc);
    if (memberName == "prevrandao")
      return b.create<mlir::sol::PrevRandaoOp>(loc);
    if (memberName == "timestamp")
      return b.create<mlir::sol::TimestampOp>(loc);

    break;
  case Type::Category::Contract: {
    // Handle external function reference: contract.func / contract.getter
    auto const *decl = memberAcc.annotation().referencedDeclaration;
    if (!dynamic_cast<FunctionDefinition const *>(decl) &&
        !dynamic_cast<VariableDeclaration const *>(decl))
      return {};

    auto const *fnTy =
        dynamic_cast<FunctionType const *>(memberAcc.annotation().type);
    assert(fnTy && fnTy->kind() == FunctionType::Kind::External);
    mlir::Value contrAddr = genRValExpr(
        memberAcc.expression(),
        mlir::sol::AddressType::get(b.getContext(), /*payable=*/false));
    uint32_t selector = fnTy->externalIdentifier().convert_to<uint32_t>();

    // Create ExtFuncConstantOp
    mlir::Type resTy = getType(memberAcc.annotation().type);
    return b.create<mlir::sol::ExtFuncConstantOp>(
        loc, resTy, contrAddr, b.getI32IntegerAttr(selector));
  }

  case Type::Category::Array:
    if (memberName == "length") {
      // via-IR reads just the size (extcodesize) here, without materializing
      // the code bytes in memory.
      if (auto const *codeAcc =
              dynamic_cast<MemberAccess const *>(&memberAcc.expression());
          codeAcc && codeAcc->memberName() == "code" &&
          codeAcc->expression().annotation().type->category() ==
              Type::Category::Address) {
        auto nonPayableAddrTy =
            mlir::sol::AddressType::get(b.getContext(), /*payable=*/false);
        return b.create<mlir::sol::CodeSizeOp>(
            loc, genRValExpr(codeAcc->expression(), nonPayableAddrTy));
      }
      return b.create<mlir::sol::LengthOp>(loc,
                                           genRValExpr(memberAcc.expression()));
    }
    break;
  case Type::Category::FixedBytes: {
    auto const *fixedBytesTy = dynamic_cast<FixedBytesType const *>(memberAccTy);
    assert(fixedBytesTy);
    assert(memberName == "length" && "Illegal fixed bytes member");
    // The length is a compile-time constant, but the base expression must
    // still be evaluated for its side effects.
    (void)genRValExpr(memberAcc.expression());
    auto ui8Ty = b.getIntegerType(8, /*isSigned=*/false);
    auto len = b.create<mlir::sol::ConstantOp>(
        loc,
        b.getIntegerAttr(ui8Ty, llvm::APInt(8, fixedBytesTy->numBytes())));
    return genCast(len, getType(memberAcc.annotation().type));
  }
  case Type::Category::Struct: {
    const auto *structTy = dynamic_cast<StructType const *>(memberAccTy);
    auto memberIdx = genUnsignedConst(structTy->index(memberAcc.memberName()),
                                      /*numBits=*/64, loc);
    return b.create<mlir::sol::GepOp>(loc, genRValExpr(memberAcc.expression()),
                                      memberIdx);
  }
  case Type::Category::Function: {
    if (memberName == "selector") {
      auto const &fnTy = dynamic_cast<FunctionType const &>(*memberAccTy);
      // The selector helpers produce plain integers. Cast to the annotated
      // fixed-bytes type (bytes4, bytes32 for events) so consumers that do
      // not convert (e.g. mapping keys) see the correct representation.
      mlir::Type resTy = getType(memberAcc.annotation().type);
      if (mlir::Value selector =
              genCompileTimeFunctionSelector(memberAcc.expression(), fnTy, loc))
        return genCast(selector, resTy);
      if (fnTy.kind() == FunctionType::Kind::External)
        return genCast(
            genRuntimeFunctionSelector(memberAcc.expression(), fnTy, loc),
            resTy);
    }
    if (memberName == "address") {
      Expression const *fnExpr = &memberAcc.expression();
      while (auto const *opts =
                 dynamic_cast<FunctionCallOptions const *>(fnExpr)) {
        // Call options are evaluated for their side effects even though the
        // address comes from the underlying reference.
        for (ASTPointer<Expression const> const &optExpr : opts->options())
          (void)genRValExpr(*optExpr);
        fnExpr = &opts->expression();
      }
      auto const &fnTy = dynamic_cast<FunctionType const &>(*memberAccTy);
      mlir::Type addrTy =
          mlir::sol::AddressType::get(b.getContext(), /*payable=*/false);
      return b.create<mlir::sol::ExtFuncAddrOp>(
          loc, addrTy, genRValExpr(*fnExpr, getType(&fnTy)));
    }
    break;
  }
  case Type::Category::TypeType: {
    auto const *typeTy = dynamic_cast<TypeType const *>(memberAccTy);
    assert(typeTy);
    Type const &actualType = *typeTy->actualType();

    // Enum member access: e.g. Status.Inactive
    if (auto const *enumTy = dynamic_cast<EnumType const *>(&actualType)) {
      unsigned ordinal = enumTy->memberValue(memberName);
      auto ui256Ty = b.getIntegerType(256, /*isSigned=*/false);
      auto ordinalConst = b.create<mlir::sol::ConstantOp>(
          loc, b.getIntegerAttr(ui256Ty, llvm::APInt(256, ordinal)));
      return genCast(ordinalConst, getType(memberAcc.annotation().type));
    }
    // UDVT wrap/unwrap standalone reference - no-op.
    if (dynamic_cast<UserDefinedValueType const *>(&actualType)) {
      assert(memberName == "wrap" || memberName == "unwrap");
      return {};
    }
    // Contract member where result is itself a TypeType (e.g.,
    // Library.EnumType, Library.StructType) - no-op.
    if (dynamic_cast<ContractType const *>(&actualType) &&
        dynamic_cast<TypeType const *>(memberAcc.annotation().type))
      return {};
    // State variable accessed via the contract type (e.g. `A.x`): constants
    // and, in the creation context, immutables. Resolves to the same
    // reference an unqualified identifier would.
    if (auto const *contractTy =
            dynamic_cast<ContractType const *>(&actualType)) {
      if (auto const *var = dynamic_cast<VariableDeclaration const *>(
              memberAcc.annotation().referencedDeclaration)) {
        assert(var->isStateVariable());
        return genLValRef(*var);
      }
      // Function referenced as a value via the contract/library type
      // (e.g. `Utils.sum` passed as a function pointer). Resolves like an
      // unqualified identifier reference would.
      if (auto const *fn = dynamic_cast<FunctionDefinition const *>(
              memberAcc.annotation().referencedDeclaration)) {
        auto const *refTy =
            dynamic_cast<FunctionType const *>(memberAcc.annotation().type);
        // Public library member: the library's link-time address plus the
        // selector. The callee lives in the separately deployed library.
        if (refTy && refTy->kind() == FunctionType::Kind::DelegateCall) {
          auto libAddr = b.create<mlir::sol::LibAddrOp>(
              loc, mlir::sol::AddressType::get(b.getContext(), false),
              contractTy->contractDefinition().fullyQualifiedName());
          uint32_t selector =
              refTy->externalIdentifier().convert_to<uint32_t>();
          return b.create<mlir::sol::ExtFuncConstantOp>(
              loc, getType(refTy), libAddr, b.getI32IntegerAttr(selector));
        }
        // External functions named through the contract type (kind
        // Declaration) are pure handles with no runtime representation.
        if (refTy && refTy->kind() == FunctionType::Kind::Declaration)
          return {};
        // `super.f` references the next override in the most-derived
        // contract's linearization, not the statically named declaration.
        if (*memberAcc.annotation().requiredLookup == VirtualLookup::Super) {
          assert(contractTy->isSuper());
          fn = &fn->resolveVirtual(
              *currContract,
              contractTy->contractDefinition().superContract(*currContract));
        }
        lowerFreeOrLibFuncIfAbsent(*fn);
        return b.create<mlir::sol::FuncConstantOp>(
            loc, getType(fn->type()), getMangledName(*fn));
      }
    }
    break;
  }
  default:
    break;
  case Type::Category::Address: {
    auto nonPayableAddrTy =
        mlir::sol::AddressType::get(b.getContext(), /*payable=*/false);
    if (memberName == "balance")
      return b.create<mlir::sol::BalanceOp>(
          loc, genRValExpr(memberAcc.expression(), nonPayableAddrTy));
    if (memberName == "code") {
      return b.create<mlir::sol::CodeOp>(
          loc, genRValExpr(memberAcc.expression(), nonPayableAddrTy));
    }
    if (memberName == "codehash") {
      return b.create<mlir::sol::CodeHashOp>(
          loc, genRValExpr(memberAcc.expression(), nonPayableAddrTy));
    }
    break;
  }
  case Type::Category::Module: {
    // Members accessed through a module alias (`import "..." as M`) resolve
    // like unqualified identifier references would.
    Declaration const *decl = memberAcc.annotation().referencedDeclaration;
    if (auto const *var = dynamic_cast<VariableDeclaration const *>(decl)) {
      assert(var->isConstant());
      return genLValRef(*var);
    }
    if (auto const *fn = dynamic_cast<FunctionDefinition const *>(decl)) {
      lowerFreeOrLibFuncIfAbsent(*fn);
      return b.create<mlir::sol::FuncConstantOp>(loc, getType(fn->type()),
                                                 getMangledName(*fn));
    }
    if (auto const *contr = dynamic_cast<ContractDefinition const *>(decl)) {
      if (contr->isLibrary())
        return b.create<mlir::sol::LibAddrOp>(
            loc, mlir::sol::AddressType::get(b.getContext(), false),
            contr->fullyQualifiedName());
    }
    // Type-valued members (e.g. `M.D` where D is a contract type) have no
    // runtime representation. The base is still evaluated for side effects.
    if (dynamic_cast<TypeType const *>(memberAcc.annotation().type)) {
      genDiscardedExpr(memberAcc.expression());
      return {};
    }
    break;
  }
  }

  // An isolated (uncalled) reference to a builtin member function (e.g.
  // `data.pop;`, `payable(this).transfer;`) yields no value. Only the base
  // expression is evaluated, matching the legacy codegen.
  if (auto const *fnTy =
          dynamic_cast<FunctionType const *>(memberAcc.annotation().type)) {
    switch (fnTy->kind()) {
    case FunctionType::Kind::ArrayPush:
    case FunctionType::Kind::ArrayPop:
    case FunctionType::Kind::Transfer:
    case FunctionType::Kind::Send:
    case FunctionType::Kind::BareCall:
    case FunctionType::Kind::BareCallCode:
    case FunctionType::Kind::BareDelegateCall:
    case FunctionType::Kind::BareStaticCall:
      (void)genRValExpr(memberAcc.expression());
      return {};
    // Builtins on magic bases (`abi.encode`, `bytes.concat`): the base has
    // no runtime representation and no side effects to evaluate.
    case FunctionType::Kind::ABIEncode:
    case FunctionType::Kind::ABIEncodePacked:
    case FunctionType::Kind::ABIEncodeWithSelector:
    case FunctionType::Kind::ABIEncodeWithSignature:
    case FunctionType::Kind::ABIDecode:
    case FunctionType::Kind::BytesConcat:
    case FunctionType::Kind::StringConcat:
      return {};
    default:
      break;
    }
  }

  llvm_unreachable("NYI");
}

bool SolidityToMLIRPass::isDirectLibraryMemberCallBase(
    MemberAccess const &memAcc, FunctionType const &calleeTy) {
  auto const *typeTy =
      dynamic_cast<TypeType const *>(memAcc.expression().annotation().type);
  return typeTy && dynamic_cast<ContractType const *>(typeTy->actualType()) &&
         calleeTy.kind() == FunctionType::Kind::DelegateCall;
}

SolidityToMLIRPass::LowLevelCallInfo
SolidityToMLIRPass::parseLowLevelCallInfo(FunctionCall const &call,
                                          FunctionType const &calleeTy) {
  LowLevelCallInfo info{};
  info.callExpr = &call.expression();
  if (const auto *fnCallOpt =
          dynamic_cast<FunctionCallOptions const *>(info.callExpr)) {
    for (const auto &[namePtr, exprPtr] :
         llvm::zip(fnCallOpt->names(), fnCallOpt->options())) {
      ASTString const &name = *namePtr;
      Expression const &optExpr = *exprPtr;
      mlir::Value loweredExpr =
          genRValExpr(optExpr, b.getIntegerType(256, /*isSigned=*/false));
      if (name == "gas")
        info.gas = loweredExpr;
      else if (name == "value")
        info.value = loweredExpr;
    }
    info.callExpr = &fnCallOpt->expression();
  }
  info.memberAcc = dynamic_cast<MemberAccess const *>(
      resolveOuterUnaryTuples(info.callExpr));

  // memberAcc is null for indirect calls via function pointers.
  if (info.memberAcc &&
      (info.memberAcc->expression().annotation().type->category() ==
           Type::Category::Contract ||
       info.memberAcc->expression().annotation().type->category() ==
           Type::Category::Address ||
       isDirectLibraryMemberCallBase(*info.memberAcc, calleeTy))) {
    // Canonicalize the call base to non-payable address at this boundary so
    // init MLIR reflects the FE cast and lowering keeps the usual low-160-bit
    // address normalization semantics before the low-level call.
    info.addr = genRValExpr(
        info.memberAcc->expression(),
        mlir::sol::AddressType::get(b.getContext(), /*payable=*/false));
  }
  return info;
}

mlir::Value SolidityToMLIRPass::materializeCallGas(
    mlir::Value gas, u256 const &gasNeededByCaller, mlir::Location loc) {
  if (gas)
    return gas;

  gas = b.create<mlir::sol::GasLeftOp>(loc);
  if (!evmVersion.canOverchargeGasForCall()) {
    mlir::Type ui256Ty = b.getIntegerType(256, /*isSigned=*/false);
    mlir::Value gasNeeded = b.create<mlir::sol::ConstantOp>(
        loc, b.getIntegerAttr(ui256Ty, getAPInt(gasNeededByCaller, 256)));
    gas = b.create<mlir::sol::SubOp>(loc, gas, gasNeeded);
  }

  return gas;
}

SolidityToMLIRPass::ExternalCallResult
SolidityToMLIRPass::genExternalCall(FunctionCall const &call) {
  mlir::Location loc = getLoc(call);
  auto const *calleeTy =
      dynamic_cast<FunctionType const *>(call.expression().annotation().type);
  assert(calleeTy && (calleeTy->kind() == FunctionType::Kind::External ||
                      calleeTy->kind() == FunctionType::Kind::DelegateCall));

  auto callInfo = parseLowLevelCallInfo(call, *calleeTy);
  mlir::Value addr = callInfo.addr;
  mlir::Value gas = callInfo.gas;
  mlir::Value value = callInfo.value;
  MemberAccess const *memberAcc = callInfo.memberAcc;

  // Attached (using-for) public library calls are direct delegatecalls to
  // the library with the member-access base as the hidden first argument.
  FunctionDefinition const *boundLibFn = nullptr;
  if (memberAcc && calleeTy->hasBoundFirstArgument() &&
      calleeTy->kind() == FunctionType::Kind::DelegateCall) {
    auto const *fn = dynamic_cast<FunctionDefinition const *>(
        memberAcc->annotation().referencedDeclaration);
    if (fn && fn->libraryFunction())
      boundLibFn = fn;
  }

  // Lower the args.
  std::vector<ASTPointer<Expression const>> const &astArgs =
      call.sortedArguments();
  std::vector<mlir::Value> args;
  if (boundLibFn) {
    Expression const &self = memberAcc->expression();
    if (dynamic_cast<ReferenceType const *>(self.annotation().type))
      args.push_back(genRValExpr(self));
    else
      args.push_back(genRValExpr(self, getType(calleeTy->selfType())));
  }
  for (auto [arg, dstTy] : llvm::zip(astArgs, calleeTy->parameterTypes())) {
    // External-call ABI encoding can consume reference arguments directly
    // from their original data location (e.g. calldata or storage), so
    // don't cast them to memory.
    if (dynamic_cast<ReferenceType const *>(arg->annotation().type))
      args.push_back(genRValExpr(*arg));
    else
      args.push_back(genRValExpr(*arg, getType(dstTy)));
  }

  // Collect the return types; prepend i1 for the status flag.
  std::vector<mlir::Type> resTys;
  resTys.push_back(b.getI1Type());
  for (Type const *ty : calleeTy->returnParameterTypes()) {
    mlir::Type resTy = getType(ty);
    // Calldata-typed returns are decoded into fresh memory at the call
    // boundary; there is no calldata for them to point into.
    if (mlir::sol::isNonPtrRefType(resTy) &&
        mlir::sol::getDataLocation(resTy) == mlir::sol::DataLocation::CallData)
      resTy = toMemoryType(resTy);
    resTys.push_back(resTy);
  }

  bool isDirectContractMemberCall =
      memberAcc && memberAcc->expression().annotation().type->category() ==
                       Type::Category::Contract;
  bool isDirectLibraryMemberCall =
      memberAcc && isDirectLibraryMemberCallBase(*memberAcc, *calleeTy);

  mlir::Operation *callOp = nullptr;
  if (!isDirectContractMemberCall && !isDirectLibraryMemberCall &&
      !boundLibFn) {
    // Indirect call via external function pointer - use sol.ext_icall.
    mlir::Value fnPtr = genRValExpr(*callInfo.callExpr);
    bool isStatic = calleeTy->stateMutability() == StateMutability::Pure ||
                    calleeTy->stateMutability() == StateMutability::View;

    if (!gas)
      gas = b.create<mlir::sol::GasLeftOp>(loc);
    if (!value)
      value = genUnsignedConst(0, /*numBits=*/256, loc);

    callOp = b.create<mlir::sol::ExtICallOp>(
        loc, resTys, fnPtr, args, gas, value, isStatic,
        call.annotation().tryCall, mlir::ArrayAttr{}, mlir::ArrayAttr{});
  } else {
    // Direct call - resolve the callee symbol and selector.
    std::string calleeSymbol;
    uint32_t selectorVal = 0;
    bool isSupportedDirectCallee = getContractMemberExternalCalleeInfo(
        memberAcc->annotation().referencedDeclaration, calleeSymbol,
        selectorVal);
    assert(isSupportedDirectCallee &&
           "NYI: unsupported external contract member call target");
    mlir::Value selector = genUnsignedConst(selectorVal, /*numBits=*/256, loc);

    u256 gasNeededByCaller = evmasm::GasCosts::callGas(evmVersion) + 10;
    size_t encodedHeadSize = 0;
    for (Type const *ty : calleeTy->returnParameterTypes())
      encodedHeadSize += ty->decodingType()->calldataHeadSize();
    if (encodedHeadSize == 0 || !evmVersion.supportsReturndata())
      gasNeededByCaller += evmasm::GasCosts::callNewAccountGas;
    gas = materializeCallGas(gas, gasNeededByCaller, loc);

    if (!value)
      value = genUnsignedConst(0, /*numBits=*/256, loc);

    // The op's calleeType must cover the hidden first argument of attached
    // library calls, so use the declared function type there.
    mlir::FunctionType opCalleeTy;
    if (boundLibFn) {
      addr = b.create<mlir::sol::LibAddrOp>(
          loc, mlir::sol::AddressType::get(b.getContext(), false),
          dynamic_cast<ContractDefinition const *>(boundLibFn->scope())
              ->fullyQualifiedName());
      FunctionType declaredTy(*boundLibFn);
      opCalleeTy = mlir::cast<mlir::FunctionType>(
          getType(&declaredTy, /*indirectFn=*/false));
    } else {
      opCalleeTy = mlir::cast<mlir::FunctionType>(
          getType(calleeTy, /*indirectFn=*/false));
    }

    callOp = b.create<mlir::sol::ExtCallOp>(
        loc, resTys, calleeSymbol, args, addr, gas, value, selector,
        /*tryCall=*/call.annotation().tryCall,
        /*staticCall=*/calleeTy->stateMutability() <= StateMutability::View,
        /*delegateCall=*/calleeTy->kind() == FunctionType::Kind::DelegateCall,
        /*libraryCall=*/isDirectLibraryMemberCall || boundLibFn != nullptr,
        /*calleeType=*/opCalleeTy, mlir::ArrayAttr{}, mlir::ArrayAttr{});
  }

  ExternalCallResult out;
  out.status = callOp->getResult(0);
  for (mlir::Value val : llvm::drop_begin(callOp->getResults()))
    out.results.push_back(val);
  return out;
}

mlir::SmallVector<mlir::Value>
SolidityToMLIRPass::genExprs(FunctionCall const &call) {
  mlir::SmallVector<mlir::Value, 2> resVals;

  // Type conversion
  if (*call.annotation().kind == FunctionCallKind::TypeConversion) {
    resVals.push_back(genRValExpr(*call.arguments().front(),
                                  getType(call.annotation().type)));
    return resVals;
  }

  std::vector<ASTPointer<Expression const>> const &astArgs =
      call.sortedArguments();

  mlir::Location loc = getLoc(call);

  if (*call.annotation().kind == FunctionCallKind::StructConstructorCall) {
    auto const &typeTy =
        dynamic_cast<TypeType const &>(*call.expression().annotation().type);
    auto const &structTy =
        dynamic_cast<StructType const &>(*typeTy.actualType());
    auto calleeTy = structTy.constructorType();
    auto members = structTy.nativeMembers(nullptr);
    assert(members.size() == astArgs.size() && "Struct parameter mismatch");

    auto resultTy =
        mlir::cast<mlir::sol::StructType>(getType(call.annotation().type));
    mlir::Value structVal = b.create<mlir::sol::MallocOp>(
        loc, resultTy, /*zeroInit=*/false, /*size=*/mlir::Value{});

    for (size_t i = 0; i < astArgs.size(); ++i) {
      mlir::Value memberAddr = b.create<mlir::sol::GepOp>(
          loc, structVal, genUnsignedConst(i, /*numBits=*/64, loc));
      genAssign(
          memberAddr,
          genRValExpr(*astArgs[i], getType(calleeTy->parameterTypes()[i])),
          loc);
    }

    resVals.push_back(structVal);
    return resVals;
  }

  const auto *calleeTy =
      dynamic_cast<FunctionType const *>(call.expression().annotation().type);
  assert(calleeTy);

  switch (calleeTy->kind()) {
  case FunctionType::Kind::Wrap:
  case FunctionType::Kind::Unwrap:
    resVals.push_back(
        genRValExpr(*astArgs.front(), getType(call.annotation().type)));
    return resVals;

  // Internal call
  case FunctionType::Kind::Internal: {
    // Lower args.
    std::vector<mlir::Value> args;
    // Attached (using-for) calls pass the member-access base as the hidden
    // first argument, which bound function types exclude from
    // parameterTypes(). Like any other internal-call argument, it is
    // converted to the callee's declared self type — including its data
    // location (e.g. an attached call on a calldata array whose callee takes
    // memory requires a copy).
    if (calleeTy->hasBoundFirstArgument()) {
      auto const *memberAcc = dynamic_cast<MemberAccess const *>(
          resolveOuterUnaryTuples(&call.expression()));
      assert(memberAcc && "Expected a member access as the bound call base");
      args.push_back(genRValExpr(memberAcc->expression(),
                                 getType(calleeTy->selfType())));
    }
    for (auto [arg, dstTy] : llvm::zip(astArgs, calleeTy->parameterTypes())) {
      args.push_back(genRValExpr(*arg, getType(dstTy)));
    }

    // Collect return types.
    std::vector<mlir::Type> resTys;
    for (Type const *ty : calleeTy->returnParameterTypes()) {
      resTys.push_back(getType(ty));
    }

    // Get callee.
    FunctionDefinition const *callee = nullptr;
    if (currContract)
      callee = ASTNode::resolveFunctionCall(call, currContract);
    else
      callee = dynamic_cast<FunctionDefinition const *>(
          ASTNode::referencedDeclaration(call.expression()));

    // Generate the call op.
    mlir::CallOpInterface callOp;
    if (!callee)
      callOp = b.create<mlir::sol::ICallOp>(
          loc, resTys, genRValExpr(call.expression()), args, mlir::ArrayAttr{},
          mlir::ArrayAttr{});
    else
      callOp = b.create<mlir::sol::CallOp>(loc, getMangledName(*callee), resTys,
                                           args);
    for (mlir::Value val : callOp->getResults())
      resVals.push_back(val);

    if (!callee)
      return resVals;

    lowerFreeOrLibFuncIfAbsent(*callee);

    return resVals;
  }

  // External call. genExprs drops the status - it's an internal lowering
  // artifact for high-level external calls (consumed by SolToYul's implicit
  // revert-on-failure) and is not user-visible at the Solidity level.
  // `lower(TryStatement)` is the one frontend consumer that needs the status
  // and calls `genExternalCall` directly.
  case FunctionType::Kind::External:
  case FunctionType::Kind::DelegateCall: {
    auto [_, results] = genExternalCall(call);
    resVals.append(results.begin(), results.end());
    return resVals;
  }

  case FunctionType::Kind::BareCall:
  case FunctionType::Kind::BareDelegateCall:
  case FunctionType::Kind::BareStaticCall: {
    assert(astArgs.size() == 1);

    auto callInfo = parseLowLevelCallInfo(call, *calleeTy);
    mlir::Value addr = callInfo.addr;
    mlir::Value gas = callInfo.gas;
    mlir::Value value = callInfo.value;

    mlir::Value inp = genRValExpr(*astArgs.front(),
                                  getType(calleeTy->parameterTypes().front()));
    std::vector<mlir::Type> resTys;
    for (Type const *ty : calleeTy->returnParameterTypes())
      resTys.push_back(getType(ty));

    u256 gasNeededByCaller = evmasm::GasCosts::callGas(evmVersion) + 10;
    if (value)
      gasNeededByCaller += evmasm::GasCosts::callValueTransferGas;
    gasNeededByCaller += evmasm::GasCosts::callNewAccountGas;
    gas = materializeCallGas(gas, gasNeededByCaller, loc);

    mlir::Operation *callOp = nullptr;
    if (calleeTy->kind() == FunctionType::Kind::BareCall) {
      if (!value)
        value = genUnsignedConst(0, /*numBits=*/256, loc);
      callOp =
          b.create<mlir::sol::BareCallOp>(loc, resTys, addr, gas, value, inp);
    } else if (calleeTy->kind() == FunctionType::Kind::BareDelegateCall) {
      assert(!value && "Value set for delegatecall.");
      callOp =
          b.create<mlir::sol::BareDelegateCallOp>(loc, resTys, addr, gas, inp);
    } else {
      assert(calleeTy->kind() == FunctionType::Kind::BareStaticCall);
      assert(!value && "Value set for staticcall.");
      callOp =
          b.create<mlir::sol::BareStaticCallOp>(loc, resTys, addr, gas, inp);
    }
    assert(callOp);
    for (mlir::Value val : callOp->getResults())
      resVals.push_back(val);
    return resVals;
  }

  case FunctionType::Kind::Send:
  case FunctionType::Kind::Transfer: {
    assert(astArgs.size() == 1);

    const auto *memberAcc = dynamic_cast<MemberAccess const *>(
        resolveOuterUnaryTuples(&call.expression()));
    assert(memberAcc);

    mlir::Value addr = genRValExpr(
        memberAcc->expression(),
        mlir::sol::AddressType::get(b.getContext(), /*payable=*/false));
    mlir::Value value = genRValExpr(*astArgs.front(),
                                    b.getIntegerType(256, /*isSigned=*/false));

    if (calleeTy->kind() == FunctionType::Kind::Send) {
      resVals.push_back(b.create<mlir::sol::SendOp>(loc, addr, value));
    } else {
      b.create<mlir::sol::TransferOp>(loc, addr, value);
    }
    return resVals;
  }

  case FunctionType::Kind::Creation: {
    ContractDefinition const &cont =
        dynamic_cast<ContractType const &>(
            *calleeTy->returnParameterTypes().front())
            .contractDefinition();
    // FIXME: We assert that the creation object's name is contract op's name.
    std::string objName = getMangledName(cont);

    // Lower args.
    std::vector<mlir::Value> args;
    for (auto [arg, dstTy] : llvm::zip(astArgs, calleeTy->parameterTypes()))
      args.push_back(genRValExpr(*arg, getType(dstTy)));

    mlir::Value salt, value;
    mlir::Type ui256Ty = b.getIntegerType(256, /*isSigned=*/false);
    if (const auto *fnCallOpt =
            dynamic_cast<FunctionCallOptions const *>(&call.expression())) {
      for (const auto &[namePtr, exprPtr] :
           llvm::zip(fnCallOpt->names(), fnCallOpt->options())) {
        ASTString const &name = *namePtr;
        Expression const &expr = *exprPtr;
        if (name == "salt") {
          // The salt is a bytes32: lower it as such (string and hex literals
          // fold to a left-aligned fixed-bytes constant) and widen to the
          // word type expected by sol.new.
          mlir::Value saltVal =
              genRValExpr(expr, getType(TypeProvider::fixedBytes(32)));
          salt = genCast(saltVal, ui256Ty);
        } else if (name == "value") {
          value = genRValExpr(expr, ui256Ty);
        }
      }
    }
    if (!value)
      value = genUnsignedConst(0, /*numBits=*/256, loc);

    // Keep the creation result typed as the FE return contract type so
    // payability is preserved (contract vs contract<payable>).
    auto newOp = b.create<mlir::sol::NewOp>(
        loc, getType(calleeTy->returnParameterTypes().front()), objName, value,
        salt, args);
    if (call.annotation().tryCall)
      newOp.setTryCall(true);
    resVals.push_back(newOp.getResult());
    return resVals;
  }

  case FunctionType::Kind::ObjectCreation: {
    mlir::Type ty =
        getType(dynamic_cast<ArrayType const *>(call.annotation().type));
    assert(astArgs.size() == 1);
    resVals.push_back(b.create<mlir::sol::MallocOp>(
        loc, ty, /*zeroInit=*/true, genRValExpr(*astArgs.front())));
    return resVals;
  }

  // Event invocation
  case FunctionType::Kind::Event: {
    const auto &event =
        dynamic_cast<EventDefinition const &>(calleeTy->declaration());

    // Lower and track the indexed and non-indexed args. The indexed-arg
    // topic computation (cleanup for value types, keccak256-of-packed-encode
    // for reference types) is handled by EmitOpLowering.
    std::vector<mlir::Value> indexedArgs, nonIndexedArgs;
    for (size_t i = 0; i < event.parameters().size(); ++i) {
      // TODO? YulUtilFunctions::conversionFunction
      mlir::Value arg =
          genRValExpr(*astArgs[i], getType(calleeTy->parameterTypes()[i]));

      if (event.parameters()[i]->isIndexed()) {
        indexedArgs.push_back(arg);
      } else {
        nonIndexedArgs.push_back(arg);
      }
    }

    // Generate sol.emit (with signature for non-anonymous events).
    if (event.isAnonymous()) {
      b.create<mlir::sol::EmitOp>(loc, indexedArgs, nonIndexedArgs);
    } else {
      b.create<mlir::sol::EmitOp>(loc, indexedArgs, nonIndexedArgs,
                                  calleeTy->externalSignature());
    }

    return {};
  }

  // Assert statement
  case FunctionType::Kind::Assert: {
    b.create<mlir::sol::AssertOp>(loc, genRValExpr(*astArgs[0]));
    return {};
  }

  // Revert function call
  case FunctionType::Kind::Revert: {
    if (astArgs.empty()) {
      // revert()
      b.create<mlir::sol::RevertOp>(loc, mlir::ValueRange{},
                                    /*signature=*/mlir::StringAttr{});
    } else if (const auto *msg =
                   dynamic_cast<Literal const *>(astArgs[0].get())) {
      // revert("reason"). An empty reason is kept distinct from revert() as it
      // still ABI-encodes Error("").
      b.create<mlir::sol::RevertOp>(loc, mlir::ValueRange{},
                                    b.getStringAttr(msg->value()));
    } else {
      // revert(<string expression>): revert with the Error(string) encoding of
      // the runtime message. When stripped, the revert carries no data and the
      // message is only evaluated for its side effects (skipped for pure
      // expressions, as in the old codegen).
      if (revertStrings == RevertStrings::Strip) {
        if (!*astArgs[0]->annotation().isPure)
          (void)genRValExpr(*astArgs[0], getType(calleeTy->parameterTypes()[0]));
        b.create<mlir::sol::RevertOp>(loc, mlir::ValueRange{},
                                      /*signature=*/mlir::StringAttr{});
      } else {
        mlir::Value msgVal =
            genRValExpr(*astArgs[0], getType(calleeTy->parameterTypes()[0]));
        b.create<mlir::sol::RevertOp>(loc, mlir::ValueRange{msgVal},
                                      b.getStringAttr("Error(string)"),
                                      /*call=*/true);
      }
    }
    return {};
  }

  // Revert invocation
  case FunctionType::Kind::Error: {
    mlir::SmallVector<mlir::Value> args;
    for (auto [arg, dstTy] : llvm::zip(astArgs, calleeTy->parameterTypes()))
      args.push_back(genRValExpr(*arg, getType(dstTy)));
    b.create<mlir::sol::RevertOp>(
        loc, args, b.getStringAttr(calleeTy->externalSignature()),
        /*call=*/true);
    return {};
  }

  // Require statement
  case FunctionType::Kind::Require: {
    if (call.arguments().size() == 2) {
      const auto *msg = dynamic_cast<Literal const *>(astArgs[1].get());
      const auto *errorCall = dynamic_cast<FunctionCall const *>(astArgs[1].get());
      const auto *errorDef =
          errorCall ? dynamic_cast<ErrorDefinition const *>(
                          ASTNode::referencedDeclaration(errorCall->expression()))
                    : nullptr;
      mlir::Value cond = genRValExpr(*astArgs[0]);
      if (msg) {
        // require(cond, "message") form.

        b.create<mlir::sol::RequireOp>(loc, cond, b.getStringAttr(msg->value()),
                                       mlir::ValueRange{});
      } else if (errorDef) {
        // require(cond, Error(...)) form.

        mlir::SmallVector<mlir::Value> args;
        for (auto [callArg, argDef] :
             llvm::zip(errorCall->arguments(), errorDef->parameters()))
          args.push_back(genRValExpr(*callArg, getType(argDef->type())));

        b.create<mlir::sol::RequireOp>(
            loc, cond,
            b.getStringAttr(errorDef->functionType(true)->externalSignature()),
            args, /*errorCall=*/true);
      } else {
        // require(cond, <string expression>) form: revert with the
        // Error(string) encoding of the runtime message. The message is
        // evaluated unconditionally after the condition, matching the old
        // codegen's call-argument evaluation order. When stripped, the
        // revert carries no data and the message is only evaluated for its
        // side effects (skipped for pure expressions, as in the old codegen).
        if (revertStrings == RevertStrings::Strip) {
          if (!*astArgs[1]->annotation().isPure)
            (void)genRValExpr(*astArgs[1],
                              getType(calleeTy->parameterTypes()[1]));
          b.create<mlir::sol::RequireOp>(loc, cond, mlir::StringAttr{},
                                         mlir::ValueRange{});
        } else {
          mlir::Value msgVal =
              genRValExpr(*astArgs[1], getType(calleeTy->parameterTypes()[1]));
          b.create<mlir::sol::RequireOp>(loc, cond,
                                         b.getStringAttr("Error(string)"),
                                         mlir::ValueRange{msgVal},
                                         /*errorCall=*/true);
        }
      }
    } else {
      b.create<mlir::sol::RequireOp>(loc, genRValExpr(*astArgs[0]),
                                     mlir::StringAttr{}, mlir::ValueRange{});
    }
    return {};
  }

  // ABI encode
  case FunctionType::Kind::ABIEncode:
  case FunctionType::Kind::ABIEncodePacked: {
    mlir::SmallVector<mlir::Value, 4> args;
    for (const auto &arg : astArgs)
      args.push_back(genRValExpr(*arg));
    resVals.push_back(b.create<mlir::sol::EncodeOp>(
        loc, /*res=*/
        mlir::sol::StringType::get(b.getContext(),
                                   mlir::sol::DataLocation::Memory),
        args,
        /*selector=*/mlir::Value{},
        /*packed=*/calleeTy->kind() == FunctionType::Kind::ABIEncodePacked));
    return resVals;
  }

  // ABI encode with selector
  case FunctionType::Kind::ABIEncodeWithSelector: {
    assert(!astArgs.empty());

    mlir::Value selector = genRValExpr(
        *astArgs.front(), getType(calleeTy->parameterTypes().front()));

    mlir::SmallVector<mlir::Value, 4> args;
    for (const auto &arg : llvm::drop_begin(astArgs))
      args.push_back(genRValExpr(*arg));
    resVals.push_back(b.create<mlir::sol::EncodeOp>(
        loc, /*res=*/
        mlir::sol::StringType::get(b.getContext(),
                                   mlir::sol::DataLocation::Memory),
        args, selector,
        /*packed=*/false));
    return resVals;
  }

  // ABI encode call
  case FunctionType::Kind::ABIEncodeCall: {
    assert(astArgs.size() == 2);

    auto const *selectorType =
        dynamic_cast<FunctionType const *>(astArgs.front()->annotation().type);
    assert(selectorType);

    mlir::Value selectorBytes4 =
        genCompileTimeFunctionSelector(*astArgs.front(), *selectorType, loc,
                                       /*stateVarGetterOnly=*/true);
    if (selectorBytes4) {
      mlir::Type bytes4Ty =
          mlir::sol::FixedBytesType::get(b.getContext(), /*size=*/4);
      selectorBytes4 = genCast(selectorBytes4, bytes4Ty);
    } else {
      selectorBytes4 =
          genRuntimeFunctionSelector(*astArgs.front(), *selectorType, loc);
    }

    auto const *externalFunctionType =
        selectorType->asExternallyCallableFunction(false);
    assert(externalFunctionType);

    mlir::SmallVector<mlir::Value, 4> args;
    if (dynamic_cast<TupleType const *>(astArgs[1]->annotation().type)) {
      auto const *tupleExpr =
          dynamic_cast<TupleExpression const *>(astArgs[1].get());
      assert(tupleExpr);
      assert(tupleExpr->components().size() ==
             externalFunctionType->parameterTypes().size());
      for (auto [component, dstTy] :
           llvm::zip(tupleExpr->components(),
                     externalFunctionType->parameterTypes())) {
        assert(component);
        args.push_back(genRValExpr(*component, getType(dstTy)));
      }
    } else {
      // Handle cases where the second abi.encodeCall argument is a single
      // parameter (e.g. abi.encodeCall(f, x) for f(uint256)).
      assert(externalFunctionType->parameterTypes().size() == 1);
      args.push_back(
          genRValExpr(*astArgs[1],
                      getType(externalFunctionType->parameterTypes().front())));
    }

    resVals.push_back(b.create<mlir::sol::EncodeOp>(
        loc, /*res=*/
        mlir::sol::StringType::get(b.getContext(),
                                   mlir::sol::DataLocation::Memory),
        args, selectorBytes4,
        /*packed=*/false));
    return resVals;
  }

  // ABI encode with signature
  case FunctionType::Kind::ABIEncodeWithSignature: {
    assert(!astArgs.empty());

    mlir::Value selector;
    Type const *signatureTy = astArgs.front()->annotation().type;
    if (auto const *stringLitTy =
            dynamic_cast<StringLiteralType const *>(signatureTy)) {
      // Materialize the compile-time selector directly.
      mlir::Type i32Ty = b.getIntegerType(32, /*isSigned=*/false);
      selector = b.create<mlir::sol::ConstantOp>(
          loc, b.getIntegerAttr(i32Ty, util::selectorFromSignatureU32(
                                           stringLitTy->value())));
    } else {
      // Runtime signature: keccak256(signature).
      mlir::Type bytes32Ty =
          mlir::sol::FixedBytesType::get(b.getContext(), /*size=*/32);
      mlir::Value signature = genRValExpr(*astArgs.front());
      mlir::sol::DataLocation signatureDataLoc =
          mlir::sol::getDataLocation(signature.getType());
      if (signatureDataLoc == mlir::sol::DataLocation::Storage ||
          signatureDataLoc == mlir::sol::DataLocation::CallData) {
        mlir::Type memStringTy = mlir::sol::StringType::get(
            b.getContext(), mlir::sol::DataLocation::Memory);
        // keccak256 expects a memory string, so copy runtime signatures there.
        signature = genCast(signature, memStringTy);
      }
      selector = b.create<mlir::sol::Keccak256Op>(loc, bytes32Ty, signature);
    }

    mlir::Type bytes4Ty =
        mlir::sol::FixedBytesType::get(b.getContext(), /*size=*/4);
    selector = genCast(selector, bytes4Ty);

    mlir::SmallVector<mlir::Value, 4> args;
    for (const auto &arg : llvm::drop_begin(astArgs))
      args.push_back(genRValExpr(*arg));

    resVals.push_back(b.create<mlir::sol::EncodeOp>(
        loc, /*res=*/
        mlir::sol::StringType::get(b.getContext(),
                                   mlir::sol::DataLocation::Memory),
        args, selector,
        /*packed=*/false));
    return resVals;
  }

  // ABI decode
  case FunctionType::Kind::ABIDecode: {
    TypePointers astTys;
    if (TupleType const *tupleTy =
            dynamic_cast<TupleType const *>(call.annotation().type))
      astTys = tupleTy->components();
    else
      astTys = TypePointers{call.annotation().type};
    mlir::SmallVector<mlir::Type, 4> resTys;
    for (const Type *astTy : astTys)
      resTys.push_back(getType(astTy));

    mlir::Value decodeArg = genRValExpr(*astArgs[0]);
    auto strTy = mlir::dyn_cast<mlir::sol::StringType>(decodeArg.getType());

    // If the argument is a storage string, copy it to memory and then do the
    // decoding.
    if (strTy && strTy.getDataLocation() == mlir::sol::DataLocation::Storage) {
      mlir::Type memStringTy = mlir::sol::StringType::get(
          b.getContext(), mlir::sol::DataLocation::Memory);
      decodeArg = genCast(decodeArg, memStringTy);
    }

    auto decodeOp = b.create<mlir::sol::DecodeOp>(loc, resTys, decodeArg);
    for (mlir::Value res : decodeOp.getResults())
      resVals.push_back(res);
    return resVals;
  }

  case FunctionType::Kind::ArrayPush:
  case FunctionType::Kind::ArrayPop: {
    const auto *memberAcc = dynamic_cast<MemberAccess const *>(
        resolveOuterUnaryTuples(&call.expression()));
    solAssert(memberAcc);

    // Lower `pop`
    if (calleeTy->kind() == FunctionType::Kind::ArrayPop) {
      b.create<mlir::sol::PopOp>(loc, genRValExpr(memberAcc->expression()));
      return resVals;
    }

    // The `bytes` type requires special handling, as pushing elements can
    // trigger a layout transition from short (packed) to long (unpacked)
    // encoding. The no-arg case (str.push() = val) falls through to PushOp
    // below, which handles it at the cost of extra storage accesses.
    const auto *arrTy = dynamic_cast<ArrayType const *>(
        memberAcc->expression().annotation().type);
    assert(arrTy);
    if (arrTy->isByteArrayOrString() && !astArgs.empty()) {
      // Handle:
      //   str.push(0x41);
      //
      // Cast the argument to the parameter type (bytes1) so that string
      // literals like "f" are folded to a single byte before the op is built.
      b.create<mlir::sol::PushStringOp>(
          loc, genRValExpr(memberAcc->expression()),
          genRValExpr(*astArgs[0], getType(calleeTy->parameterTypes()[0])));
      return resVals;
    }

    // Lower `push`
    auto newAddr =
        b.create<mlir::sol::PushOp>(loc, genRValExpr(memberAcc->expression()));
    if (!astArgs.empty())
      genAssign(newAddr, genRValExpr(*astArgs[0]), loc);
    resVals.push_back(newAddr);
    return resVals;
  }

  case FunctionType::Kind::BytesConcat:
  case FunctionType::Kind::StringConcat: {
    llvm::SmallVector<mlir::Value, 4> args;
    for (const auto &arg : astArgs)
      args.push_back(genRValExpr(*arg));

    resVals.push_back(b.create<mlir::sol::ConcatOp>(
        loc, getType(calleeTy->returnParameterTypes()[0]), args));
    return resVals;
  }

  case FunctionType::Kind::AddMod:
  case FunctionType::Kind::MulMod: {
    std::vector<mlir::Value> args;
    for (auto [arg, dstTy] : llvm::zip(astArgs, calleeTy->parameterTypes()))
      args.push_back(genRValExpr(*arg, getType(dstTy)));

    if (calleeTy->kind() == FunctionType::Kind::AddMod)
      resVals.push_back(b.create<mlir::sol::AddModOp>(loc, args));
    else
      resVals.push_back(b.create<mlir::sol::MulModOp>(loc, args));

    return resVals;
  }

  case FunctionType::Kind::KECCAK256:
  case FunctionType::Kind::SHA256:
  case FunctionType::Kind::ECRecover:
  case FunctionType::Kind::RIPEMD160: {
    BuilderExt bExt(b);
    std::vector<mlir::Type> resTys;
    for (Type const *ty : calleeTy->returnParameterTypes())
      resTys.push_back(getType(ty));

    std::vector<mlir::Value> args;
    for (auto [arg, dstTy] : llvm::zip(astArgs, calleeTy->parameterTypes()))
      args.push_back(genRValExpr(*arg, getType(dstTy)));

    if (calleeTy->kind() == FunctionType::Kind::KECCAK256)
      resVals.push_back(b.create<mlir::sol::Keccak256Op>(loc, resTys, args));
    else if (calleeTy->kind() == FunctionType::Kind::SHA256)
      resVals.push_back(b.create<mlir::sol::Sha256Op>(loc, resTys, args));
    else if (calleeTy->kind() == FunctionType::Kind::RIPEMD160)
      resVals.push_back(b.create<mlir::sol::Ripemd160Op>(loc, resTys, args));
    else if (calleeTy->kind() == FunctionType::Kind::ECRecover)
      resVals.push_back(b.create<mlir::sol::EcrecoverOp>(loc, resTys, args));

    return resVals;
  }

  case FunctionType::Kind::GasLeft:
    resVals.push_back(b.create<mlir::sol::GasLeftOp>(loc));
    return resVals;

  case FunctionType::Kind::BlockHash: {
    mlir::Value arg =
        genRValExpr(*astArgs[0], getType(calleeTy->parameterTypes()[0]));
    resVals.push_back(b.create<mlir::sol::BlockHashOp>(
        loc, getType(calleeTy->returnParameterTypes()[0]), arg));
    return resVals;
  }

  case FunctionType::Kind::BlobHash: {
    mlir::Value arg =
        genRValExpr(*astArgs[0], getType(calleeTy->parameterTypes()[0]));
    resVals.push_back(b.create<mlir::sol::BlobHashOp>(
        loc, getType(calleeTy->returnParameterTypes()[0]), arg));
    return resVals;
  }

  case FunctionType::Kind::Selfdestruct: {
    mlir::Value arg =
        genRValExpr(*astArgs[0], getType(calleeTy->parameterTypes()[0]));
    b.create<mlir::sol::SelfdestructOp>(loc, arg);
    return resVals;
  }

  default:
    break;
  }

  llvm_unreachable("NYI");
}

mlir::SmallVector<mlir::Value>
SolidityToMLIRPass::genExprs(TupleExpression const &tuple) {
  mlir::SmallVector<mlir::Value, 2> vals;

  // Array literal
  if (tuple.isInlineArray()) {
    const auto *const arrTy =
        dynamic_cast<ArrayType const *>(tuple.annotation().type);
    for (const ASTPointer<Expression> &subExpr : tuple.components())
      vals.push_back(genRValExpr(*subExpr, getType(arrTy->baseType())));
    mlir::SmallVector<mlir::Value, 1> res;
    res.push_back(
        b.create<mlir::sol::ArrayLitOp>(getLoc(tuple), getType(arrTy), vals));
    return res;
  }

  for (const ASTPointer<Expression> &subExpr : tuple.components()) {
    // Wildcard component of a destructuring assignment: keep a null
    // placeholder so the flattened sides stay aligned.
    if (!subExpr) {
      vals.push_back({});
      continue;
    }
    // Nested tuples and multi-value components (calls, conditionals) are
    // flattened, matching the component-wise assignment semantics.
    llvm::append_range(vals, genLValExprs(*subExpr));
  }
  return vals;
}

void SolidityToMLIRPass::genAssign(mlir::Value lhs, mlir::Value rhs,
                                   mlir::Location loc) {
  // Aggregate copy into storage. Pointer LHS falls through so genCast can fold
  // a literal RHS to the element type (e.g. bytes.push() = "G").
  if (mlir::sol::isNonPtrRefType(lhs.getType()) &&
      mlir::sol::isNonPtrRefType(rhs.getType()) &&
      mlir::sol::getDataLocation(lhs.getType()) ==
          mlir::sol::DataLocation::Storage) {
    b.create<mlir::sol::CopyOp>(loc, rhs, lhs);
  } else {
    mlir::Value castedRhs = rhs;
    if (mlir::isa<mlir::sol::PointerType>(lhs.getType()))
      castedRhs = genCast(rhs, mlir::sol::getEltType(lhs.getType()));
    b.create<mlir::sol::StoreOp>(loc, castedRhs, lhs);
  }
}

/// Number of scalar leaves a type contributes to a flattened tuple.
static size_t tupleLeafCount(Type const *ty) {
  if (const auto *tupleTy = dynamic_cast<TupleType const *>(ty)) {
    size_t n = 0;
    for (Type const *comp : tupleTy->components())
      // A tuple type slot may be null for an omitted (wildcard) component.
      n += comp ? tupleLeafCount(comp) : 1;
    return n;
  }
  return 1;
}

/// Flattens the (possibly nested) left-hand side of an assignment into its
/// non-tuple components. Null entries denote wildcards of a destructuring
/// assignment. \p rhsTy is the type of the right-hand side paired with \p
/// expr: a wildcard's slot type is not recorded on the left-hand side (the
/// LHS type of `(a, ) = (4, (8, 16, 32))` is `tuple(int,)`), so the number of
/// leaves a wildcard absorbs is taken from the right-hand side instead.
static void
flattenLHSComponents(Expression const &expr, Type const *rhsTy,
                     mlir::SmallVectorImpl<Expression const *> &out) {
  const auto *tuple = dynamic_cast<TupleExpression const *>(&expr);
  if (!tuple || tuple->isInlineArray()) {
    out.push_back(&expr);
    return;
  }
  const auto *rhsTupleTy = dynamic_cast<TupleType const *>(rhsTy);
  bool haveRhsSlots =
      rhsTupleTy &&
      rhsTupleTy->components().size() == tuple->components().size();
  for (size_t i = 0; i < tuple->components().size(); ++i) {
    ASTPointer<Expression> const &comp = tuple->components()[i];
    Type const *compRhsTy = haveRhsSlots ? rhsTupleTy->components()[i] : nullptr;
    if (comp)
      flattenLHSComponents(*comp, compRhsTy, out);
    else
      out.append(compRhsTy ? tupleLeafCount(compRhsTy) : 1, nullptr);
  }
}

mlir::SmallVector<mlir::Value>
SolidityToMLIRPass::lower(Assignment const &asgnStmt) {
  mlir::Location loc = getLoc(asgnStmt);

  if (asgnStmt.assignmentOperator() == Token::Assign) {
    // The right-hand side is evaluated first (left to right); the left-hand
    // side components are then resolved and assigned right to left, matching
    // the old codegen (e.g. `(y, y, y) = (1, 2, 3)` leaves y == 1).
    mlir::SmallVector<mlir::Value> rhsVals =
        genRValExprs(asgnStmt.rightHandSide());
    mlir::SmallVector<Expression const *> lhsComps;
    flattenLHSComponents(asgnStmt.leftHandSide(),
                         asgnStmt.rightHandSide().annotation().type, lhsComps);
    assert(lhsComps.size() == rhsVals.size());

    // The left-hand side lvalue references are evaluated left to right before
    // any store happens (e.g. `(s[1], s) = (4, [0])` resolves s[1] against the
    // old array), and the stores are then performed right to left — the old
    // codegen's stack discipline. Wildcard components discard the
    // corresponding right-hand side value.
    mlir::SmallVector<mlir::Value> lhsVals(lhsComps.size());
    for (size_t i = 0; i < lhsComps.size(); ++i)
      if (lhsComps[i])
        lhsVals[i] = genLValExpr(*lhsComps[i]);
    for (size_t i = lhsComps.size(); i-- > 0;)
      if (lhsVals[i])
        genAssign(lhsVals[i], rhsVals[i], loc);
    return lhsVals;
  }

  // Compound assignment statement. The right-hand side is evaluated before
  // the left-hand side, as in the old codegen.
  mlir::Value rhs = genRValExpr(asgnStmt.rightHandSide());
  mlir::Value lhs = genLValExpr(asgnStmt.leftHandSide());
  mlir::Value lhsAsRVal = genRValExpr(lhs, loc);
  Token binOp =
      TokenTraits::AssignmentToBinaryOp(asgnStmt.assignmentOperator());
  b.create<mlir::sol::StoreOp>(
      loc, genBinExpr(binOp, lhsAsRVal, genCast(rhs, lhsAsRVal.getType()), loc),
      lhs);
  return {lhs};
}

mlir::Value SolidityToMLIRPass::genLValExpr(Expression const &expr) {
  // TODO: We should do a faster dispatch here. We could:
  // (a) Get frontend::ASTConstVisitor and ASTNode::accept to be able to
  // return mlir::Value(s). (b) Adopt llvm's rtti in the ast so that we can
  // switch over the enum that discriminates the derived ast's.

  // Literal
  if (const auto *lit = dynamic_cast<Literal const *>(&expr))
    return genExpr(*lit);

  // Elementary type names in expression position (e.g. a stray `uint256;`
  // statement or the callee of a cast) are pure compile-time handles.
  if (dynamic_cast<ElementaryTypeNameExpression const *>(&expr))
    return {};

  // Identifier
  if (const auto *ident = dynamic_cast<Identifier const *>(&expr))
    return genExpr(*ident);

  // Index access
  if (const auto *idxAcc = dynamic_cast<IndexAccess const *>(&expr))
    return genExpr(*idxAcc);

  // Index range access (array slice)
  if (const auto *idxRangeAcc = dynamic_cast<IndexRangeAccess const *>(&expr))
    return genExpr(*idxRangeAcc);

  // Member access
  if (const auto *memAcc = dynamic_cast<MemberAccess const *>(&expr))
    return genExpr(*memAcc);

  // (Compound) Assignment statement
  if (const auto *asgnStmt = dynamic_cast<Assignment const *>(&expr)) {
    mlir::SmallVector<mlir::Value> lhsVals = lower(*asgnStmt);
    // For chained scalar assignments (a = b = c), return b's address so
    // the outer assignment can load the just written value. Tuple assignments
    // cannot be chained through a scalar lvalue, so return {} in that case.
    if (lhsVals.size() == 1)
      return lhsVals.front();
    return {};
  }

  // Unary operation
  if (const auto *unaryOp = dynamic_cast<UnaryOperation const *>(&expr))
    return genExpr(*unaryOp);

  // Binary operation
  if (const auto *binOp = dynamic_cast<BinaryOperation const *>(&expr))
    return genExpr(*binOp);

  // Tuple
  if (const auto *tuple = dynamic_cast<TupleExpression const *>(&expr)) {
    mlir::SmallVector<mlir::Value, 1> res = genExprs(*tuple);
    assert(res.size() == 1);
    return res.front();
  }

  // Function call
  if (const auto *call = dynamic_cast<FunctionCall const *>(&expr)) {
    mlir::SmallVector<mlir::Value> exprs = genExprs(*call);
    assert(exprs.size() < 2);
    if (exprs.size() == 1)
      return exprs[0];
    return {};
  }

  // Conditional (ternary operator)
  if (const auto *cond = dynamic_cast<Conditional const *>(&expr)) {
    mlir::SmallVector<mlir::Value> exprs = genExprs(*cond);
    assert(exprs.size() == 1);
    return exprs[0];
  }

  llvm_unreachable("NYI");
}

mlir::SmallVector<mlir::Value>
SolidityToMLIRPass::genLValExprs(Expression const &expr) {
  // Tuple
  if (const auto *tuple = dynamic_cast<TupleExpression const *>(&expr))
    return genExprs(*tuple);

  // Function call
  if (const auto *call = dynamic_cast<FunctionCall const *>(&expr))
    return genExprs(*call);

  // Conditional (ternary)
  if (const auto *cond = dynamic_cast<Conditional const *>(&expr))
    return genExprs(*cond);

  mlir::SmallVector<mlir::Value, 1> vals;
  vals.push_back(genLValExpr(expr));
  return vals;
}

mlir::Value SolidityToMLIRPass::genRValExpr(mlir::Value val, mlir::Location loc,
                                            std::optional<mlir::Type> resTy) {
  if (mlir::isa<mlir::sol::PointerType>(val.getType()))
    val = b.create<mlir::sol::LoadOp>(loc, val);
  if (resTy)
    return genCast(val, *resTy);
  return val;
}

mlir::Value SolidityToMLIRPass::genRValExpr(Expression const &expr,
                                            std::optional<mlir::Type> resTy) {
  mlir::Value lVal = genLValExpr(expr);
  assert(lVal);
  return genRValExpr(lVal, getLoc(expr), resTy);
}

mlir::SmallVector<mlir::Value>
SolidityToMLIRPass::genRValExprs(Expression const &expr,
                                 mlir::TypeRange resTys) {
  mlir::SmallVector<mlir::Value> lVals = genLValExprs(expr);
  assert(!lVals.empty());
  assert(resTys.empty() || lVals.size() == resTys.size());

  mlir::SmallVector<mlir::Value, 2> rVals;
  if (resTys.empty()) {
    for (mlir::Value lVal : lVals)
      rVals.push_back(lVal ? genRValExpr(lVal, getLoc(expr)) : mlir::Value());
  } else {
    for (auto [lVal, resTy] : llvm::zip(lVals, resTys))
      rVals.push_back(lVal ? genRValExpr(lVal, getLoc(expr), resTy)
                           : mlir::Value());
  }

  return rVals;
}

static bool needsDiscardedLoad(Expression const &expr) {
  return dynamic_cast<Identifier const *>(&expr) ||
         dynamic_cast<IndexAccess const *>(&expr) ||
         dynamic_cast<MemberAccess const *>(&expr);
}

void SolidityToMLIRPass::genDiscardedExpr(Expression const &expr) {
  // Tuple statements discard each component individually so that the
  // load-side semantics below apply per component.
  if (const auto *tuple = dynamic_cast<TupleExpression const *>(&expr)) {
    if (!tuple->isInlineArray()) {
      for (ASTPointer<Expression> const &comp : tuple->components())
        if (comp)
          genDiscardedExpr(*comp);
      return;
    }
  }

  // Multi-value expressions (e.g. a discarded call returning several values)
  // are lowered for their side effects only.
  mlir::SmallVector<mlir::Value> vals = genLValExprs(expr);
  if (vals.size() != 1)
    return;
  mlir::Value val = vals.front();
  if (val && mlir::isa<mlir::sol::PointerType>(val.getType()) &&
      needsDiscardedLoad(expr))
    // Discarded lvalue expressions still need a load so calldata validation
    // and other load-side cleanup semantics are preserved.
    (void)b.create<mlir::sol::LoadOp>(getLoc(expr), val);
}

void SolidityToMLIRPass::lower(ExpressionStatement const &exprStmt) {
  genDiscardedExpr(exprStmt.expression());
}

void SolidityToMLIRPass::lower(
    VariableDeclarationStatement const &varDeclStmt) {
  mlir::Location loc = getLoc(varDeclStmt);

  mlir::SmallVector<mlir::Value> initExprs(varDeclStmt.declarations().size());
  if (Expression const *initExpr = varDeclStmt.initialValue())
    initExprs = genRValExprs(*initExpr);

  for (auto [varDeclPtr, initExpr] :
       llvm::zip(varDeclStmt.declarations(), initExprs)) {
    if (!varDeclPtr)
      continue;

    mlir::Type varTy = getType(varDeclPtr->type(), /*indirectFn=*/true);
    mlir::Type allocTy = mlir::sol::PointerType::get(
        b.getContext(), varTy, mlir::sol::DataLocation::Stack);

    auto addr = b.create<mlir::sol::AllocaOp>(loc, allocTy);
    trackLocalVarAddr(*varDeclPtr, addr);
    if (initExpr)
      b.create<mlir::sol::StoreOp>(loc, genCast(initExpr, varTy), addr);
    else
      genDefaultVal(addr);
  }
}

void SolidityToMLIRPass::lower(EmitStatement const &emit) {
  genLValExprs(emit.eventCall());
}

void SolidityToMLIRPass::lower(RevertStatement const &rev) {
  genLValExprs(rev.errorCall());
}

void SolidityToMLIRPass::lower(Break const &brkStmt) {
  b.create<mlir::sol::BreakOp>(getLoc(brkStmt));
  mlir::Block *newBlock = b.getBlock()->splitBlock(b.getInsertionPoint());
  b.setInsertionPointToStart(newBlock);
}

void SolidityToMLIRPass::lower(Continue const &contStmt) {
  b.create<mlir::sol::ContinueOp>(getLoc(contStmt));
  mlir::Block *newBlock = b.getBlock()->splitBlock(b.getInsertionPoint());
  b.setInsertionPointToStart(newBlock);
}

void SolidityToMLIRPass::lower(PlaceholderStatement const &placeholder) {
  b.create<mlir::sol::PlaceholderOp>(getLoc(placeholder));
}

void SolidityToMLIRPass::lower(Return const &ret) {
  // annotation().function is null for a return inside a modifier, which is
  // necessarily expression-less.
  Expression const *astExpr = ret.expression();
  if (astExpr) {
    mlir::SmallVector<mlir::Type> fnResTys;
    for (ASTPointer<VariableDeclaration> const &retParam :
         ret.annotation().function->returnParameters())
      fnResTys.push_back(getType(retParam->type()));
    b.create<mlir::sol::ReturnOp>(getLoc(ret),
                                  genRValExprs(*astExpr, fnResTys));
  } else
    b.create<mlir::sol::ReturnOp>(getLoc(ret));
  b.setInsertionPointToStart(b.createBlock(b.getBlock()->getParent()));
}

void SolidityToMLIRPass::lower(IfStatement const &ifStmt) {
  mlir::Value cond = genRValExpr(ifStmt.condition());
  auto ifOp = b.create<mlir::sol::IfOp>(getLoc(ifStmt), cond);
  mlir::OpBuilder::InsertionGuard insertGuard(b);

  b.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
  lower(ifStmt.trueStatement());
  b.create<mlir::sol::YieldOp>(ifOp.getLoc());
  if (ifStmt.falseStatement()) {
    b.setInsertionPointToStart(&ifOp.getElseRegion().emplaceBlock());
    lower(*ifStmt.falseStatement());
    b.create<mlir::sol::YieldOp>(ifOp.getLoc());
  }
}

void SolidityToMLIRPass::lower(WhileStatement const &whileStmt) {
  mlir::sol::LoopOpInterface whileOp;
  if (whileStmt.isDoWhile())
    whileOp = b.create<mlir::sol::DoWhileOp>(getLoc(whileStmt));
  else
    whileOp = b.create<mlir::sol::WhileOp>(getLoc(whileStmt));
  mlir::OpBuilder::InsertionGuard insertGuard(b);

  // Lower condition.
  b.setInsertionPointToStart(&whileOp.getCond().emplaceBlock());
  mlir::Value cond = genRValExpr(whileStmt.condition());
  b.create<mlir::sol::ConditionOp>(getLoc(whileStmt.condition()), cond);

  // Lower body.
  b.setInsertionPointToStart(&whileOp.getBody().emplaceBlock());
  lower(whileStmt.body());
  b.create<mlir::sol::YieldOp>(whileOp.getLoc());
}

void SolidityToMLIRPass::lower(ForStatement const &forStmt) {
  BuilderExt bExt(b);

  // Lower init expression.
  if (forStmt.initializationExpression())
    lower(*forStmt.initializationExpression());

  auto forOp = b.create<mlir::sol::ForOp>(getLoc(forStmt));
  mlir::OpBuilder::InsertionGuard insertGuard(b);

  // Lower condition.
  b.setInsertionPointToStart(&forOp.getCond().emplaceBlock());
  mlir::Value cond = forStmt.condition() ? genRValExpr(*forStmt.condition())
                                         : bExt.genBool(true, forOp.getLoc());
  b.create<mlir::sol::ConditionOp>(cond.getLoc(), cond);

  // Lower body.
  b.setInsertionPointToStart(&forOp.getBody().emplaceBlock());
  lower(forStmt.body());
  b.create<mlir::sol::YieldOp>(forOp.getLoc());

  // Lower loop expression.
  b.setInsertionPointToStart(&forOp.getStep().emplaceBlock());
  if (forStmt.loopExpression()) {
    llvm::SaveAndRestore<bool> g(inUnchecked, true);
    lower(*forStmt.loopExpression());
  }
  b.create<mlir::sol::YieldOp>(forOp.getLoc());
}

void SolidityToMLIRPass::lower(TryStatement const &tryStmt) {
  mlir::Location loc = getLoc(tryStmt);

  auto const *callExpr =
      dynamic_cast<FunctionCall const *>(&tryStmt.externalCall());
  assert(callExpr && "Expected FunctionCall expression in TryStatement");
  auto const *calleeTy = dynamic_cast<FunctionType const *>(
      callExpr->expression().annotation().type);
  assert(calleeTy);

  mlir::Value status;
  mlir::SmallVector<mlir::Value> results;
  if (calleeTy->kind() == FunctionType::Kind::Creation) {
    // genExprs propagates tryCall onto the sol.new (suppressing the
    // forwarding-revert); status is `addr != 0`
    results.append(genExprs(*callExpr));
    assert(results.size() == 1 && "new returns exactly one contract value");
    auto addrTy =
        mlir::sol::AddressType::get(b.getContext(), /*payable=*/false);
    auto ui160Ty = b.getIntegerType(160, /*isSigned=*/false);
    mlir::Value addr =
        b.create<mlir::sol::AddressCastOp>(loc, addrTy, results.front());
    mlir::Value addrUi = b.create<mlir::sol::AddressCastOp>(loc, ui160Ty, addr);
    mlir::Value zero = genUnsignedConst(0, /*numBits=*/160, loc);
    status = b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::ne,
                                        addrUi, zero);
  } else {
    auto extResult = genExternalCall(*callExpr);
    status = extResult.status;
    results = std::move(extResult.results);
  }

  auto tryOp = b.create<mlir::sol::TryOp>(loc, status);

  // Lower success clause.
  if (TryCatchClause const *successClause = tryStmt.successClause()) {
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    b.setInsertionPointToStart(&tryOp.getSuccessRegion().emplaceBlock());

    // Bind success-clause parameters to the call's result values.
    if (successClause->parameters()) {
      assert(successClause->parameters()->parameters().size() ==
             results.size());
      for (auto &&[param, resultVal] :
           llvm::zip(successClause->parameters()->parameters(), results)) {
        mlir::Location loc = getLoc(*param);
        mlir::Type allocTy =
            mlir::sol::PointerType::get(b.getContext(), getType(param->type()),
                                        mlir::sol::DataLocation::Stack);
        auto addr = b.create<mlir::sol::AllocaOp>(loc, allocTy);
        trackLocalVarAddr(*param, addr);
        b.create<mlir::sol::StoreOp>(loc, resultVal, addr);
      }
    }

    lower(successClause->block());
    b.create<mlir::sol::YieldOp>(loc);
  }

  // Lower panic clause.
  if (TryCatchClause const *panicClause = tryStmt.panicClause()) {
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    mlir::Block *blk = &tryOp.getPanicRegion().emplaceBlock();
    b.setInsertionPointToStart(blk);

    // Add block argument for the error code which is expected to be replaced
    // by the error code from the external call by the sol.try lowering.
    assert(panicClause->parameters() &&
           panicClause->parameters()->parameters().size() == 1);
    auto ui256 = b.getIntegerType(256, /*isSigned=*/false);
    ASTPointer<VariableDeclaration> const &codeParam =
        panicClause->parameters()->parameters()[0];
    mlir::Location codeParamLoc = getLoc(*codeParam);
    mlir::BlockArgument codeParamBlkArg = blk->addArgument(ui256, codeParamLoc);

    mlir::Type allocTy = mlir::sol::PointerType::get(
        b.getContext(), ui256, mlir::sol::DataLocation::Stack);
    auto codeParamAddr = b.create<mlir::sol::AllocaOp>(codeParamLoc, allocTy);
    trackLocalVarAddr(*codeParam, codeParamAddr);
    b.create<mlir::sol::StoreOp>(loc, codeParamBlkArg, codeParamAddr);

    lower(panicClause->block());
    b.create<mlir::sol::YieldOp>(loc);
  }

  // Lower panic clause.
  if (TryCatchClause const *errorClause = tryStmt.errorClause()) {
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    mlir::Block *blk = &tryOp.getErrorRegion().emplaceBlock();
    b.setInsertionPointToStart(blk);

    // Add a block argument for the error message which is expected to be
    // replaced by the error message from the external call by the sol.try
    // lowering.
    assert(errorClause->parameters() &&
           errorClause->parameters()->parameters().size() == 1);
    auto memStrTy = mlir::sol::StringType::get(b.getContext(),
                                               mlir::sol::DataLocation::Memory);
    ASTPointer<VariableDeclaration> const &msgParam =
        errorClause->parameters()->parameters()[0];
    mlir::Location msgParamLoc = getLoc(*msgParam);
    mlir::BlockArgument msgParamBlkArg =
        blk->addArgument(memStrTy, msgParamLoc);

    mlir::Type allocTy = mlir::sol::PointerType::get(
        b.getContext(), memStrTy, mlir::sol::DataLocation::Stack);
    auto msgParamAddr = b.create<mlir::sol::AllocaOp>(msgParamLoc, allocTy);
    trackLocalVarAddr(*msgParam, msgParamAddr);
    b.create<mlir::sol::StoreOp>(loc, msgParamBlkArg, msgParamAddr);
    lower(errorClause->block());
    b.create<mlir::sol::YieldOp>(loc);
  }

  // Lower fallback clause.
  if (TryCatchClause const *fallbackClause = tryStmt.fallbackClause()) {
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    mlir::Block *blk = &tryOp.getFallbackRegion().emplaceBlock();
    b.setInsertionPointToStart(blk);

    // `catch (bytes memory data)` binds the raw revert data as a memory bytes
    // string. Bind the block argument to the declared parameter.
    if (fallbackClause->parameters()) {
      assert(fallbackClause->parameters()->parameters().size() == 1);
      auto memBytesTy = mlir::sol::StringType::get(
          b.getContext(), mlir::sol::DataLocation::Memory);
      ASTPointer<VariableDeclaration> const &dataParam =
          fallbackClause->parameters()->parameters()[0];
      mlir::Location dataLoc = getLoc(*dataParam);
      mlir::BlockArgument dataBlkArg = blk->addArgument(memBytesTy, dataLoc);
      mlir::Type allocTy = mlir::sol::PointerType::get(
          b.getContext(), memBytesTy, mlir::sol::DataLocation::Stack);
      auto dataAddr = b.create<mlir::sol::AllocaOp>(dataLoc, allocTy);
      trackLocalVarAddr(*dataParam, dataAddr);
      b.create<mlir::sol::StoreOp>(loc, dataBlkArg, dataAddr);
    }

    lower(fallbackClause->block());
    b.create<mlir::sol::YieldOp>(loc);
  }
}

void SolidityToMLIRPass::lower(InlineAssembly const &inAsm) {
  mlir::Location loc = getLoc(inAsm.location());
  mlir::Type i256Ty = b.getIntegerType(256);
  auto yulPtrTy = mlir::yul::PtrType::get(b.getContext());

  auto memorySafeAttr = inAsm.annotation().markedMemorySafe
                            ? mlir::UnitAttr::get(b.getContext())
                            : mlir::UnitAttr();
  auto inAsmOp = b.create<mlir::sol::InlineAsmOp>(loc, memorySafeAttr);
  mlir::OpBuilder::InsertionGuard guard(b);
  mlir::Block *body = b.createBlock(&inAsmOp.getBody());
  b.setInsertionPointToStart(body);

  std::function<mlir::Value(yul::Identifier const *)> externalRefResolver =
      [&](yul::Identifier const *id) -> mlir::Value {
    auto it = inAsm.annotation().externalReferences.find(id);
    if (it == inAsm.annotation().externalReferences.end())
      return {};

    auto const &info = it->second;
    auto const *decl =
        dynamic_cast<VariableDeclaration const *>(info.declaration);
    assert(decl);
    std::string const &suffix = info.suffix;

    if (decl->isConstant()) {
      mlir::Value cst = genRValExpr(*decl->value(), getType(decl->type()));
      return b.create<mlir::sol::YulValCastOp>(loc, i256Ty, cst);
    }

    if (decl->isStateVariable()) {
      auto sym =
          mlir::FlatSymbolRefAttr::get(b.getContext(), getMangledName(*decl));
      if (suffix == "slot")
        return b.create<mlir::sol::YulStateVarSlotOp>(loc, i256Ty, sym);
      if (suffix == "offset")
        return b.create<mlir::sol::YulStateVarOffsetOp>(loc, i256Ty, sym);
    }

    mlir::Value localAddr = getLocalVarAddr(*decl);

    if (suffix == "slot")
      return b.create<mlir::sol::YulStorageSlotOp>(loc, yulPtrTy, localAddr);

    if (suffix == "offset") {
      auto const *refTy = dynamic_cast<ReferenceType const *>(decl->type());
      if (refTy && refTy->location() == DataLocation::CallData)
        return b.create<mlir::sol::YulCallDataOffsetOp>(loc, yulPtrTy,
                                                        localAddr);
      return b.create<mlir::sol::YulStorageOffsetOp>(loc, i256Ty, localAddr);
    }

    if (suffix == "length")
      return b.create<mlir::sol::YulCallDataLengthOp>(loc, yulPtrTy, localAddr);

    if (suffix == "selector")
      return b.create<mlir::sol::YulSelectorOp>(loc, yulPtrTy, localAddr);
    if (suffix == "address")
      return b.create<mlir::sol::YulFuncAddrOp>(loc, yulPtrTy, localAddr);

    return b.create<mlir::sol::YulPtrCastOp>(loc, yulPtrTy, localAddr);
  };

  // TODO: YulToMLIRPass has an expensive ctor (Due to things like
  // populateBuiltinGenMap() etc.). Can we ctor once?
  runYulToMLIRPass(inAsm.operations(), *stream, externalRefResolver, b);
}

void SolidityToMLIRPass::lower(Statement const &stmt) {
  // Expression
  if (const auto *exprStmt = dynamic_cast<ExpressionStatement const *>(&stmt))
    lower(*exprStmt);

  // Variable declaration
  else if (const auto *varDeclStmt =
               dynamic_cast<VariableDeclarationStatement const *>(&stmt))
    lower(*varDeclStmt);

  // Emit
  else if (const auto *emitStmt = dynamic_cast<EmitStatement const *>(&stmt))
    lower(*emitStmt);

  // Revert
  else if (const auto *revStmt = dynamic_cast<RevertStatement const *>(&stmt))
    lower(*revStmt);

  // Placeholder
  else if (const auto *placeholderStmt =
               dynamic_cast<PlaceholderStatement const *>(&stmt))
    lower(*placeholderStmt);

  // Return
  else if (const auto *retStmt = dynamic_cast<Return const *>(&stmt))
    lower(*retStmt);

  // Break
  else if (const auto *brkStmt = dynamic_cast<Break const *>(&stmt))
    lower(*brkStmt);

  // Continue
  else if (const auto *contStmt = dynamic_cast<Continue const *>(&stmt))
    lower(*contStmt);

  // If-then-else
  else if (const auto *ifStmt = dynamic_cast<IfStatement const *>(&stmt))
    lower(*ifStmt);

  // While and do-while
  else if (const auto *whileStmt = dynamic_cast<WhileStatement const *>(&stmt))
    lower(*whileStmt);

  // For
  else if (const auto *forStmt = dynamic_cast<ForStatement const *>(&stmt))
    lower(*forStmt);

  // Try
  else if (const auto *tryStmt = dynamic_cast<TryStatement const *>(&stmt))
    lower(*tryStmt);

  // Inline assembly
  else if (const auto *inAsm = dynamic_cast<InlineAssembly const *>(&stmt))
    lower(*inAsm);

  // Block
  else if (const auto *blk = dynamic_cast<Block const *>(&stmt))
    lower(*blk);

  else
    llvm_unreachable("NYI");
}

void SolidityToMLIRPass::lower(Block const &blk) {
  // Unchecked-ness is lexical: it must survive into nested plain blocks
  // (the type checker forbids nested `unchecked`, so it can never unset).
  llvm::SaveAndRestore<bool> g(inUnchecked, inUnchecked || blk.unchecked());
  for (const ASTPointer<Statement> &stmt : blk.statements())
    lower(*stmt);
}

/// Returns the mlir::sol::StateMutability of the function
static mlir::sol::StateMutability
getStateMutability(FunctionDefinition const &fn) {
  switch (fn.stateMutability()) {
  case StateMutability::Pure:
    return mlir::sol::StateMutability::Pure;
  case StateMutability::View:
    return mlir::sol::StateMutability::View;
  case StateMutability::NonPayable:
    return mlir::sol::StateMutability::NonPayable;
  case StateMutability::Payable:
    return mlir::sol::StateMutability::Payable;
  }
}

/// Returns true if 'fn' is inherited into 'currContract' but overridden by a
/// more-derived implementation in that contract's inheritance tree.
static bool isOverriddenByFunctionInCurrentContract(
    FunctionDefinition const &fn, ContractDefinition const &currContract) {
  auto const *declaringContract =
      dynamic_cast<ContractDefinition const *>(fn.scope());
  if (!declaringContract || declaringContract == &currContract)
    return false;
  if (!fn.virtualSemantics() || fn.isConstructor() || fn.isFree() ||
      fn.libraryFunction() || !fn.isOrdinary() || fn.name().empty())
    return false;
  return &fn.resolveVirtual(currContract) != &fn;
}

/// Returns true if a public state variable declared in 'currContract' overrides
/// 'fn', implying that the getter owns the external selector.
static bool
isOverriddenByPublicStateVarGetter(FunctionDefinition const &fn,
                                   ContractDefinition const &currContract) {
  for (auto const *stateVar : currContract.stateVariables()) {
    if (!stateVar->isPartOfExternalInterface())
      continue;
    if (stateVar->annotation().baseFunctions.count(&fn))
      return true;
  }
  return false;
}

void SolidityToMLIRPass::lower(ModifierDefinition const &modifier) {
  std::vector<mlir::Type> inpTys;
  std::vector<mlir::Location> inpLocs;

  for (const auto &param : modifier.parameters()) {
    inpTys.push_back(getType(param->annotation().type));
    inpLocs.push_back(getLoc(*param));
  }
  auto funcType = b.getFunctionType(inpTys, {});
  auto op = b.create<mlir::sol::ModifierOp>(getLoc(modifier),
                                            getMangledName(modifier), funcType);

  mlir::Block *entryBlk = b.createBlock(&op.getRegion());
  b.setInsertionPointToStart(entryBlk);
  for (auto &&[inpTy, inpLoc, param] :
       ranges::views::zip(inpTys, inpLocs, modifier.parameters())) {
    mlir::Value arg = entryBlk->addArgument(inpTy, inpLoc);
    auto addr = b.create<mlir::sol::AllocaOp>(
        inpLoc, mlir::sol::PointerType::get(b.getContext(), inpTy,
                                            mlir::sol::DataLocation::Stack));
    trackLocalVarAddr(*param, addr);
    b.create<mlir::sol::StoreOp>(inpLoc, arg, addr);
  }

  lower(modifier.body());
  b.create<mlir::sol::ReturnOp>(getLoc(modifier));

  b.setInsertionPointAfter(op);
}

void SolidityToMLIRPass::genBaseCtorCall(ContractDefinition const &curCont,
                                         FunctionDefinition const &nextCtor,
                                         mlir::Location loc) {
  auto const &argMap = currContract->annotation().baseConstructorArguments;

  auto lowerArgsNode = [&](ASTNode const *argsNode,
                           FunctionDefinition const &target) {
    std::vector<ASTPointer<Expression>> const *args = nullptr;
    if (const auto *inheritanceSpec =
            dynamic_cast<InheritanceSpecifier const *>(argsNode))
      args = inheritanceSpec->arguments();
    else if (const auto *modifierInvoc =
                 dynamic_cast<ModifierInvocation const *>(argsNode))
      args = modifierInvoc->arguments();
    assert(args);
    mlir::SmallVector<mlir::Value> vals;
    for (auto [arg, param] : llvm::zip(*args, target.parameters()))
      vals.push_back(genRValExpr(*arg, getType(param->annotation().type)));
    return vals;
  };

  // Lower every argument list this contract provides, in chain order, so the
  // expressions reference constructor parameters local to this region.
  size_t curPos = ctorChainPos(curCont);
  for (size_t j = curPos + 1; j < ctorChain.size(); ++j) {
    FunctionDefinition const *target = ctorChain[j].second;
    auto provider = baseCtorArgProviders.find(target);
    if (provider == baseCtorArgProviders.end() ||
        provider->second != &curCont)
      continue;
    pendingBaseCtorArgs[target] = lowerArgsNode(argMap.at(target), *target);
  }

  // Argument values for the next constructor: threaded or just lowered
  // above. When the provider has no constructor of its own (e.g. a plain
  // `is A(1)` on a constructor-less contract), the expressions cannot
  // reference constructor parameters and are lowered right here.
  mlir::SmallVector<mlir::Value> callArgs;
  auto pendingFound = pendingBaseCtorArgs.find(&nextCtor);
  if (pendingFound != pendingBaseCtorArgs.end()) {
    callArgs = pendingFound->second;
  } else if (auto argsFound = argMap.find(&nextCtor);
             argsFound != argMap.end()) {
    callArgs = lowerArgsNode(argsFound->second, nextCtor);
  }

  // Forward the pending values for deeper constructors the callee expects.
  for (FunctionDefinition const *target :
       pendingBaseCtorTargets(curPos + 1)) {
    auto found = pendingBaseCtorArgs.find(target);
    assert(found != pendingBaseCtorArgs.end() &&
           "Pending base-constructor arguments not yet lowered");
    llvm::append_range(callArgs, found->second);
  }

  b.create<mlir::sol::CallOp>(loc, getMangledName(nextCtor),
                              /*resTys=*/mlir::TypeRange{}, callArgs);
}

mlir::sol::FuncOp SolidityToMLIRPass::lower(FunctionDefinition const &fn) {
  assert(fn.isImplemented());

  // Immutable references resolve differently in the creation context (lvalues
  // into the reserved immutables memory vs. sol.load_immutable rvalues), and
  // every constructor body executes in it. Base constructors are lowered
  // through the generic function path, so derive the flag from the function
  // itself instead of relying on the caller to set it.
  llvm::SaveAndRestore<bool> inCtorGuard(inCtor, fn.isConstructor());

  // Create the function type.
  std::vector<mlir::Type> inpTys, outTys;
  std::vector<mlir::Location> inpLocs;
  for (const auto &param : fn.parameters()) {
    inpTys.push_back(getType(param->annotation().type));
    inpLocs.push_back(getLoc(*param));
  }
  // Constructors additionally receive the pending base-constructor argument
  // values threaded down the chain from more-derived providers.
  mlir::SmallVector<FunctionDefinition const *> pendingTargets;
  if (fn.isConstructor() && !ctorChain.empty()) {
    auto const &ctorCont =
        dynamic_cast<ContractDefinition const &>(*fn.scope());
    pendingTargets = pendingBaseCtorTargets(ctorChainPos(ctorCont));
    for (FunctionDefinition const *target : pendingTargets)
      for (const auto &param : target->parameters()) {
        inpTys.push_back(getType(param->annotation().type));
        inpLocs.push_back(getLoc(*param));
      }
  }
  for (const auto &param : fn.returnParameters())
    outTys.push_back(getType(param->annotation().type));
  auto fnTy = b.getFunctionType(inpTys, outTys);

  // Generate sol.func.
  mlir::Location fnLoc = getLoc(fn);
  auto op = b.create<mlir::sol::FuncOp>(fnLoc, getMangledName(fn), fnTy,
                                        getStateMutability(fn));

  // Set id.
  op.setId(fn.id());

  if (fn.isPartOfExternalInterface()) {
    auto selectorIt = selectorMap.find(&fn);
    if (selectorIt != selectorMap.end()) {
      op.setSelectorAttr(b.getIntegerAttr(
          b.getIntegerType(32), mlir::APInt(32, selectorIt->second.hex(), 16)));
      op.setOrigFnType(fnTy);
    } else {
      assert(currContract && "function lowering outside a contract");
      solAssert(
          isOverriddenByFunctionInCurrentContract(fn, *currContract) ||
              isOverriddenByPublicStateVarGetter(fn, *currContract),
          "missing selector for a non-overridden external interface function");
    }
  }

  // Set function kind.
  if (fn.isReceive()) {
    op.setKind(mlir::sol::FunctionKind::Receive);
  } else if (fn.isFallback()) {
    op.setKind(mlir::sol::FunctionKind::Fallback);
  }

  mlir::Block *entryBlk = b.createBlock(&op.getRegion());
  b.setInsertionPointToStart(entryBlk);

  // Lower the args.
  for (auto &&[inpTy, inpLoc, param] :
       ranges::views::zip(inpTys, inpLocs, fn.parameters())) {
    mlir::Value arg = entryBlk->addArgument(inpTy, inpLoc);
    auto addr = b.create<mlir::sol::AllocaOp>(
        inpLoc, mlir::sol::PointerType::get(b.getContext(), inpTy,
                                            mlir::sol::DataLocation::Stack));
    trackLocalVarAddr(*param, addr);
    b.create<mlir::sol::StoreOp>(inpLoc, arg, addr);
  }

  // Bind the threaded pending base-constructor argument values.
  pendingBaseCtorArgs.clear();
  for (FunctionDefinition const *target : pendingTargets) {
    auto &vals = pendingBaseCtorArgs[target];
    for (const auto &param : target->parameters())
      vals.push_back(entryBlk->addArgument(getType(param->annotation().type),
                                           getLoc(*param)));
  }

  // Allocate and zero-initialize return parameters so they can be loaded at
  // implicit-return sites. Unnamed parameters get the same treatment: the old
  // codegen allocates (and for memory reference types, heap-allocates) every
  // return variable at function entry, which is observable through the free
  // memory pointer.
  for (const auto &param : fn.returnParameters()) {
    mlir::Location paramLoc = getLoc(*param);
    mlir::Type paramTy = getType(param->annotation().type);
    auto addr = b.create<mlir::sol::AllocaOp>(
        paramLoc, mlir::sol::PointerType::get(b.getContext(), paramTy,
                                              mlir::sol::DataLocation::Stack));
    trackLocalVarAddr(*param, addr);
    genDefaultVal(addr);
  }

  // Generate the call to the next ctor (if any) if `fn` is a ctor.
  if (fn.isConstructor()) {
    // Get base contract of `currContract`
    auto const &baseCont =
        dynamic_cast<ContractDefinition const &>(*fn.scope());

    if (FunctionDefinition const *nextCtor =
            baseCont.nextConstructor(*currContract))
      genBaseCtorCall(baseCont, *nextCtor, fnLoc);
  }

  // Lower modifier invocations. Argument expressions reference the param and
  // named-return allocas above, so this must follow them.
  for (const ASTPointer<ModifierInvocation> &modifier : fn.modifiers()) {
    Declaration const *refDecl =
        modifier->name().annotation().referencedDeclaration;
    ModifierDefinition const *modifierDef =
        dynamic_cast<ModifierDefinition const *>(refDecl);
    if (!modifierDef) {
      // Base constructor invocations are lowered via the next-ctor call.
      assert(dynamic_cast<ContractDefinition const *>(refDecl));
      continue;
    }
    if (*modifier->name().annotation().requiredLookup == VirtualLookup::Virtual)
      modifierDef = &modifierDef->resolveVirtual(*currContract);

    // A library function lowered on-demand into a non-library contract brings
    // its modifiers along: they are otherwise only lowered in the library's
    // own module.
    auto const *modifierContr =
        dynamic_cast<ContractDefinition const *>(modifierDef->scope());
    if (modifierContr && modifierContr->isLibrary() &&
        (!currContract || !currContract->isLibrary())) {
      auto *symTableOp = mlir::SymbolTable::getNearestSymbolTable(op);
      assert(symTableOp);
      if (!mlir::SymbolTable::lookupSymbolIn(symTableOp,
                                             getMangledName(*modifierDef))) {
        mlir::OpBuilder::InsertionGuard insertGuard(b);
        b.setInsertionPoint(op);
        lower(*modifierDef);
      }
    }

    mlir::Location loc = getLoc(*modifier);

    auto invocation = b.create<mlir::sol::ModifierInvocationOp>(
        loc, mlir::FlatSymbolRefAttr::get(b.getContext(),
                                          getMangledName(*modifierDef)));
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    b.setInsertionPointToStart(&invocation.getArgsRegion().front());

    std::vector<mlir::Value> loweredArgs;
    if (modifier->arguments()) {
      loweredArgs.reserve(modifier->arguments()->size());
      unsigned i = 0;
      for (const ASTPointer<Expression> &arg : *modifier->arguments()) {
        mlir::Type reqTy = getType(modifierDef->parameters()[i++]->type());
        loweredArgs.push_back(genRValExpr(*arg, reqTy));
      }
    }
    b.create<mlir::sol::YieldOp>(loc, loweredArgs);
  }

  // Lower the body.
  lower(fn.body());

  mlir::Block *currBlk = b.getBlock();
  assert(currBlk && "insertion point lost after lowering function body");
  if (!currBlk->empty() &&
      currBlk->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    b.setInsertionPointAfter(op);
    return op;
  }

  // A return statement lowers into a new trailing block for post-return
  // code. If nothing follows the return, that block is empty and can be
  // dropped.
  if (currBlk->empty() && op.getBody().getBlocks().size() > 1) {
    op.getBody().back().erase();
    b.setInsertionPointAfter(op);
    return op;
  }

  // Handle void function.
  if (fn.returnParameters().empty()) {
    b.create<mlir::sol::ReturnOp>(fnLoc);
    b.setInsertionPointAfter(op);
    return op;
  }

  mlir::SmallVector<mlir::Value> retVals;
  // Load all return params (named and unnamed) from the zero-initialized
  // local variables allocated at function entry.
  for (const auto &param : fn.returnParameters())
    retVals.push_back(
        b.create<mlir::sol::LoadOp>(fnLoc, getLocalVarAddr(*param)));
  b.create<mlir::sol::ReturnOp>(fnLoc, retVals);

  b.setInsertionPointAfter(op);
  return op;
}

/// Returns the mlir::sol::ContractKind of the contract
static mlir::sol::ContractKind getContractKind(ContractDefinition const &cont) {
  switch (cont.contractKind()) {
  case ContractKind::Interface:
    return mlir::sol::ContractKind::Interface;
  case ContractKind::Contract:
    return mlir::sol::ContractKind::Contract;
  case ContractKind::Library:
    return mlir::sol::ContractKind::Library;
  }
}

void SolidityToMLIRPass::lower(ContractDefinition const &cont) {
  currContract = &cont;
  mlir::Location loc = getLoc(cont);

  // This function works on the full inheritance tree from `cont` but we only
  // generate the sol.contract op for `cont`.

  // Track selectors of interface functions.
  const auto &interfaceFnInfos = cont.interfaceFunctions();
  for (const auto &i : interfaceFnInfos)
    selectorMap[&i.second->declaration()] = i.first;

  // Build the constructor chain and the argument-list providers.
  ctorChain.clear();
  baseCtorArgProviders.clear();
  if (!cont.isLibrary()) {
    ctorChain.emplace_back(&cont, cont.constructor());
    ContractDefinition const *chainCont = &cont;
    while (FunctionDefinition const *next = chainCont->nextConstructor(cont)) {
      chainCont = dynamic_cast<ContractDefinition const *>(next->scope());
      assert(chainCont);
      ctorChain.emplace_back(chainCont, next);
    }
    for (auto const &[ctor, argsNode] :
         cont.annotation().baseConstructorArguments) {
      for (ContractDefinition const *q :
           cont.annotation().linearizedBaseContracts) {
        bool provides = false;
        for (ASTPointer<InheritanceSpecifier> const &spec : q->baseContracts())
          if (spec.get() == argsNode)
            provides = true;
        if (q->constructor())
          for (ASTPointer<ModifierInvocation> const &mi :
               q->constructor()->modifiers())
            if (mi.get() == argsNode)
              provides = true;
        if (provides) {
          baseCtorArgProviders[ctor] = q;
          break;
        }
      }
    }
  }

  // Create the contract op.
  mlir::sol::ContractOp contOp = b.create<mlir::sol::ContractOp>(
      loc, getMangledName(cont), getContractKind(cont));
  b.setInsertionPointToStart(&contOp.getBodyRegion().emplaceBlock());

  // Build a map from state variable to {slot, byteOffset}.
  llvm::DenseMap<VariableDeclaration const *, std::pair<u256, unsigned>>
      stateVarSlots;
  for (auto dataLoc : {DataLocation::Storage, DataLocation::Transient})
    for (auto &[var, slot, byteOffset] :
         ContractType(cont).linearizedStateVariables(dataLoc))
      stateVarSlots[var] = {slot, byteOffset};

  // Lower immutables and state variables; Generate getters.
  for (ContractDefinition const *baseCont :
       cont.annotation().linearizedBaseContracts) {
    for (VariableDeclaration const *stateVar : baseCont->stateVariables()) {
      if (stateVar->immutable())
        b.create<mlir::sol::ImmutableOp>(getLoc(*stateVar),
                                         getMangledName(*stateVar),
                                         getType(stateVar->type()));
      else if (!stateVar->isConstant()) {
        auto [slot, byteOffset] = stateVarSlots[stateVar];
        bool isTransient = stateVar->referenceLocation() ==
                           VariableDeclaration::Location::Transient;
        b.create<mlir::sol::StateVarOp>(
            getLoc(*stateVar), getMangledName(*stateVar),
            getType(stateVar->type()), mlirgen::getAPInt(slot, 256), byteOffset,
            isTransient);
      }

      if (stateVar->isPartOfExternalInterface())
        genGetter(*stateVar);
    }
  }

  // Lower/generate ctor if not library. Note that lower() of functions
  // generates the call to the next ctor.
  if (!cont.isLibrary()) {
    mlir::sol::FuncOp ctorFn;
    inCtor = true;
    if (FunctionDefinition const *ctor = cont.constructor()) {
      ctorFn = lower(*ctor);
    } else {
      mlir::OpBuilder::InsertionGuard insertGuard(b);
      ctorFn = b.create<mlir::sol::FuncOp>(
          loc, contOp.getName(), b.getFunctionType({}, {}),
          mlir::sol::StateMutability::NonPayable);
      b.setInsertionPointToStart(b.createBlock(&ctorFn.getRegion()));
      // The synthesized ctor must still evaluate base-constructor arguments
      // given in the inheritance specifier (e.g. `contract B is A(...) {}`).
      pendingBaseCtorArgs.clear();
      if (FunctionDefinition const *nextCtor = cont.nextConstructor(cont))
        genBaseCtorCall(cont, *nextCtor, loc);
      b.create<mlir::sol::ReturnOp>(loc);
    }
    ctorFn.setKind(mlir::sol::FunctionKind::Constructor);
    ctorFn.setOrigFnType(ctorFn.getFunctionType());

    // Generate state variable init in the ctor. Iterate base-to-derived in
    // declaration order so that storage, transient and immutable initializers
    // run interleaved in the same order as the old codegen
    // (linearizedStateVariables only tracks slot-assigned variables and would
    // skip immutables).
    b.setInsertionPointToStart(&ctorFn.getBody().front());
    for (ContractDefinition const *baseCont :
         llvm::reverse(cont.annotation().linearizedBaseContracts)) {
      for (VariableDeclaration const *stateVar : baseCont->stateVariables()) {
        if (!stateVar->isConstant() && stateVar->value()) {
          genAssign(genStateVarRef(*stateVar, /*inCreationContext=*/true),
                    genRValExpr(*stateVar->value()), getLoc(*stateVar));
        }
      }
    }
    b.setInsertionPointAfter(ctorFn);
    inCtor = false;
  }

  // Lower all other functions and modifiers.
  auto lowerFnsAndMods = [&](ContractDefinition const &baseCont) {
    // Lower functions.
    for (auto *f : baseCont.definedFunctions()) {
      // Skip the current contract's ctor since it is already lowered.
      if (baseCont == *currContract && f->isConstructor())
        continue;
      if (f->isImplemented())
        lower(*f);
    }

    // Lower modifiers.
    for (auto *modifier : baseCont.functionModifiers()) {
      if (modifier->isImplemented())
        lower(*modifier);
    }
  };
  for (ContractDefinition const *baseCont :
       cont.annotation().linearizedBaseContracts) {
    lowerFnsAndMods(*baseCont);
  }
  currContract = nullptr;
}

void SolidityToMLIRPass::lowerFreeFuncs(SourceUnit const &srcUnit) {
  for (const auto *func :
       ASTNode::filteredNodes<FunctionDefinition>(srcUnit.nodes())) {
    lower(*func);
  }
}

static void loadDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::sol::SolDialect>();
  ctx.getOrLoadDialect<mlir::yul::YulDialect>();
  // For lowering yul in inline-asm.
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
}

bool CompilerStack::runMlirPipeline() {
  llvm::DefaultThreadPool threadPool;
  // For sync'ing the output collection.
  std::mutex outMtx;
  // For sync'ing the error printing.
  std::mutex errMtx;
  // Tracks if any thread had an error.
  std::atomic<bool> hadError{false};

  // Maps contract ast to requested output. Updates are sync'ed by `outMtx`.
  std::map<ContractDefinition const *, std::string, ASTNode::CompareByID>
      outputMap;
  std::map<ContractDefinition const *, evm::UnlinkedObj, ASTNode::CompareByID>
      unlinkedObjMap;

  for (Source const *src : m_sourceOrder) {
    // Lower requested contracts per thread.
    bool hasContract = false;
    for (const auto *contr :
         ASTNode::filteredNodes<ContractDefinition>(src->ast->nodes())) {
      if (!contr->canBeDeployed())
        continue;
      hasContract = true;
      if (isRequestedContract(*contr)) {
        threadPool.async([&, src, contr]() {
          // Create the mlir context.
          mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
          loadDialects(ctx);

          // Run the ast lowering pass.
          SolidityToMLIRPass gen(ctx, m_evmVersion, m_revertStrings,
                                 /*genUnkLoc=*/m_mlirGenJob.action ==
                                     Action::GenObj);
          gen.init(src->charStream);
          // Free functions are emitted on-demand inside the contract scope by
          // the call-site handler when they are first referenced. Do NOT call
          // lowerFreeFuncs here: it would place them at module level where they
          // are invisible to callers inside sol.contract, and multiple call
          // sites would produce duplicate symbols.
          //
          // An exception escaping the thread-pool task would silently drop
          // this contract's output while the job still reports success -
          // record it as an error instead.
          try {
            gen.lower(*contr);
          } catch (std::exception const &e) {
            hadError.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> g(errMtx);
            llvm::errs() << "Exception lowering contract " << contr->name()
                         << ": " << e.what() << "\n";
            return;
          }
          mlir::ModuleOp mod = gen.getModule();

          // Verify the module.
          if (failed(mlir::verify(mod))) {
            hadError.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> g(errMtx);
            mod.dump();
            mod.emitError("Module verification error");
            return;
          }

          if (requiresLinking(m_mlirGenJob.action)) {
            // Create the llvm target machine.
            std::unique_ptr<llvm::TargetMachine> tgtMach =
                createTargetMachine(m_mlirGenJob.tgt);
            setTgtMachOpt(tgtMach.get(), m_mlirGenJob.optLevel);

            // Generate the object.
            evm::UnlinkedObj obj =
                genEvmObj(mod, m_mlirGenJob.optLevel, *tgtMach);
            std::lock_guard<std::mutex> g(outMtx);
            unlinkedObjMap[contr] = obj;

          } else {
            // Generate the print output.
            std::string out = printJob(m_mlirGenJob, mod);
            std::lock_guard<std::mutex> g(outMtx);
            outputMap[contr] = out;
          }
        });
      }
    }

    if (!hasContract) {
      mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
      loadDialects(ctx);

      SolidityToMLIRPass gen(ctx, m_evmVersion, m_revertStrings,
                             /*genUnkLoc=*/m_mlirGenJob.action ==
                                 Action::GenObj);
      // Then lower free functions. This is handy in testing.
      gen.init(src->charStream);
      gen.lowerFreeFuncs(*src->ast);

      mlir::ModuleOp mod = gen.getModule();
      if (failed(mlir::verify(mod))) {
        std::lock_guard<std::mutex> g(errMtx);
        mod.dump();
        mod.emitError("Module verification error");
        return false;
      }

      if (m_mlirGenJob.action != Action::GenObj) {
        std::lock_guard<std::mutex> g(outMtx);
        llvm::outs() << printJob(m_mlirGenJob, mod);
      }
    }
  }

  threadPool.wait();
  if (hadError)
    return false;

  // Combine all the outputs.
  if (!requiresLinking(m_mlirGenJob.action)) {
    for (auto const &i : outputMap)
      llvm::outs() << i.second;
  } else {
    evm::BytecodeGen bcGen(unlinkedObjMap, m_libraries);
    for (auto const &i : unlinkedObjMap) {
      ContractDefinition const *cont = i.first;
      m_contracts.at(cont->fullyQualifiedName()).mlirPipeline =
          bcGen.genEvmBytecode(i.first);
    }

    if (m_mlirGenJob.action == Action::PrintObj) {
      for (auto const &i : unlinkedObjMap) {
        ContractDefinition const *cont = i.first;
        Bytecode const &bc =
            m_contracts.at(cont->fullyQualifiedName()).mlirPipeline;
        llvm::outs() << "Binary:" << "\n";
        llvm::outs() << llvm::toHex(bc.creation, /*LowerCase=*/true) << "\n";
        llvm::outs() << "Binary of the runtime part:" << "\n";
        llvm::outs() << llvm::toHex(bc.runtime, /*LowerCase=*/true) << "\n";
      }
    }

    for (auto const &i : unlinkedObjMap) {
      evm::UnlinkedObj obj = i.second;
      if (obj.creationPart)
        LLVMDisposeMemoryBuffer(obj.creationPart);
      if (obj.runtimePart)
        LLVMDisposeMemoryBuffer(obj.runtimePart);
    }
  }

  return true;
}

// TODO: Move the following functions somewhere else.

void solidity::mlirgen::registerMLIRCLOpts() {
  // FIXME: Verifier's InFlightDiagnostic doesn't work with --mmlir
  // -mlir-print-op-on-diagnostic!
  mlir::registerMLIRContextCLOptions();

  mlir::registerAsmPrinterCLOptions();
  mlir::registerPassManagerCLOptions();
}

bool solidity::mlirgen::parseMLIROpts(std::vector<const char *> &argv) {
  // ParseCommandLineOptions() expects argv[0] to be the name of a program
  std::vector<const char *> fooArgv{"foo"};
  for (const char *arg : argv) {
    fooArgv.push_back(arg);
  }

  return llvm::cl::ParseCommandLineOptions(fooArgv.size(), fooArgv.data(),
                                           "Generic MLIR flags\n");
}

solidity::mlirgen::Target
solidity::mlirgen::strToTarget(std::string const &str) {
  std::string inLowerCase = str;
  std::transform(inLowerCase.begin(), inLowerCase.end(), inLowerCase.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (inLowerCase == "evm")
    return Target::EVM;
  return Target::Undefined;
}
