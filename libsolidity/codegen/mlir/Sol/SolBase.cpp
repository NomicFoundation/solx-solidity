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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Sol/SolInterfaces.cpp.inc"
#include "Sol/SolOpsDialect.cpp.inc"
#include "Sol/SolOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::sol;

namespace {

struct SolOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto contrKindAttr = dyn_cast<ContractKindAttr>(attr)) {
      os << stringifyContractKind(contrKindAttr.getValue());
      return AliasResult::OverridableAlias;
    }
    if (auto stateMutAttr = dyn_cast<StateMutabilityAttr>(attr)) {
      os << stringifyStateMutability(stateMutAttr.getValue());
      return AliasResult::OverridableAlias;
    }
    if (auto fnKindAttr = dyn_cast<FunctionKindAttr>(attr)) {
      os << stringifyFunctionKind(fnKindAttr.getValue());
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

} // namespace

void SolDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Sol/SolOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Sol/SolOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "Sol/SolOpsAttributes.cpp.inc"
      >();

  addInterfaces<SolOpAsmDialectInterface>();
}

// TODO: Implement these!
// We should track the evm version in the module op.
bool mlir::sol::evmhasStaticCall(ModuleOp mod) { return true; }
bool mlir::sol::evmSupportsReturnData(ModuleOp mod) { return true; }
bool mlir::sol::evmCanOverchargeGasForCall(ModuleOp mod) { return true; }

Type mlir::sol::getEltType(Type ty, Index structTyIdx) {
  if (auto ptrTy = dyn_cast<sol::PointerType>(ty)) {
    return ptrTy.getPointeeType();
  }
  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    return arrTy.getEltType();
  }
  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    return structTy.getMemberTypes()[structTyIdx];
  }
  llvm_unreachable("Invalid type");
}

DataLocation mlir::sol::getDataLocation(Type ty) {
  return TypeSwitch<Type, DataLocation>(ty)
      .Case<PointerType>(
          [](sol::PointerType ptrTy) { return ptrTy.getDataLocation(); })
      .Case<ArrayType>(
          [](sol::ArrayType arrTy) { return arrTy.getDataLocation(); })
      .Case<StringType>(
          [](sol::StringType strTy) { return strTy.getDataLocation(); })
      .Case<StructType>(
          [](sol::StructType structTy) { return structTy.getDataLocation(); })
      .Case<MappingType>(
          [](sol::MappingType) { return sol::DataLocation::Storage; })
      .Default([&](Type) { return DataLocation::Stack; });
}

// TODO? Should we exclude sol.pointer from reference types?

bool mlir::sol::isRefType(Type ty) {
  return isa<ArrayType>(ty) || isa<StringType>(ty) || isa<StructType>(ty) ||
         isa<PointerType>(ty) || isa<MappingType>(ty);
}

bool mlir::sol::isNonPtrRefType(Type ty) {
  return isRefType(ty) && !isa<PointerType>(ty);
}

bool mlir::sol::isLeftAligned(Type ty) {
  if (isa<IntegerType>(ty))
    return false;
  llvm_unreachable("NYI: isLeftAligned of other types");
}

bool mlir::sol::isDynamicallySized(Type ty) {
  if (isa<StringType>(ty))
    return true;

  if (auto arrTy = dyn_cast<ArrayType>(ty))
    return arrTy.isDynSized();

  return false;
}

bool mlir::sol::hasDynamicallySizedElt(Type ty) {
  if (isa<StringType>(ty))
    return true;

  if (auto arrTy = dyn_cast<ArrayType>(ty))
    return arrTy.isDynSized() || hasDynamicallySizedElt(arrTy.getEltType());

  if (auto structTy = dyn_cast<StructType>(ty))
    return llvm::any_of(structTy.getMemberTypes(),
                        [](Type ty) { return hasDynamicallySizedElt(ty); });

  return false;
}

static ParseResult parseDataLocation(AsmParser &parser,
                                     DataLocation &dataLocation) {
  StringRef dataLocationTok;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&dataLocationTok))
    return failure();

  auto parsedDataLoc = symbolizeDataLocation(dataLocationTok);
  if (!parsedDataLoc) {
    parser.emitError(loc, "Invalid data-location");
    return failure();
  }

  dataLocation = *parsedDataLoc;
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/// Parses a sol.array type.
///
///   array-type ::= `<` size `x` elt-ty `,` data-location `>`
///   size ::= fixed-size | `?`
///
Type ArrayType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  int64_t size = -1;
  if (parser.parseOptionalQuestion()) {
    if (parser.parseInteger(size))
      return {};
  }

  if (parser.parseKeyword("x"))
    return {};

  Type eleTy;
  if (parser.parseType(eleTy))
    return {};

  if (parser.parseComma())
    return {};

  DataLocation dataLocation = DataLocation::Memory;
  if (parseDataLocation(parser, dataLocation))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), size, eleTy, dataLocation);
}

/// Prints a sol.array type.
void ArrayType::print(AsmPrinter &printer) const {
  printer << "<";

  if (getSize() == -1)
    printer << "?";
  else
    printer << getSize();

  printer << " x " << getEltType() << ", "
          << stringifyDataLocation(getDataLocation()) << ">";
}

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

/// Parses a sol.string type.
///
///   string-type ::= `<` data-location `>`
///
Type StringType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  DataLocation dataLocation = DataLocation::Memory;
  if (parseDataLocation(parser, dataLocation))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), dataLocation);
}

/// Prints a sol.string type.
void StringType::print(AsmPrinter &printer) const {
  printer << "<" << stringifyDataLocation(this->getDataLocation()) << ">";
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

/// Parses a sol.struct type.
///
///   struct-type ::= `<` `(` member-types `)` `,` data-location `>`
///
Type StructType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  if (parser.parseLParen())
    return {};

  SmallVector<Type, 4> memTys;
  do {
    Type memTy;
    if (parser.parseType(memTy))
      return {};
    memTys.push_back(memTy);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return {};

  if (parser.parseComma())
    return {};

  DataLocation dataLocation = DataLocation::Memory;
  if (parseDataLocation(parser, dataLocation))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), memTys, dataLocation);
}

/// Prints a sol.array type.
void StructType::print(AsmPrinter &printer) const {
  printer << "<(";
  llvm::interleaveComma(getMemberTypes(), printer.getStream(),
                        [&](Type memTy) { printer << memTy; });
  printer << "), " << stringifyDataLocation(getDataLocation()) << ">";
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

/// Parses a sol.ptr type.
///
///   ptr-type ::= `<` pointee-ty, data-location `>`
///
Type PointerType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  Type pointeeTy;
  if (parser.parseType(pointeeTy))
    return {};

  if (parser.parseComma())
    return {};

  DataLocation dataLocation = DataLocation::Memory;
  if (parseDataLocation(parser, dataLocation))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), pointeeTy, dataLocation);
}

/// Prints a sol.ptr type.
void PointerType::print(AsmPrinter &printer) const {
  printer << "<" << getPointeeType() << ", "
          << stringifyDataLocation(getDataLocation()) << ">";
}

#define GET_ATTRDEF_CLASSES
#include "Sol/SolOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Sol/SolOpsTypes.cpp.inc"
