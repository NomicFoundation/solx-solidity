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
// MLIR utilities
//

#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/ErrorHandling.h"

namespace solidity {
namespace mlirgen {

/// Converts a solidity u256 to an llvm::APInt
inline llvm::APInt getAPInt(solidity::u256 const &val, unsigned numBits) {
  switch (numBits) {
  case 8:
    return llvm::APInt(numBits, val.convert_to<uint8_t>());
  case 16:
    return llvm::APInt(numBits, val.convert_to<uint16_t>());
  case 32:
    return llvm::APInt(numBits, val.convert_to<uint32_t>());
  case 64:
    return llvm::APInt(numBits, val.convert_to<uint64_t>());
  case 128:
    return llvm::APInt(numBits, val.str().substr(128, 128), /*radix=*/10);
  case 256:
    return llvm::APInt(numBits, val.str(), /*radix=*/10);
  }
  llvm_unreachable("Unsupported bit width");
}

/// Extension of mlir::OpBuilder with APIs helpful for codegen in solidity.
class BuilderExt {
  mlir::OpBuilder &b;
  mlir::Location defLoc;

  mlir::Value genConst(llvm::APInt const &val,
                       std::optional<mlir::Location> locArg) {
    mlir::IntegerType ty = b.getIntegerType(val.getBitWidth());
    auto op = b.create<mlir::arith::ConstantOp>(locArg ? *locArg : defLoc,
                                                b.getIntegerAttr(ty, val));
    return op.getResult();
  }

public:
  explicit BuilderExt(mlir::OpBuilder &b) : b(b), defLoc(b.getUnknownLoc()) {}

  explicit BuilderExt(mlir::OpBuilder &b, mlir::Location loc)
      : b(b), defLoc(loc) {}

  mlir::Value genBool(bool val,
                      std::optional<mlir::Location> locArg = std::nullopt) {
    mlir::IntegerType ty = b.getIntegerType(1);
    auto op = b.create<mlir::arith::ConstantOp>(locArg ? *locArg : defLoc,
                                                b.getIntegerAttr(ty, val));
    return op.getResult();
  }

  mlir::Value
  genI256Const(int64_t val,
               std::optional<mlir::Location> locArg = std::nullopt) {
    return genConst(llvm::APInt(256, val, /*isSigned=*/true), locArg);
  }

  mlir::Value
  genI256Const(solidity::u256 const &val,
               std::optional<mlir::Location> locArg = std::nullopt) {
    return genConst(getAPInt(val, 256), locArg);
  }
};

} // namespace mlirgen
} // namespace solidity
