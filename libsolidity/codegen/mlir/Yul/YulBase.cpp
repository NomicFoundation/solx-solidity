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

#include "Yul.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "Yul/YulOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::yul;

void YulDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Yul/YulOps.cpp.inc"
      >();
}

Operation *YulDialect::materializeConstant(OpBuilder &builder, Attribute val,
                                           Type type, Location loc) {
  return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(val));
}
