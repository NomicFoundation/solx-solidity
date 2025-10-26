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
// Yul dialect
//

#ifndef MLIR_YUL_YULBASE_TD
#define MLIR_YUL_YULBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Yul_Dialect : Dialect {
  let name = "yul";
  let cppNamespace = "::mlir::yul";
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def I256 : I<256>;

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def I256Attr : SignlessIntegerAttrBase<I256,
                                       "256-bit signless integer attribute">;

//===----------------------------------------------------------------------===//
// Op classes
//===----------------------------------------------------------------------===//

class Yul_Op<string mnemonic, list<Trait> traits = []> : Op<Yul_Dialect,
                                                            mnemonic, traits>;

#endif // MLIR_YUL_YULBASE_TD
