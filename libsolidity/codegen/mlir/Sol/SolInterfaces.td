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
// Sol dialect interfaces
//

#ifndef MLIR_SOL_SOLINTERFACES_TD
#define MLIR_SOL_SOLINTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

// (Copied and modified from clangir).
//
// Loops in solidity are C-style loops.
// TODO: Inherit from LoopLikeOpInterface.
def LoopOpInterface : OpInterface<"LoopOpInterface", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface>
]> {
  let cppNamespace = "::mlir::sol";

  // Do we need the getRegionsInExecutionOrder interface function from the
  // clangir version?
  let methods = [
    InterfaceMethod<[{
        Returns the loop's conditional region.
      }],
      /*retTy=*/"mlir::Region &",
      /*methodName=*/"getCond"
    >,
    InterfaceMethod<[{
        Returns the loop's body region.
      }],
      /*retTy=*/"mlir::Region &",
      /*methodName=*/"getBody"
    >,
    InterfaceMethod<[{
        Returns a pointer to the loop's step region or nullptr.
      }],
      /*retTy=*/"mlir::Region *",
      /*methodName=*/"maybeGetStep",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return nullptr;"
    >,
    InterfaceMethod<[{
        Returns the first region to be executed in the loop.
      }],
      /*retTy=*/"mlir::Region &",
      /*methodName=*/"getEntry",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return $_op.getCond();"
    >,
    InterfaceMethod<[{
        Recursively walks the body of the loop in pre-order while skipping
        nested loops and executing a callback on every other operation.
      }],
      /*retTy=*/"mlir::WalkResult",
      /*methodName=*/"walkBodySkippingNestedLoops",
      /*args=*/(ins "::llvm::function_ref<mlir::WalkResult (Operation *)>":$callback),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getBody().template walk<WalkOrder::PreOrder>([&](Operation *op) {
          if (isa<LoopOpInterface>(op))
            return mlir::WalkResult::skip();
          return callback(op);
        });
      }]
    >
  ];

  let extraClassDeclaration = [{
    /// Generic method to retrieve the successors of a LoopOpInterface operation.
    static void getLoopOpSuccessorRegions(
        sol::LoopOpInterface op, RegionBranchPoint point,
        SmallVectorImpl<RegionSuccessor> &regions);
  }];
}

#endif // MLIR_SOL_SOLINTERFACES_TD
