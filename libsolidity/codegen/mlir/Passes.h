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
// MLIR Passes
//

#pragma once

#include "libsolidity/ast/AST.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "libsolidity/codegen/mlir/Target/EVM/Util.h"
#include "libsolutil/FixedHash.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>

namespace mlir {
class Pass;

namespace sol {

/// Creates a pass to lower sol dialect to standard dialects.
std::unique_ptr<Pass> createConvertSolToStandardPass();
std::unique_ptr<Pass>
createConvertSolToStandardPass(solidity::mlirgen::Target tgt);

std::unique_ptr<Pass> createModifierOpLoweringPass();
std::unique_ptr<Pass> createFuseFreePtrPass();
std::unique_ptr<Pass> createLoopInvariantCodeMotionPass();

// TODO: Is mlir::sol the right namespace for this?
/// Creates a pass to convert standard dialects to llvm dialect.
std::unique_ptr<Pass> createConvertStandardToLLVMPass(StringRef triple,
                                                      unsigned indexBitwidth,
                                                      StringRef dataLayout);

} // namespace sol

} // namespace mlir

namespace solidity::mlirgen {

/// Adds dialect conversion passes for the target.
void addConversionPasses(mlir::PassManager &, Target tgt, bool enableDI = true);

/// Performs the JobSpec of ir/asm printing.
std::string printJob(JobSpec const &, mlir::ModuleOp);

/// Creates and return the llvm::TargetMachine for `tgt`
std::unique_ptr<llvm::TargetMachine> createTargetMachine(Target tgt);

void setTgtMachOpt(llvm::TargetMachine *tgtMach, char levelChar);

/// Sets target specific info in `llvmMod` from `tgt`
void setTgtSpecificInfoInModule(Target tgt, llvm::Module &llvmMod,
                                llvm::TargetMachine const &tgtMach);

/// Lowers the module and returns the evm object.
evm::UnlinkedObj genEvmObj(mlir::ModuleOp, char optLevel,
                           llvm::TargetMachine &tgtMach);

/// Returns the bytecode of `obj` for evm.  This function is not thread-safe
/// across LLVM threads.  It calls into lld, which mutates global llvm state
/// internally.
Bytecode
genEvmBytecode(frontend::ContractDefinition const *cont,
               std::map<frontend::ContractDefinition const *, evm::UnlinkedObj,
                        frontend::ASTNode::CompareByID> &objMap,
               std::map<std::string, util::h160> const &libAddrMap);

} // namespace solidity::mlirgen

namespace llvm {
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::TargetMachine, LLVMTargetMachineRef)
}
