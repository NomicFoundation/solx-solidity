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
// EVM specific utility
//

#pragma once

#include "libsolidity/ast/AST.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "llvm-c/Core.h"

namespace evm {

/// The unlinked elf objects.
struct UnlinkedObj {
  LLVMMemoryBufferRef creationPart;
  LLVMMemoryBufferRef runtimePart;
  std::string creationId;
  std::string runtimeId;
};

/// This class creates and caches "assembled" objects ("assembled" here means
/// linked with dependendent bytecodes) and uses it to generate the bytecode. It
/// internally uses evm specific lld functions.
struct BytecodeGen {
  using UnlinkedMap =
      std::map<solidity::frontend::ContractDefinition const *, UnlinkedObj,
               solidity::frontend::ASTNode::CompareByID>;
  using AssembledMap =
      std::map<solidity::frontend::ContractDefinition const *,
               LLVMMemoryBufferRef, solidity::frontend::ASTNode::CompareByID>;

  /// Maps contracts to their unlinked objects.
  UnlinkedMap const &unlinkedMap;
  /// Maps contracts to their assembled objects.
  AssembledMap creationAssembledMap;
  AssembledMap runtimeAssembledMap;
  /// Maps libraries to their address.
  std::map<std::string, solidity::util::h160> const &libAddrMap;

  BytecodeGen(UnlinkedMap const &objMap,
              std::map<std::string, solidity::util::h160> const &libAddrMap)
      : unlinkedMap(objMap), libAddrMap(libAddrMap) {}

  ~BytecodeGen() {
    // Dispose all LLVM memory buffers
    for (auto &i : creationAssembledMap)
      LLVMDisposeMemoryBuffer(i.second);
    for (auto &i : runtimeAssembledMap)
      LLVMDisposeMemoryBuffer(i.second);
  }

  /// Returns the bytecode of a fully contained unlinked object.
  solidity::mlirgen::Bytecode genEvmBytecode(UnlinkedObj);
  /// Returns the bytecode from the assembled objects.
  solidity::mlirgen::Bytecode
  genEvmBytecode(LLVMMemoryBufferRef creationAssembled,
                 LLVMMemoryBufferRef runtimeAssembled);
  /// Returns the bytecode of contract.
  solidity::mlirgen::Bytecode
  genEvmBytecode(solidity::frontend::ContractDefinition const *cont);

  /// Returns the assembled object of the contract. Also caches the assembled
  /// dependent objects.
  LLVMMemoryBufferRef
  genAssembledObj(solidity::frontend::ContractDefinition const *cont,
                  bool isCreationRequested);

  /// Return true if the ast node representation is a creation bytecode
  /// dependency.
  bool isCreationDep(solidity::frontend::ASTNode const *ast);
};

} // namespace evm
