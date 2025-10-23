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

#include "libsolidity/codegen/mlir/Target/EVM/Util.h"

#include "libsolidity/ast/AST.h"
#include "libsolidity/ast/CallGraph.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "libsolidity/codegen/mlir/Passes.h"
#include "lld-c/LLDAsLibraryC.h"
#include "llvm-c/Core.h"
#include "llvm-c/Types.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace evm;
using namespace solidity;
using namespace solidity::frontend;

mlirgen::Bytecode BytecodeGen::genEvmBytecode(ContractDefinition const *cont) {
  // Generate the assembled objs.
  LLVMMemoryBufferRef creationAssembled =
      genAssembledObj(cont, /*isCreationRequested=*/true);
  LLVMMemoryBufferRef runtimeAssembled = runtimeAssembledMap[cont];

  // Convert libAddrMap for lld-c.
  auto numLibs = libAddrMap.size();
  auto **libNames = new const char *[numLibs];
  auto *libAddrs = new char[numLibs][LINKER_SYMBOL_SIZE];
  int64_t i = 0;
  for (const auto &[name, addr] : libAddrMap) {
    libNames[i] = name.c_str();
    std::memcpy(libAddrs[i], addr.data(), LINKER_SYMBOL_SIZE);
    ++i;
  }

  // Generate bytecode.
  LLVMMemoryBufferRef creationBytecode, runtimeBytecode;
  char *errMsg = nullptr;
  if (LLVMLinkEVM(/*inBuffer=*/creationAssembled,
                  /*outBuffer=*/&creationBytecode, libNames, libAddrs, numLibs,
                  &errMsg))
    llvm_unreachable(errMsg);
  if (LLVMLinkEVM(/*inBuffer=*/runtimeAssembled,
                  /*outBuffer=*/&runtimeBytecode, libNames, libAddrs, numLibs,
                  &errMsg))
    llvm_unreachable(errMsg);

  delete[] libNames;
  delete[] libAddrs;

  mlirgen::Bytecode ret;
  ret.creation = llvm::unwrap(creationBytecode)->getBuffer();
  ret.runtime = llvm::unwrap(runtimeBytecode)->getBuffer();

  LLVMDisposeMemoryBuffer(creationBytecode);
  LLVMDisposeMemoryBuffer(runtimeBytecode);

  return ret;
}

LLVMMemoryBufferRef BytecodeGen::genAssembledObj(ContractDefinition const *cont,
                                                 bool isCreationRequested) {
  // Return if memoized.
  if (isCreationRequested) {
    if (creationAssembledMap.find(cont) != creationAssembledMap.end())
      return creationAssembledMap[cont];
  } else {
    if (runtimeAssembledMap.find(cont) != runtimeAssembledMap.end())
      return runtimeAssembledMap[cont];
  }

  LLVMMemoryBufferRef *objs = nullptr;
  const char **objIds = nullptr;
  int numObjs = 0;
  std::map<ContractDefinition const *, ASTNode const *,
           ASTCompareByID<ContractDefinition>>
      deps;

  // Track the dependency from the ast.
  if (isCreationRequested) {
    deps = cont->annotation().creationCallGraph->get()->bytecodeDependency;
    numObjs = deps.size() + 2;
  } else {
    deps = cont->annotation().deployedCallGraph->get()->bytecodeDependency;
    numObjs = deps.size() + 1;
  }

  // Populate the first object(s) from `cont`'s unlinked obj.
  objs = new LLVMMemoryBufferRef[numObjs];
  objIds = new const char *[numObjs];
  int newObjIdx = 0;
  UnlinkedObj evmObj = unlinkedMap.at(cont);
  if (isCreationRequested) {
    objs[0] = evmObj.creationPart;
    objIds[0] = evmObj.creationId.data();
    objs[1] = genAssembledObj(cont, /*isCreationRequested=*/false);
    objIds[1] = evmObj.runtimeId.data();
    newObjIdx = 2;
  } else {
    objs[0] = evmObj.runtimePart;
    objIds[0] = evmObj.runtimeId.data();
    newObjIdx = 1;
  }

  // Populate the unlinked objs from the dependencies.
  for (auto dep : deps) {
    auto *depCont = dep.first;
    auto *depAst = dep.second;
    objs[newObjIdx] = genAssembledObj(depCont, isCreationDep(depAst));
    objIds[newObjIdx] = isCreationDep(depAst)
                            ? unlinkedMap.at(depCont).creationId.data()
                            : unlinkedMap.at(depCont).runtimeId.data();
    newObjIdx++;
  }

  // Assemble the obj.
  LLVMMemoryBufferRef assembled;
  char *errMsg = nullptr;
  if (LLVMAssembleEVM(isCreationRequested ? 0 : 1, /*inBuffers=*/objs,
                      /*inBuffersIDs=*/objIds,
                      /*inBuffersNum=*/numObjs,
                      /*outBuffer=*/&assembled,
                      /*errorMessage=*/&errMsg))
    llvm_unreachable(errMsg);

  delete[] objs;
  delete[] objIds;

  // Memoize the assembled obj.
  if (isCreationRequested)
    creationAssembledMap[cont] = assembled;
  else
    runtimeAssembledMap[cont] = assembled;
  return assembled;
}

bool BytecodeGen::isCreationDep(ASTNode const *ast) {
  if (auto *memAcc = dynamic_cast<MemberAccess const *>(ast)) {
    ASTString const &memName = memAcc->memberName();
    assert(memName == "creationCode" || memName == "runtimeCode");
    assert(dynamic_cast<MagicType const *>(
        memAcc->expression().annotation().type));
    return memName == "creationCode";
  }
  auto *newExpr = dynamic_cast<NewExpression const *>(ast);
  assert(newExpr);
  assert(dynamic_cast<ContractType const *>(
      newExpr->typeName().annotation().type));
  return true;
}
