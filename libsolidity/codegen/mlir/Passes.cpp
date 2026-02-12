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

#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "lld-c/LLDAsLibraryC.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Sol/Transforms/Immutables.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm-c/Core.h"
#include "llvm-c/Target.h"
#include "llvm-c/Transforms/PassBuilder.h"
#include "llvm-c/Types.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <mutex>

void solidity::mlirgen::addConversionPasses(mlir::PassManager &passMgr,
                                            Target tgt, bool enableDI) {
  passMgr.addPass(mlir::createCanonicalizerPass());
  passMgr.addPass(mlir::sol::createModifierOpLoweringPass());
  passMgr.addPass(mlir::sol::createConvertSolToStandardPass());
  // Canonicalizer removes unreachable blocks, which is important for getting
  // the translation to llvm-ir working correctly.
  passMgr.addPass(mlir::createCanonicalizerPass());

  passMgr.addPass(mlir::createSCFToControlFlowPass());
  passMgr.addPass(
      mlir::createConvertFuncToLLVMPass(mlir::ConvertFuncToLLVMPassOptions(
          /*useBarePtrCallConv*/ false, /*indexBitwidth*/ 256)));
  passMgr.addPass(mlir::createArithToLLVMConversionPass(
      mlir::ArithToLLVMConversionPassOptions(/*indexBitwidth*/ 256)));
  passMgr.addPass(mlir::createConvertControlFlowToLLVMPass(
      mlir::ConvertControlFlowToLLVMPassOptions(/*indexBitwidth*/ 256)));
  passMgr.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (enableDI)
    passMgr.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
}

static std::mutex tgtMachMtx;

std::unique_ptr<llvm::TargetMachine>
solidity::mlirgen::createTargetMachine(Target tgt) {
  static std::once_flag initTargetOnceFlag;

  switch (tgt) {
  case Target::EVM: {
    // Initialize and register the target.
    std::call_once(initTargetOnceFlag, []() {
      LLVMInitializeEVMTarget();
      LLVMInitializeEVMTargetInfo();
      LLVMInitializeEVMTargetMC();
      LLVMInitializeEVMAsmPrinter();
    });

    // Lookup llvm::Target.
    std::string errMsg;
    llvm::Target const *llvmTgt =
        llvm::TargetRegistry::lookupTarget("evm", errMsg);
    if (!llvmTgt)
      llvm_unreachable(errMsg.c_str());

    // Create and return the llvm::TargetMachine.
    llvm::TargetOptions options;
    std::lock_guard<std::mutex> l(tgtMachMtx);
    return std::unique_ptr<llvm::TargetMachine>(llvmTgt->createTargetMachine(
        llvm::Triple("evm"), /*CPU=*/"", /*Features=*/"", options,
        /*Reloc::Model=*/std::nullopt));

    // TODO: Set code-model?
    // tgtMach->setCodeModel(?);
  }
  case Target::Undefined:
    llvm_unreachable("Invalid target");
  }
}

void solidity::mlirgen::setTgtSpecificInfoInModule(
    Target tgt, llvm::Module &llvmMod, llvm::TargetMachine const &tgtMach) {
  std::string triple;
  switch (tgt) {
  case Target::EVM:
    triple = "evm-unknown-unknown";
    break;
  case Target::Undefined:
    llvm_unreachable("Undefined target");
  }

  llvmMod.setTargetTriple(llvm::Triple(triple));
  llvmMod.setDataLayout(tgtMach.createDataLayout());
}

/// Sets the optimization level of the llvm::TargetMachine.
void solidity::mlirgen::setTgtMachOpt(llvm::TargetMachine *tgtMach,
                                      char levelChar) {
  // Set level to 2 if size optimization is specified.
  if (levelChar == 's' || levelChar == 'z')
    levelChar = '2';
  auto level = llvm::CodeGenOpt::parseLevel(levelChar);
  assert(level);
  tgtMach->setOptLevel(*level);
}

/// Runs the llvm optimization pipeline.
static void runLLVMOptPipeline(llvm::Module *mod, char level,
                               llvm::TargetMachine *tgtMach) {
  char pipeline[12];
  std::sprintf(pipeline, "default<O%c>", level);
  LLVMErrorRef status =
      LLVMRunPasses(reinterpret_cast<LLVMModuleRef>(mod), pipeline,
                    reinterpret_cast<LLVMTargetMachineRef>(tgtMach),
                    LLVMCreatePassBuilderOptions());
  if (status != LLVMErrorSuccess) {
    llvm_unreachable(LLVMGetErrorMessage(status));
  }
}

static std::unique_ptr<llvm::Module>
genLLVMIR(mlir::ModuleOp mod, solidity::mlirgen::Target tgt, char optLevel,
          llvm::TargetMachine &tgtMach, llvm::LLVMContext &llvmCtx) {
  // Translate the llvm dialect to llvm-ir.
  mlir::registerLLVMDialectTranslation(*mod.getContext());
  mlir::registerBuiltinDialectTranslation(*mod.getContext());
  std::unique_ptr<llvm::Module> llvmMod =
      mlir::translateModuleToLLVMIR(mod, llvmCtx);
  assert(llvmMod);

  // Set target specfic info in the llvm module.
  setTgtSpecificInfoInModule(tgt, *llvmMod, tgtMach);

  runLLVMOptPipeline(llvmMod.get(), optLevel, &tgtMach);

  return llvmMod;
}

static std::string getAsm(llvm::Module &llvmMod, llvm::TargetMachine &tgtMach) {
  llvm::legacy::PassManager llvmPassMgr;
  llvm::SmallString<4096> ret;
  llvm::raw_svector_ostream ss(ret);
  tgtMach.addPassesToEmitFile(llvmPassMgr, ss,
                              /*DwoOut=*/nullptr,
                              llvm::CodeGenFileType::AssemblyFile);
  llvmPassMgr.run(llvmMod);
  return std::string(ret);
}

static mlir::ModuleOp extractRuntimeModule(mlir::ModuleOp creationMod) {
  for (mlir::Operation &op : *creationMod.getBody()) {
    if (mlir::isa<mlir::ModuleOp>(op)) {
      mlir::ModuleOp runtimeMod;
      runtimeMod = mlir::cast<mlir::ModuleOp>(op);
      // Remove the runtime module from the creation module.
      runtimeMod->remove();
      return runtimeMod;
    }
  }
  return {};
}

static LLVMMemoryBufferRef genObj(llvm::Module &mod,
                                  llvm::TargetMachine &tgtMach) {
  LLVMTargetMachineRef tgtMachWrapped = wrap(&tgtMach);
  LLVMModuleRef modWrapped = wrap(&mod);

  char *errMsg = nullptr;
  LLVMMemoryBufferRef obj;
  if (LLVMTargetMachineEmitToMemoryBuffer(tgtMachWrapped, modWrapped,
                                          LLVMObjectFile, &errMsg, &obj))
    llvm_unreachable(errMsg);
  return obj;
}

evm::UnlinkedObj solidity::mlirgen::genEvmObj(mlir::ModuleOp mod, char optLevel,
                                              llvm::TargetMachine &tgtMach) {
  mlir::PassManager passMgr(mod.getContext());
  llvm::LLVMContext llvmCtx;

  // Convert the module's ir to llvm dialect.
  // FIXME: enableDI UNREACHABLE executed at
  // llvm/lib/Target/EVM/MCTargetDesc/EVMAsmBackend.cpp:112!
  addConversionPasses(passMgr, Target::EVM, /*enableDI=*/false);
  if (mlir::failed(passMgr.run(mod)))
    llvm_unreachable("Conversion to llvm dialect failed");

  auto creationMod = mod;
  mlir::ModuleOp runtimeMod = extractRuntimeModule(creationMod);
  assert(runtimeMod);

  // Lower runtime object. This is a dependency for lowering the setimmutable
  // ops in the creation object.
  std::unique_ptr<llvm::Module> runtimeLlvmMod =
      genLLVMIR(runtimeMod, Target::EVM, optLevel, tgtMach, llvmCtx);
  LLVMMemoryBufferRef runtimeObj = genObj(*runtimeLlvmMod, tgtMach);

  assert(creationMod.getName() && runtimeMod.getName());

  // Lower setimmutable ops in the creation object.
  char **immIDs = nullptr;
  uint64_t *immOffsets = nullptr;
  uint64_t immCount = LLVMGetImmutablesEVM(runtimeObj, &immIDs, &immOffsets);
  llvm::StringMap<mlir::SmallVector<uint64_t>> immMap;
  for (uint64_t i = 0; i < immCount; ++i)
    immMap[immIDs[i]].push_back(immOffsets[i]);
  if (immCount) {
    LLVMDisposeImmutablesEVM(immIDs, immOffsets, immCount);
    mlir::evm::lowerSetImmutables(creationMod, immMap);
  } else {
    // llvm might have optimized away the immutable references.
    mlir::evm::removeSetImmutables(creationMod);
  }

  // Lower the creation object.
  std::unique_ptr<llvm::Module> creationLlvmMod =
      genLLVMIR(creationMod, Target::EVM, optLevel, tgtMach, llvmCtx);
  LLVMMemoryBufferRef creationObj = genObj(*creationLlvmMod, tgtMach);

  return {creationObj, runtimeObj, creationMod.getName()->str(),
          runtimeMod.getName()->str()};
}

std::string solidity::mlirgen::printJob(JobSpec const &job,
                                        mlir::ModuleOp mod) {
  assert(job.action != Action::GenObj);

  mlir::MLIRContext *mlirCtx = mod.getContext();
  mlir::PassManager passMgr(mlirCtx);
  llvm::LLVMContext llvmCtx;
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  // Register a diagnostic handler to capture the diagnostic so that we can
  // check it later.
  std::unique_ptr<mlir::Diagnostic> diagnostic;
  mlirCtx->getDiagEngine().registerHandler([&](mlir::Diagnostic &diag) {
    diagnostic = std::make_unique<mlir::Diagnostic>(std::move(diag));
  });

  switch (job.action) {
  case Action::PrintInitStg:
    mod.print(ss);
    return ret;

  case Action::PrintStandardMLIR:
    assert(job.tgt != Target::Undefined);
    passMgr.addPass(mlir::createCanonicalizerPass());
    passMgr.addPass(mlir::sol::createModifierOpLoweringPass());
    passMgr.addPass(mlir::sol::createConvertSolToStandardPass());
    passMgr.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(passMgr.run(mod))) {
      mod.print(ss);
      llvm::errs() << ret << '\n';
      llvm::errs() << diagnostic->str() << '\n';
      llvm_unreachable("Conversion to standard dialects failed");
    }
    mod.print(ss);
    return ret;

  case Action::PrintLLVMIR: {
    assert(job.tgt != Target::Undefined);

    // Convert the module's ir to llvm dialect.
    addConversionPasses(passMgr, job.tgt);
    if (mlir::failed(passMgr.run(mod))) {
      mod.print(ss);
      llvm::errs() << ret << '\n';
      llvm::errs() << diagnostic->str() << '\n';
      llvm_unreachable("Conversion to llvm dialect failed");
    }

    std::unique_ptr<llvm::TargetMachine> tgtMach = createTargetMachine(job.tgt);
    setTgtMachOpt(tgtMach.get(), job.optLevel);

    switch (job.tgt) {
    case Target::EVM: {
      auto creationMod = mod;
      mlir::ModuleOp runtimeMod = extractRuntimeModule(creationMod);
      assert(runtimeMod);

      mlir::evm::removeSetImmutables(creationMod);

      std::unique_ptr<llvm::Module> creationLlvmMod =
          genLLVMIR(creationMod, job.tgt, job.optLevel, *tgtMach, llvmCtx);
      std::unique_ptr<llvm::Module> runtimeLlvmMod =
          genLLVMIR(runtimeMod, job.tgt, job.optLevel, *tgtMach, llvmCtx);

      ss << *creationLlvmMod;
      ss << *runtimeLlvmMod;
      return ret;
    }
    default:
      break;
    }
    break;
  }

  case Action::PrintAsm: {
    // Convert the module's ir to llvm dialect.
    addConversionPasses(passMgr, job.tgt);
    if (mlir::failed(passMgr.run(mod)))
      llvm_unreachable("Conversion to llvm dialect failed");

    std::unique_ptr<llvm::TargetMachine> tgtMach = createTargetMachine(job.tgt);
    setTgtMachOpt(tgtMach.get(), job.optLevel);

    switch (job.tgt) {
    case Target::EVM: {
      auto creationMod = mod;
      mlir::ModuleOp runtimeMod = extractRuntimeModule(creationMod);

      mlir::evm::removeSetImmutables(creationMod);

      std::string ret;
      ret = getAsm(
          *genLLVMIR(creationMod, job.tgt, job.optLevel, *tgtMach, llvmCtx),
          *tgtMach);
      if (runtimeMod)
        ret += getAsm(
            *genLLVMIR(runtimeMod, job.tgt, job.optLevel, *tgtMach, llvmCtx),
            *tgtMach);
      return ret;
    }
    default:
      break;
    };
    break;
  }
  default:
    break;
  }

  llvm_unreachable("Undefined action/target");
}
