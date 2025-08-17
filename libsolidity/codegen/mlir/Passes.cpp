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
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm-c/Core.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm-c/Transforms/PassBuilder.h"
#include "llvm-c/Types.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
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

// FIXME: Define an interface for targets!

void solidity::mlirgen::addConversionPasses(mlir::PassManager &passMgr,
                                            Target tgt, bool enableDI) {
  passMgr.addPass(mlir::createCanonicalizerPass());
  passMgr.addPass(mlir::sol::createModifierOpLoweringPass());
  passMgr.addPass(mlir::sol::createConvertSolToStandardPass(tgt));
  // Canonicalizer removes unreachable blocks, which is important for getting
  // the translation to llvm-ir working correctly.
  passMgr.addPass(mlir::createCanonicalizerPass());

  // FIXME: Adding individual conversion passes for each dialects causes
  // unrealized_conversion_cast's with index types.
  //
  // FIXME: `Target` should track triple, index bitwidth and data-layout.

  switch (tgt) {
  case Target::EVM:
    passMgr.addPass(mlir::sol::createConvertStandardToLLVMPass(
        /*triple=*/"evm-unknown-unknown",
        /*indexBitwidth=*/256,
        /*dataLayout=*/"E-p:256:256-i256:256:256-S256-a:256:256"));
    break;
  case Target::EraVM:
    passMgr.addPass(mlir::sol::createConvertStandardToLLVMPass(
        /*triple=*/"eravm-unknown-unknown",
        /*indexBitwidth=*/256,
        /*dataLayout=*/"E-p:256:256-i256:256:256-S32-a:256:256"));
    break;
  default:
    llvm_unreachable("");
  }

  if (enableDI)
    passMgr.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
}

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
    return std::unique_ptr<llvm::TargetMachine>(
        llvmTgt->createTargetMachine("evm", /*CPU=*/"", /*Features=*/"",
                                     options, /*Reloc::Model=*/std::nullopt));

    // TODO: Set code-model?
    // tgtMach->setCodeModel(?);
  }
  case Target::EraVM: {
    // Initialize and register the target.
    std::call_once(initTargetOnceFlag, []() {
      LLVMInitializeEraVMTarget();
      LLVMInitializeEraVMTargetInfo();
      LLVMInitializeEraVMTargetMC();
      LLVMInitializeEraVMAsmPrinter();
    });

    // Lookup llvm::Target.
    std::string errMsg;
    llvm::Target const *llvmTgt =
        llvm::TargetRegistry::lookupTarget("eravm", errMsg);
    if (!llvmTgt)
      llvm_unreachable(errMsg.c_str());

    // Create and return the llvm::TargetMachine.
    llvm::TargetOptions options;
    return std::unique_ptr<llvm::TargetMachine>(
        llvmTgt->createTargetMachine("eravm", /*CPU=*/"", /*Features=*/"",
                                     options, /*Reloc::Model=*/std::nullopt));

    // TODO: Set code-model?
    // tgtMach->setCodeModel(?);
    break;
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
  case Target::EraVM:
    triple = "eravm-unknown-unknown";
    break;
  case Target::Undefined:
    llvm_unreachable("Undefined target");
  }

  llvmMod.setTargetTriple(llvm::Triple::normalize(triple));
  llvmMod.setDataLayout(tgtMach.createDataLayout());
}

/// Sets the optimization level of the llvm::TargetMachine.
static void setTgtMachOpt(llvm::TargetMachine *tgtMach, char levelChar) {
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

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::TargetMachine, LLVMTargetMachineRef)

static solidity::mlirgen::Output genEvmBytecode(llvm::Module &creationMod,
                                                llvm::Module &runtimeMod,
                                                llvm::StringRef creationObjName,
                                                llvm::StringRef runtimeObjName,
                                                llvm::TargetMachine &tgtMach) {
  LLVMTargetMachineRef tgtMachWrapped = wrap(&tgtMach);
  LLVMMemoryBufferRef objs[2];
  const char *objIds[2];
  char *errMsg = nullptr;

  // Generate creation obj.
  LLVMModuleRef creationModWrapped = wrap(&creationMod);
  if (LLVMTargetMachineEmitToMemoryBuffer(tgtMachWrapped, creationModWrapped,
                                          LLVMObjectFile, &errMsg, &objs[0]))
    llvm_unreachable(errMsg);
  objIds[0] = creationObjName.data();

  // Generate runtime obj.
  LLVMModuleRef runtimeModWrapped = wrap(&runtimeMod);
  if (LLVMTargetMachineEmitToMemoryBuffer(tgtMachWrapped, runtimeModWrapped,
                                          LLVMObjectFile, &errMsg, &objs[1]))
    llvm_unreachable(errMsg);
  objIds[1] = runtimeObjName.data();

  // "Assemble" (see comments in LLVMAssembleEVM) them.
  LLVMMemoryBufferRef creationObj, runtimeObj;
  if (LLVMAssembleEVM(/*codeSegment=*/0, /*inBuffers=*/objs,
                      /*inBuffersIDs=*/objIds, /*inBuffersNum=*/2,
                      /*outBuffer=*/&creationObj, /*errorMessage=*/&errMsg))
    llvm_unreachable(errMsg);
  if (LLVMAssembleEVM(/*codeSegment=*/1, /*inBuffers=*/&objs[1],
                      /*inBuffersIDs=*/&objIds[1], /*inBuffersNum=*/1,
                      /*outBuffer=*/&runtimeObj, /*errorMessage=*/&errMsg))
    llvm_unreachable(errMsg);

  // Generate bytecode.
  LLVMMemoryBufferRef creationBytecode, runtimeBytecode;
  if (LLVMLinkEVM(/*inBuffer=*/creationObj, /*outBuffer=*/&creationBytecode,
                  /*linkerSymbolNames=*/nullptr, /*linkerSymbolValues=*/nullptr,
                  /*numLinkerSymbols=*/0, &errMsg))
    llvm_unreachable(errMsg);
  if (LLVMLinkEVM(/*inBuffer=*/runtimeObj, /*outBuffer=*/&runtimeBytecode,
                  /*linkerSymbolNames=*/nullptr, /*linkerSymbolValues=*/nullptr,
                  /*numLinkerSymbols=*/0, &errMsg))
    llvm_unreachable(errMsg);

  solidity::mlirgen::Output ret;
  ret.creationBytecode = llvm::unwrap(creationBytecode)->getBuffer();
  ret.runtimeBytecode = llvm::unwrap(runtimeBytecode)->getBuffer();

  LLVMDisposeMemoryBuffer(objs[0]);
  LLVMDisposeMemoryBuffer(objs[1]);
  LLVMDisposeMemoryBuffer(creationObj);
  LLVMDisposeMemoryBuffer(runtimeObj);
  LLVMDisposeMemoryBuffer(creationBytecode);
  LLVMDisposeMemoryBuffer(runtimeBytecode);

  return ret;
}

static solidity::mlirgen::Output
genEraVMBytecode(llvm::Module &llvmMod, llvm::TargetMachine &tgtMach) {
  llvm::legacy::PassManager llvmPassMgr;
  llvm::SmallString<0> outStreamData;
  llvm::raw_svector_ostream outStream(outStreamData);
  tgtMach.addPassesToEmitFile(llvmPassMgr, outStream,
                              /*DwoOut=*/nullptr,
                              llvm::CodeGenFileType::ObjectFile);
  llvmPassMgr.run(llvmMod);

  LLVMMemoryBufferRef obj = LLVMCreateMemoryBufferWithMemoryRange(
      outStream.str().data(), outStream.str().size(), "Input",
      /*RequiresNullTerminator=*/0);
  LLVMMemoryBufferRef bytecode = nullptr;
  char *errMsg = nullptr;
  if (LLVMLinkEraVM(obj, &bytecode, /*linkerSymbolNames=*/nullptr,
                    /*linkerSymbolValues=*/nullptr, /*numLinkerSymbols=*/0,
                    /*factoryDependencySymbolNames=*/nullptr,
                    /*factoryDependencySymbolValues=*/nullptr,
                    /*numFactoryDependencySymbols=*/0, &errMsg))
    llvm_unreachable(errMsg);

  solidity::mlirgen::Output ret;
  ret.creationBytecode = llvm::unwrap(bytecode)->getBuffer();
  ret.runtimeBytecode = ret.creationBytecode;

  LLVMDisposeMemoryBuffer(obj);
  LLVMDisposeMemoryBuffer(bytecode);
  return ret;
}

std::string solidity::mlirgen::printJob(JobSpec const &job,
                                        mlir::ModuleOp mod) {
  assert(job.action != Action::GenObj);

  mlir::PassManager passMgr(mod.getContext());
  llvm::LLVMContext llvmCtx;
  std::string ret;
  llvm::raw_string_ostream ss(ret);

  switch (job.action) {
  case Action::PrintInitStg:
    mod.print(ss);
    return ret;

  case Action::PrintStandardMLIR:
    assert(job.tgt != Target::Undefined);
    passMgr.addPass(mlir::createCanonicalizerPass());
    passMgr.addPass(mlir::sol::createModifierOpLoweringPass());
    passMgr.addPass(mlir::sol::createConvertSolToStandardPass(job.tgt));
    passMgr.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(passMgr.run(mod)))
      llvm_unreachable("Conversion to standard dialects failed");
    mod.print(ss);
    return ret;

  case Action::PrintLLVMIR: {
    assert(job.tgt != Target::Undefined);

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
      assert(runtimeMod);

      // TODO: Run in parallel?
      std::unique_ptr<llvm::Module> creationLlvmMod =
          genLLVMIR(creationMod, job.tgt, job.optLevel, *tgtMach, llvmCtx);
      std::unique_ptr<llvm::Module> runtimeLlvmMod =
          genLLVMIR(runtimeMod, job.tgt, job.optLevel, *tgtMach, llvmCtx);

      ss << *creationLlvmMod;
      ss << *runtimeLlvmMod;
      return ret;
    }
    case Target::EraVM: {
      std::unique_ptr<llvm::Module> llvmMod =
          genLLVMIR(mod, job.tgt, job.optLevel, *tgtMach, llvmCtx);

      ss << *llvmMod;
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
      assert(runtimeMod);

      // TODO: Run in parallel?
      std::unique_ptr<llvm::Module> creationLlvmMod =
          genLLVMIR(creationMod, job.tgt, job.optLevel, *tgtMach, llvmCtx);
      std::unique_ptr<llvm::Module> runtimeLlvmMod =
          genLLVMIR(runtimeMod, job.tgt, job.optLevel, *tgtMach, llvmCtx);

      std::string ret;
      ret = getAsm(*creationLlvmMod, *tgtMach);
      ret += getAsm(*runtimeLlvmMod, *tgtMach);
      return ret;
    }
    case Target::EraVM: {
      std::unique_ptr<llvm::Module> llvmMod =
          genLLVMIR(mod, job.tgt, job.optLevel, *tgtMach, llvmCtx);
      return getAsm(*llvmMod, *tgtMach);
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

solidity::mlirgen::Output solidity::mlirgen::bytecodeJob(JobSpec const &job,
                                                         mlir::ModuleOp mod) {
  assert(job.action == Action::GenObj);

  mlir::PassManager passMgr(mod.getContext());
  llvm::LLVMContext llvmCtx;

  // Convert the module's ir to llvm dialect.
  // FIXME: enableDI UNREACHABLE executed at
  // llvm/lib/Target/EVM/MCTargetDesc/EVMAsmBackend.cpp:112!
  addConversionPasses(passMgr, job.tgt, /*enableDI=*/false);
  if (mlir::failed(passMgr.run(mod)))
    llvm_unreachable("Conversion to llvm dialect failed");

  std::unique_ptr<llvm::TargetMachine> tgtMach = createTargetMachine(job.tgt);
  setTgtMachOpt(tgtMach.get(), job.optLevel);

  switch (job.tgt) {
  case Target::EVM: {
    auto creationMod = mod;
    mlir::ModuleOp runtimeMod = extractRuntimeModule(creationMod);
    assert(runtimeMod);

    // TODO: Run in parallel?
    std::unique_ptr<llvm::Module> creationLlvmMod =
        genLLVMIR(creationMod, job.tgt, job.optLevel, *tgtMach, llvmCtx);
    std::unique_ptr<llvm::Module> runtimeLlvmMod =
        genLLVMIR(runtimeMod, job.tgt, job.optLevel, *tgtMach, llvmCtx);

    assert(creationMod.getName() && runtimeMod.getName());
    return genEvmBytecode(*creationLlvmMod, *runtimeLlvmMod,
                          *creationMod.getName(), *runtimeMod.getName(),
                          *tgtMach);
  }
  case Target::EraVM: {
    std::unique_ptr<llvm::Module> llvmMod =
        genLLVMIR(mod, job.tgt, job.optLevel, *tgtMach, llvmCtx);
    return genEraVMBytecode(*llvmMod, *tgtMach);
  }
  default:
    break;
  };
  llvm_unreachable("Invalid target");
}
