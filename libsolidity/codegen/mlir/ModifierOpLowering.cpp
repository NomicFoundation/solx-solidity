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
// A pass that lowers sol.modifier and sol.modifier_call_blk ops
//

// TODO: This pass expects/generates non-mem2reg ir.

#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Sol/SolOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace mlir;

struct ModifierOpLowering
    : public PassWrapper<ModifierOpLowering, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModifierOpLowering)

  // TODO: Move this to general utils.
  StringAttr getNearestUnusedSymFrom(Operation *op, StringAttr sym) {
    if (!SymbolTable::lookupNearestSymbolFrom(op, sym))
      return sym;

    unsigned i = 0;
    StringAttr newSym;
    do {
      newSym = StringAttr::get(op->getContext(),
                               sym.getValue() + std::to_string(i++));
    } while (SymbolTable::lookupNearestSymbolFrom(op, newSym));
    return newSym;
  }

  //
  // Terminologies, etc.
  //
  // - Modifier: The modifier.
  // - Caller function: The function that calls the modifier.
  // - Modifier function: The function that contains the body of the modifier
  // (in the context of a sol.modifier_call_blk).
  //
  // The caller function and the modifier function has the same signature. We
  // generate a modifier function for each sol.modifier_call_blk even if the
  // same modifier is called (since the call args can be different).

  /// Creates and returns a function that contains the body of the modifier in
  /// the context of the sol.modifier_call_blk. This removes the
  /// sol.modifier_call_blk as well.
  sol::FuncOp createModifierFn(sol::ModifierCallBlkOp modifierCallBlk,
                               OpBuilder b) {
    OpBuilder::InsertionGuard guard(b);

    auto modifierCall = cast<sol::CallOp>(modifierCallBlk.getBody()->back());
    auto modifier = cast<sol::ModifierOp>(SymbolTable::lookupNearestSymbolFrom(
        modifierCall, modifierCall.getCalleeAttr()));
    auto callerFn = cast<sol::FuncOp>(modifierCallBlk->getParentOp());
    Block &callerFnEntry = callerFn.getBody().front();

    // Record the respective block argument of all the stack loads in the
    // sol.modifier_call_blk.
    llvm::DenseMap<sol::LoadOp, unsigned> modifierCallBlkArgMap;
    modifierCallBlk.getBody()->walk([&](sol::LoadOp ld) {
      bool foundBlkArg = false;
      if (auto allocaAddr =
              dyn_cast<sol::AllocaOp>(ld.getAddr().getDefiningOp())) {
        for (Operation *user : allocaAddr->getUsers()) {
          auto st = dyn_cast<sol::StoreOp>(user);
          if (st && st->getBlock() == &callerFnEntry &&
              isa<BlockArgument>(st.getVal())) {
            foundBlkArg = true;
            modifierCallBlkArgMap[ld] =
                cast<BlockArgument>(st.getVal()).getArgNumber();
          }
        }
        assert(foundBlkArg);
      }
    });

    // Create the modifier function with caller function signature.
    StringRef modifierFnName =
        getNearestUnusedSymFrom(modifier, modifier.getNameAttr()).getValue();
    b.setInsertionPoint(modifier);
    auto modifierFn = b.create<sol::FuncOp>(modifier.getLoc(), modifierFnName,
                                            callerFn.getFunctionType());

    // Clone the modifier's body and move the the sol.modifier_call_blk into the
    // modifier function.
    IRMapping mapper;
    modifier.getBody().cloneInto(&modifierFn.getBody(), mapper);
    Block &modifierFnEntry = modifierFn.getBody().front();
    modifierFnEntry.getOperations().splice(
        modifierFnEntry.begin(), modifierCallBlk.getBody()->getOperations());

    // Replace uses of the original modifier args with the operands of the
    // modifier call.
    assert(modifierCall.getArgOperands().size() ==
           modifierFnEntry.getNumArguments());
    for (auto blkArg : modifierFnEntry.getArguments()) {
      auto respectiveModifierCallArg =
          modifierCall.getArgOperands()[blkArg.getArgNumber()];
      blkArg.replaceAllUsesWith(respectiveModifierCallArg);
    }
    modifierCall.erase();

    // Change the block arguments in the modifier function to conform to the
    // caller function signature.
    modifierFnEntry.eraseArguments(0, modifierFnEntry.getNumArguments());
    for (BlockArgument arg : callerFnEntry.getArguments())
      modifierFnEntry.addArgument(arg.getType(), arg.getLoc());

    // Replace stack load uses with the respective block argument.
    for (auto i : modifierCallBlkArgMap) {
      sol::LoadOp allocaLd = i.first;
      unsigned respectiveBlkArgNum = i.second;
      allocaLd.replaceAllUsesWith(
          modifierFnEntry.getArgument(respectiveBlkArgNum));
      allocaLd.erase();
    }

    modifierCallBlk.erase();
    return modifierFn;
  }

  /// Replaces the sol.placeholders in the modifier function with a call to
  /// callee. This generates the return arg handling.  The callee is expected to
  /// have the same type of the modifier function as we forward args in the
  /// modifier function to the callee.
  void replacePlaceholders(sol::FuncOp modifierFn, sol::FuncOp callee,
                           OpBuilder b) {
    OpBuilder::InsertionGuard guard(b);

    FunctionType modifierFnTy = modifierFn.getFunctionType();
    assert(modifierFnTy == callee.getFunctionType());
    assert(modifierFnTy.getNumResults() <= 1 && "NYI");

    // Generate the alloca for the return arg.
    sol::AllocaOp retAddr;
    if (modifierFnTy.getNumResults() == 1) {
      b.setInsertionPointToStart(&modifierFn.getBlocks().front());
      retAddr = b.create<sol::AllocaOp>(
          modifierFn.getLoc(),
          sol::PointerType::get(b.getContext(), modifierFnTy.getResult(0),
                                sol::DataLocation::Stack));
    }

    // Replace sol.placeholders with a call to the callee.
    modifierFn.walk([&](sol::PlaceholderOp placeholder) {
      b.setInsertionPoint(placeholder);
      auto call = b.create<sol::CallOp>(placeholder.getLoc(), callee,
                                        modifierFn.getArguments());
      if (modifierFnTy.getNumResults() == 1)
        b.create<sol::StoreOp>(placeholder.getLoc(), call.getResult(0),
                               retAddr);
      placeholder.erase();
    });

    // Add the return arg in the sol.returns.
    if (modifierFnTy.getNumResults() == 1) {
      modifierFn.walk([&](sol::ReturnOp ret) {
        b.setInsertionPoint(ret);
        auto retVal = b.create<sol::LoadOp>(ret.getLoc(), retAddr);
        b.create<sol::ReturnOp>(ret.getLoc(), retVal.getResult());
        ret.erase();
      });
    }
  }

  /// Lowers all the modifier calls in the function.
  void lowerModifierCalls(sol::FuncOp callerFn) {
    if (callerFn.getBlocks().empty())
      return;

    OpBuilder b(callerFn.getContext());

    // Create modifier functions from the sol.modifier_call_blks.
    // TODO: We could .reserve here if we track the modifier count.
    SmallVector<sol::FuncOp, 4> modifierFns;
    Block &entryBlk = callerFn.getBlocks().front();
    for (auto &op : llvm::make_early_inc_range(entryBlk)) {
      auto modifierCallBlk = dyn_cast<sol::ModifierCallBlkOp>(op);
      if (!modifierCallBlk)
        continue;
      modifierFns.push_back(createModifierFn(modifierCallBlk, b));
    }
    if (modifierFns.empty())
      return;

    // Replace `callerFn` with a new function that calls the first modifier.
    sol::FuncOp newCallerFn = callerFn.cloneWithoutRegions();
    b.setInsertionPoint(callerFn);
    b.insert(newCallerFn);
    callerFn.setSymName(
        getNearestUnusedSymFrom(callerFn, callerFn.getSymNameAttr()));
    b.setInsertionPointToStart(newCallerFn.addEntryBlock());
    // The sol.placeholder will be replaced with the first modifier. This
    // simplifies the following placeholder replacement loop.
    b.create<sol::PlaceholderOp>(b.getUnknownLoc());
    b.create<sol::ReturnOp>(b.getUnknownLoc());

    // Generate the chain of calls of modifiers and the caller function by
    // replacing the sol.placeholders.
    SmallVector<sol::FuncOp, 4> callOrder{newCallerFn};
    callOrder.reserve(modifierFns.size() + 1);
    callOrder.append(modifierFns);
    callOrder.push_back(callerFn);
    for (auto *it = callOrder.begin(); it != callOrder.end(); ++it) {
      if (it + 1 != callOrder.end())
        replacePlaceholders(*it, *(it + 1), b);
    }
  }

  void runOnOperation() override {
    getOperation().walk([&](sol::FuncOp fn) { lowerModifierCalls(fn); });
    getOperation().walk([&](sol::ModifierOp modifier) { modifier.erase(); });
    // getOperation().dump();
  }

  StringRef getArgument() const override { return "lower-modifier"; }
};

std::unique_ptr<Pass> sol::createModifierOpLoweringPass() {
  return std::make_unique<ModifierOpLowering>();
}
