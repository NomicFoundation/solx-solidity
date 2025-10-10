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
// Sol dialect loop invariant code motion pass.
//

#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Sol/Sol.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

using namespace mlir;

struct LoopInvariantCodeMotion
    : public PassWrapper<LoopInvariantCodeMotion, OperationPass<>> {
  LoopInvariantCodeMotion() = default;
  LoopInvariantCodeMotion(const LoopInvariantCodeMotion &) : PassWrapper() {}

  Statistic count{this, "count", "Number of ops licm'd"};

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      if (isa<scf::ForOp, scf::WhileOp, sol::DoWhileOp>(op))
        run(cast<LoopLikeOpInterface>(op));
    });
  }

  SmallVector<MemoryEffects::EffectInstance> getMemEffs(Operation *op) {
    SmallVector<MemoryEffects::EffectInstance> effs;
    if (auto effIfc = dyn_cast<MemoryEffectOpInterface>(op))
      effIfc.getEffects(effs);
    return effs;
  }

  /// Returns true if we can statically determine whether the loop has 1 or more
  /// iterations.
  bool loopsAtLeastOnce(LoopLikeOpInterface op) {
    // sol.do_while that has no break/continue/return
    if (isa<sol::DoWhileOp>(op))
      return op.getLoopRegions().size() == 1;

    // Statically check if scf.for iterates once.
    if (auto forOp = dyn_cast<scf::ForOp>(op.getOperation())) {
      auto lb = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
      auto ub = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
      auto step = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
      if (lb && ub && step)
        return llvm::divideCeil(ub.value() - lb.value(), step.value()) > 0;
      return false;
    }

    // Statically check if scf.while iterates once.
    if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
      auto *terminator = &whileOp.getBefore().front().back();
      auto cond = cast<scf::ConditionOp>(terminator);
      if (auto constOp = cond.getCondition().getDefiningOp<arith::ConstantOp>())
        if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue()))
          return boolAttr.getValue();
      return false;
    }

    return false;
  }

  void run(LoopLikeOpInterface loopLike) {
    // Collect resources that are written to in the loop.
    llvm::SmallDenseSet<SideEffects::Resource *, 4> writtenResources;
    loopLike->walk([&](Operation *op) {
      SmallVector<MemoryEffects::EffectInstance> effs = getMemEffs(op);
      for (auto &eff : effs)
        if (isa<MemoryEffects::Write>(eff.getEffect()))
          writtenResources.insert(eff.getResource());
    });

    bool knownToLoopAtLeastOnce = loopsAtLeastOnce(loopLike);

    count += moveLoopInvariantCode(
        loopLike.getLoopRegions(),
        /*isDefinedOutsideRegion=*/
        [&](Value value, Region *) {
          return loopLike.isDefinedOutsideOfLoop(value);
        },
        /*shouldMoveOutOfRegion=*/
        [&](Operation *op, Region *) {
          if (isPure(op))
            return true;
          if (!isSpeculatable(op))
            return false;
          if (!knownToLoopAtLeastOnce)
            return false;

          // Don’t hoist ops that read from resources written inside the loop.
          SmallVector<MemoryEffects::EffectInstance> effs = getMemEffs(op);
          for (auto &eff : effs) {
            // Speculative ops have memory effects iff it reads.
            assert(isa<MemoryEffects::Read>(eff.getEffect()));
            if (writtenResources.contains(eff.getResource()))
              return false;
          }
          return true;
        },
        /*moveOutOfRegion=*/
        [&](Operation *op, Region *) { loopLike.moveOutOfLoop(op); });
  }

  StringRef getArgument() const override { return "sol-licm"; }
};

std::unique_ptr<Pass> sol::createLoopInvariantCodeMotionPass() {
  return std::make_unique<LoopInvariantCodeMotion>();
}
