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
// Yul dialect free-ptr fusing pass.
//

#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Yul/Yul.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

struct FuseFreePtr : public PassWrapper<FuseFreePtr, OperationPass<>> {
  StringRef getArgument() const override { return "sol-fuse-free-ptr"; }
  Statistic count{this, "count", "Number free-ptr updates fused"};

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      if (auto fnOp = dyn_cast<FunctionOpInterface>(op)) {
        for (Block &blk : fnOp.getFunctionBody()) {
          run(blk);
        }
      }
    });
  }

  void run(Block &blk) {
    if (blk.empty())
      return;

    SmallVector<yul::UpdFreePtrOp, 4> updOps;
    SmallPtrSet<Operation *, 4> updOpsUsers;

    auto fuseAndReset = [&]() {
      fuse(updOps);

      // Start over again.
      updOps.clear();
      updOpsUsers.clear();
    };

    for (Operation &op : llvm::make_early_inc_range(blk)) {
      if (updOpsUsers.contains(&op)) {
        fuseAndReset();
        continue;
      }
      if (auto updOp = dyn_cast<yul::UpdFreePtrOp>(op)) {
        for (Operation *user : updOp->getUsers())
          updOpsUsers.insert(user);
        updOps.push_back(updOp);
      }
    }
    fuse(updOps);
  }

  /// Fuses all the UpdFreePtrOps into one at the last op so that the use-graph
  /// is not broken. This expects the user-graph to be dominated by the last op.
  void fuse(ArrayRef<yul::UpdFreePtrOp> updOps) {
    if (updOps.size() < 2)
      return;

    // Generate the total size of allocated area after all the UpdFreePtrOps.
    count += updOps.size();
    yul::UpdFreePtrOp firstUpd = updOps.front();
    IRRewriter r(firstUpd);
    SmallVector<Location> fusedLocs{firstUpd.getLoc()};
    mlir::Value totalSize = firstUpd.getSize();
    for (yul::UpdFreePtrOp updOp : llvm::drop_begin(updOps)) {
      fusedLocs.push_back(updOp.getLoc());
      r.setInsertionPoint(updOp);
      totalSize =
          r.create<arith::AddIOp>(updOp.getLoc(), updOp.getSize(), totalSize);
    }

    // Generate the fused UpdFreePtrOp. The users of this will be the users of
    // the original first UpdFreePtrOp.
    auto fusedUpd = r.create<yul::UpdFreePtrOp>(
        FusedLoc::get(r.getContext(), fusedLocs), totalSize);
    r.replaceAllOpUsesWith(updOps.front(), fusedUpd);

    // The remaining UpdFreePtrOps are generated as add ops and the users are
    // updated accordingly.
    yul::UpdFreePtrOp prevUpdOp = updOps.front();
    mlir::Value prevFreePtr = fusedUpd;
    for (yul::UpdFreePtrOp updOp : llvm::drop_begin(updOps)) {
      auto add = r.create<arith::AddIOp>(updOp.getLoc(), prevFreePtr,
                                         prevUpdOp.getSize());
      r.replaceAllOpUsesWith(updOp, add);
      prevFreePtr = add;
      prevUpdOp = updOp;
    }

    // Erase all original UpdFreePtrOps.
    for (yul::UpdFreePtrOp updOp : updOps)
      r.eraseOp(updOp);
  }

  FuseFreePtr() = default;
  FuseFreePtr(const FuseFreePtr &other) : PassWrapper(other) {}
};

std::unique_ptr<Pass> sol::createFuseFreePtrPass() {
  return std::make_unique<FuseFreePtr>();
}
