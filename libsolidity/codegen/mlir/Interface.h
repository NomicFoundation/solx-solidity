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
// MLIR codegen interface
//

#pragma once

#include "liblangutil/EVMVersion.h"
#include "libsolutil/FixedHash.h"
#include <string>
#include <vector>

namespace solidity::langutil {
class CharStream;
} // namespace solidity::langutil

namespace solidity::yul {
class Dialect;
class Object;
class AST;
struct Identifier;
}; // namespace solidity::yul

namespace mlir {
class Value;
class OpBuilder;
}; // namespace mlir

namespace solidity::mlirgen {

enum class Action {
  /// Print the MLIR generated from the AST.
  PrintInitStg,

  /// Print the standard MLIR after lowering dialect(s) in solc.
  PrintStandardMLIR,

  /// Print the LLVM-IR.
  PrintLLVMIR,

  /// Print the assembly.
  PrintAsm,

  /// Print the object.
  PrintObj,

  /// Generate the object.
  GenObj,

  Undefined,
};

constexpr bool requiresLinking(Action a) {
  switch (a) {
  case Action::GenObj:
  case Action::PrintObj:
    return true;
  default:
    return false;
  }
}

enum class Target {
  EVM,
  EraVM,
  Undefined,
};

/// Return the enum Target from the string (case insensitive).
Target strToTarget(std::string const &str);

struct JobSpec {
  // TODO: Add other codegen info like debug info, output file?
  Action action = Action::Undefined;
  Target tgt = Target::EVM;
  char optLevel = '3';
};

struct Bytecode {
  std::string creation;
  std::string runtime;
};

/// Registers required command line options in the MLIR framework
extern void registerMLIRCLOpts();

/// Parses command line options in `argv` for the MLIR framework
extern bool parseMLIROpts(std::vector<const char *> &argv);

extern void
runYulToMLIRPass(yul::AST const &, langutil::CharStream const &,
                 std::function<mlir::Value(yul::Identifier const *)> const &,
                 mlir::OpBuilder &);
extern Bytecode runYulToMLIRPass(yul::Object const &,
                                 langutil::CharStream const &,
                                 yul::Dialect const &, JobSpec const &,
                                 langutil::EVMVersion,
                                 std::map<std::string, util::h160> const &);

} // namespace solidity::mlirgen
