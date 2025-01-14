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

#include "liblangutil/EVMVersion.h"
#include "libsolidity/SolidityExecutionFramework.h"
#include "libsolutil/Common.h"
#include "libsolutil/CommonData.h"
#include <cstdio>
#include <iostream>
#include <optional>

using namespace solidity;

// FIXME:
#define VM_PATH "/home/abinavpp/pj/build/evmone/lib/libevmone.so"

struct ExecFramework: frontend::test::SolidityExecutionFramework
{
	ExecFramework(): SolidityExecutionFramework(langutil::EVMVersion::cancun(), std::nullopt, {VM_PATH}) {}

	void dump(bytes b) { std::cerr << util::toHex(b) << "\n"; }

	void run()
	{
		std::string src = R"(
		contract C {
			function f() public returns (uint) { return 42; }
		}
	  )";

		// m_optimiserSettings = solidity::frontend::OptimiserSettings::full();
		// m_compileViaYul = true;

		m_compileViaMlir = true;

		m_showMessages = true;
		// FIXME: How do I query the m_vm of EVMHost?
		m_evmcHost->getVM(VM_PATH).set_option("trace", "yes");

		// OK:
		// bytes bytecode = compileContract(src, "C");
		// dump(bytecode);
		// sendMessage(bytecode, /*_isCreation=*/false);
		// dump(callContractFunction("f()"));

		compileAndRun(src, 0, "C");
		callContractFunction("f()");
	}
};

int main()
{
	auto options = std::make_unique<test::CommonOptions>();
	// options->validate();
	test::CommonOptions::setSingleton(std::move(options));

	ExecFramework exec;
	exec.run();
}
