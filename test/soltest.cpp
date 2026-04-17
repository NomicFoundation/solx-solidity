/*
	This file is part of solidity.

	solidity is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	solidity is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with solidity.  If not, see <http://www.gnu.org/licenses/>.
*/
// SPDX-License-Identifier: GPL-3.0
/** @file boostTest.cpp
 * @author Marko Simovic <markobarko@gmail.com>
 * @date 2014
 * Stub for generating main boost.test module.
 * Original code taken from boost sources.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4535) // calling _set_se_translator requires /EHa
#endif
#include <boost/test/unit_test.hpp>
#include <boost/test/tree/traverse.hpp>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#pragma GCC diagnostic pop
#pragma GCC diagnostic pop

#include <test/InteractiveTests.h>
#include <test/Common.h>
#include <test/EVMHost.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>

#include <array>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>

using namespace boost::unit_test;
using namespace solidity::frontend::test;
namespace fs = boost::filesystem;

namespace
{
enum class IsolatedRunStatus
{
	Success,
	Failure,
	FatalError,
	Exception,
	Skipped,
};

struct IsolatedRunResult
{
	IsolatedRunStatus status = IsolatedRunStatus::Success;
	std::string message;
};

std::string readAllFromFd(int _fd)
{
	std::string content;
	char buffer[4096];
	while (true)
	{
		ssize_t bytesRead = ::read(_fd, buffer, sizeof(buffer));
		if (bytesRead == 0)
			break;
		if (bytesRead < 0)
		{
			if (errno == EINTR)
				continue;
			break;
		}
		content.append(buffer, static_cast<size_t>(bytesRead));
	}
	return content;
}

void writeAllToFd(int _fd, std::string const& _content)
{
	size_t totalWritten = 0;
	while (totalWritten < _content.size())
	{
		ssize_t written = ::write(
			_fd,
			_content.data() + totalWritten,
			_content.size() - totalWritten
		);
		if (written < 0)
		{
			if (errno == EINTR)
				continue;
			return;
		}
		totalWritten += static_cast<size_t>(written);
	}
}

std::string readFile(fs::path const& _path)
{
	std::ifstream input(_path.string(), std::ios::binary);
	if (!input)
		return {};
	return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string serializeResult(IsolatedRunResult const& _result)
{
	return std::to_string(static_cast<int>(_result.status)) + "\n" + _result.message;
}

std::optional<IsolatedRunResult> deserializeResult(std::string const& _payload)
{
	if (_payload.empty())
		return std::nullopt;

	size_t newline = _payload.find('\n');
	if (newline == std::string::npos)
		return std::nullopt;

	IsolatedRunResult result;
	try
	{
		result.status = static_cast<IsolatedRunStatus>(std::stoi(_payload.substr(0, newline)));
	}
	catch (...)
	{
		return std::nullopt;
	}
	result.message = _payload.substr(newline + 1);
	return result;
}

std::string appendChildOutput(std::string _message, std::string const& _childOutput)
{
	if (_childOutput.empty())
		return _message;

	if (!_message.empty() && _message.back() != '\n')
		_message += "\n";
	_message += "Child process output:\n";
	_message += _childOutput;
	return _message;
}

void restoreDefaultSignalHandlers()
{
	struct sigaction defaultAction;
	std::memset(&defaultAction, 0, sizeof(defaultAction));
	defaultAction.sa_handler = SIG_DFL;
	sigemptyset(&defaultAction.sa_mask);

	// Boost installs handlers in the parent. Reset them in the child so a hard
	// backend failure terminates the isolated worker instead of unwinding through
	// inherited Boost state.
	std::array<int, 6> const signalsToReset{SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGPIPE, SIGSEGV};
	for (int signalNumber: signalsToReset)
		::sigaction(signalNumber, &defaultAction, nullptr);
}

[[noreturn]] void runTestCaseInChild(
	TestCase::Config const& _config,
	TestCase::TestCaseCreator const& _testCaseCreator,
	int _writeFd,
	fs::path const& _logPath
)
{
	restoreDefaultSignalHandlers();

	int logFd = ::open(_logPath.string().c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
	if (logFd >= 0)
	{
		::dup2(logFd, STDOUT_FILENO);
		::dup2(logFd, STDERR_FILENO);
		::close(logFd);
	}

	IsolatedRunResult result;
	try
	{
		std::stringstream errorStream;
		auto testCase = _testCaseCreator(_config);
		if (!testCase->shouldRun())
			result.status = IsolatedRunStatus::Skipped;
		else
			// The child only computes the result payload. The parent process must
			// translate it into Boost failures for the suite run.
			switch (testCase->run(errorStream))
			{
				case TestCase::TestResult::Success:
					result.status = IsolatedRunStatus::Success;
					break;
				case TestCase::TestResult::Failure:
					result.status = IsolatedRunStatus::Failure;
					result.message = errorStream.str();
					break;
				case TestCase::TestResult::FatalError:
					result.status = IsolatedRunStatus::FatalError;
					result.message = errorStream.str();
					break;
			}
	}
	catch (...)
	{
		result.status = IsolatedRunStatus::Exception;
		result.message = boost::current_exception_diagnostic_information();
	}

	writeAllToFd(_writeFd, serializeResult(result));
	::close(_writeFd);
	::_exit(0);
}

void removeTestSuite(std::string const& _name)
{
	master_test_suite_t& master = framework::master_test_suite();
	auto id = master.get(_name);
	soltestAssert(id != INV_TEST_UNIT_ID, "Removing non-existent test suite!");
	master.remove(id);
}

/**
 * Class that traverses the boost test tree and removes unit tests that are
 * not in the current batch.
 */
class BoostBatcher: public test_tree_visitor
{
public:
	BoostBatcher(solidity::test::Batcher& _batcher):
		m_batcher(_batcher)
	{}

	void visit(test_case const& _testCase) override
	{
		if (!m_batcher.checkAndAdvance())
			// disabling them would be nicer, but it does not work like this:
			// const_cast<test_case&>(_testCase).p_run_status.value = test_unit::RS_DISABLED;
			m_path.back()->remove(_testCase.p_id);
	}
	bool test_suite_start(test_suite const& _testSuite) override
	{
		m_path.push_back(&const_cast<test_suite&>(_testSuite));
		return test_tree_visitor::test_suite_start(_testSuite);
	}
	void test_suite_finish(test_suite const& _testSuite) override
	{
		m_path.pop_back();
		test_tree_visitor::test_suite_finish(_testSuite);
	}

private:
	solidity::test::Batcher& m_batcher;
	std::vector<test_suite*> m_path;
};


void runTestCase(TestCase::Config const& _config, TestCase::TestCaseCreator const& _testCaseCreator)
{
	try
	{
		std::stringstream errorStream;
		auto testCase = _testCaseCreator(_config);
		if (testCase->shouldRun())
			switch (testCase->run(errorStream))
			{
				case TestCase::TestResult::Success:
					break;
				case TestCase::TestResult::Failure:
					BOOST_ERROR("Test expectation mismatch.\n" + errorStream.str());
					break;
				case TestCase::TestResult::FatalError:
					BOOST_ERROR("Fatal error during test.\n" + errorStream.str());
					break;
			}
	}
	catch (...)
	{
		BOOST_ERROR("Exception during extracted test: " << boost::current_exception_diagnostic_information());
	}
}

void runTestCaseIsolated(TestCase::Config const& _config, TestCase::TestCaseCreator const& _testCaseCreator)
{
	int pipeFds[2];
	if (::pipe(pipeFds) != 0)
	{
		BOOST_ERROR("Failed to create isolation pipe: " + std::string(std::strerror(errno)));
		return;
	}

	fs::path logPath = fs::temp_directory_path() / fs::unique_path("soltest-isolated-%%%%-%%%%-%%%%.log");
	std::fflush(nullptr);

	pid_t pid = ::fork();
	if (pid < 0)
	{
		::close(pipeFds[0]);
		::close(pipeFds[1]);
		BOOST_ERROR("Failed to fork isolated test: " + std::string(std::strerror(errno)));
		return;
	}

	if (pid == 0)
	{
		::close(pipeFds[0]);
		runTestCaseInChild(_config, _testCaseCreator, pipeFds[1], logPath);
	}

	::close(pipeFds[1]);
	std::string payload = readAllFromFd(pipeFds[0]);
	::close(pipeFds[0]);

	int waitStatus = 0;
	while (::waitpid(pid, &waitStatus, 0) < 0)
	{
		if (errno != EINTR)
		{
			waitStatus = -1;
			break;
		}
	}

	std::string childOutput = readFile(logPath);
	boost::system::error_code removeError;
	fs::remove(logPath, removeError);

	if (waitStatus == -1)
	{
		BOOST_ERROR("Failed waiting for isolated test child: " + std::string(std::strerror(errno)));
		return;
	}

	if (WIFSIGNALED(waitStatus))
	{
		int signalNumber = WTERMSIG(waitStatus);
		std::string message = "Isolated test terminated by signal " + std::to_string(signalNumber);
		if (char const* signalName = ::strsignal(signalNumber))
			message += " (" + std::string(signalName) + ")";
		BOOST_ERROR(appendChildOutput(message, childOutput));
		return;
	}

	if (!WIFEXITED(waitStatus))
	{
		BOOST_ERROR(appendChildOutput("Isolated test terminated unexpectedly.", childOutput));
		return;
	}

	// Reporting in the child would be lost when the isolated worker exits, so
	// the parent owns the final Boost bookkeeping.
	auto result = deserializeResult(payload);
	if (!result.has_value())
	{
		BOOST_ERROR(appendChildOutput(
			"Isolated test exited without a valid result payload.",
			childOutput
		));
		return;
	}

	switch (result->status)
	{
	case IsolatedRunStatus::Success:
	case IsolatedRunStatus::Skipped:
		break;
	case IsolatedRunStatus::Failure:
		BOOST_ERROR("Test expectation mismatch.\n" + appendChildOutput(result->message, childOutput));
		break;
	case IsolatedRunStatus::FatalError:
		BOOST_ERROR("Fatal error during test.\n" + appendChildOutput(result->message, childOutput));
		break;
	case IsolatedRunStatus::Exception:
		BOOST_ERROR("Exception during extracted test: " + appendChildOutput(result->message, childOutput));
		break;
	}
}

int registerTests(
	boost::unit_test::test_suite& _suite,
	boost::filesystem::path const& _basepath,
	boost::filesystem::path const& _path,
	std::vector<std::string> const& _labels,
	TestCase::TestCaseCreator _testCaseCreator,
	solidity::test::Batcher& _batcher
)
{
	int numTestsAdded = 0;
	fs::path fullpath = _basepath / _path;
	TestCase::Config config{
		fullpath.string(),
		solidity::test::CommonOptions::get().evmVersion(),
		solidity::test::CommonOptions::get().eofVersion(),
		solidity::test::CommonOptions::get().vmPaths,
		solidity::test::CommonOptions::get().enforceGasTest,
		solidity::test::CommonOptions::get().enforceGasTestMinValue,
	};
	if (fs::is_directory(fullpath))
	{
		test_suite* sub_suite = BOOST_TEST_SUITE(_path.filename().string());
		for (auto const& entry: boost::iterator_range<fs::directory_iterator>(
			fs::directory_iterator(fullpath),
			fs::directory_iterator()
		))
			if (
				solidity::test::isValidSemanticTestPath(entry) &&
				(fs::is_directory(entry.path()) || TestCase::isTestFilename(entry.path().filename()))
			)
				numTestsAdded += registerTests(
					*sub_suite,
					_basepath, _path / entry.path().filename(),
					_labels,
					_testCaseCreator,
					_batcher
				);
		_suite.add(sub_suite);
	}
	else
	{
		// TODO would be better to set the test to disabled.
		if (_batcher.checkAndAdvance())
		{
			// This must be a vector of unique_ptrs because Boost.Test keeps the equivalent of a string_view to the filename
			// that is passed in. If the strings were stored directly in the vector, pointers/references to them would be
			// invalidated on reallocation.
			static std::vector<std::unique_ptr<std::string const>> filenames;

			filenames.emplace_back(std::make_unique<std::string>(_path.string()));
			auto test_case = make_test_case(
				[config, _testCaseCreator]
				{
					BOOST_REQUIRE_NO_THROW({
						if (solidity::test::CommonOptions::get().isolateTests)
							runTestCaseIsolated(config, _testCaseCreator);
						else
							runTestCase(config, _testCaseCreator);
					});
				},
				_path.stem().string(),
				*filenames.back(),
				0
			);
			for (auto const& _label: _labels)
				test_case->add_label(_label);
			_suite.add(test_case);
			numTestsAdded = 1;
		}
	}
	return numTestsAdded;
}

bool initializeOptions()
{
	auto const& suite = boost::unit_test::framework::master_test_suite();

	auto options = std::make_unique<solidity::test::CommonOptions>();
	bool shouldContinue = options->parse(suite.argc, suite.argv);
	if (!shouldContinue)
		return false;
	options->validate();

	solidity::test::CommonOptions::setSingleton(std::move(options));
	return true;
}

}

// TODO: Prototype -- why isn't this declared in the boost headers?
// TODO: replace this with a (global) fixture.
test_suite* init_unit_test_suite(int /*argc*/, char* /*argv*/[]);

test_suite* init_unit_test_suite(int /*argc*/, char* /*argv*/[])
{
	using namespace solidity::test;

	master_test_suite_t& master = framework::master_test_suite();
	master.p_name.value = "SolidityTests";

	try
	{
		bool shouldContinue = initializeOptions();
		if (!shouldContinue)
			exit(EXIT_SUCCESS);

		if (!solidity::test::loadVMs(solidity::test::CommonOptions::get()))
			exit(EXIT_FAILURE);

		if (solidity::test::CommonOptions::get().disableSemanticTests)
			std::cout << std::endl << "--- SKIPPING ALL SEMANTICS TESTS ---" << std::endl << std::endl;

		Batcher batcher(CommonOptions::get().selectedBatch, CommonOptions::get().batches);
		if (CommonOptions::get().batches > 1)
			std::cout << "Batch " << CommonOptions::get().selectedBatch << " out of " << CommonOptions::get().batches << std::endl;

		// Batch the boost tests
		BoostBatcher boostBatcher(batcher);
		traverse_test_tree(master, boostBatcher, true);

		// Include the interactive tests in the automatic tests as well
		for (auto const& ts: g_interactiveTestsuites)
		{
			auto const& options = solidity::test::CommonOptions::get();

			if (ts.smt && options.disableSMT)
				continue;

			if (ts.needsVM && solidity::test::CommonOptions::get().disableSemanticTests)
				continue;

			//TODO
			//solAssert(
			registerTests(
				master,
				options.testPath / ts.path,
				ts.subpath,
				ts.labels,
				ts.testCaseCreator,
				batcher
			);
			// > 0, std::string("no ") + ts.title + " tests found");
		 }

		if (solidity::test::CommonOptions::get().disableSemanticTests)
		{
			for (auto suite: {
				"ABIDecoderTest",
				"ABIEncoderTest",
				"SolidityAuctionRegistrar",
				"SolidityWallet",
				"GasMeterTests",
				"GasCostTests",
				"SolidityEndToEndTest",
				"SolidityOptimizer"
			})
				removeTestSuite(suite);
		}
	}
	catch (solidity::test::ConfigException const& exception)
	{
		std::cerr << exception.what() << std::endl;
		exit(EXIT_FAILURE);
	}
	catch (std::runtime_error const& exception)
	{
		std::cerr << exception.what() << std::endl;
		exit(EXIT_FAILURE);
	}

	return nullptr;
}

// BOOST_TEST_DYN_LINK should be defined if user want to link against shared boost test library
#ifdef BOOST_TEST_DYN_LINK

// Because we want to have customized initialization function and support shared boost libraries at the same time,
// we are forced to customize the entry point.
// see: https://www.boost.org/doc/libs/1_67_0/libs/test/doc/html/boost_test/adv_scenarios/shared_lib_customizations/init_func.html

int main(int argc, char* argv[])
{
	auto init_unit_test = []() -> bool { init_unit_test_suite(0, nullptr); return true; };
	return boost::unit_test::unit_test_main(init_unit_test, argc, argv);
}

#endif
