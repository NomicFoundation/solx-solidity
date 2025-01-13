add_executable(soltest-mlir
	Main.cpp
	../../Common.cpp
	../../EVMHost.cpp
	../../ExecutionFramework.cpp
	../../ExecutionFramework.h
	../../libsolidity/util/Common.cpp
	../../libsolidity/SolidityExecutionFramework.cpp
)

target_link_libraries(soltest-mlir PRIVATE solcli libsolc yul solidity smtutil solutil phaser Boost::boost yulInterpreter evmasm Boost::filesystem Boost::program_options Boost::unit_test_framework evmc)
