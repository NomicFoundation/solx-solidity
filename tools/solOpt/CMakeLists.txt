include_directories(${PROJECT_BINARY_DIR}/libsolidity/codegen/mlir)

add_executable(sol-opt main.cpp)

target_link_libraries(sol-opt PUBLIC
  MLIROptLib
  solidity
)
