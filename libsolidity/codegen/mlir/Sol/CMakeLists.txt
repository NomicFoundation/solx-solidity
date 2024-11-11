add_mlir_doc(SolDialect SolDialect Sol/ -gen-dialect-doc)
add_mlir_doc(SolOps SolOps Sol/ -gen-op-doc)
add_mlir_doc(SolAttrs SolAttrs Dialects/ -gen-attrdef-doc)
add_mlir_doc(SolTypes SolTypes Dialects/ -gen-typedef-doc)

# Create MLIRSolOpsIncGen.
add_mlir_dialect(SolOps sol)

# Generate sources from SolOps.td.
set(LLVM_TARGET_DEFINITIONS SolOps.td)
mlir_tablegen(SolOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(SolOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(SolOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=sol)
mlir_tablegen(SolOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=sol)

# Generate sources from SolInterfaces.td.
set(LLVM_TARGET_DEFINITIONS SolInterfaces.td)
mlir_tablegen(SolInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SolInterfaces.cpp.inc -gen-op-interface-defs)

# Create MLIRSolExtraIncGen.
add_public_tablegen_target(MLIRSolExtraIncGen)
