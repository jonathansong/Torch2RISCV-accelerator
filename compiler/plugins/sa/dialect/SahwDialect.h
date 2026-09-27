// The sahw dialect (SahwOps.td; docs/iree_compiler_plan.md §8.3).
#ifndef SA_DIALECT_SAHWDIALECT_H_
#define SA_DIALECT_SAHWDIALECT_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

// clang-format off
#include "SahwDialect.h.inc"  // IWYU pragma: keep
#define GET_TYPEDEF_CLASSES
#include "SahwTypes.h.inc"  // IWYU pragma: keep
#define GET_OP_CLASSES
#include "SahwOps.h.inc"  // IWYU pragma: keep
// clang-format on

#endif  // SA_DIALECT_SAHWDIALECT_H_
