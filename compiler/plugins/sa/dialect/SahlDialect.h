// The sahl dialect (SahlOps.td; docs/iree_compiler_plan.md §8.3).
#ifndef SA_DIALECT_SAHLDIALECT_H_
#define SA_DIALECT_SAHLDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "SahlDialect.h.inc"  // IWYU pragma: keep
#define GET_OP_CLASSES
#include "SahlOps.h.inc"  // IWYU pragma: keep
// clang-format on

#endif  // SA_DIALECT_SAHLDIALECT_H_
