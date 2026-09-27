// The sahw dialect (SahwOps.td).
#include "SahwDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::iree_compiler::sa::sahw;

#include "SahwDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "SahwTypes.cpp.inc"
#define GET_OP_CLASSES
#include "SahwOps.cpp.inc"

void SahwDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "SahwTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "SahwOps.cpp.inc"
      >();
}
