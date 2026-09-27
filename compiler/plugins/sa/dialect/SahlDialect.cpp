// The sahl dialect (SahlOps.td).
#include "SahlDialect.h"

using namespace mlir;
using namespace mlir::iree_compiler::sa::sahl;

#include "SahlDialect.cpp.inc"
#define GET_OP_CLASSES
#include "SahlOps.cpp.inc"

void SahlDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SahlOps.cpp.inc"
      >();
}
