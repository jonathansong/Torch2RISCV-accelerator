// Descriptor-template generation for one dispatch (docs/iree_compiler_plan.md
// §6.8): the stage-C3 code generator of the sa backend.
#ifndef SA_TARGET_CODEGEN_H_
#define SA_TARGET_CODEGEN_H_

#include <cstdint>
#include <string>
#include <vector>

#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::sa {

// One register setup entry of sa-desc version 2 (compiler/runtime/tools/sadesc.py):
// value = ((push constant * mul) >> shift) + add (constant -1: add only),
// for BASE also + the binding's physical address.
struct SetupEntry {
  enum Kind : uint8_t { BASE = 0, PARAM = 1 } kind;
  uint8_t reg, binding;
  int16_t constant;
  int32_t mul;
  uint8_t shift;
  int32_t add;
};

struct Generated {
  std::string templ;                 // descriptor bytes (ends with RET)
  std::vector<SetupEntry> setup;
  // sa-desc version 3 (compiler/runtime/tools/sadesc.py): the template may start with a prefix (prefix
  // descriptors, its RET included; the body follows) of loads that the driver
  // may run before the FENCE ordering the dispatch after the previous one,
  // when the bindings it reads (prefixReads) were not written since the last
  // FENCE. writes: the bindings the dispatch stores to.
  uint32_t prefix = 0, prefixReads = 0, writes = 0;
  // the bindings loaded from (LD, LDPARAM); head: the body's leading loads
  // (descriptors, RET included; 0: none) callable on their own, the rest
  // follows; usesSpadB: the dispatch touches SPAD_B (where prefixes load);
  // prefixReg: the BASE register the prefix loads were generated for (they
  // use BASE15, which the driver sets to its value)
  uint32_t reads = 0, head = 0;
  bool usesSpadB = false;
  int prefixReg = -1;
};

struct CodegenOptions {
  int64_t d = 8;
  int64_t maxDynamic = 256;          // upper bound of a dynamic dimension (local memory planning)
};

// Generates the template of |func| (a dispatch function). On failure returns
// false with |why| set.
bool generateDispatch(FunctionOpInterface func, const CodegenOptions &options, Generated &out, std::string &why);

}  // namespace mlir::iree_compiler::sa

#endif  // SA_TARGET_CODEGEN_H_
