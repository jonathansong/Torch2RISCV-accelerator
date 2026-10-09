// Passes and serialization on the sahw dialect (docs/iree_compiler_plan.md §8.4).
#ifndef SA_TRANSFORMS_SAHWPASSES_H_
#define SA_TRANSFORMS_SAHWPASSES_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "SahwDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

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

// The hardware parameters of an sa executable target (#hal.executable.target
// configuration; §8.9 item 1): the code generator reads them from here only.
struct TargetConfig {
  int64_t d = 8;
  int64_t spadBytes = 128 * 1024;    // per SPAD (two banks)
  int64_t accBytes = 256 * 1024;     // ACC (two banks)
  int64_t maxDynamic = 256;          // upper bound of a dynamic dimension
  int64_t dynFields = 2;             // dynamic fields per descriptor
  int64_t bases = 16, params = 8;    // BASE / PARAM registers (BASE15: prefixes)
  int64_t gemvPorts = 0;             // K2b streaming GEMV unit (LD mode GEMV) and its read ports; 0: none
  static TargetConfig fromAttr(DictionaryAttr config);
  // What the command encoding allows (empty if valid): D 8 or 16, the local
  // memories powers of two of at most 2^16 words (16-bit local addresses).
  std::string invalid() const;
  void addTo(Builder &b, SmallVectorImpl<NamedAttribute> &attrs) const;
  // not a hardware parameter: the micro-kernels sahl-to-sahw may use (§8.8)
  bool ukernel(StringRef name) const;
  std::string ukernels = "all";
};

// Code generation of every export (SaCodegen.cpp): the C5 pipeline
// (SahlPasses.h) on a copy of the dispatch function, the sahw.template into
// the variant's inner module. A dispatch it cannot compile is an error, or
// with allowUnsupported a warning and an export that faults when run.
// report: one line per export on stderr.
std::unique_ptr<Pass> createSaDropHostVariantsPass();
std::unique_ptr<Pass> createSaCodegenPass(bool allowUnsupported, const std::string &ukernels = "all",
                                          bool report = false, bool hostFallback = false,
                                          const std::string &hostDispatches = "");
// The body's leading loads -> sahw.head, for exports that do not touch SPAD_B.
std::unique_ptr<Pass> createSahwSplitHeadPass();
// BASE registers from 0, register-setup PARAMs from 0, private PARAMs from 7 down.
std::unique_ptr<Pass> createSahwAssignRegistersPass();
void registerSahwPasses();

// One export as sa-desc v3 wants it (SahwSerialize.cpp).
struct SerializedExport {
  std::string templ;                 // descriptors: prefix (+ RET), head (+ RET), body (+ RET)
  std::vector<SetupEntry> setup;
  uint32_t prefix = 0, prefixReads = 0, writes = 0, reads = 0, head = 0, flags = 0, prefixReg = 0;
  uint32_t cycles = 0;
};
LogicalResult serializeTemplate(sahw::TemplateOp t, SerializedExport &out);

}  // namespace mlir::iree_compiler::sa

#endif  // SA_TRANSFORMS_SAHWPASSES_H_
