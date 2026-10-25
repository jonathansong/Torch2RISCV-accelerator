// Passes of the sa backend plugin (docs/iree_compiler_plan.md §6).
#ifndef SA_TARGET_SAPASSES_H_
#define SA_TARGET_SAPASSES_H_

#include <cstdint>
#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::sa {

// Preprocessing: constant int8 linear weights -> the B-tile layout
// (PackLinearWeights.cpp).
std::unique_ptr<Pass> createPackLinearWeightsPass(int64_t d);
void registerPackLinearWeightsPass();

// Preprocessing: index / scalar-only element-wise values are recomputed in
// each consumer (CloneCheapProducers.cpp).
std::unique_ptr<Pass> createCloneCheapProducersPass();
void registerCloneCheapProducersPass();

// Preprocessing: decode's gated FFN wrapped in one flow.dispatch.region
// (FormFfnRegions.cpp, docs/fusion_design.md §4.2).
std::unique_ptr<Pass> createFormFfnRegionsPass(bool enabled);
void registerFormFfnRegionsPass();

// MoE: a router's Top-2 in a dispatch region of its own (docs/moe_native_lowering_design.md §5.2).
std::unique_ptr<Pass> createFormMoeRegionsPass(bool enabled);
void registerFormMoeRegionsPass();
// Each MoE expert's down-projection epilogue kept in its own dispatch with k > 2
// (FormMoeRegions.cpp, docs/moe_native_lowering_design.md §23).
std::unique_ptr<Pass> createIsolateMoeExpertsPass();
void registerIsolateMoeExpertsPass();
// Its barriers removed after dispatch region formation (FormMoeRegions.cpp).
std::unique_ptr<Pass> createDropMoeIsolationPass();
void registerDropMoeIsolationPass();

// Dispatch creation, after the regions are formed: decode's attention (q path,
// scores, probabilities, p . V, int8 output) merged into one flow.dispatch.region
// (FormAttnRegions.cpp, docs/fusion_design.md §6.4.1).
std::unique_ptr<Pass> createFormAttnRegionsPass(bool report = false, int64_t maxDynamic = 0, bool chunks = false);
void registerFormAttnRegionsPass();

// The module targets the sa device (the preprocessing passes act only then).
bool targetsSA(ModuleOp module);

// Translation: every export runs as one workgroup (a dispatch is one
// descriptor template), so the workgroup count regions become (1, 1, 1)
// (LowerWorkgroupCount.cpp).
std::unique_ptr<Pass> createLowerWorkgroupCountPass();
void registerLowerWorkgroupCountPass();

}  // namespace mlir::iree_compiler::sa

#endif  // SA_TARGET_SAPASSES_H_
