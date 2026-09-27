// Passes of the sa backend plugin (docs/iree_compiler_plan.md §6).
#ifndef SA_TARGET_SAPASSES_H_
#define SA_TARGET_SAPASSES_H_

#include <cstdint>
#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::sa {

// Preprocessing: constant int8 linear weights -> the B-tile layout
// (PackLinearWeights.cpp).
std::unique_ptr<Pass> createPackLinearWeightsPass(int64_t d);
void registerPackLinearWeightsPass();

// Translation: every export runs as one workgroup (a dispatch is one
// descriptor template), so the workgroup count regions become (1, 1, 1)
// (LowerWorkgroupCount.cpp).
std::unique_ptr<Pass> createLowerWorkgroupCountPass();
void registerLowerWorkgroupCountPass();

}  // namespace mlir::iree_compiler::sa

#endif  // SA_TARGET_SAPASSES_H_
