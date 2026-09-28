// The C5 pipeline from linalg to sahw (docs/iree_compiler_plan.md §8.4).
#ifndef SA_TRANSFORMS_SAHLPASSES_H_
#define SA_TRANSFORMS_SAHLPASSES_H_

#include <memory>
#include <string>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::sa {

struct TargetConfig;

// The DDR binding subspan a memref view comes from (null: not DDR).
Value ddrRoot(Value v);

// Bufferized dispatch -> linalg on local buffers + sahl.load / sahl.store; the
// operations of each micro-kernel / generic contraction grouped into a sahl.kernel.
std::unique_ptr<Pass> createSaToSahlPass(const TargetConfig &config);
void registerSaToSahlPass();
// Splits an element-wise dispatch too large for ACC into pieces (sahl.scope).
std::unique_ptr<Pass> createSahlTilePass(const TargetConfig &config);
void registerSahlTilePass();
// Places each local buffer and chooses its layout (sa.mem, sa.layout attributes).
std::unique_ptr<Pass> createSahlPlanMemoryPass();
void registerSahlPlanMemoryPass();
// The schedule of each linear micro-kernel (chunk_tiles, loop on its sahl.kernel).
std::unique_ptr<Pass> createSahlSchedulePass(const TargetConfig &config);
void registerSahlSchedulePass();
// sahl -> one sahw.template per function (the lowering: registers, local
// memory, one single-stage VE per arith / math operation).
std::unique_ptr<Pass> createSahlToSahwPass(const TargetConfig &config);
void registerSahlToSahwPass();
// Merges single-stage VEs into multi-stage ones (FUNC(OP(s1', s2) * A + B)).
std::unique_ptr<Pass> createSahwFuseVePass();
void registerSahwFuseVePass();

// The whole C5 pipeline for one dispatch function (in a module of its own):
// bufferize, sa-to-sahl, sahl-to-sahw, sahw-fuse-ve.
void buildSahlPipeline(OpPassManager &pm, const TargetConfig &config);

}  // namespace mlir::iree_compiler::sa

#endif  // SA_TRANSFORMS_SAHLPASSES_H_
