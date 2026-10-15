// sa.advance_final (docs/iree_compiler_plan.md §8.22 step c) offsets an
// operation by a sahl.loop's final PARAM value: only right after the loop.
// RUN: iree-opt --iree-sahl-to-sahw="d=16" --verify-diagnostics %s -o /dev/null
// expected-error @below {{sa.advance_final not right after a sahl.loop}}
func.func @final_without_loop() {
  %a = memref.alloc() {sa.layout = "packed", sa.mem = "acc"} : memref<16xi32>
  %b = memref.alloc() {sa.layout = "packed", sa.mem = "acc"} : memref<16xi32>
  memref.copy %a, %b {sa.advance_final = 0 : i64} : memref<16xi32> to memref<16xi32>
  return
}
