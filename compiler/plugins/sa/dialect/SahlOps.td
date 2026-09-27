// The sahl dialect: the tile level of the sa backend (docs/iree_compiler_plan.md
// §8.3). After bufferization a dispatch is linalg on memrefs; sahl makes the
// DDR traffic explicit: computation (linalg.generic / fill) reads and writes
// local buffers only (memref.alloc, placed in SPAD / ACC by the lowering), and
// sahl.load / sahl.store move data between DDR (binding subspans and views of
// them) and local buffers. sahl-to-sahw lowers this to commands.
#ifndef SA_DIALECT_SAHL_OPS
#define SA_DIALECT_SAHL_OPS

include "mlir/IR/OpBase.td"

def Sahl_Dialect : Dialect {
  let name = "sahl";
  let summary = "Tile level of the sa backend: explicit DDR <-> local traffic around linalg on local buffers";
  let cppNamespace = "::mlir::iree_compiler::sa::sahl";
}

class Sahl_Op<string mnemonic, list<Trait> traits = []> : Op<Sahl_Dialect, mnemonic, traits>;

def Sahl_LoadOp : Sahl_Op<"load"> {
  let summary = "DDR -> local buffer (the whole view)";
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}
def Sahl_StoreOp : Sahl_Op<"store"> {
  let summary = "local buffer -> DDR (the whole view)";
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

#endif  // SA_DIALECT_SAHL_OPS
