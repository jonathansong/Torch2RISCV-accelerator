// The sahl dialect: the tile level of the sa backend (docs/iree_compiler_plan.md
// §8.3). After bufferization a dispatch is linalg on memrefs; sahl makes the
// DDR traffic explicit: computation (linalg.generic / fill) reads and writes
// local buffers only (memref.alloc, placed in SPAD / ACC by the lowering), and
// sahl.load / sahl.store move data between DDR (binding subspans and views of
// them) and local buffers; sahl.kernel groups the operations lowered together
// (a micro-kernel, a generic contraction). sahl-to-sahw lowers this to commands.
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

def Sahl_KernelOp : Sahl_Op<"kernel", [SingleBlock, NoTerminator]> {
  let summary = "Operations lowered together: a micro-kernel or a generic contraction";
  let description = [{
    sa-to-sahl groups the operations of each match of the kernel matcher
    (SahlKernels.h): `kind` is "linear" or "attention" (micro-kernels, as
    --iree-sa-ukernels allows) or "contraction" (the generic lowering). The
    operation marked `sahl.anchor` is the contraction the plan is built from;
    the others are its loads, extensions, fill, epilogue and store, in their
    original order.
  }];
  let arguments = (ins StrAttr:$kind);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$kind attr-dict-with-keyword $body";
}

#endif  // SA_DIALECT_SAHL_OPS
