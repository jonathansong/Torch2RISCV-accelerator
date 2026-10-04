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
include "mlir/Interfaces/SideEffectInterfaces.td"

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

def Sahl_ToI8Op : Sahl_Op<"to_i8", [Pure]> {
  let summary = "fp32 -> int8 as qllama.to_i8: clamp(round_even(nan_to_num(x)), -127, 127)";
  let description = [{
    Inside a linalg body: the whole quantization chain (NaN -> 0, +-inf ->
    +-FLT_MAX, round to nearest even, clamp to +-127, fptosi), which the VE's
    int8 output conversion does in one step. sa-to-sahl replaces each matched
    chain by this operation.
  }];
  let arguments = (ins F32:$x);
  let results = (outs I8:$y);
  let assemblyFormat = "$x attr-dict";
}

def Sahl_GatherOp : Sahl_Op<"gather"> {
  let summary = "A data-dependent read inside a linalg body";
  let description = [{
    Element `indices` of `source` (a buffer captured by the body), the indices
    computed from the body's iteration indices and an i64 scalar input. `kind`
    is the form sa-to-sahl chose (gatherForm in SahlKernels.h): "row",
    "scalar", "packed", "swap", or "none" (no form: the lowering reports why).
  }];
  let arguments = (ins Arg<AnyMemRef, "the table", [MemRead]>:$source, Variadic<Index>:$indices, StrAttr:$kind);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$kind $source `[` $indices `]` attr-dict `:` type($source) `->` type($result)";
}

def Sahl_ScatterOp : Sahl_Op<"scatter"> {
  let summary = "One row written into DDR at a dynamic index (the KV cache update)";
  let description = [{
    dest[index, ...] = updates[0, ...], index = the i64 in `indices` (DDR).
    sa-to-sahl makes it from an iree_linalg_ext.scatter that overwrites one
    row along dimension 0.
  }];
  let arguments = (ins Arg<AnyMemRef, "the row", [MemRead]>:$updates,
                       Arg<AnyMemRef, "the index", [MemRead]>:$indices,
                       Arg<AnyMemRef, "the table", [MemWrite]>:$dest);
  let assemblyFormat = "$updates `,` $indices `,` $dest attr-dict `:` type($updates) `,` type($indices) `,` type($dest)";
}

def Sahl_ScopeOp : Sahl_Op<"scope", [SingleBlock, NoTerminator]> {
  let summary = "A piece of a dispatch: the local memory allocated inside is released at its end";
  let description = [{
    sahl-tile splits an element-wise dispatch too large for ACC into pieces,
    one sahl.scope each: its operations work on views and local buffers of the
    piece. Buffers shared by the pieces are allocated before them (sahl.reserve).
    With `keep` (an expanded kernel's step, sahl-expand-kernels) the buffers
    allocated before the scope stay usable inside it (a piece sees only the
    reserved ones). With `spad_from` the SPAD_A buffers allocated inside it
    start at that word (the generic contraction's int8 results: after x).
  }];
  let arguments = (ins UnitAttr:$keep, OptionalAttr<I64Attr>:$spad_from);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def Sahl_ReserveOp : Sahl_Op<"reserve"> {
  let summary = "Allocates a local buffer now (shared by the sahl.scope pieces that follow)";
  let arguments = (ins AnyMemRef:$buffer, UnitAttr:$bcast);
  let assemblyFormat = "$buffer attr-dict `:` type($buffer)";
}

def Sahl_StripOp : Sahl_Op<"strip"> {
  let summary = "x -> an A strip of EX in SPAD_A";
  let description = [{
    The A operand of sahl.mma. From rows of int8 x in DDR ([D, K], prefill's
    rows): element k of row r at word k, lane r (one LD with INTERLEAVE).
    From a local vector [K] (decode): every element over the D rows (a VE
    copy in the DIV mode).
  }];
  let arguments = (ins Arg<AnyMemRef, "the rows", [MemRead]>:$src, Arg<AnyMemRef, "the strip", [MemWrite]>:$dst);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

def Sahl_MmaOp : Sahl_Op<"mma"> {
  let summary = "acc[D, nc, D] = A strip [D, K] x B tiles [nc, K, D] (one EX; + acc with `accumulate`)";
  let description = [{
    The systolic array's matrix multiply: `a` an A strip in SPAD_A (sahl.strip),
    `b` nc weight tiles in a SPAD_B bank (packed [nc, K, D], loaded by
    sahl.load into a buffer placed "spad_b"), `acc` int32 in ACC.
  }];
  let arguments = (ins Arg<AnyMemRef, "the A strip", [MemRead]>:$a, Arg<AnyMemRef, "the B tiles", [MemRead]>:$b,
                       Arg<AnyMemRef, "the accumulator", [MemRead, MemWrite]>:$acc, UnitAttr:$accumulate);
  let assemblyFormat = "$a `,` $b `,` $acc attr-dict `:` type($a) `,` type($b) `,` type($acc)";
}

def Sahl_BcastOp : Sahl_Op<"bcast"> {
  let summary = "A local vector [n] or scalar -> its broadcast layout (one word per element, every lane the element)";
  let description = [{
    The layout a per-row (or scalar) operand of an element-wise operation
    reads with the DIV index mode. sahl-expand-kernels makes it explicit for
    the inputs of a kernel's epilogue that every step reads.
  }];
  let arguments = (ins Arg<AnyMemRef, "the values", [MemRead]>:$src, Arg<AnyMemRef, "the broadcast words", [MemWrite]>:$dst);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

def Sahl_LoopOp : Sahl_Op<"loop", [SingleBlock, NoTerminator]> {
  let summary = "A hardware loop (LOOP_END): the body `count` times, its DDR views advancing each time";
  let description = [{
    Each DMA of the body whose operation carries `sa.advance = j` moves on by
    `strides[j]` bytes per iteration (decode's linear layer: two chunks of
    weights, two chunks of per-chunk inputs and results).
  }];
  let arguments = (ins I64Attr:$count, DenseI64ArrayAttr:$strides);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def Sahl_ClaimSpadOp : Sahl_Op<"claim_spad"> {
  let summary = "SPAD_A words [0, words) belong to the placed buffers that follow (SPAD_A unused before)";
  let description = [{
    The generic contraction's layout: the A strip at word 0, x or the raw rows
    of a transposed matrix in bank 1, placed with sa.word. Other SPAD_A
    buffers are allocated after `words`.
  }];
  let arguments = (ins I64Attr:$words);
  let assemblyFormat = "attr-dict";
}

def Sahl_TransposeOp : Sahl_Op<"transpose"> {
  let summary = "Rows [R, C] in SPAD_A -> weight tiles in a SPAD_B bank (TRANSPOSE: B tiles of a row-major matrix)";
  let arguments = (ins Arg<AnyMemRef, "the rows", [MemRead]>:$src, Arg<AnyMemRef, "the tiles", [MemWrite]>:$dst);
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
