// The sahw dialect: the command level of the sa backend (docs/iree_compiler_plan.md
// §8.3). One operation is one command of the accelerator (LD, ST, EX, VE,
// TRANSPOSE, LDPARAM, FENCE, SETREG, LOOP_END); a sahw.template holds one
// export (its prefix, head and body, as in sa-desc v3). Local addresses are
// words with the memory in bits [31:28] (MEM_SPAD_A = 1, SPAD_B = 2, ACC = 3);
// DDR addresses are offsets from a BASE register (!sahw.base) or absolute.
// Registers are values until sahw-assign-registers numbers them (`reg`).
#ifndef SA_DIALECT_SAHW_OPS
#define SA_DIALECT_SAHW_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"

def Sahw_Dialect : Dialect {
  let name = "sahw";
  let summary = "Commands of the PYNQ-Z1 systolic accelerator (sa backend, command level)";
  let cppNamespace = "::mlir::iree_compiler::sa::sahw";
  let useDefaultTypePrinterParser = 1;
}

class Sahw_Type<string name, string typeMnemonic> : TypeDef<Sahw_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Sahw_BaseType : Sahw_Type<"Base", "base"> {
  let summary = "A BASE register: a binding's address plus an offset (DDR relocation)";
}
def Sahw_ParamType : Sahw_Type<"Param", "param"> {
  let summary = "A PARAM register (a dynamic value)";
}

class Sahw_Op<string mnemonic, list<Trait> traits = []> : Op<Sahw_Dialect, mnemonic, traits>;

// Fields of a command taken from PARAM registers: dyn_fields[i] is the field
// number (DescList DYN_*), dyn_add[i] adds the PARAM instead of replacing.
defvar DynArgs = (ins Variadic<Sahw_ParamType>:$dyn,
                  DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$dyn_fields,
                  DefaultValuedAttr<DenseBoolArrayAttr, "{}">:$dyn_add,
                  UnitAttr:$fence_before);

//===----------------------------------------------------------------------===//
// Structure
//===----------------------------------------------------------------------===//

def Sahw_TemplateOp : Sahw_Op<"template", [IsolatedFromAbove, SingleBlock, NoTerminator]> {
  let summary = "One export: register definitions, then sahw.prefix / head / body";
  let arguments = (ins StrAttr:$export_name, I64Attr:$bindings, I64Attr:$constants,
                       DefaultValuedAttr<I64Attr, "0">:$est_cycles, UnitAttr:$unsupported);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "$export_name attr-dict-with-keyword $region";
}

class Sahw_SectionOp<string mnemonic, string doc>
    : Sahw_Op<mnemonic, [SingleBlock, NoTerminator, HasParent<"TemplateOp">]> {
  let summary = doc;
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "attr-dict-with-keyword $region";
}
def Sahw_PrefixOp : Sahw_SectionOp<"prefix",
    "Loads the driver may run early (through BASE15), inside earlier dispatches or before the FENCE">;
def Sahw_HeadOp : Sahw_SectionOp<"head",
    "The body's leading loads, callable on their own (another dispatch's prefix may follow)">;
def Sahw_BodyOp : Sahw_SectionOp<"body", "The rest of the export">;

def Sahw_LoopOp : Sahw_Op<"loop", [SingleBlock, NoTerminator]> {
  let summary = "LOOP_END: runs the region `count` times, adding s1 to PARAM k1 and s2 to PARAM k2 each time";
  let arguments = (ins I64Attr:$count, Sahw_ParamType:$k1, I64Attr:$s1, Sahw_ParamType:$k2, I64Attr:$s2);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "$count `,` $k1 `step` $s1 `,` $k2 `step` $s2 attr-dict-with-keyword $region";
}

//===----------------------------------------------------------------------===//
// Registers
//===----------------------------------------------------------------------===//

// value = ((push constant `constant` * mul) >> shift) + add (constant -1: add only)
defvar LinArgs = (ins DefaultValuedAttr<I64Attr, "-1">:$constant, DefaultValuedAttr<I64Attr, "1">:$mul,
                      DefaultValuedAttr<I64Attr, "0">:$shift, DefaultValuedAttr<I64Attr, "0">:$add,
                      OptionalAttr<I64Attr>:$reg);

def Sahw_BaseOp : Sahw_Op<"base", [HasParent<"TemplateOp">]> {
  let summary = "BASE register = binding address + f(push constant) (a register setup entry)";
  let arguments = !con((ins I64Attr:$binding), LinArgs);
  let results = (outs Sahw_BaseType:$result);
  let assemblyFormat = "`binding` `(` $binding `)` attr-dict";
}
def Sahw_ParamOp : Sahw_Op<"param", [HasParent<"TemplateOp">]> {
  let summary = "PARAM register = f(push constant) (a register setup entry)";
  let arguments = LinArgs;
  let results = (outs Sahw_ParamType:$result);
  let assemblyFormat = "attr-dict";
}
def Sahw_PrivateOp : Sahw_Op<"private", [HasParent<"TemplateOp">]> {
  let summary = "A PARAM register owned by the template (loop offsets, LDPARAM results)";
  let arguments = (ins OptionalAttr<I64Attr>:$reg);
  let results = (outs Sahw_ParamType:$result);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Commands
//===----------------------------------------------------------------------===//

def Sahw_LdOp : Sahw_Op<"ld", [AttrSizedOperandSegments]> {
  let summary = "LD: DDR -> local (rows x row_bytes, DDR stride pitch; mode LINEAR / INTERLEAVE / GEMV)";
  let description = [{
    Mode GEMV (2, docs/k2b_gemv_design.md): `rows` weight strips of row_bytes
    (K x D) stream into the GEMV unit; ACC word laddr + s * cstep = x . strip s
    (+ itself with gemv_acc), x = K int8 at SPAD_A word xword.
  }];
  let arguments = !con((ins Optional<Sahw_BaseType>:$base, I64Attr:$ddr, I64Attr:$laddr, I64Attr:$rows,
                            I64Attr:$row_bytes, I64Attr:$pitch, DefaultValuedAttr<I64Attr, "0">:$mode), DynArgs,
                       (ins DefaultValuedAttr<I64Attr, "0">:$xword, DefaultValuedAttr<I64Attr, "0">:$cstep,
                            UnitAttr:$gemv_acc));
  let assemblyFormat = "($base^)? (`dyn` `(` $dyn^ `)`)? attr-dict";
}
def Sahw_StOp : Sahw_Op<"st", [AttrSizedOperandSegments]> {
  let summary = "ST: local -> DDR";
  let arguments = !con((ins Optional<Sahw_BaseType>:$base, I64Attr:$ddr, I64Attr:$laddr, I64Attr:$rows,
                            I64Attr:$row_bytes, I64Attr:$pitch), DynArgs);
  let assemblyFormat = "($base^)? (`dyn` `(` $dyn^ `)`)? attr-dict";
}
def Sahw_ExOp : Sahw_Op<"ex"> {
  let summary = "EX: int8 A strip x B tiles -> int32 C (repeat tiles, B / C steps)";
  let arguments = !con((ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$kt, UnitAttr:$accumulate,
                            I64Attr:$repeat, I64Attr:$bstep, I64Attr:$cstep, I64Attr:$crow), DynArgs);
  let assemblyFormat = "(`dyn` `(` $dyn^ `)`)? attr-dict";
}
def Sahw_VeOp : Sahw_Op<"ve"> {
  let summary = "VE: y = FUNC(OP(s1', s2) * A + B), with index modes, reduction, VALID and type conversion";
  let arguments = !con((ins I64Attr:$src1, I64Attr:$src2, I64Attr:$dst, I64Attr:$length, I64Attr:$op,
                            I64Attr:$types, DefaultValuedAttr<I64Attr, "0">:$period, BoolAttr:$fp,
                            DefaultValuedAttr<I64Attr, "0">:$func, DefaultValuedAttr<I64Attr, "0">:$m1,
                            DefaultValuedAttr<I64Attr, "0">:$m2, DefaultValuedAttr<I64Attr, "0">:$reduce,
                            UnitAttr:$swapneg, F32Attr:$imm, F32Attr:$a, F32Attr:$b,
                            DefaultValuedAttr<I64Attr, "0">:$rowlen, DefaultValuedAttr<I64Attr, "0">:$valid,
                            DefaultValuedAttr<I64Attr, "0">:$p1, DefaultValuedAttr<I64Attr, "1">:$scale,
                            DefaultValuedAttr<I64Attr, "0">:$shift, DefaultValuedAttr<I64Attr, "0">:$zp,
                            DefaultValuedAttr<I64Attr, "-2147483648">:$lo,
                            DefaultValuedAttr<I64Attr, "2147483647">:$hi), DynArgs);
  let assemblyFormat = "(`dyn` `(` $dyn^ `)`)? attr-dict";
}
def Sahw_TransposeOp : Sahw_Op<"transpose"> {
  let summary = "TRANSPOSE: D x D blocks of `length` elements, `stride` words apart";
  let arguments = !con((ins I64Attr:$src, I64Attr:$dst, I64Attr:$length, I64Attr:$types, I64Attr:$stride), DynArgs);
  let assemblyFormat = "(`dyn` `(` $dyn^ `)`)? attr-dict";
}
def Sahw_LdParamOp : Sahw_Op<"ldparam", [AttrSizedOperandSegments]> {
  let summary = "LDPARAM: target = mem32[addr] * mul + add";
  let arguments = !con((ins Optional<Sahw_BaseType>:$base, Sahw_ParamType:$target, I64Attr:$addr, I64Attr:$mul,
                            I64Attr:$add), DynArgs);
  let assemblyFormat = "$target (`,` $base^)? (`dyn` `(` $dyn^ `)`)? attr-dict";
}
def Sahw_FenceOp : Sahw_Op<"fence"> {
  let summary = "FENCE: wait until the scheduler queue is empty and the engines in mask are idle (0: all)";
  let arguments = (ins DefaultValuedAttr<I64Attr, "0">:$mask);
  let assemblyFormat = "attr-dict";
}
def Sahw_SetRegOp : Sahw_Op<"setreg", [AttrSizedOperandSegments]> {
  let summary = "SETREG: up to three PARAMs = (or += with add_mask) values";
  let arguments = !con((ins Variadic<Sahw_ParamType>:$targets, DenseI64ArrayAttr:$values,
                            DefaultValuedAttr<I64Attr, "0">:$add_mask), DynArgs);
  let assemblyFormat = "$targets (`dyn` `(` $dyn^ `)`)? attr-dict";
}

#endif  // SA_DIALECT_SAHW_OPS
