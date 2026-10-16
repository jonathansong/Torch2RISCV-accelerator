// sahw.template -> LLVM IR of a function that issues the export's commands
// over PCPI from the PicoRV32 (docs/riscv_backend_design.md §9: plan A's code
// generator, S1). One function per export:
//   void <export>(const uint32_t *bindings, const uint32_t *push_constants)
// - registers become stack slots (promoted by the optimizer): sahw.base =
//   binding address + f(push constant), sahw.param = f(push constant) with
//   f(c) = add + ((c * mul) >> shift) in 64-bit arithmetic (sa_loader.c),
//   sahw.private = 0;
// - each command is specialized at compile time: its descriptor words as the
//   serializer encodes them are constants, the dynamic fields applied in the
//   IR (the fetch unit's rule), the DDR address relocated by its BASE; then
//   the PCPI operations of the command (the configuration keys it uses, then
//   the command: sa_rv_mat_cfg / sa_rv_mat_load / ..., one custom instruction
//   each in firmware/rv/sa_rv_rt.c). Static fields fold to immediates; a key
//   whose constant value the code already wrote is not written again (the
//   values tracked through the template; unknown at entry and at a loop's
//   head, the body's end state after it: the body runs at least once);
// - control flow is compiled: LOOP_END a loop (the count, possibly from a
//   PARAM, taken when it is first reached; k1 += s1, k2 += s2 after every
//   pass), SETREG assignments, LDPARAM a call sa_rv_ldparam(address) (a fence,
//   then a load) times mul plus add.
// The template's prefix, head and body run in that order. The IR has no
// target triple or data layout (i32 / i64 values and the two pointers only):
// the same file compiles for riscv32 and for the host's checks.
#include <cstring>

#include "SahwPasses.h"
#include <map>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#ifdef SA_HAVE_RISCV
#include <mutex>

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
extern "C" {
void LLVMInitializeRISCVTargetInfo();
void LLVMInitializeRISCVTarget();
void LLVMInitializeRISCVTargetMC();
void LLVMInitializeRISCVAsmPrinter();
void LLVMInitializeRISCVAsmParser();
}
#endif

namespace mlir::iree_compiler::sa {
namespace {

constexpr uint32_t OP_LOOP_END = 0x13, OP_RET = 0x16;

struct RvEmitter {
  llvm::LLVMContext &ctx;
  llvm::Module &mod;
  // false: the PCPI primitives as calls of sa_rv_* (portable IR: the host's checks, the sim);
  // true: as riscv32 inline assembly (.insn), compiled in-process (S4)
  bool insn;
  llvm::IRBuilder<> b{ctx};
  llvm::Function *fn = nullptr, *ldparamFn = nullptr;
  llvm::StringMap<llvm::Function *> pcpi;   // sa_rv_<operation>: one PCPI instruction each
  // the configuration keys' values written so far, when constant (mat_cfg, vec_cfg, gemv_cfg)
  std::map<uint32_t, uint64_t> known[3];
  llvm::Type *i32 = nullptr, *i64 = nullptr, *ptr = nullptr;
  llvm::Value *bindArg = nullptr, *pcArg = nullptr;
  llvm::BasicBlock *entry = nullptr;
  llvm::DenseMap<Value, llvm::AllocaInst *> slots;
  const std::string &templ;
  size_t cursor = 0, rows = 0;
  std::string err;

  RvEmitter(llvm::LLVMContext &c, llvm::Module &m, bool asInsn, const std::string &t)
      : ctx(c), mod(m), insn(asInsn), templ(t), rows(t.size() / 64) {
    i32 = llvm::Type::getInt32Ty(ctx);
    i64 = llvm::Type::getInt64Ty(ctx);
    ptr = llvm::PointerType::getUnqual(ctx);
  }

  bool fail(Operation *op, const std::string &m) {
    if (err.empty()) {
      err = m;
      if (op) op->emitError() << "sahw to riscv: " << m;
    }
    return false;
  }
  uint64_t word(size_t row, int i) const {
    uint64_t v;
    std::memcpy(&v, templ.data() + 64 * row + 8 * i, 8);
    return v;
  }
  llvm::Value *c32(uint64_t v) { return llvm::ConstantInt::get(i32, v & 0xFFFFFFFFull); }
  llvm::Value *c64(int64_t v) { return llvm::ConstantInt::get(i64, uint64_t(v)); }

  llvm::AllocaInst *newSlot(const char *name) {
    llvm::IRBuilder<> eb(entry, entry->begin());
    return eb.CreateAlloca(i32, nullptr, name);
  }
  llvm::Value *load(Operation *op, Value v) {
    auto it = slots.find(v);
    if (it == slots.end()) {
      fail(op, "a register defined outside the template");
      return c32(0);
    }
    return b.CreateLoad(i32, it->second);
  }
  void store(Operation *op, Value v, llvm::Value *x) {
    auto it = slots.find(v);
    if (it == slots.end()) {
      fail(op, "a register defined outside the template");
      return;
    }
    b.CreateStore(x, it->second);
  }
  // f(c) = add + ((c * mul) >> shift) (constant -1: add), as sa_loader.c's setup
  llvm::Value *lin(int64_t constant, int64_t mul, int64_t shift, int64_t add) {
    llvm::Value *v = c64(add);
    if (constant >= 0) {
      llvm::Value *p = b.CreateLoad(i32, b.CreateConstGEP1_32(i32, pcArg, unsigned(constant)));
      llvm::Value *x = b.CreateMul(b.CreateZExt(p, i64), c64(mul));
      v = b.CreateAdd(v, b.CreateAShr(x, c64(shift)));
    }
    return b.CreateTrunc(v, i32);
  }
  // a control descriptor's field with its dynamic slots applied (the fetch unit's rule)
  llvm::Value *dynField(Operation *op, size_t row, uint32_t field, uint64_t value, uint64_t mask, ValueRange dyn) {
    llvm::Value *v = c32(value & mask);
    uint32_t hdr = uint32_t(word(row, 0));
    for (unsigned s = 0; s < 2; ++s) {
      uint32_t slot = (hdr >> (16 + 8 * s)) & 0xFF;
      if ((slot & 0xF) != field) continue;
      if (s >= dyn.size()) {
        fail(op, "a dynamic slot without its PARAM");
        return v;
      }
      llvm::Value *p = load(op, dyn[s]);
      v = b.CreateAnd((slot & 0x80) ? b.CreateAdd(v, p) : p, c32(mask));
    }
    return v;
  }
  bool takeRow(Operation *op, size_t &row) {
    if (cursor >= rows) return fail(op, "more operations than descriptors");
    row = cursor++;
    return true;
  }
  llvm::Function *prim(const char *name, unsigned nargs) {
    auto it = pcpi.find(name);
    if (it != pcpi.end()) return it->second;
    SmallVector<llvm::Type *> args(nargs, i32);
    auto *f = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), args, false),
                                     llvm::GlobalValue::ExternalLinkage, std::string("sa_rv_") + name, mod);
    pcpi[name] = f;
    return f;
  }
  // .insn r CUSTOM_0 (0x0B), funct3, funct7 (firmware/include/sysarray_intrinsics.h)
  void insnCall(unsigned f3, unsigned f7, llvm::Value *rs1, llvm::Value *rs2, bool memory) {
    auto *fty = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), {i32, i32}, false);
    std::string text = ".insn r 0x0B, " + std::to_string(f3) + ", " + std::to_string(f7) + ", x0, $0, $1";
    b.CreateCall(llvm::InlineAsm::get(fty, text, memory ? "r,r,~{memory}" : "r,r", /*hasSideEffects=*/true),
                 {rs1, rs2});
  }
  void fenceInsn(llvm::Value *mask) {
    auto *fty = llvm::FunctionType::get(i32, {i32}, false);
    b.CreateCall(llvm::InlineAsm::get(fty, ".insn r 0x0B, 4, 1, $0, $1, x0", "=r,r,~{memory}", true), {mask});
  }
  void call(const char *name, std::initializer_list<llvm::Value *> args) {
    if (!insn) {
      b.CreateCall(prim(name, unsigned(args.size())), args);
      return;
    }
    std::vector<llvm::Value *> a(args);
    std::string n(name);
    if (n == "mat_cfg") insnCall(0, 1, a[0], a[1], false);
    else if (n == "mat_load") insnCall(1, 1, a[0], a[1], true);
    else if (n == "mat_store") insnCall(2, 1, a[0], a[1], true);
    else if (n == "mat_exec")                    // rs1 = b << 16 | a, rs2 = acc << 28 | kt << 16 | c (as mat_exec)
      insnCall(3, 1, b.CreateOr(b.CreateShl(a[1], c32(16)), b.CreateAnd(a[0], c32(0xFFFF))),
               b.CreateOr(b.CreateOr(b.CreateShl(a[4], c32(28)), b.CreateShl(b.CreateAnd(a[3], c32(0xFFF)), c32(16))),
                          b.CreateAnd(a[2], c32(0xFFFF))),
               true);
    else if (n == "mat_fence") fenceInsn(a[0]);
    else if (n == "vec_cfg") insnCall(0, 2, a[0], a[1], false);
    else if (n == "vec_run") insnCall(1, 2, a[0], a[1], true);
    else if (n == "gemv_cfg") insnCall(0, 3, a[0], a[1], false);
    else if (n == "gemv_run")                    // rs2 = acc << 28 | (strips - 1) << 16 | c (as gemv_run)
      insnCall(1, 3, a[0],
               b.CreateOr(b.CreateOr(b.CreateShl(a[3], c32(28)),
                                     b.CreateShl(b.CreateAnd(b.CreateSub(a[2], c32(1)), c32(7)), c32(16))),
                          b.CreateAnd(a[1], c32(0xFFFF))),
               true);
    else fail(nullptr, "no instruction for " + n);
  }
  // LDPARAM: the word at a DDR address once the engines are idle (it may have just been stored)
  llvm::Value *ldparam(llvm::Value *addr) {
    if (!insn) {
      if (!ldparamFn)
        ldparamFn = llvm::Function::Create(llvm::FunctionType::get(i32, {i32}, false),
                                           llvm::GlobalValue::ExternalLinkage, "sa_rv_ldparam", mod);
      return b.CreateCall(ldparamFn, {addr});
    }
    fenceInsn(c32(0));
    auto *ld = b.CreateLoad(i32, b.CreateIntToPtr(addr, ptr));
    ld->setVolatile(true);
    return ld;
  }
  // a configuration key (table 0 mat_cfg, 1 vec_cfg, 2 gemv_cfg), unless the same constant is there
  void cfg(unsigned table, uint32_t key, llvm::Value *v) {
    static const char *names[3] = {"mat_cfg", "vec_cfg", "gemv_cfg"};
    if (auto *c = llvm::dyn_cast<llvm::ConstantInt>(v)) {
      auto it = known[table].find(key);
      if (it != known[table].end() && it->second == c->getZExtValue()) return;
      known[table][key] = c->getZExtValue();
    } else {
      known[table].erase(key);
    }
    call(names[table], {c32(key), v});
  }
  // bits [lo, lo + n) of a descriptor word (i64) as i32
  llvm::Value *bits(llvm::Value *w, unsigned lo, unsigned n) {
    uint64_t mask = n >= 64 ? ~0ull : ((1ull << n) - 1);
    return b.CreateTrunc(b.CreateAnd(b.CreateLShr(w, c64(lo)), c64(int64_t(mask))), i32);
  }
  // dynamic fields: field -> (word, lsb, width) per opcode (llm/sa_funcsim.py DYN_FIELDS)
  static bool dynField(uint32_t op, uint32_t f, unsigned &wd, unsigned &lsb, unsigned &width) {
    static const uint8_t DMA[6][3] = {{0, 0, 0}, {1, 0, 32}, {2, 0, 16}, {2, 32, 16}, {2, 48, 16}, {3, 0, 32}};
    static const uint8_t EX[8][3] = {{0, 0, 0}, {1, 0, 16}, {1, 16, 16}, {1, 32, 16}, {1, 48, 12},
                                     {2, 0, 12}, {2, 16, 16}, {2, 32, 16}};
    static const uint8_t VE[12][3] = {{0, 0, 0}, {1, 0, 16}, {1, 32, 16}, {2, 0, 16}, {2, 32, 32}, {7, 16, 16},
                                      {7, 0, 16}, {7, 32, 16}, {3, 16, 16}, {6, 0, 32}, {6, 32, 32}, {5, 32, 32}};
    const uint8_t(*t)[3] = op == 1 || op == 2 ? DMA : op == 3 ? EX : op == 4 ? VE : nullptr;
    unsigned n = op == 1 || op == 2 ? 6 : op == 3 ? 8 : op == 4 ? 12 : 0;
    if (!t || f == 0 || f >= n) return false;
    wd = t[f][0];
    lsb = t[f][1];
    width = t[f][2];
    return true;
  }
  bool command(Operation *op, Value base, ValueRange dyn) {
    size_t row;
    if (!takeRow(op, row)) return false;
    if (dyn.size() > 2) return fail(op, "more than two dynamic fields");
    uint32_t hdr = uint32_t(word(row, 0)), opc = hdr & 0xFF;
    llvm::Value *w[8];
    for (int i = 0; i < 8; ++i) w[i] = c64(int64_t(word(row, i)));
    for (unsigned s = 0; s < 2; ++s) {           // the dynamic fields (the fetch unit's rule)
      uint32_t slot = (hdr >> (16 + 8 * s)) & 0xFF, f = slot & 0xF;
      if (!f) continue;
      unsigned wd, lsb, width;
      if (!dynField(opc, f, wd, lsb, width)) return fail(op, "a dynamic field undefined for its command");
      if (s >= dyn.size()) return fail(op, "a dynamic slot without its PARAM");
      uint64_t mask = width >= 64 ? ~0ull : ((1ull << width) - 1);
      llvm::Value *p = b.CreateZExt(load(op, dyn[s]), i64);
      llvm::Value *old = b.CreateAnd(b.CreateLShr(w[wd], c64(lsb)), c64(int64_t(mask)));
      llvm::Value *nv = b.CreateAnd((slot & 0x80) ? b.CreateAdd(old, p) : p, c64(int64_t(mask)));
      w[wd] = b.CreateOr(b.CreateAnd(w[wd], c64(int64_t(~(mask << lsb)))), b.CreateShl(nv, c64(lsb)));
    }
    if (opc != 0x10 && ((hdr >> 11) & 1)) call("mat_fence", {c32(0)});   // FENCE_BEFORE
    llvm::Value *ddr = bits(w[1], 0, 32);
    if ((hdr >> 8) & 1) {
      if (!base) return fail(op, "a relocated command without its BASE");
      ddr = b.CreateAdd(ddr, load(op, base));
    }
    auto isGemv = [&] {
      auto *c = llvm::dyn_cast<llvm::ConstantInt>(bits(w[3], 32, 2));
      return c && c->getZExtValue() == 2;
    };
    if (opc == 0x01 && isGemv()) {
      if (((word(row, 2) >> 28) & 0xF) != 3) return fail(op, "a GEMV result outside ACC (gemv_run takes ACC words)");
      cfg(2, 0, bits(w[2], 48, 16));            // SA_GCFG_KBYTES
      cfg(2, 1, bits(w[3], 0, 32));             // SA_GCFG_PITCH
      cfg(2, 2, bits(w[3], 34, 16));            // SA_GCFG_XWORD
      cfg(2, 3, bits(w[3], 50, 8));             // SA_GCFG_CSTEP
      call("gemv_run", {ddr, b.CreateAnd(bits(w[2], 0, 32), c32(0xFFFF)), bits(w[2], 32, 16), bits(w[3], 58, 1)});
    } else if (opc == 0x01) {
      if (!llvm::isa<llvm::ConstantInt>(bits(w[3], 32, 2))) return fail(op, "a dynamic LD mode");
      cfg(0, 0, bits(w[2], 32, 16));            // SA_CFG_LD_ROWS
      cfg(0, 1, bits(w[2], 48, 16));            // SA_CFG_LD_ROW_BYTES
      cfg(0, 2, bits(w[3], 0, 32));             // SA_CFG_LD_PITCH
      cfg(0, 3, bits(w[3], 32, 2));             // SA_CFG_LD_MODE
      call("mat_load", {ddr, bits(w[2], 0, 32)});
    } else if (opc == 0x02) {
      cfg(0, 4, bits(w[2], 32, 16));            // SA_CFG_ST_ROWS
      cfg(0, 5, bits(w[2], 48, 16));            // SA_CFG_ST_ROW_BYTES
      cfg(0, 6, bits(w[3], 0, 32));             // SA_CFG_ST_PITCH
      call("mat_store", {ddr, bits(w[2], 0, 32)});
    } else if (opc == 0x03) {
      cfg(0, 7, bits(w[2], 0, 12));             // SA_CFG_EX_REPEAT
      cfg(0, 8, bits(w[2], 16, 16));            // SA_CFG_EX_B_STEP
      cfg(0, 9, bits(w[2], 32, 16));            // SA_CFG_EX_C_STEP
      cfg(0, 10, bits(w[2], 48, 16));           // SA_CFG_EX_C_ROW
      call("mat_exec", {bits(w[1], 0, 16), bits(w[1], 16, 16), bits(w[1], 32, 16), bits(w[1], 48, 12),
                        bits(w[1], 60, 1)});
    } else if (opc == 0x04) {
      static const unsigned K[16][3] = {{3, 0, 8},   {2, 32, 32}, {2, 0, 32},  {3, 8, 6},  {3, 16, 16}, {3, 32, 16},
                                        {3, 48, 5},  {4, 0, 32},  {4, 32, 32}, {5, 0, 32}, {3, 53, 11}, {5, 32, 32},
                                        {6, 0, 32},  {6, 32, 32}, {7, 0, 32},  {7, 32, 32}};
      for (unsigned k = 0; k < 16; ++k) cfg(1, k, bits(w[K[k][0]], K[k][1], K[k][2]));   // SA_VCFG_*
      call("vec_run", {bits(w[1], 0, 32), bits(w[1], 32, 32)});
    } else if (opc == 0x10) {
      call("mat_fence", {bits(w[1], 0, 4)});
    } else {
      return fail(op, "an unexpected command");
    }
    return err.empty();
  }

  bool emitBlock(Block &blk) {
    for (Operation &opRef : blk) {
      Operation *op = &opRef;
      if (auto o = dyn_cast<sahw::LoopOp>(op)) {
        llvm::AllocaInst *iSlot = newSlot("loop.i"), *cntSlot = newSlot("loop.count");
        b.CreateStore(c32(0), iSlot);
        b.CreateStore(c32(0), cntSlot);
        auto *body = llvm::BasicBlock::Create(ctx, "loop", fn);
        auto *exit = llvm::BasicBlock::Create(ctx, "loop.end", fn);
        b.CreateBr(body);
        b.SetInsertPoint(body);
        for (auto &k : known) k.clear();          // (reached from before and from the body's end)
        if (!emitBlock(o.getRegion().front())) return false;
        size_t row;
        if (!takeRow(op, row)) return false;
        if ((word(row, 0) & 0xFF) != OP_LOOP_END) return fail(op, "LOOP_END expected");
        llvm::Value *i = b.CreateLoad(i32, iSlot);
        llvm::Value *count = dynField(op, row, 1, uint64_t(o.getCount()), 0xFFFF, o.getDyn());
        llvm::Value *cnt = b.CreateSelect(b.CreateICmpEQ(i, c32(0)), count, b.CreateLoad(i32, cntSlot));
        b.CreateStore(cnt, cntSlot);
        store(op, o.getK1(), b.CreateAdd(load(op, o.getK1()), c32(uint64_t(o.getS1()))));
        store(op, o.getK2(), b.CreateAdd(load(op, o.getK2()), c32(uint64_t(o.getS2()))));
        llvm::Value *next = b.CreateAdd(i, c32(1));
        b.CreateStore(next, iSlot);
        b.CreateCondBr(b.CreateICmpULT(next, cnt), body, exit);
        b.SetInsertPoint(exit);
      } else if (auto o = dyn_cast<sahw::SetRegOp>(op)) {
        size_t row;
        if (!takeRow(op, row)) return false;
        auto vals = o.getValues();
        if (vals.size() != o.getTargets().size()) return fail(op, "SETREG values and targets differ");
        // the values with their dynamic fields first, then the assignments
        SmallVector<llvm::Value *> vs;
        for (size_t k = 0; k < vals.size(); ++k)
          vs.push_back(dynField(op, row, uint32_t(k + 1), uint64_t(vals[k]), 0xFFFFFFFFull, o.getDyn()));
        for (size_t k = 0; k < vals.size(); ++k) {
          Value t = o.getTargets()[k];
          store(op, t, ((o.getAddMask() >> k) & 1) ? b.CreateAdd(load(op, t), vs[k]) : vs[k]);
        }
      } else if (auto o = dyn_cast<sahw::LdParamOp>(op)) {
        size_t row;
        if (!takeRow(op, row)) return false;
        llvm::Value *a = dynField(op, row, 1, uint64_t(o.getAddr()), 0xFFFFFFFFull, o.getDyn());
        if (o.getBase()) a = b.CreateAdd(a, load(op, o.getBase()));
        llvm::Value *v = ldparam(a);
        v = b.CreateAdd(b.CreateMul(v, c32(uint64_t(o.getMul()) & 0xFFFF)), c32(uint64_t(o.getAdd())));
        store(op, o.getTarget(), v);
      } else if (auto o = dyn_cast<sahw::LdOp>(op)) {
        if (!command(op, o.getBase(), o.getDyn())) return false;
      } else if (auto o = dyn_cast<sahw::StOp>(op)) {
        if (!command(op, o.getBase(), o.getDyn())) return false;
      } else if (auto o = dyn_cast<sahw::ExOp>(op)) {
        if (!command(op, Value(), o.getDyn())) return false;
      } else if (auto o = dyn_cast<sahw::VeOp>(op)) {
        if (!command(op, Value(), o.getDyn())) return false;
      } else if (auto o = dyn_cast<sahw::TransposeOp>(op)) {
        if (!command(op, Value(), o.getDyn())) return false;
      } else if (isa<sahw::FenceOp>(op)) {
        if (!command(op, Value(), ValueRange())) return false;
      } else {
        return fail(op, "unexpected operation " + op->getName().getStringRef().str());
      }
      if (!err.empty()) return false;
    }
    return true;
  }
  // the RET closing a section
  bool skipRet(Operation *op) {
    if (cursor >= rows || (word(cursor, 0) & 0xFF) != OP_RET) return fail(op, "RET expected after a section");
    ++cursor;
    return true;
  }
};

}  // namespace

std::string riscvSymbol(StringRef exportName) {
  std::string s;
  for (char c : exportName) s.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
  return s;
}

namespace {
// The export's function into the emitter's module.
LogicalResult buildFunction(RvEmitter &e, sahw::TemplateOp t) {
  if (t.getUnsupported()) return t.emitError("sahw to riscv: an unsupported export");
  auto *voidTy = llvm::Type::getVoidTy(e.ctx);
  e.fn = llvm::Function::Create(llvm::FunctionType::get(voidTy, {e.ptr, e.ptr}, false),
                                llvm::GlobalValue::ExternalLinkage, riscvSymbol(t.getExportName()), e.mod);
  e.fn->addFnAttr(llvm::Attribute::NoUnwind);      // (no unwind tables: the code is a flat image)
  e.fn->addFnAttr("no-jump-tables", "true");        // (position independent, no data)
  e.bindArg = e.fn->getArg(0);
  e.pcArg = e.fn->getArg(1);
  e.bindArg->setName("bindings");
  e.pcArg->setName("push_constants");
  e.entry = llvm::BasicBlock::Create(e.ctx, "entry", e.fn);
  e.b.SetInsertPoint(e.entry);
  sahw::PrefixOp prefix;
  sahw::HeadOp head;
  sahw::BodyOp body;
  for (Operation &op : t.getRegion().front()) {
    if (auto o = dyn_cast<sahw::BaseOp>(op)) {
      e.slots[o.getResult()] = e.newSlot("base");
      llvm::Value *a = e.b.CreateLoad(e.i32, e.b.CreateConstGEP1_32(e.i32, e.bindArg, unsigned(o.getBinding())));
      e.b.CreateStore(e.b.CreateAdd(a, e.lin(o.getConstant(), o.getMul(), o.getShift(), o.getAdd())),
                      e.slots[o.getResult()]);
    } else if (auto o = dyn_cast<sahw::ParamOp>(op)) {
      e.slots[o.getResult()] = e.newSlot("param");
      e.b.CreateStore(e.lin(o.getConstant(), o.getMul(), o.getShift(), o.getAdd()), e.slots[o.getResult()]);
    } else if (auto o = dyn_cast<sahw::PrivateOp>(op)) {
      e.slots[o.getResult()] = e.newSlot("private");
      e.b.CreateStore(e.c32(0), e.slots[o.getResult()]);
    } else if (auto o = dyn_cast<sahw::PrefixOp>(op)) {
      prefix = o;
    } else if (auto o = dyn_cast<sahw::HeadOp>(op)) {
      head = o;
    } else if (auto o = dyn_cast<sahw::BodyOp>(op)) {
      body = o;
    }
  }
  if (!body) return t.emitError("sahw to riscv: a template without a body");
  for (Operation *sec : {prefix.getOperation(), head.getOperation(), body.getOperation()}) {
    if (!sec) continue;
    if (!e.emitBlock(sec->getRegion(0).front()) || !e.skipRet(sec)) return failure();
  }
  if (e.cursor != e.rows) return t.emitError("sahw to riscv: descriptors left over");
  e.b.CreateRetVoid();
  return success();
}
}  // namespace

LogicalResult emitRiscvLL(sahw::TemplateOp t, const SerializedExport &se, std::string &ll) {
  llvm::LLVMContext ctx;
  llvm::Module mod(t.getExportName().str(), ctx);
  RvEmitter e(ctx, mod, /*insn=*/false, se.templ);
  if (failed(buildFunction(e, t))) return failure();
  std::string verr;
  llvm::raw_string_ostream vos(verr);
  if (llvm::verifyModule(mod, &vos)) return t.emitError("sahw to riscv: invalid LLVM IR: ") << verr;
  ll.clear();
  llvm::raw_string_ostream os(ll);
  os << "; " << t.getExportName() << ": the sa export's commands over PCPI (docs/riscv_backend_design.md §9)\n";
  mod.print(os, nullptr);
  return success();
}

#ifdef SA_HAVE_RISCV
// S4: the exports of one executable compiled in-process for rv32imc (the PCPI
// instructions as inline assembly, -O2, medany, no linker relaxation, no jump
// tables, one section per function); each function's code taken out of the
// object as it is, which requires it to have no relocations and no data (the
// specialized code: immediates, PC-relative branches only).
LogicalResult compileRiscv(ArrayRef<sahw::TemplateOp> ts, ArrayRef<const SerializedExport *> ses,
                           std::vector<std::string> &code, Operation *where) {
  static std::once_flag once;
  std::call_once(once, [] {
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
    LLVMInitializeRISCVAsmPrinter();
    LLVMInitializeRISCVAsmParser();
  });
  llvm::Triple triple("riscv32-unknown-elf");
  std::string terr;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, terr);
  if (!target) return where->emitError("sa riscv: ") << terr;
  llvm::TargetOptions opts;
  opts.FunctionSections = true;
  std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
      triple, "generic-rv32", "+m,+c,-relax", opts, llvm::Reloc::Static, llvm::CodeModel::Medium,
      llvm::CodeGenOptLevel::Default));
  if (!tm) return where->emitError("sa riscv: no target machine");
  llvm::LLVMContext ctx;
  llvm::Module mod("sa_riscv", ctx);
  mod.setTargetTriple(triple);
  mod.setDataLayout(tm->createDataLayout());
  for (auto [t, se] : llvm::zip(ts, ses)) {
    if (sahw::TemplateOp(t).getUnsupported()) continue;   // (on the host: no code, the table's 0 / 0)
    RvEmitter e(ctx, mod, /*insn=*/true, se->templ);
    if (failed(buildFunction(e, t))) return failure();
  }
  std::string verr;
  llvm::raw_string_ostream vos(verr);
  if (llvm::verifyModule(mod, &vos)) return where->emitError("sa riscv: invalid LLVM IR: ") << verr;
  {                                                 // -O2 (the slots promoted, the loops simplified)
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
    llvm::PassBuilder pb(tm.get());
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2).run(mod, mam);
  }
  llvm::SmallVector<char, 0> obj;
  {
    llvm::raw_svector_ostream os(obj);
    llvm::legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, os, nullptr, llvm::CodeGenFileType::ObjectFile))
      return where->emitError("sa riscv: cannot emit an object");
    pm.run(mod);
  }
  auto file = llvm::object::ObjectFile::createObjectFile(
      llvm::MemoryBufferRef(llvm::StringRef(obj.data(), obj.size()), "sa_riscv.o"));
  if (!file) return where->emitError("sa riscv: ") << llvm::toString(file.takeError());
  llvm::StringMap<std::string> byName;
  for (const llvm::object::SectionRef &sec : (*file)->sections()) {
    auto name = sec.getName();
    if (!name) continue;
    if (sec.relocations().begin() != sec.relocations().end()) {
      auto target = sec.getRelocatedSection();
      std::string what = target && *target != (*file)->section_end() ? std::string(*(*target)->getName()) : "?";
      return where->emitError("sa riscv: relocations in ") << what << " (the code is taken without linking)";
    }
    bool data = name->starts_with(".rodata") || name->starts_with(".srodata") || name->starts_with(".data") ||
                name->starts_with(".sdata") || name->starts_with(".bss") || name->starts_with(".sbss");
    if (data && sec.getSize()) return where->emitError("sa riscv: data in ") << *name;
    if (name->starts_with(".text.")) {
      auto contents = sec.getContents();
      if (!contents) return where->emitError("sa riscv: ") << llvm::toString(contents.takeError());
      byName[name->drop_front(6)] = contents->str();
    }
  }
  code.clear();
  for (auto t : ts) {
    if (t.getUnsupported()) {
      code.emplace_back();
      continue;
    }
    auto it = byName.find(riscvSymbol(t.getExportName()));
    if (it == byName.end()) return t.emitError("sa riscv: no code for this export in the object");
    code.push_back(it->second);
  }
  return success();
}
#else
LogicalResult compileRiscv(ArrayRef<sahw::TemplateOp>, ArrayRef<const SerializedExport *>, std::vector<std::string> &,
                           Operation *where) {
  return where->emitError("sa riscv: iree-compile was built without LLVM's RISCV target "
                          "(compiler/scripts/build_iree_compiler.sh with SA_WITH_RISCV=1)");
}
#endif

}  // namespace mlir::iree_compiler::sa
