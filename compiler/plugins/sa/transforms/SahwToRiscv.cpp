// sahw.template -> LLVM IR of a function that issues the export's commands
// over PCPI from the PicoRV32 (docs/riscv_backend_design.md §9: plan A's code
// generator, S1). One function per export:
//   void <export>(const uint32_t *bindings, const uint32_t *push_constants)
// - registers become stack slots (promoted by the optimizer): sahw.base =
//   binding address + f(push constant), sahw.param = f(push constant) with
//   f(c) = add + ((c * mul) >> shift) in 64-bit arithmetic (sa_loader.c),
//   sahw.private = 0;
// - each command is one call sa_rv_cmd(descriptor, base, p0, p1): the 64-byte
//   descriptor exactly as the serializer encodes it (a constant), the value of
//   its BASE register and of its dynamic-field PARAMs (slot 0, slot 1); the
//   runtime (firmware/rv/sa_rv_rt.c) applies them as the fetch unit does and
//   issues the command over PCPI;
// - control flow is compiled: LOOP_END a loop (the count, possibly from a
//   PARAM, taken when it is first reached; k1 += s1, k2 += s2 after every
//   pass), SETREG assignments, LDPARAM a call sa_rv_ldparam(address) (a fence,
//   then a load) times mul plus add.
// The template's prefix, head and body run in that order. The IR has no
// target triple or data layout (i32 / i64 values and the two pointers only):
// the same file compiles for riscv32 and for the host's checks.
#include <cstring>

#include "SahwPasses.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::iree_compiler::sa {
namespace {

constexpr uint32_t OP_LOOP_END = 0x13, OP_RET = 0x16;

struct RvEmitter {
  llvm::LLVMContext ctx;
  std::unique_ptr<llvm::Module> mod;
  llvm::IRBuilder<> b{ctx};
  llvm::Function *fn = nullptr, *cmdFn = nullptr, *ldparamFn = nullptr;
  llvm::Type *i32 = nullptr, *i64 = nullptr, *ptr = nullptr;
  llvm::Value *bindArg = nullptr, *pcArg = nullptr;
  llvm::BasicBlock *entry = nullptr;
  llvm::DenseMap<Value, llvm::AllocaInst *> slots;
  const std::string &templ;
  size_t cursor = 0, rows = 0;
  unsigned ndesc = 0;
  std::string err;

  explicit RvEmitter(const std::string &t) : templ(t), rows(t.size() / 64) {}

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
  // the descriptor of row `row` as a constant [16 x i32]
  llvm::Constant *descriptor(size_t row) {
    std::vector<uint32_t> w(16);
    std::memcpy(w.data(), templ.data() + 64 * row, 64);
    auto *ty = llvm::ArrayType::get(i32, 16);
    auto *init = llvm::ConstantDataArray::get(ctx, llvm::ArrayRef<uint32_t>(w));
    auto *g = new llvm::GlobalVariable(*mod, ty, /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage, init,
                                       "desc" + std::to_string(ndesc++));
    g->setAlignment(llvm::Align(8));
    return g;
  }
  bool takeRow(Operation *op, size_t &row) {
    if (cursor >= rows) return fail(op, "more operations than descriptors");
    row = cursor++;
    return true;
  }
  bool command(Operation *op, Value base, ValueRange dyn) {
    size_t row;
    if (!takeRow(op, row)) return false;
    if (dyn.size() > 2) return fail(op, "more than two dynamic fields");
    llvm::Value *bv = base ? load(op, base) : c32(0);
    llvm::Value *p0 = dyn.size() > 0 ? load(op, dyn[0]) : c32(0);
    llvm::Value *p1 = dyn.size() > 1 ? load(op, dyn[1]) : c32(0);
    b.CreateCall(cmdFn, {descriptor(row), bv, p0, p1});
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
        llvm::Value *v = b.CreateCall(ldparamFn, {a});
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

LogicalResult emitRiscvLL(sahw::TemplateOp t, const SerializedExport &se, std::string &ll) {
  if (t.getUnsupported()) return t.emitError("sahw to riscv: an unsupported export");
  RvEmitter e(se.templ);
  e.mod = std::make_unique<llvm::Module>(t.getExportName().str(), e.ctx);
  e.i32 = llvm::Type::getInt32Ty(e.ctx);
  e.i64 = llvm::Type::getInt64Ty(e.ctx);
  e.ptr = llvm::PointerType::getUnqual(e.ctx);
  auto *voidTy = llvm::Type::getVoidTy(e.ctx);
  e.cmdFn = llvm::Function::Create(llvm::FunctionType::get(voidTy, {e.ptr, e.i32, e.i32, e.i32}, false),
                                   llvm::GlobalValue::ExternalLinkage, "sa_rv_cmd", *e.mod);
  e.ldparamFn = llvm::Function::Create(llvm::FunctionType::get(e.i32, {e.i32}, false),
                                       llvm::GlobalValue::ExternalLinkage, "sa_rv_ldparam", *e.mod);
  e.fn = llvm::Function::Create(llvm::FunctionType::get(voidTy, {e.ptr, e.ptr}, false),
                                llvm::GlobalValue::ExternalLinkage, riscvSymbol(t.getExportName()), *e.mod);
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
  std::string verr;
  llvm::raw_string_ostream vos(verr);
  if (llvm::verifyModule(*e.mod, &vos)) return t.emitError("sahw to riscv: invalid LLVM IR: ") << verr;
  ll.clear();
  llvm::raw_string_ostream os(ll);
  os << "; " << t.getExportName() << ": the sa export's commands over PCPI (docs/riscv_backend_design.md §9)\n";
  e.mod->print(os, nullptr);
  return success();
}

}  // namespace mlir::iree_compiler::sa
